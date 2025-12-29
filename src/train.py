"""
Self-play training for Sequence AI.
Generates training data through self-play and trains the neural network.
Optimized for GPU utilization using Multiprocessing and Batched Inference.
"""
import argparse
import os
import time
import warnings
from pathlib import Path
from collections import deque
from typing import List, Tuple, Optional, Dict
import random
import torch.multiprocessing as mp
import queue # For queue.Empty exception
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter

# Suppress the writable buffer warnings from C-extensions
warnings.filterwarnings("ignore", message="The given buffer is not writable")
warnings.filterwarnings("ignore", message="The given NumPy array is not writable")

from .fast_game import FastSequenceGame as SequenceGame
from .game import Move
from .cards import Card
from .model import SequenceNet, create_model, save_model, load_model
from .mcts import MCTS, RandomPlayer


class ThreadedRemoteModelProxy:
    """
    Proxy that multiplexes thread requests onto a single process queue.
    """
    def __init__(self, worker_id: int, request_queue: mp.Queue, result_queue: mp.Queue):
        self.worker_id = worker_id
        self.request_queue = request_queue
        self.result_queue = result_queue
        self.thread_queues: Dict[int, queue.Queue] = {}
        self.lock = threading.Lock()
        
    def start_dispatcher(self):
        """Starts a background thread to route results to the correct thread."""
        t = threading.Thread(target=self._dispatch_loop, daemon=True)
        t.start()
        
    def _dispatch_loop(self):
        while True:
            try:
                # Result from server: (thread_id, policy, value)
                data = self.result_queue.get()
                thread_id, policy, value = data
                
                with self.lock:
                    if thread_id in self.thread_queues:
                        self.thread_queues[thread_id].put((policy, value))
            except Exception as e:
                print(f"Dispatcher error: {e}")
                break

    def get_thread_queue(self, thread_id: int) -> queue.Queue:
        with self.lock:
            if thread_id not in self.thread_queues:
                self.thread_queues[thread_id] = queue.Queue()
            return self.thread_queues[thread_id]

    def predict_batch(self, states: List[torch.Tensor], thread_id: int) -> Tuple[torch.Tensor, torch.Tensor]:
        q = self.get_thread_queue(thread_id)
        
        # Convert to numpy for IPC to avoid torch shared memory file descriptor exhaustion
        # shape: (B, 8, 10, 10)
        batch_np = torch.stack(states).numpy()
        
        # Send request: (worker_id, thread_id, batch_numpy)
        self.request_queue.put((self.worker_id, thread_id, batch_np))
        
        # Wait for result (comes back as numpy, convert to torch)
        policy_np, value_np = q.get()
        return torch.from_numpy(policy_np), torch.from_numpy(value_np)


class ThreadAwareModel:
    """Wrapper for MCTS to call predict with implicit thread ID."""
    def __init__(self, proxy: ThreadedRemoteModelProxy, thread_id: int):
        self.proxy = proxy
        self.thread_id = thread_id
        
    def eval(self): pass
    
    def predict_batch(self, states: List[torch.Tensor]):
        return self.proxy.predict_batch(states, self.thread_id)


def run_game_loop(worker_id: int, thread_id: int, proxy: ThreadedRemoteModelProxy, 
                  game_queue: mp.Queue, simulations: int, batch_size: int = 8):
    """
    Single thread game loop running MULTIPLE games in parallel (Micro-Batching).
    This hides IPC latency by sending larger batches to the GPU.
    """
    import c_sequence
    
    # Unique seed for this thread
    seed = worker_id * 100000 + thread_id * 1000 + int(time.time())
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # Model wrapper
    model = ThreadAwareModel(proxy, thread_id)
    
    # Initialize parallel environments
    games = [SequenceGame() for _ in range(batch_size)]
    mcts_list = [c_sequence.CMCTS() for _ in range(batch_size)]
    game_histories = [[] for _ in range(batch_size)]
    move_counts = [0] * batch_size
    
    for g in games: g.reset()
    
    # State tracking for MCTS loop
    # We need to know if a game is currently "thinking" (running simulations) 
    # or if it's ready to make a move.
    
    # Since we want to batch everything, we align them:
    # All games will run 'simulations' steps. 
    # If a game finishes early (or was just reset), it waits or starts next.
    # To keep it simple: We run 'simulations' passes. In each pass, we advance all games.
    
    sim_counts = [0] * batch_size
    
    # Pre-initialize MCTS roots for new games
    for i in range(batch_size):
        board_bytes = games[i].c_state.get_tensor()
        hand_ids = [c.to_int() for c in games[i].hands[games[i].current_player]]
        mcts_list[i].reset(board_bytes, hand_ids, games[i].current_player)
    
    while True:
        try:
            # --- MCTS SIMULATION PHASE ---
            # We assume all games need to run simulations now.
            # We run until all games have reached 'simulations' count.
            
            while True:
                # Identify games that still need simulations
                active_indices = [i for i in range(batch_size) if sim_counts[i] < simulations]
                if not active_indices:
                    break
                
                requests = []
                req_indices = []
                leaves = []
                
                # 1. Select Leaf for all active games
                for i in active_indices:
                    leaf_node, tensor_bytes, is_terminal, value = mcts_list[i].select_leaf()
                    
                    if is_terminal:
                        # Backpropagate immediately (no GPU needed)
                        mcts_list[i].backpropagate(leaf_node, b'', value)
                        sim_counts[i] += 1
                    else:
                        # Queue for inference
                        # Create tensor copy
                        # No clone needed: torch.stack will copy the data
                        state_tensor = torch.frombuffer(tensor_bytes, dtype=torch.float32).reshape(8, 10, 10)
                        requests.append(state_tensor)
                        req_indices.append(i)
                        leaves.append(leaf_node)
                
                # 2. Batch Inference
                if requests:
                    # Returns stacked tensors on CPU
                    policy_batch, value_batch = model.predict_batch(requests)
                    
                    # 3. Backpropagate results
                    # policy_batch: (B, 100), value_batch: (B, 1)
                    policies_np = policy_batch.numpy()
                    values_np = value_batch.numpy()
                    
                    for k, idx in enumerate(req_indices):
                        p_bytes = policies_np[k].astype(np.float32).tobytes()
                        v = values_np[k].item()
                        mcts_list[idx].backpropagate(leaves[k], p_bytes, v)
                        sim_counts[idx] += 1
            
            # --- ACTION PHASE ---
            # All games have finished thinking. Make moves.
            for i in range(batch_size):
                temp = 1.0 if move_counts[i] < 10 else 0.1
                res = mcts_list[i].get_action(temp)
                
                if res is None:
                    # Should not happen unless no moves legal
                    games[i].reset()
                    game_histories[i] = []
                    move_counts[i] = 0
                    
                    # Reset MCTS
                    board_bytes = games[i].c_state.get_tensor()
                    hand_ids = [c.to_int() for c in games[i].hands[games[i].current_player]]
                    mcts_list[i].reset(board_bytes, hand_ids, games[i].current_player)
                    sim_counts[i] = 0
                    continue
                
                card_int, r, c, is_rem, policy_bytes = res
                policy = np.frombuffer(policy_bytes, dtype=np.float32)
                
                # Store history
                state_tensor = games[i].get_state_tensor(games[i].current_player)
                game_histories[i].append((state_tensor, policy, games[i].current_player))
                
                # Apply move
                move = Move(card=Card.from_int(card_int), row=r, col=c, is_removal=(is_rem != 0))
                games[i].make_move(move)
                move_counts[i] += 1
                
                # Check Game Over
                if games[i].game_over or move_counts[i] >= 200:
                    # Save Data
                    training_data = []
                    for state, pi, player in game_histories[i]:
                        value = games[i].get_result(player)
                        training_data.append((state, pi, value))
                    
                    # Blocking put is fine here, happens rarely per game
                    if training_data:
                        game_queue.put(training_data)
                    
                    # Reset Game
                    games[i].reset()
                    game_histories[i] = []
                    move_counts[i] = 0
                
                # Reset MCTS for next state (or new game)
                board_bytes = games[i].c_state.get_tensor()
                hand_ids = [c.to_int() for c in games[i].hands[games[i].current_player]]
                mcts_list[i].reset(board_bytes, hand_ids, games[i].current_player)
                sim_counts[i] = 0

        except Exception as e:
            print(f"Worker {worker_id} Thread {thread_id} error: {e}")
            import traceback
            traceback.print_exc()
            time.sleep(1)

def process_worker(worker_id: int, 
                   request_queue: mp.Queue, 
                   result_queue: mp.Queue, 
                   game_queue: mp.Queue,
                   args_dict: Dict):
    """
    Process worker that spawns threads.
    """
    threads_per_worker = args_dict.get('threads', 4)
    games_per_thread = args_dict.get('games_per_thread', 8)
    simulations = args_dict['simulations']
    
    # Shared proxy for this process
    proxy = ThreadedRemoteModelProxy(worker_id, request_queue, result_queue)
    proxy.start_dispatcher()
    
    # Initialize static layout in this process
    import c_sequence
    from .game import SequenceGame as OriginalSequenceGame
    temp_game = OriginalSequenceGame()
    for card, positions in temp_game.card_to_positions.items():
        for i, pos in enumerate(positions):
            c_sequence.setup_layout(card.to_int(), i, pos[0], pos[1])
    
    with ThreadPoolExecutor(max_workers=threads_per_worker) as executor:
        futures = []
        for i in range(threads_per_worker):
            futures.append(
                executor.submit(run_game_loop, worker_id, i, proxy, game_queue, simulations, games_per_thread)
            )
        
        # Keep process alive
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as e:
                print(f"Thread crashed: {e}")


class InferenceServer:
    """
    Single-threaded inference server with numpy IPC.
    Collects requests, runs inference, distributes results.
    """
    def __init__(self, model: SequenceNet, device: torch.device, 
                 request_queue: mp.Queue, result_queues: Dict[int, mp.Queue],
                 batch_size: int = 256, timeout: float = 0.01,
                 num_threads: int = 4):  # num_threads ignored, kept for API compat
        self.model = model
        self.device = device
        self.request_queue = request_queue
        self.result_queues = result_queues
        self.batch_size = batch_size
        self.timeout = timeout
        self.model.eval()
        
        # Statistics
        self.total_processed = 0
        self.total_batches = 0

    def step(self) -> int:
        """Process one batch of requests. Returns number processed."""
        batch_items = []
        
        # Non-blocking collect
        try:
            item = self.request_queue.get(timeout=0.001)
            batch_items.append(item)
        except queue.Empty:
            return 0

        # Collect more
        start_time = time.time()
        current_samples = batch_items[0][2].shape[0]  # numpy array
        
        while current_samples < self.batch_size:
            try:
                remaining = self.timeout - (time.time() - start_time)
                if remaining <= 0:
                    break
                item = self.request_queue.get(timeout=remaining)
                batch_items.append(item)
                current_samples += item[2].shape[0]
            except queue.Empty:
                break
        
        if not batch_items:
            return 0

        # item is (worker_id, thread_id, numpy_array)
        worker_ids = []
        thread_ids = []
        all_arrays = []
        sizes = []
        
        for item in batch_items:
            w_id, t_id, arr = item
            worker_ids.append(w_id)
            thread_ids.append(t_id)
            all_arrays.append(arr)
            sizes.append(arr.shape[0])
        
        # Mega-batch from numpy
        mega_batch = torch.from_numpy(np.concatenate(all_arrays, axis=0)).to(self.device, non_blocking=True)
        
        with torch.no_grad():
            policies, values = self.model.predict(mega_batch)
            
        policies = policies.cpu().numpy()
        values = values.cpu().numpy()
        
        # Distribute results (as numpy)
        curr = 0
        for i in range(len(batch_items)):
            w_id = worker_ids[i]
            t_id = thread_ids[i]
            sz = sizes[i]
            
            p_slice = policies[curr : curr+sz]
            v_slice = values[curr : curr+sz]
            
            self.result_queues[w_id].put((t_id, p_slice, v_slice))
            curr += sz
        
        self.total_processed += current_samples
        self.total_batches += 1
            
        return current_samples
    
    def get_stats(self) -> Tuple[int, int]:
        """Returns (total_processed, total_batches)."""
        return self.total_processed, self.total_batches
    
    def stop(self):
        """No-op for single-threaded server."""
        pass


def train_step(model: SequenceNet,
               optimizer: optim.Optimizer,
               states: torch.Tensor,
               target_policies: torch.Tensor,
               target_values: torch.Tensor,
               scaler: GradScaler = None,
               device: torch.device = None) -> Tuple[float, float, float]:
    """Perform one training step."""
    model.train()
    optimizer.zero_grad()
    
    device_type = 'cuda' if device and device.type == 'cuda' else 'cpu'
    
    with autocast(device_type=device_type, enabled=(scaler is not None)):
        policy_logits, values = model(states)
        policy_loss = -torch.mean(torch.sum(target_policies * policy_logits, dim=1))
        value_loss = torch.mean((values.squeeze() - target_values) ** 2)
        total_loss = policy_loss + value_loss
    
    if scaler is not None:
        scaler.scale(total_loss).backward()
        scaler.step(optimizer)
        scaler.update()
    else:
        total_loss.backward()
        optimizer.step()
    
    return total_loss.item(), policy_loss.item(), value_loss.item()


def train(args):
    """Main training loop."""
    # Setup multiprocessing
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass

    print("=" * 60)
    print("Sequence AI Training (High-Throughput Micro-Batching)")
    print("=" * 60)
    
    # Create model
    model, device = create_model(compile_model=not args.no_compile)
    print(f"Device: {device}")
    
    # Load checkpoint
    models_dir = Path(__file__).parent.parent / "models"
    models_dir.mkdir(exist_ok=True)
    checkpoint_path = models_dir / "latest.pt"
    
    start_epoch = 0
    if checkpoint_path.exists() and not args.fresh:
        print(f"Loading checkpoint: {checkpoint_path}")
        model, device, checkpoint = load_model(str(checkpoint_path), compile_model=not args.no_compile)
        start_epoch = checkpoint.get('epoch', 0)
        print(f"Resuming from epoch {start_epoch}")
    
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scaler = GradScaler('cuda') if device.type == 'cuda' else None
    
    # Queues
    # request_queue gets bigger items now, but fewer of them
    request_queue = mp.Queue(maxsize=10000)
    game_queue = mp.Queue(maxsize=50000)
    result_queues = {i: mp.Queue() for i in range(args.workers)}
    
    # Start Workers
    print(f"Starting {args.workers} workers...")
    print(f"  Threads per worker: {args.threads}")
    print(f"  Games per thread:   {args.games_per_thread}")
    print(f"  Total concurrent:   {args.workers * args.threads * args.games_per_thread}")
    
    # TensorBoard writer
    writer = SummaryWriter(log_dir=models_dir.parent / "runs" / f"run_{int(time.time())}")
    
    workers = []
    args_dict = {
        'simulations': args.simulations,
        'threads': args.threads,
        'games_per_thread': args.games_per_thread
    }
    
    for i in range(args.workers):
        p = mp.Process(target=process_worker, 
                       args=(i, request_queue, result_queues[i], game_queue, args_dict))
        p.daemon = True
        p.start()
        workers.append(p)
        
    # Inference Server
    server = InferenceServer(
        model, device, request_queue, result_queues,
        batch_size=args.batch_size_inference, 
        timeout=0.01
    )
    
    replay_buffer = deque(maxlen=args.buffer_size)
    
    print(f"\nStarting training for {args.epochs} epochs")
    print("-" * 60)
    
    try:
        for epoch in range(start_epoch, start_epoch + args.epochs):
            epoch_start = time.time()
            games_collected_epoch = 0
            
            print(f"\nEpoch {epoch + 1}: Collecting {args.games_per_epoch} games...")
            
            # Collection Loop
            while games_collected_epoch < args.games_per_epoch:
                # Run inference step
                processed = server.step()
                
                # Check for completed games
                try:
                    while True:
                        game_data = game_queue.get_nowait()
                        replay_buffer.extend(game_data)
                        games_collected_epoch += 1
                        if args.verbose:
                            total_items, total_calls = server.get_stats()
                            avg_batch = total_items / total_calls if total_calls > 0 else 0
                            print(f"  Game {games_collected_epoch}/{args.games_per_epoch} collected | Avg Batch: {avg_batch:.1f}", end='\r')
                except queue.Empty:
                    pass
                
                # Avoid busy wait if idle
                if processed == 0:
                    time.sleep(0.0001)
            
            if args.verbose:
                print()

            print(f"  Buffer size: {len(replay_buffer)}")
            
            # Training Phase
            if len(replay_buffer) >= args.batch_size:
                print(f"Epoch {epoch + 1}: Training...")
                
                # Sample batch
                sample_size = min(len(replay_buffer), args.batch_size * 20)
                # Optimization: Convert buffer to numpy array once if possible, or just sample list
                batch = random.sample(list(replay_buffer), sample_size)
                
                states = torch.stack([torch.from_numpy(s) for s, _, _ in batch]).to(device)
                policies = torch.stack([torch.from_numpy(p) for _, p, _ in batch]).to(device)
                values = torch.tensor([v for _, _, v in batch], dtype=torch.float32).to(device)
                
                total_losses = []
                for _ in range(args.train_steps):
                    indices = torch.randperm(len(states))[:args.batch_size]
                    loss, p_loss, v_loss = train_step(
                        model, optimizer,
                        states[indices], policies[indices], values[indices],
                        scaler=scaler, device=device
                    )
                    total_losses.append(loss)
                
                avg_loss = np.mean(total_losses)
                print(f"  Avg loss: {avg_loss:.4f}")
                
                epoch_time = time.time() - epoch_start
                games_per_sec = args.games_per_epoch / epoch_time
                print(f"  Epoch time: {epoch_time:.1f}s ({games_per_sec:.2f} games/s)")

                # Log to TensorBoard
                writer.add_scalar("Loss/Total", avg_loss, epoch + 1)
                writer.add_scalar("Loss/Policy", p_loss, epoch + 1)
                writer.add_scalar("Loss/Value", v_loss, epoch + 1)
                writer.add_scalar("Perf/GamesPerSec", games_per_sec, epoch + 1)
            
            # Save checkpoint
            model_to_save = model._orig_mod if hasattr(model, '_orig_mod') else model
            save_model(model_to_save, str(checkpoint_path), optimizer, epoch + 1)
            
            if (epoch + 1) % 10 == 0:
                numbered_path = models_dir / f"checkpoint_{epoch + 1:04d}.pt"
                save_model(model_to_save, str(numbered_path), optimizer, epoch + 1)
            
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        # Cleanup
        server.stop()
        for p in workers:
            p.terminate()
        for p in workers:
            p.join()

    print("\nTraining complete!")


def evaluate(args):
    """Evaluate model."""
    print("Evaluating model...")
    # For now, simple evaluation with local model (slow but works)
    # Ideally, reuse the MP architecture
    model, device = create_model()
    
    models_dir = Path(__file__).parent.parent / "models"
    checkpoint_path = models_dir / "latest.pt"
    
    if checkpoint_path.exists():
        model, device, _ = load_model(str(checkpoint_path), compile_model=False)
        print("Loaded trained model")
    
    wins = 0
    losses = 0
    draws = 0
    
    for game_idx in range(args.games):
        game = SequenceGame()
        game.reset()
        
        mcts = MCTS(model, device, num_simulations=args.simulations, temperature=0.1)
        random_player = RandomPlayer()
        
        while not game.game_over:
            if game.current_player == 1:
                move, _ = mcts.search(game, 1)
            else:
                move = random_player.get_move(game, 2)
            
            if move:
                game.make_move(move)
            else:
                break
        
        if game.winner == 1:
            wins += 1
        elif game.winner == 2:
            losses += 1
        else:
            draws += 1
        print(f"Game {game_idx + 1}: {'Win' if game.winner == 1 else 'Loss' if game.winner == 2 else 'Draw'}")
            
    print(f"\nResults: {wins}W / {losses}L / {draws}D")


def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest='command')
    
    train_parser = subparsers.add_parser('train')
    train_parser.add_argument('--epochs', type=int, default=100)
    train_parser.add_argument('--games-per-epoch', type=int, default=20)
    train_parser.add_argument('--simulations', type=int, default=50)
    train_parser.add_argument('--batch-size', type=int, default=64)
    train_parser.add_argument('--batch-size-inference', type=int, default=256)
    train_parser.add_argument('--buffer-size', type=int, default=10000)
    train_parser.add_argument('--train-steps', type=int, default=10)
    train_parser.add_argument('--lr', type=float, default=0.001)
    train_parser.add_argument('--fresh', action='store_true')
    train_parser.add_argument('--no-compile', action='store_true')
    train_parser.add_argument('--verbose', action='store_true')
    train_parser.add_argument('--workers', type=int, default=8)
    train_parser.add_argument('--threads', type=int, default=2, help='Threads per worker')
    train_parser.add_argument('--games-per-thread', type=int, default=8, help='Games per thread (micro-batch)')
    train_parser.add_argument('--inference-threads', type=int, default=4, help='Number of inference threads')
    
    eval_parser = subparsers.add_parser('eval')
    eval_parser.add_argument('--games', type=int, default=10)
    eval_parser.add_argument('--simulations', type=int, default=50)
    
    args = parser.parse_args()
    
    if args.command == 'train':
        train(args)
    elif args.command == 'eval':
        evaluate(args)
    else:
        parser.print_help()

if __name__ == '__main__':
    mp.freeze_support()
    main()