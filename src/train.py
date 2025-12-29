"""
Self-play training for Sequence AI.
Robust multiprocessing architecture for high throughput on multi-core systems.

Architecture:
- N worker processes, each running M games independently
- Central inference server on GPU
- Workers batch locally before sending to server (reduces IPC frequency)
- Per-worker result queues (no dispatcher needed)
"""
import argparse
import os
import time
import warnings
from pathlib import Path
from collections import deque
from typing import List, Tuple, Optional, Dict
import random
import queue

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp
from torch.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter

# Suppress warnings
warnings.filterwarnings("ignore", message="The given buffer is not writable")
warnings.filterwarnings("ignore", message="The given NumPy array is not writable")

from .fast_game import FastSequenceGame as SequenceGame
from .game import Move
from .cards import Card
from .model import SequenceNet, create_model, save_model, load_model
from .mcts import MCTS, RandomPlayer


def worker_process(worker_id: int,
                   request_queue: mp.Queue,
                   result_queue: mp.Queue,
                   game_queue: mp.Queue,
                   ready_event: mp.Event,
                   args_dict: Dict):
    """
    Worker process that runs multiple games independently.
    Batches requests locally before sending to inference server.
    """
    try:
        # Imports inside worker to avoid CUDA forking issues
        import c_sequence
        from .game import SequenceGame as OriginalSequenceGame
        
        num_games = args_dict['games_per_worker']
        simulations = args_dict['simulations']
        
        # Initialize card layout
        temp_game = OriginalSequenceGame()
        for card, positions in temp_game.card_to_positions.items():
            for i, pos in enumerate(positions):
                c_sequence.setup_layout(card.to_int(), i, pos[0], pos[1])
        
        # Initialize games
        games = [SequenceGame() for _ in range(num_games)]
        mcts_list = [c_sequence.CMCTS() for _ in range(num_games)]
        game_histories = [[] for _ in range(num_games)]
        move_counts = [0] * num_games
        
        for g in games:
            g.reset()
        
        def reset_mcts(idx):
            board_bytes = games[idx].c_state.get_tensor()
            hand_ids = [c.to_int() for c in games[idx].hands[games[idx].current_player]]
            mcts_list[idx].reset(board_bytes, hand_ids, games[idx].current_player)
        
        for i in range(num_games):
            reset_mcts(i)
        
        # Signal ready
        ready_event.set()
        
        # Main game loop
        while True:
            # Run simulations for all games until all have enough
            sim_counts = [0] * num_games
            
            # BATCH SIZE for IPC - accumulate this many leaves before sending
            IPC_BATCH_SIZE = 256
            
            # Accumulated leaves waiting for inference
            pending_leaves = []  # (game_idx, leaf_node)
            pending_tensors = []  # numpy arrays
            
            while True:
                # Find games needing more simulations
                active = [i for i in range(num_games) if sim_counts[i] < simulations]
                if not active:
                    # Flush remaining pending leaves
                    if pending_leaves:
                        batch_np = np.stack(pending_tensors)
                        request_queue.put((worker_id, batch_np))
                        policies_np, values_np = result_queue.get()
                        
                        for k, (i, leaf) in enumerate(pending_leaves):
                            p_bytes = policies_np[k].astype(np.float32).tobytes()
                            v = float(values_np[k])
                            mcts_list[i].backpropagate(leaf, p_bytes, v)
                            sim_counts[i] += 1
                        pending_leaves = []
                        pending_tensors = []
                    break
                
                # Select leaf for each active game
                for i in active:
                    leaf, tensor_bytes, is_terminal, value = mcts_list[i].select_leaf()
                    if is_terminal:
                        mcts_list[i].backpropagate(leaf, b'', value)
                        sim_counts[i] += 1
                    else:
                        tensor = np.frombuffer(tensor_bytes, dtype=np.float32).reshape(8, 10, 10)
                        pending_leaves.append((i, leaf))
                        pending_tensors.append(tensor)
                
                # If we have enough pending, do IPC
                if len(pending_leaves) >= IPC_BATCH_SIZE:
                    batch_np = np.stack(pending_tensors)
                    request_queue.put((worker_id, batch_np))
                    policies_np, values_np = result_queue.get()
                    
                    for k, (i, leaf) in enumerate(pending_leaves):
                        p_bytes = policies_np[k].astype(np.float32).tobytes()
                        v = float(values_np[k])
                        mcts_list[i].backpropagate(leaf, p_bytes, v)
                        sim_counts[i] += 1
                    pending_leaves = []
                    pending_tensors = []
            
            # All simulations done - make moves for all games
            for i in range(num_games):
                temp = 1.0 if move_counts[i] < 10 else 0.1
                res = mcts_list[i].get_action(temp)
                
                if res is None:
                    games[i].reset()
                    game_histories[i] = []
                    move_counts[i] = 0
                    reset_mcts(i)
                    continue
                
                card_int, r, c, is_rem, policy_bytes = res
                policy = np.frombuffer(policy_bytes, dtype=np.float32).copy()
                
                # Store state (as numpy to avoid SHM leak)
                state_np = games[i].get_state_numpy(games[i].current_player)
                game_histories[i].append((state_np, policy, games[i].current_player))
                
                # Apply move
                move = Move(card=Card.from_int(card_int), row=r, col=c, is_removal=(is_rem != 0))
                games[i].make_move(move)
                move_counts[i] += 1
                
                # Check game over
                if games[i].game_over or move_counts[i] >= 200:
                    # Package training data and send to main process
                    training_data = []
                    for state, pi, player in game_histories[i]:
                        value = games[i].get_result(player)
                        training_data.append((state, pi, value))
                    
                    if training_data:
                        game_queue.put(training_data)
                    
                    # Reset for next game
                    games[i].reset()
                    game_histories[i] = []
                    move_counts[i] = 0
                
                reset_mcts(i)
                
    except Exception as e:
        import traceback
        print(f"[Worker {worker_id}] ERROR: {e}", flush=True)
        traceback.print_exc()


class InferenceServer:
    """Batches inference requests from multiple workers."""
    
    def __init__(self, model, device, result_queues: Dict[int, mp.Queue],
                 batch_size: int = 512, timeout: float = 0.05):
        self.model = model
        self.device = device
        self.result_queues = result_queues
        self.batch_size = batch_size
        self.timeout = timeout
        self.total_processed = 0
        self.total_batches = 0
    
    def step(self, request_queue: mp.Queue) -> int:
        """Process pending requests. Returns count processed."""
        batch_items = []
        
        # Collect requests with timeout
        try:
            item = request_queue.get(timeout=self.timeout)
            batch_items.append(item)
        except queue.Empty:
            return 0
        
        # Collect more until batch is full or timeout
        start_time = time.time()
        while len(batch_items) < self.batch_size:
            remaining = self.timeout - (time.time() - start_time)
            if remaining <= 0:
                break
            try:
                item = request_queue.get(timeout=min(remaining, 0.005))
                batch_items.append(item)
            except queue.Empty:
                break
        
        if not batch_items:
            return 0
        
        # Build mega-batch
        worker_ids = []
        all_arrays = []
        sizes = []
        
        for (w_id, arr) in batch_items:
            worker_ids.append(w_id)
            all_arrays.append(arr)
            sizes.append(arr.shape[0])
        
        mega_batch = torch.from_numpy(np.concatenate(all_arrays, axis=0)).to(self.device)
        
        # Inference
        with torch.no_grad():
            policies, values = self.model.predict(mega_batch)
        
        policies = policies.cpu().numpy()
        values = values.cpu().numpy()
        
        # Distribute results
        curr = 0
        for i in range(len(batch_items)):
            w_id = worker_ids[i]
            sz = sizes[i]
            
            p_slice = policies[curr:curr+sz]
            v_slice = values[curr:curr+sz].flatten()
            
            self.result_queues[w_id].put((p_slice, v_slice))
            curr += sz
        
        self.total_processed += sum(sizes)
        self.total_batches += 1
        
        return sum(sizes)


def train_step(model, optimizer, states, target_policies, target_values,
               scaler=None, device=None):
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
    """Main training loop with multiprocessing."""
    print("=" * 60)
    print("Sequence AI Training (Multiprocess Architecture)")
    print("=" * 60)
    
    # Use spawn for CUDA compatibility
    mp.set_start_method('spawn', force=True)
    
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
    
    model.share_memory()
    
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scaler = GradScaler('cuda') if device.type == 'cuda' else None
    
    # TensorBoard
    writer = SummaryWriter(log_dir=models_dir.parent / "runs" / f"run_{int(time.time())}")
    
    # Queues
    request_queue = mp.Queue()
    game_queue = mp.Queue()
    result_queues = {i: mp.Queue() for i in range(args.workers)}
    ready_events = [mp.Event() for _ in range(args.workers)]
    
    # Worker config
    args_dict = {
        'games_per_worker': args.games_per_worker,
        'simulations': args.simulations,
    }
    
    print(f"Starting {args.workers} workers...")
    print(f"  Games per worker: {args.games_per_worker}")
    print(f"  Total concurrent: {args.workers * args.games_per_worker}")
    
    # Start workers
    workers = []
    for i in range(args.workers):
        p = mp.Process(target=worker_process, args=(
            i, request_queue, result_queues[i], game_queue, ready_events[i], args_dict
        ))
        p.daemon = True
        p.start()
        workers.append(p)
    
    # Wait for workers to be ready (with timeout)
    print("Waiting for workers to initialize...")
    for i, event in enumerate(ready_events):
        if not event.wait(timeout=30):
            print(f"WARNING: Worker {i} did not signal ready in 30s")
    print("All workers ready!")
    
    # Inference server
    server = InferenceServer(model, device, result_queues, batch_size=args.batch_size)
    
    replay_buffer = deque(maxlen=args.buffer_size)
    
    print(f"\nStarting training for {args.epochs} epochs")
    print("-" * 60)
    
    try:
        for epoch in range(start_epoch, start_epoch + args.epochs):
            epoch_start = time.time()
            games_collected = 0
            
            print(f"\nEpoch {epoch + 1}: Collecting {args.games_per_epoch} games...")
            
            # Collection loop
            model.eval()
            while games_collected < args.games_per_epoch:
                # Process inference requests
                server.step(request_queue)
                
                # Collect completed games
                try:
                    while True:
                        training_data = game_queue.get_nowait()
                        replay_buffer.extend(training_data)
                        games_collected += 1
                        if args.verbose:
                            print(f"  Game {games_collected}/{args.games_per_epoch} collected", end='\r')
                        if games_collected >= args.games_per_epoch:
                            break
                except queue.Empty:
                    pass
            
            collect_time = time.time() - epoch_start
            print(f"  Buffer: {len(replay_buffer)} | Collect: {collect_time:.1f}s")
            
            # Training phase
            if len(replay_buffer) >= args.batch_size:
                train_start = time.time()
                print(f"Epoch {epoch + 1}: Training...")
                
                sample_size = min(len(replay_buffer), args.batch_size * 20)
                batch = random.sample(list(replay_buffer), sample_size)
                
                states = torch.stack([torch.from_numpy(s) for s, _, _ in batch]).to(device)
                policies = torch.stack([torch.from_numpy(p) for _, p, _ in batch]).to(device)
                values = torch.tensor([v for _, _, v in batch], dtype=torch.float32).to(device)
                
                total_losses = []
                for _ in range(args.train_steps):
                    indices = torch.randperm(len(states))[:args.batch_size]
                    loss, _, _ = train_step(
                        model, optimizer,
                        states[indices], policies[indices], values[indices],
                        scaler=scaler, device=device
                    )
                    total_losses.append(loss)
                
                train_time = time.time() - train_start
                epoch_time = time.time() - epoch_start
                games_per_sec = args.games_per_epoch / epoch_time
                
                print(f"  Loss: {np.mean(total_losses):.4f} | Train: {train_time:.1f}s | Total: {epoch_time:.1f}s ({games_per_sec:.2f} g/s)")
                
                writer.add_scalar("Loss/Total", np.mean(total_losses), epoch + 1)
                writer.add_scalar("Perf/GamesPerSec", games_per_sec, epoch + 1)
            
            # Save checkpoint
            model_to_save = model._orig_mod if hasattr(model, '_orig_mod') else model
            save_model(model_to_save, str(checkpoint_path), optimizer, epoch + 1)
            
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        # Cleanup
        for p in workers:
            p.terminate()
    
    print("\nTraining complete!")


def evaluate(args):
    """Evaluate model."""
    print("Evaluating model...")
    model, device = create_model()
    
    models_dir = Path(__file__).parent.parent / "models"
    checkpoint_path = models_dir / "latest.pt"
    
    if checkpoint_path.exists():
        model, device, _ = load_model(str(checkpoint_path), compile_model=False)
        print("Loaded trained model")
    
    wins, losses, draws = 0, 0, 0
    
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
    train_parser.add_argument('--games-per-epoch', type=int, default=50)
    train_parser.add_argument('--simulations', type=int, default=50)
    train_parser.add_argument('--batch-size', type=int, default=256)
    train_parser.add_argument('--buffer-size', type=int, default=50000)
    train_parser.add_argument('--train-steps', type=int, default=20)
    train_parser.add_argument('--lr', type=float, default=0.001)
    train_parser.add_argument('--fresh', action='store_true')
    train_parser.add_argument('--no-compile', action='store_true')
    train_parser.add_argument('--verbose', action='store_true')
    train_parser.add_argument('--workers', type=int, default=8,
                              help='Number of worker processes')
    train_parser.add_argument('--games-per-worker', type=int, default=64,
                              help='Games per worker process')
    
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
    main()