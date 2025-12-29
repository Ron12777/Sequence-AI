"""
Self-play training for Sequence AI.
Single-process batched architecture - no IPC, no multiprocessing issues.
All games run in the main process with batched GPU inference.
"""
import argparse
import os
import time
import warnings
from pathlib import Path
from collections import deque
from typing import List, Tuple, Optional, Dict
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
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


class BatchedSelfPlay:
    """
    Runs many games in parallel with batched neural network inference.
    All in a single process - no IPC overhead.
    """
    def __init__(self, model: SequenceNet, device: torch.device, 
                 num_games: int = 128, simulations: int = 50):
        import c_sequence
        from .game import SequenceGame as OriginalSequenceGame
        
        self.model = model
        self.device = device
        self.num_games = num_games
        self.simulations = simulations
        
        # Initialize card layout for C extension
        temp_game = OriginalSequenceGame()
        for card, positions in temp_game.card_to_positions.items():
            for i, pos in enumerate(positions):
                c_sequence.setup_layout(card.to_int(), i, pos[0], pos[1])
        
        # Initialize games
        self.games = [SequenceGame() for _ in range(num_games)]
        self.mcts_list = [c_sequence.CMCTS() for _ in range(num_games)]
        self.game_histories = [[] for _ in range(num_games)]
        self.move_counts = [0] * num_games
        
        # Reset all games
        for g in self.games:
            g.reset()
        
        # Initialize MCTS roots
        for i in range(num_games):
            self._reset_mcts(i)
    
    def _reset_mcts(self, idx: int):
        """Reset MCTS for game at index."""
        board_bytes = self.games[idx].c_state.get_tensor()
        hand_ids = [c.to_int() for c in self.games[idx].hands[self.games[idx].current_player]]
        self.mcts_list[idx].reset(board_bytes, hand_ids, self.games[idx].current_player)
    
    def collect_games(self, target_games: int, verbose: bool = False) -> List[Tuple]:
        """
        Collect training data from self-play games.
        Returns list of (state, policy, value) tuples.
        """
        all_training_data = []
        games_completed = 0
        
        self.model.eval()
        
        while games_completed < target_games:
            # Run simulations for all games
            sim_counts = [0] * self.num_games
            
            while True:
                # Find games that need more simulations
                active = [i for i in range(self.num_games) if sim_counts[i] < self.simulations]
                if not active:
                    break
                
                # Select leaves for all active games
                requests = []
                req_info = []  # (game_idx, leaf_node)
                
                for i in active:
                    leaf_node, tensor_bytes, is_terminal, value = self.mcts_list[i].select_leaf()
                    
                    if is_terminal:
                        self.mcts_list[i].backpropagate(leaf_node, b'', value)
                        sim_counts[i] += 1
                    else:
                        state_tensor = torch.frombuffer(tensor_bytes, dtype=torch.float32).reshape(8, 10, 10)
                        requests.append(state_tensor)
                        req_info.append((i, leaf_node))
                
                # Batch inference
                if requests:
                    batch = torch.stack(requests).to(self.device)
                    with torch.no_grad():
                        policies, values = self.model.predict(batch)
                    
                    policies = policies.cpu().numpy()
                    values = values.cpu().numpy()
                    
                    # Backpropagate results
                    for k, (idx, leaf) in enumerate(req_info):
                        p_bytes = policies[k].astype(np.float32).tobytes()
                        v = values[k].item()
                        self.mcts_list[idx].backpropagate(leaf, p_bytes, v)
                        sim_counts[idx] += 1
            
            # All simulations done - make moves for all games
            for i in range(self.num_games):
                temp = 1.0 if self.move_counts[i] < 10 else 0.1
                res = self.mcts_list[i].get_action(temp)
                
                if res is None:
                    # No legal moves - reset
                    self.games[i].reset()
                    self.game_histories[i] = []
                    self.move_counts[i] = 0
                    self._reset_mcts(i)
                    continue
                
                card_int, r, c, is_rem, policy_bytes = res
                policy = np.frombuffer(policy_bytes, dtype=np.float32)
                
                # Store state for training
                state_tensor = self.games[i].get_state_tensor(self.games[i].current_player)
                self.game_histories[i].append((state_tensor, policy, self.games[i].current_player))
                
                # Apply move
                move = Move(card=Card.from_int(card_int), row=r, col=c, is_removal=(is_rem != 0))
                self.games[i].make_move(move)
                self.move_counts[i] += 1
                
                # Check game over
                if self.games[i].game_over or self.move_counts[i] >= 200:
                    # Collect training data
                    for state, pi, player in self.game_histories[i]:
                        value = self.games[i].get_result(player)
                        all_training_data.append((state, pi, value))
                    
                    games_completed += 1
                    if verbose:
                        print(f"  Game {games_completed}/{target_games} collected", end='\r')
                    
                    if games_completed >= target_games:
                        break
                    
                    # Reset game
                    self.games[i].reset()
                    self.game_histories[i] = []
                    self.move_counts[i] = 0
                
                # Reset MCTS for next move
                self._reset_mcts(i)
        
        if verbose:
            print()
        
        return all_training_data


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
    print("=" * 60)
    print("Sequence AI Training (Single-Process Batched)")
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
    
    # TensorBoard
    writer = SummaryWriter(log_dir=models_dir.parent / "runs" / f"run_{int(time.time())}")
    
    # Self-play engine
    print(f"Initializing {args.concurrent_games} concurrent games...")
    selfplay = BatchedSelfPlay(
        model, device, 
        num_games=args.concurrent_games,
        simulations=args.simulations
    )
    
    replay_buffer = deque(maxlen=args.buffer_size)
    
    print(f"\nStarting training for {args.epochs} epochs")
    print("-" * 60)
    
    try:
        for epoch in range(start_epoch, start_epoch + args.epochs):
            epoch_start = time.time()
            
            print(f"\nEpoch {epoch + 1}: Collecting {args.games_per_epoch} games...")
            
            # Collect games
            training_data = selfplay.collect_games(args.games_per_epoch, verbose=args.verbose)
            replay_buffer.extend(training_data)
            
            print(f"  Buffer size: {len(replay_buffer)}")
            
            # Training phase
            if len(replay_buffer) >= args.batch_size:
                print(f"Epoch {epoch + 1}: Training...")
                
                sample_size = min(len(replay_buffer), args.batch_size * 20)
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
                
                writer.add_scalar("Loss/Total", avg_loss, epoch + 1)
                writer.add_scalar("Perf/GamesPerSec", games_per_sec, epoch + 1)
            
            # Save checkpoint
            model_to_save = model._orig_mod if hasattr(model, '_orig_mod') else model
            save_model(model_to_save, str(checkpoint_path), optimizer, epoch + 1)
            
    except KeyboardInterrupt:
        print("\nStopping...")
    
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
    train_parser.add_argument('--batch-size', type=int, default=128)
    train_parser.add_argument('--buffer-size', type=int, default=50000)
    train_parser.add_argument('--train-steps', type=int, default=20)
    train_parser.add_argument('--lr', type=float, default=0.001)
    train_parser.add_argument('--fresh', action='store_true')
    train_parser.add_argument('--no-compile', action='store_true')
    train_parser.add_argument('--verbose', action='store_true')
    train_parser.add_argument('--concurrent-games', type=int, default=256,
                              help='Number of games running simultaneously')
    
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