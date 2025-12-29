import time
import os
import psutil
import c_sequence
import numpy as np
from src.fast_game import FastSequenceGame

def get_memory_usage():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024  # MB

def test_leak():
    print(f"Initial Memory: {get_memory_usage():.2f} MB")
    
    # Initialize basic game structures
    games = [FastSequenceGame() for _ in range(16)]
    mcts_list = [c_sequence.CMCTS() for _ in range(16)]
    
    # Initial Setup
    for g in games:
        g.reset()
    
    for i in range(16):
        board_bytes = games[i].c_state.get_tensor()
        hand_ids = [c.to_int() for c in games[i].hands[games[i].current_player]]
        mcts_list[i].reset(board_bytes, hand_ids, games[i].current_player)

    start_time = time.time()
    
    # Run loop
    for step in range(100000):
        # Select leaf
        leaves = []
        for i in range(16):
            leaf, tensor_bytes, is_terminal, value = mcts_list[i].select_leaf()
            if is_terminal:
                mcts_list[i].backpropagate(leaf, b'', value)
            else:
                leaves.append((i, leaf))
        
        # Fake inference (just random policy)
        for i, leaf in leaves:
            # Policy: 100 floats
            policy = np.random.random(100).astype(np.float32)
            p_bytes = policy.tobytes()
            value = float(np.random.random())
            mcts_list[i].backpropagate(leaf, p_bytes, value)
            
        # Reset occasionally
        if step % 200 == 0:
            for i in range(16):
                # Fake reset
                board_bytes = games[i].c_state.get_tensor()
                hand_ids = [c.to_int() for c in games[i].hands[games[i].current_player]]
                mcts_list[i].reset(board_bytes, hand_ids, games[i].current_player)
                
        if step % 1000 == 0:
            mem = get_memory_usage()
            print(f"Step {step}: {mem:.2f} MB", end='\r')
            
    print(f"\nFinal Memory: {get_memory_usage():.2f} MB")

if __name__ == "__main__":
    test_leak()
