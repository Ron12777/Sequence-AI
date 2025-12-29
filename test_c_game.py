from src.fast_game import FastSequenceGame
from src.cards import Card, Suit, Rank
from src.game import Move

def test_game():
    print("Initializing game...")
    g = FastSequenceGame()
    g.reset()
    
    print("Getting moves...")
    moves = g.get_legal_moves(1)
    print(f"Found {len(moves)} legal moves")
    
    if not moves:
        print("No moves!")
        return

    m = moves[0]
    print(f"Making move: {m}")
    g.make_move(m)
    
    print("Checking board...")
    # Verify board changed in C
    t = g.get_state_tensor(1)
    print("Tensor shape:", t.shape)
    
    print("Cloning...")
    g2 = g.clone()
    print("Clone success")
    
    print("Test Complete")

if __name__ == "__main__":
    test_game()
