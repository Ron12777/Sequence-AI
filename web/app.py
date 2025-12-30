"Flask web server for Sequence game UI."
import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from flask import Flask, jsonify, request, send_from_directory
import torch
import numpy as np
import threading
import time

from src.game import SequenceGame, Move
from src.cards import Card
from src.model import create_model, load_model
from src.mcts import MCTS, RandomPlayer, GreedyPlayer


app = Flask(__name__, static_folder='static')

# Global game state
game = None
model = None
device = None
mcts = None
mcts = None
ai_player = 2  # AI plays as player 2 by default

# Async AI Status
ai_status_lock = threading.Lock()
ai_status = {
    'thinking': False,
    'simulations': 0,
    'top_moves': [] 
}

def process_policy(policy):
    """
    Convert raw 100-dim policy to list of top moves.
    Filters to return only 'significant' moves (e.g. > 50% of top score)
    to avoid cluttering the UI with low-probability noise.
    """
    top_moves = []
    if policy is not None and len(policy) > 0:
        # Policy is 100-dim array (board positions)
        indices = np.argsort(policy)[::-1]
        
        best_score = float(policy[indices[0]]) if len(indices) > 0 else 0
        
        for idx in indices:
            score = float(policy[idx])
            # Visualization threshold:
            # 1. Must be at least 1% probability
            # 2. Must be at least 30% of the best move's score (adaptive filtering)
            if score < 0.01: break
            if score < (best_score * 0.3): break 
            
            top_moves.append({
                'row': int(idx // 10),
                'col': int(idx % 10),
                'score': score
            })
            
            # Limit to max 5 visuals to prevent overload
            if len(top_moves) >= 5: break
            
    return top_moves

def mcts_progress_callback(mcts_conn, sims):
    """Callback from MCTS to update global status."""
    global ai_status
    try:
        # Get current policy estimate
        # Note: Temperature 1.0 is standard for policy viewing
        # cmcts.get_action returns: (card, row, col, is_rem, policy_bytes)
        if mcts_conn.cmcts:
            _, _, _, _, policy_bytes = mcts_conn.cmcts.get_action(1.0)
            if policy_bytes:
                policy = np.frombuffer(policy_bytes, dtype=np.float32)
                top_moves = process_policy(policy)
                
                with ai_status_lock:
                    ai_status['simulations'] = sims
                    ai_status['top_moves'] = top_moves
                    ai_status['thinking'] = True
    except Exception as e:
        print(f"Error in progress callback: {e}")



def init_ai():
    """Initialize AI model."""
    global model, device, mcts
    
    print("Initializing AI...")
    # Disable compilation locally to avoid Windows hang
    model, device = create_model(compile_model=False)
    
    # Try to load trained model
    models_dir = Path(__file__).parent.parent / "models"
    checkpoint_path = models_dir / "latest.pt"
    
    if checkpoint_path.exists():
        print(f"Loading trained model from {checkpoint_path}")
        try:
            model, device, _ = load_model(str(checkpoint_path), compile_model=False)
            print("Model loaded successfully")
        except Exception as e:
            print(f"Error loading model: {e}")
    else:
        print("No trained model found, using untrained network")
    
    mcts = MCTS(model, device, num_simulations=50, temperature=0)  # temperature=0 for greedy (deterministic)
    print(f"AI initialized on {device}")


def make_ai_move_internal(player_id, simulations=None):
    """Make AI move for the specified player and return the move details."""
    global game
    initial_game_state = game
    
    print(f"AI thinking for Player {player_id}... (Simulations: {simulations})")
    if mcts is None:
        print("MCTS is None, using greedy")
        # Fallback to greedy player if no model
        player = GreedyPlayer()
        move = player.get_move(game, player_id)
    else:
        try:
            print(f"Starting MCTS search (Player {player_id})...")
            
            # Reset status
            with ai_status_lock:
                ai_status['thinking'] = True
                ai_status['simulations'] = 0
                ai_status['top_moves'] = []
            
            move, policy = mcts.search(game, player_id, simulations=simulations, progress_callback=mcts_progress_callback)
            
            with ai_status_lock:
                ai_status['thinking'] = False

            print(f"MCTS returned move: {move}")
        except Exception as e:
            with ai_status_lock:
                ai_status['thinking'] = False
            print(f"MCTS Error: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    # Check if game has changed (New Game clicked during thinking)
    if game is not initial_game_state:
        print("Game instance changed during AI thinking - aborting status update")
        return None

    if move and game.make_move(move):
        print(f"AI made move: {move}")
        
        # Log move
        if not hasattr(game, 'move_history_log'):
            game.move_history_log = []
        game.move_history_log.append({
            'player': player_id,
            'move': str(move),
            'card': str(move.card),
            'r': move.row,
            'c': move.col
        })
        
        # Process policy for visualization
        top_moves = process_policy(policy)
        
        # [NEW] Ensure the move we report back is actually the one we just made, 
        # and it should match the top_moves[0] if we want to be strict.
        # But for now, just reporting the actual 'move' applied.
        
        return {
            'card': str(move.card),
            'row': move.row,
            'col': move.col,
            'is_removal': move.is_removal,
            'top_moves': top_moves
        }
    print(f"AI failed to make a legal move for player {player_id}")
    return None


@app.route('/')
def index():
    """Serve the main game page."""
    return send_from_directory('static', 'index.html')


@app.route('/api/ai_status')
def get_ai_status():
    """Get current AI thinking status."""
    with ai_status_lock:
        return jsonify(ai_status)

@app.route('/static/<path:path>')
def serve_static(path):
    """Serve static files."""
    return send_from_directory('static', path)


@app.route('/api/new_game', methods=['POST'])
def new_game():
    """Start a new game."""
    global game, ai_player, ai_status
    
    # Reset AI status immediately to stop stale polling data
    with ai_status_lock:
        ai_status['thinking'] = False
        ai_status['simulations'] = 0
        ai_status['top_moves'] = []
    
    data = request.get_json() or {}
    ai_player = data.get('ai_player', 2)
    
    game = SequenceGame()
    game.reset()
    game.move_history_log = []
    
    return jsonify(get_game_state())


@app.route('/api/state', methods=['GET'])
def get_state():
    """Get current game state."""
    if game is None:
        return jsonify({'error': 'No game in progress'}), 400
    return jsonify(get_game_state())


def get_game_state():
    """Build game state dictionary."""
    if game is None:
        return {}
        
    # Convert board to list
    board = game.board.tolist()
    
    # Get board layout (card labels)
    board_layout = []
    for row in game.board_layout:
        layout_row = []
        for card in row:
            layout_row.append(str(card) if card else 'FREE')
        board_layout.append(layout_row)
    
    # Get human player's hand
    human_player = 3 - ai_player
    hand = [str(card) for card in game.hands.get(human_player, [])]
    
    # Get legal moves for highlighting
    legal_moves = []
    if game.current_player == human_player and not game.game_over:
        for move in game.get_legal_moves(human_player):
            legal_moves.append({
                'card': str(move.card),
                'row': move.row,
                'col': move.col,
                'is_removal': move.is_removal
            })
    
    return {
        'board': board,
        'board_layout': board_layout,
        'hand': hand,
        'current_player': game.current_player,
        'human_player': human_player,
        'ai_player': ai_player,
        'game_over': game.game_over,
        'winner': game.winner,
        'legal_moves': legal_moves,
        'sequences': {
            1: [list(seq) for seq in game.completed_sequences[1]],
            2: [list(seq) for seq in game.completed_sequences[2]]
        },
        'hands': {
            1: [str(c) for c in game.hands.get(1, [])],
            2: [str(c) for c in game.hands.get(2, [])]
        },
        'history': getattr(game, 'move_history_log', []),
        'game_id': getattr(game, 'id', None)
    }


@app.route('/api/move', methods=['POST'])
def make_move():
    """Make a player move."""
    global game
    
    if game is None:
        return jsonify({'error': 'No game in progress'}), 400
    
    data = request.get_json()
    if not data:
        return jsonify({'error': 'No move data provided'}), 400
    
    human_player = 3 - ai_player
    
    if game.current_player != human_player:
        return jsonify({'error': 'Not your turn'}), 400
    
    if game.game_over:
        return jsonify({'error': 'Game is over'}), 400
    
    # [NEW] Validate Game ID
    provided_id = data.get('game_id')
    if provided_id and provided_id != getattr(game, 'id', None):
        print(f"Bailing out: provided_id={provided_id}, actual_id={getattr(game, 'id', None)}")
        return jsonify({'error': 'Game instance mismatch. Please refresh or start a new game.', 'mismatch': True}), 409
    
    # Parse move
    try:
        card_str = data['card']
        row = data['row']
        col = data['col']
        is_removal = data.get('is_removal', False)
        
        # Find card in hand
        card = None
        for c in game.hands[human_player]:
            if str(c) == card_str:
                card = c
                break
        
        if card is None:
            return jsonify({'error': f'Card {card_str} not in hand'}), 400
        
        move = Move(card=card, row=row, col=col, is_removal=is_removal)
        
        if not game.make_move(move):
            return jsonify({'error': 'Invalid move'}), 400
            
        # Reset AI status immediately on human move
        with ai_status_lock:
            ai_status['thinking'] = False
            ai_status['top_moves'] = []
            
        # Log move
        if not hasattr(game, 'move_history_log'):
            game.move_history_log = []
        game.move_history_log.append({
            'player': human_player,
            'move': str(move),
            'card': str(move.card),
            'r': row,
            'c': col
        })
        
    except (KeyError, ValueError) as e:
        return jsonify({'error': f'Invalid move data: {e}'}), 400
    
    response = get_game_state()
    
    print(f"Post-Move Check: GameOver={game.game_over}, Current={game.current_player}, AI={ai_player}")
    
    return jsonify(response)


@app.route('/api/ai_move', methods=['POST'])
def request_ai_move():
    """Request AI to make a move (for async play, hints, or watch mode)."""
    if game is None:
        return jsonify({'error': 'No game in progress'}), 400
    
    if game.game_over:
        return jsonify({'error': 'Game is over'}), 400
    
    # Get difficulty from request
    data = request.get_json() or {}
    
    # [NEW] Validate Game ID
    provided_id = data.get('game_id')
    if provided_id and provided_id != getattr(game, 'id', None):
        print(f"Bailing out AI: provided_id={provided_id}, actual_id={getattr(game, 'id', None)}")
        return jsonify({'error': 'Game instance mismatch during AI request.', 'mismatch': True}), 409
        
    difficulty = data.get('difficulty', 'medium')
    
    # Map difficulty to simulations
    sim_map = {
        'easy': 50,
        'medium': 400,
        'hard': 1600
    }
    simulations = sim_map.get(difficulty, 400)
    
    # Make move for the current player
    current_p = game.current_player
    ai_move = make_ai_move_internal(player_id=current_p, simulations=simulations)
    
    response = get_game_state()
    response['ai_move'] = ai_move
    
    return jsonify(response)


@app.route('/api/hint', methods=['GET'])
def get_hint():
    """Get AI's recommended move for the human player."""
    if game is None or mcts is None:
        return jsonify({'error': 'Game or AI not initialized'}), 400
    
    human_player = 3 - ai_player
    
    if game.current_player != human_player:
        return jsonify({'error': 'Not your turn'}), 400
    
    move, _ = mcts.search(game, human_player)
    
    return jsonify({
        'card': str(move.card),
        'row': move.row,
        'col': move.col,
        'is_removal': move.is_removal
    })


def main():
    """Run the web server."""
    init_ai()
    
    port = int(os.environ.get('PORT', 5000))
    print(f"\n{'='*60}")
    print(f"Sequence AI Web Interface")
    print(f"{ '='*60}")
    print(f"Open http://localhost:{port} in your browser")
    print(f"{ '='*60}\n")
    
    app.run(host='0.0.0.0', port=port, debug=False)


if __name__ == '__main__':
    main()