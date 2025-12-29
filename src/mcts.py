"""
Information Set Monte Carlo Tree Search for Sequence.
Handles imperfect information through determinization.
"""
import math
import random
import numpy as np
import torch
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
import time

from .game import SequenceGame, Move
from .cards import Card, Deck

# Try to import C extension
try:
    import c_sequence
    C_EXTENSION_LOADED = True
except ImportError:
    print("Warning: c_sequence extension not found. Using slow Python implementation.")
    C_EXTENSION_LOADED = False


@dataclass
class MCTSNode:
    """Node in the MCTS tree."""
    game_state: SequenceGame
    parent: Optional['MCTSNode'] = None
    move: Optional[Move] = None
    children: Dict[Tuple, 'MCTSNode'] = field(default_factory=dict)
    
    visits: int = 0
    total_value: float = 0.0
    prior: float = 0.0
    
    @property
    def q_value(self) -> float:
        """Mean action value."""
        if self.visits == 0:
            return 0.0
        return self.total_value / self.visits
    
    @property
    def ucb_score(self) -> float:
        """Upper Confidence Bound score for selection."""
        if self.parent is None:
            return 0.0
        
        c_puct = 2.0  # Exploration constant
        
        u = c_puct * self.prior * math.sqrt(self.parent.visits) / (1 + self.visits)
        return self.q_value + u
    
    def is_leaf(self) -> bool:
        return len(self.children) == 0
    
    def is_terminal(self) -> bool:
        return self.game_state.game_over


class MCTS:
    """
    Information Set MCTS with neural network guidance.
    Uses determinization to handle hidden information.
    """
    _layout_initialized = False
    
    def __init__(self, 
                 model: torch.nn.Module,
                 device: torch.device,
                 num_simulations: int = 100,
                 temperature: float = 1.0,
                 dirichlet_alpha: float = 0.3,
                 dirichlet_epsilon: float = 0.25):
        self.model = model
        self.device = device
        self.num_simulations = num_simulations
        self.temperature = temperature
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_epsilon = dirichlet_epsilon
        
        self.cmcts = None
        if C_EXTENSION_LOADED:
            self._init_c_layout()
            self.cmcts = c_sequence.CMCTS()

    def _init_c_layout(self):
        """Initialize C extension board layout map."""
        if MCTS._layout_initialized:
            return
            
        # Create a dummy game to access layout
        game = SequenceGame()
        counts = {}
        
        for r, row in enumerate(game.board_layout):
            for c, card in enumerate(row):
                if card:
                    card_int = card.to_int()
                    idx = counts.get(card_int, 0)
                    c_sequence.setup_layout(card_int, idx, r, c)
                    counts[card_int] = idx + 1
                    
        MCTS._layout_initialized = True

    def search(self, game: SequenceGame, player: int, simulations: Optional[int] = None) -> Tuple[Move, np.ndarray]:
        """
        Run MCTS search from current game state.
        
        Args:
            game: Current game state
            player: Player to find move for
            simulations: Optional override for number of simulations
            
        Returns:
            best_move: Selected move
            policy: Visit count distribution over moves
        """
        num_sims = simulations if simulations is not None else self.num_simulations
        
        if self.cmcts:
            return self._search_c(game, player, num_sims)
        else:
            return self._search_python(game, player, num_sims)

    def _search_c(self, game: SequenceGame, player: int, num_simulations: int) -> Tuple[Move, np.ndarray]:
        """Run MCTS using C extension."""
        start_time = time.time()
        
        # Prepare inputs
        board_bytes = game.board.tobytes()
        hand_ints = [c.to_int() for c in game.hands[player]]
        
        # Reset C tree
        self.cmcts.reset(board_bytes, hand_ints, player)
        
        # Simulation loop
        for i in range(num_simulations):
            # Select leaf (traversal in C)
            # Returns: (capsule, tensor_bytes, value, is_terminal_int)
            capsule, tensor_bytes, value, is_terminal_int = self.cmcts.select_leaf()
            is_terminal = bool(is_terminal_int)
            
            policy_bytes = b''
            
            if not is_terminal:
                # Run inference
                # Note: C returns 800 floats (8x10x10)
                tensor = torch.frombuffer(tensor_bytes, dtype=torch.float32).reshape(1, 8, 10, 10).to(self.device)
                
                self.model.eval()
                with torch.no_grad():
                    policy, val = self.model.predict(tensor)
                
                # Squeeze batch dim
                policy = policy.squeeze(0).cpu().numpy()
                value = val.item()
                policy_bytes = policy.tobytes()
            
            # Backpropagate (expansion + backprop in C)
            self.cmcts.backpropagate(capsule, policy_bytes, value)
        
        # Get action
        # Returns: (card_int, row, col, is_removal_int, policy_bytes)
        card_int, row, col, is_rem, policy_bytes = self.cmcts.get_action(self.temperature)
        
        if policy_bytes is None:
            # Should not happen unless no moves
            return None, np.array([])
            
        # Convert back to Python objects
        card = Card.from_int(card_int)
        move = Move(card=card, row=row, col=col, is_removal=bool(is_rem))
        
        # Reconstruct full policy array (100)
        # Note: The C get_action returns a policy over BOARD POSITIONS (100), 
        # but pure Python returns it over... well, it returns a 100-dim array too.
        # Wait, Python version returns 'policy' from model which is 100-dim.
        # C version builds policy from visit counts.
        policy = np.frombuffer(policy_bytes, dtype=np.float32)
        
        print(f"C-MCTS finished in {time.time() - start_time:.2f}s ({num_simulations} sims)")
        return move, policy

    def _search_python(self, game: SequenceGame, player: int, num_simulations: int) -> Tuple[Move, np.ndarray]:
        """Legacy Python implementation of MCTS."""
        # Determinize once at the start (Single Observer MCTS)
        # This fixes the opponent's hand to one possible instantiation for the duration of the search
        det_game = self._determinize(game, player)
        root = MCTSNode(game_state=det_game)
        
        # Get prior probabilities from neural network
        self._expand_node(root, player)
        
        # Handle no legal moves
        if not root.children:
            legal_moves = game.get_legal_moves(player)
            if legal_moves:
                return legal_moves[0], np.array([1.0])
            return None, np.array([])
        
        # Add Dirichlet noise to root for exploration
        self._add_dirichlet_noise(root)
        
        # Run simulations
        start_time = time.time()
        
        for i in range(num_simulations):
            if i % 100 == 0:
                print(f"MCTS Simulation {i}/{num_simulations}...")
                
            node = root
            search_path = [node]
            
            # Selection - traverse tree using UCB
            while not node.is_leaf() and not node.is_terminal():
                node = self._select_child(node)
                search_path.append(node)
            
            # Expansion and evaluation
            if not node.is_terminal():
                value = self._expand_node(node, node.game_state.current_player)
            else:
                value = node.game_state.get_result(player)
            
            # Backpropagation
            self._backpropagate(search_path, value, player)
            
        print(f"Py-MCTS finished in {time.time() - start_time:.2f}s")
        
        # Select move based on visit counts
        return self._select_move(root)
    
    def _expand_node(self, node: MCTSNode, player: int) -> float:
        """Expand node by adding children and return value estimate."""
        game = node.game_state
        
        if game.game_over:
            return game.get_result(player)
        
        # Get neural network predictions
        state_tensor = torch.from_numpy(game.get_state_tensor(player))
        state_tensor = state_tensor.unsqueeze(0).to(self.device)
        
        self.model.eval()
        with torch.no_grad():
            policy, value = self.model.predict(state_tensor)
        
        policy = policy.squeeze(0).cpu().numpy()
        value = value.item()
        
        # Get legal moves (game is already determinized)
        legal_moves = game.get_legal_moves()
        
        if not legal_moves:
            return value
        
        # Create children for each legal move
        for move in legal_moves:
            child_game = game.clone()
            child_game.make_move(move)
            
            # Map move to policy index
            policy_idx = move.row * 10 + move.col
            prior = policy[policy_idx] if policy_idx < len(policy) else 1.0 / len(legal_moves)
            
            move_key = (str(move.card), move.row, move.col, move.is_removal)
            child = MCTSNode(
                game_state=child_game,
                parent=node,
                move=move,
                prior=prior
            )
            node.children[move_key] = child
        
        return value
    
    def _determinize(self, game: SequenceGame, observer: int) -> SequenceGame:
        """
        Create determinized game state by sampling hidden information.
        The observer's hand is known; opponent's hand is sampled.
        """
        det_game = game.clone()
        opponent = 3 - observer
        
        # Get all cards that could be in opponent's hand
        # (cards not on board, not in observer's hand, not in discard)
        all_known_cards = set()
        
        # Cards on board
        for r in range(10):
            for c in range(10):
                card = det_game.board_layout[r][c]
                if card and det_game.board[r, c] != 0:  # Position is occupied
                    all_known_cards.add(card)
        
        # Observer's hand
        for card in det_game.hands[observer]:
            all_known_cards.add(card)
        
        # Discard piles
        for pile in det_game.discard_piles.values():
            for card in pile:
                all_known_cards.add(card)
        
        # Sample opponent hand from remaining deck cards
        available = [c for c in det_game.deck.cards if c not in all_known_cards]
        hand_size = len(det_game.hands[opponent])
        
        if len(available) >= hand_size:
            sampled_hand = random.sample(available, hand_size)
            det_game.hands[opponent] = sampled_hand
        
        return det_game
    
    def _select_child(self, node: MCTSNode) -> MCTSNode:
        """Select child with highest UCB score."""
        best_score = float('-inf')
        best_child = None
        
        for child in node.children.values():
            score = child.ucb_score
            if score > best_score:
                best_score = score
                best_child = child
        
        return best_child
    
    def _add_dirichlet_noise(self, node: MCTSNode):
        """Add Dirichlet noise to root node priors for exploration."""
        if not node.children:
            return
        
        noise = np.random.dirichlet([self.dirichlet_alpha] * len(node.children))
        
        for i, child in enumerate(node.children.values()):
            child.prior = (1 - self.dirichlet_epsilon) * child.prior + \
                         self.dirichlet_epsilon * noise[i]
    
    def _backpropagate(self, path: List[MCTSNode], value: float, root_player: int):
        """Backpropagate value through the search path."""
        for node in reversed(path):
            # Flip value for opponent's perspective
            if node.game_state.current_player != root_player:
                node.total_value += -value
            else:
                node.total_value += value
            node.visits += 1
    
    def _select_move(self, root: MCTSNode) -> Tuple[Move, np.ndarray]:
        """Select move based on visit counts and temperature."""
        if not root.children:
            return None, np.array([])
        
        moves = []
        visit_counts = []
        
        for move_key, child in root.children.items():
            moves.append(child.move)
            visit_counts.append(child.visits)
        
        if not moves:
            return None, np.array([])
        
        visit_counts = np.array(visit_counts, dtype=np.float32)
        
        # Create policy distribution
        if self.temperature == 0:
            # Greedy selection
            best_idx = np.argmax(visit_counts)
            policy = np.zeros_like(visit_counts)
            policy[best_idx] = 1.0
        else:
            # Temperature-scaled selection
            visit_counts = visit_counts ** (1.0 / self.temperature)
            total = visit_counts.sum()
            if total == 0:
                policy = np.ones_like(visit_counts) / len(visit_counts)
            else:
                policy = visit_counts / total
        
        # Sample move
        chosen_idx = np.random.choice(len(moves), p=policy)
        
        return moves[chosen_idx], policy


class RandomPlayer:
    """Random baseline player for testing."""
    
    def get_move(self, game: SequenceGame, player: int) -> Optional[Move]:
        legal_moves = game.get_legal_moves(player)
        if not legal_moves:
            return None
        return random.choice(legal_moves)


class GreedyPlayer:
    """Greedy player that prioritizes sequence-building moves."""
    
    def get_move(self, game: SequenceGame, player: int) -> Optional[Move]:
        legal_moves = game.get_legal_moves(player)
        if not legal_moves:
            return None
        
        # Score moves by how many chips are adjacent
        best_move = None
        best_score = -1
        
        for move in legal_moves:
            if move.is_removal:
                score = 10  # Prioritize removals
            else:
                score = self._count_adjacent(game, player, move.row, move.col)
            
            if score > best_score:
                best_score = score
                best_move = move
        
        return best_move
    
    def _count_adjacent(self, game: SequenceGame, player: int, row: int, col: int) -> int:
        """Count adjacent friendly chips."""
        count = 0
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                if dr == 0 and dc == 0:
                    continue
                r, c = row + dr, col + dc
                if 0 <= r < 10 and 0 <= c < 10:
                    if game.board[r, c] == player or game.board[r, c] == -1:
                        count += 1
        return count
