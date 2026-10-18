"""
Determinized Monte Carlo Tree Search for Sequence.
Handles imperfect information through determinization.
"""

import math
import random
import numpy as np
import torch
from typing import List, Dict, Optional, Tuple, Callable
from dataclasses import dataclass, field
import time

from .game import SequenceGame, Move
from .cards import Card
from .belief import unseen_cards, search_class
from .model import STATE_CHANNELS

# The Python reference remains available when the optional native module is absent.
try:
    import sequence_native  # noqa: F401 -- validate the extension can actually load

    NATIVE_EXTENSION_LOADED = True
except ImportError:
    print("Rust extension not found. Install the project to enable search.")
    NATIVE_EXTENSION_LOADED = False


@dataclass
class MCTSNode:
    """Node in the MCTS tree."""

    game_state: SequenceGame
    parent: Optional["MCTSNode"] = None
    move: Optional[Move] = None
    children: Dict[Tuple, "MCTSNode"] = field(default_factory=dict)

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
        return -self.q_value + u

    def is_leaf(self) -> bool:
        return len(self.children) == 0

    def is_terminal(self) -> bool:
        return self.game_state.game_over


class MCTS:
    """
    Determinized MCTS with neural network guidance.
    Uses determinization to handle hidden information.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        device: torch.device,
        num_simulations: int = 128,
        temperature: float = 1.0,
        dirichlet_alpha: float = 0.3,
        dirichlet_epsilon: float = 0.25,
    ):
        self.model = model
        self.device = device
        self.num_simulations = num_simulations
        self.temperature = temperature
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_epsilon = dirichlet_epsilon

        self.native_search = None
        if NATIVE_EXTENSION_LOADED:
            self.native_search = search_class("ensemble")()

    def search(
        self,
        game: SequenceGame,
        player: int,
        simulations: Optional[int] = None,
        progress_callback: Optional[Callable] = None,
    ) -> Tuple[Move, np.ndarray]:
        """
        Run MCTS search from current game state.

        Args:
            game: Current game state
            player: Player to find move for
            simulations: Optional override for number of simulations
            progress_callback: Optional callback(mcts_instance, current_sim) -> None

        Returns:
            best_move: Selected move
            policy: Visit count distribution over moves
        """
        if player != game.current_player:
            raise ValueError("Search player must be the player to move")
        num_sims = simulations if simulations is not None else self.num_simulations
        if num_sims < 1:
            raise ValueError("Search requires at least one simulation")
        if not game.get_legal_moves(player):
            return None, np.zeros(100, dtype=np.float32)

        if self.native_search:
            return self._search_native(game, player, num_sims, progress_callback)
        else:
            raise RuntimeError(
                "Four-deal search requires sequence_native; install the project with pip install ."
            )

    def _search_native(
        self,
        game: SequenceGame,
        player: int,
        num_simulations: int,
        progress_callback: Optional[Callable] = None,
    ) -> Tuple[Move, np.ndarray]:
        """Run MCTS using the Rust backend."""
        start_time = time.time()

        # Prepare inputs
        board_bytes = game.board.tobytes()
        hand_ints = [c.to_int() for c in game.hands[player]]

        # Reset C tree
        sequences = [
            [sorted(r * 10 + c for r, c in seq) for seq in game.completed_sequences[p]]
            for p in (1, 2)
        ]
        unseen = unseen_cards(game, player)
        self.native_search.reset(
            board_bytes,
            hand_ints,
            player,
            sequences,
            unseen,
            len(game.hands[3 - player]),
        )

        for i in range(num_simulations):
            capsule, tensor_bytes, value, is_terminal_int = (
                self.native_search.select_leaf()
            )
            is_terminal = bool(is_terminal_int)
            policy_bytes = b""

            if not is_terminal:
                # Run inference
                tensor = (
                    torch.from_numpy(
                        np.frombuffer(tensor_bytes, dtype=np.float32).copy()
                    )
                    .reshape(1, STATE_CHANNELS, 10, 10)
                    .to(self.device)
                )

                self.model.eval()
                with torch.no_grad():
                    policy, val = self.model.predict(tensor)

                # Squeeze batch dim
                policy = policy.squeeze(0).cpu().numpy()
                value = val.item()
                policy_bytes = policy.tobytes()

            self.native_search.backpropagate(capsule, policy_bytes, value)

            if progress_callback and (i + 1) % 10 == 0:
                progress_callback(self, i + 1)

        result = self.native_search.get_action(self.temperature)
        if result is None:
            return None, np.zeros(100, dtype=np.float32)
        card_int, row, col, is_rem, policy_bytes = result

        if policy_bytes is None:
            return None, np.array([])

        # Reconstruct policy and build move object
        policy = np.frombuffer(policy_bytes, dtype=np.float32)
        card = Card.from_int(card_int)
        move = Move(card=card, row=row, col=col, is_removal=bool(is_rem))

        print(
            f"Rust MCTS finished in {time.time() - start_time:.2f}s ({num_simulations} sims)"
        )
        return move, policy

    def _search_python(
        self, game: SequenceGame, player: int, num_simulations: int
    ) -> Tuple[Move, np.ndarray]:
        """Reference implementation of determinized MCTS."""
        det_game = self._determinize(game, player)
        root = MCTSNode(game_state=det_game)
        root.total_value = self._expand_node(root, player)
        root.visits = 1

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

            while not node.is_leaf() and not node.is_terminal():
                node = self._select_child(node)
                search_path.append(node)

            if not node.is_terminal():
                value = self._expand_node(node, node.game_state.current_player)
            else:
                value = node.game_state.get_result(node.game_state.current_player)

            self._backpropagate(search_path, value, node.game_state.current_player)

        print(f"Py-MCTS finished in {time.time() - start_time:.2f}s")

        # Select move based on visit counts
        return self._select_move(root)

    def _expand_node(self, node: MCTSNode, player: int) -> float:
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
            return 0.0

        counts = np.bincount([m.row * 10 + m.col for m in legal_moves], minlength=100)
        priors = np.array(
            [
                max(0.0, policy[m.row * 10 + m.col]) / counts[m.row * 10 + m.col]
                for m in legal_moves
            ]
        )
        priors = (
            priors / priors.sum()
            if priors.sum() > 0
            else np.full(len(legal_moves), 1 / len(legal_moves))
        )
        # Create children for each legal move
        for move, prior in zip(legal_moves, priors):
            child_game = game.clone()
            child_game.make_move(move)

            # Map move to policy index

            move_key = (str(move.card), move.row, move.col, move.is_removal)
            child = MCTSNode(game_state=child_game, parent=node, move=move, prior=prior)
            node.children[move_key] = child

        return value

    def _determinize(self, game: SequenceGame, observer: int) -> SequenceGame:
        det_game = game.clone()
        opponent = 3 - observer

        # Shuffle the hidden multiset without changing card counts or consulting its split.
        hidden = [Card.from_int(c) for c in unseen_cards(game, observer)]
        random.shuffle(hidden)
        hand_size = len(det_game.hands[opponent])
        det_game.hands[opponent] = hidden[:hand_size]
        det_game.deck.cards = hidden[hand_size:]

        return det_game

    def _select_child(self, node: MCTSNode) -> MCTSNode:
        """Select child with highest UCB score."""
        best_score = float("-inf")
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
            child.prior = (
                1 - self.dirichlet_epsilon
            ) * child.prior + self.dirichlet_epsilon * noise[i]

    def _backpropagate(self, path: List[MCTSNode], value: float, evaluated_player: int):
        """Backpropagate value through the search path."""
        for node in reversed(path):
            # Flip value for opponent's perspective
            if node.game_state.current_player != evaluated_player:
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
            visit_counts = (visit_counts / max(1.0, visit_counts.max())) ** (
                1.0 / self.temperature
            )
            total = visit_counts.sum()
            if total == 0:
                policy = np.ones_like(visit_counts) / len(visit_counts)
            else:
                policy = visit_counts / total

        # Sample move
        chosen_idx = np.random.choice(len(moves), p=policy)

        spatial_policy = np.zeros(100, dtype=np.float32)
        for move, probability in zip(moves, policy):
            spatial_policy[move.row * 10 + move.col] += probability
        return moves[chosen_idx], spatial_policy


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

    def _count_adjacent(
        self, game: SequenceGame, player: int, row: int, col: int
    ) -> int:
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
