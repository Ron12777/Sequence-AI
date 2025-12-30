/**
 * Monte Carlo Tree Search (MCTS) - JavaScript Implementation
 * Uses ONNX.js for neural network inference in the browser.
 */

// MCTS Node
class MCTSNode {
    constructor(game, parent = null, move = null, player = null) {
        this.game = game; // Game state (cloned)
        this.parent = parent;
        this.move = move; // Move that led to this node
        this.player = player; // Player who made the move

        this.children = [];
        this.visits = 0;
        this.value = 0;
        this.prior = 0; // Neural network prior probability
        this.isExpanded = false;
    }

    /**
     * UCB1 score for selection
     */
    ucbScore(cPuct = 1.4) {
        if (this.visits === 0) {
            return Infinity;
        }
        const exploitation = this.value / this.visits;
        const exploration = cPuct * this.prior * Math.sqrt(this.parent.visits) / (1 + this.visits);
        return exploitation + exploration;
    }

    /**
     * Select best child by UCB score
     */
    selectChild() {
        let bestChild = null;
        let bestScore = -Infinity;
        for (const child of this.children) {
            const score = child.ucbScore();
            if (score > bestScore) {
                bestScore = score;
                bestChild = child;
            }
        }
        return bestChild;
    }

    /**
     * Get best move by visit count (for final selection)
     */
    getBestMove() {
        let bestChild = null;
        let mostVisits = -1;
        for (const child of this.children) {
            if (child.visits > mostVisits) {
                mostVisits = child.visits;
                bestChild = child;
            }
        }
        return bestChild ? bestChild.move : null;
    }

    /**
     * Get policy distribution (normalized visit counts)
     */
    getPolicy() {
        const policy = new Float32Array(100).fill(0);
        let totalVisits = 0;

        for (const child of this.children) {
            if (!child.move.is_removal) {
                const idx = child.move.row * 10 + child.move.col;
                policy[idx] = child.visits;
                totalVisits += child.visits;
            }
        }

        if (totalVisits > 0) {
            for (let i = 0; i < 100; i++) {
                policy[i] /= totalVisits;
            }
        }

        return policy;
    }
}

/**
 * Deep clone a game state
 */
function cloneGame(game) {
    const clone = new SequenceGame();
    clone.board = game.board.map(row => [...row]);
    clone.currentPlayer = game.currentPlayer;
    clone.deck = [...game.deck];
    clone.hands = {
        1: [...game.hands[1]],
        2: [...game.hands[2]]
    };
    clone.completedSequences = {
        1: game.completedSequences[1].map(seq => [...seq]),
        2: game.completedSequences[2].map(seq => [...seq])
    };
    clone.gameOver = game.gameOver;
    clone.winner = game.winner;
    return clone;
}

/**
 * MCTS Search
 */
class MCTS {
    constructor(onnxSession = null, numSimulations = 50) {
        this.session = onnxSession;
        this.numSimulations = numSimulations;
    }

    /**
     * Run MCTS search and return best move
     */
    /**
     * Run MCTS search and return best move
     */
    async search(game, progressCallback = null) {
        const root = new MCTSNode(cloneGame(game), null, null, game.currentPlayer);

        for (let i = 0; i < this.numSimulations; i++) {
            await this.simulate(root);

            if (progressCallback) {
                // Calculate root value (win chance)
                // Normalize roughly to [-1, 1] -> [0, 1] relative to current player
                // But typically node.value accumulates +1/-1. 
                // So average is between -1 and 1.
                const visits = root.visits || 1;
                const avgValue = root.value / visits;
                progressCallback(i + 1, this.numSimulations, root.getPolicy(), avgValue);
            }

            // critical: Yield to main thread to allow UI rendering
            // If we don't do this, the browser freezes and no progress bar/highlights appear
            if (i % 5 === 0) {
                await new Promise(resolve => setTimeout(resolve, 0));
            }
        }

        return {
            move: root.getBestMove(),
            policy: root.getPolicy()
        };
    }

    /**
     * Single MCTS simulation: select -> expand -> evaluate -> backprop
     */
    async simulate(root) {
        let node = root;

        // Selection: traverse to leaf
        while (node.isExpanded && node.children.length > 0 && !node.game.gameOver) {
            node = node.selectChild();
        }

        // Check terminal
        if (node.game.gameOver) {
            // Value must be from the perspective of the player whose turn it WOULD be at this node.
            // (Strictly, this node represents the state AFTER 'node.player' moved).
            // So if node.player (Just Moved) Won, then Current Perspective (Opponent) Lost (-1).

            // Note: node.game.winner is 1 or 2.
            // node.game.currentPlayer is the next player.

            let value = 0;
            if (node.game.winner === node.game.currentPlayer) {
                value = 1; // Unlikely if they just moved, but handling consistency
            } else {
                value = -1; // The player who just moved won, so "Current Perspective" lost.
            }

            this.backpropagate(node, value);
            return;
        }

        // Expansion
        const moves = node.game.getLegalMoves();
        if (moves.length === 0) {
            this.backpropagate(node, 0);
            return;
        }

        // Shuffle moves to ensure random exploration order
        // This prevents the AI from only exploring the top-left of the board
        // when simulations are low and priors are flat (e.g. Jacks).
        for (let i = moves.length - 1; i > 0; i--) {
            const j = Math.floor(Math.random() * (i + 1));
            [moves[i], moves[j]] = [moves[j], moves[i]];
        }

        // Get neural network evaluation
        let priors, value;
        if (this.session) {
            const result = await this.evaluate(node.game);
            priors = result.policy;
            value = result.value;
        } else {
            // Fallback: uniform priors, random value
            priors = new Float32Array(100).fill(1 / 100);
            value = Math.random() * 2 - 1;
        }

        // Create child nodes
        for (const move of moves) {
            const childGame = cloneGame(node.game);
            childGame.makeMove(move);

            const child = new MCTSNode(childGame, node, move, node.game.currentPlayer);

            // Set prior from policy
            if (!move.is_removal) {
                const idx = move.row * 10 + move.col;
                child.prior = priors[idx];
            } else {
                child.prior = 0.01; // Small prior for removals
            }

            node.children.push(child);
        }

        node.isExpanded = true;

        // Backpropagate
        this.backpropagate(node, value);
    }

    /**
     * Neural network evaluation using ONNX.js
     */
    async evaluate(game) {
        if (!this.session) {
            return {
                policy: new Float32Array(100).fill(1 / 100),
                value: 0
            };
        }

        // Canonical form: If P2, rotate board 180 degrees
        const isP2 = game.currentPlayer === 2;

        // Get state tensor (rotate if P2)
        const stateTensor = game.getStateTensor(game.currentPlayer, isP2);

        // Create ONNX tensor - shape [1, 8, 10, 10]
        const inputTensor = new ort.Tensor('float32', stateTensor, [1, 8, 10, 10]);

        // Run inference
        const results = await this.session.run({ input: inputTensor });

        // Extract policy and value
        const policyLogits = results.policy.data;
        const value = results.value.data[0];

        // Softmax on policy
        const policy = softmax(policyLogits);

        // If P2, we must un-rotate the policy (reverse the array) 
        // to match the real board coordinates
        if (isP2) {
            policy.reverse();
        }

        return { policy, value };
    }

    /**
     * Backpropagate value through tree
     */
    backpropagate(node, value) {
        while (node !== null) {
            node.visits += 1;
            node.value += value;

            // Value is relative to the player at 'node'
            // When moving to parent (opponent), we must negate the value
            value = -value;
            node = node.parent;
        }
    }
}

/**
 * Softmax function
 */
function softmax(logits) {
    const maxLogit = Math.max(...logits);
    const exps = logits.map(x => Math.exp(x - maxLogit));
    const sum = exps.reduce((a, b) => a + b, 0);
    return exps.map(x => x / sum);
}

// Export
if (typeof module !== 'undefined' && module.exports) {
    module.exports = { MCTS, MCTSNode, cloneGame };
}
