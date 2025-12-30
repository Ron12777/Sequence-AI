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
    async search(game, progressCallback = null) {
        const root = new MCTSNode(cloneGame(game), null, null, game.currentPlayer);

        for (let i = 0; i < this.numSimulations; i++) {
            await this.simulate(root);

            if (progressCallback) {
                progressCallback(i + 1, this.numSimulations);
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
            const value = node.game.winner === root.game.currentPlayer ? 1 : -1;
            this.backpropagate(node, value);
            return;
        }

        // Expansion
        const moves = node.game.getLegalMoves();
        if (moves.length === 0) {
            this.backpropagate(node, 0);
            return;
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

        // Get state tensor
        const stateTensor = game.getStateTensor(game.currentPlayer);

        // Create ONNX tensor - shape [1, 8, 10, 10]
        const inputTensor = new ort.Tensor('float32', stateTensor, [1, 8, 10, 10]);

        // Run inference
        const results = await this.session.run({ input: inputTensor });

        // Extract policy and value
        const policyLogits = results.policy.data;
        const value = results.value.data[0];

        // Softmax on policy
        const policy = softmax(policyLogits);

        return { policy, value };
    }

    /**
     * Backpropagate value through tree
     */
    backpropagate(node, value) {
        while (node !== null) {
            node.visits += 1;
            // Value is from perspective of root player
            node.value += value;
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
