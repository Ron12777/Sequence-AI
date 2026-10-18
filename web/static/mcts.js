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
        const exploitation = this.visits ? this.value / this.visits : 0;
        const exploration = cPuct * this.prior * Math.sqrt(Math.max(1, this.parent.visits)) / (1 + this.visits);
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
            if (child.visits > mostVisits || (child.visits === mostVisits && child.prior > (bestChild?.prior ?? -1))) {
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
            const idx = child.move.row * 10 + child.move.col;
            policy[idx] += child.visits;
            totalVisits += child.visits;
        }
        if (totalVisits > 0) {
            for (let i = 0; i < 100; i++) policy[i] /= totalVisits;
        } else {
            for (const child of this.children) {
                policy[child.move.row * 10 + child.move.col] += child.prior;
            }
        }

        return policy;
    }
}

/**
 * Deep clone a game state
 */
function cloneGame(game) {
    const clone = Object.create(SequenceGame.prototype);
    clone.board = game.board.map(row => [...row]);
    clone.currentPlayer = game.currentPlayer;
    clone.deck = [...game.deck];
    clone.hands = {
        1: [...game.hands[1]],
        2: [...game.hands[2]]
    };
    clone.discardPiles = {1: [...game.discardPiles[1]], 2: [...game.discardPiles[2]]};
    clone.publicEvents = game.publicEvents.map(event => [...event]);
    clone.publicHistoryComplete = game.publicHistoryComplete;
    clone.completedSequences = {
        1: game.completedSequences[1].map(seq => [...seq]),
        2: game.completedSequences[2].map(seq => [...seq])
    };
    clone.gameOver = game.gameOver;
    clone.winner = game.winner;
    return clone;
}

// Build the hidden multiset without inspecting actual hidden card identities.
function unseenCards(game, observer) {
    const counts = new Map();
    for (const suit of SUITS) for (const rank of [...RANKS, 'J']) counts.set(rank + suit, 2);
    for (const card of [...game.hands[observer], ...game.discardPiles[1], ...game.discardPiles[2]]) {
        const remaining = (counts.get(card) ?? 0) - 1;
        if (remaining < 0) throw new Error('Public card history exceeds two-deck counts');
        counts.set(card, remaining);
    }
    return [...counts].flatMap(([card, count]) => Array(count).fill(card)).sort();
}

// Keep card identity when combining actions; only the training policy is spatial.
function aggregateRoots(roots) {
    const actions = new Map();
    for (const root of roots) for (const child of root.children) {
        const m = child.move;
        const key = JSON.stringify([m.card, m.row, m.col, Boolean(m.is_removal)]);
        if (!actions.has(key)) actions.set(key, {move: m, visits: 0, prior: 0});
        const action = actions.get(key);
        action.visits += child.visits;
        action.prior += child.prior;
    }
    const total = [...actions.values()].reduce((n, a) => n + a.visits, 0);
    const priorTotal = [...actions.values()].reduce((n, a) => n + a.prior, 0);
    const policy = new Float32Array(100);
    let best = null;
    for (const action of actions.values()) {
        action.weight = total ? action.visits / total : action.prior / Math.max(priorTotal, 1e-12);
        policy[action.move.row * 10 + action.move.col] += action.weight;
        if (!best || action.weight > best.weight ||
            (action.weight === best.weight && action.prior > best.prior)) best = action;
    }
    return {move: best?.move ?? null, policy};
}

/**
 * MCTS Search
 */
class MCTS {
    constructor(onnxSession = null, numSimulations = 128, numDeals = 4) {
        this.session = onnxSession;
        this.numSimulations = numSimulations;
        this.numDeals = numDeals;
    }

    async search(game, progressCallback = null) {
        if (!Number.isInteger(this.numSimulations) || this.numSimulations < 1 ||
            !Number.isInteger(this.numDeals) || this.numDeals < 1 || this.numDeals > 32) {
            throw new Error('Search needs positive simulations and 1–32 hidden deals');
        }
        const opponent = 3 - game.currentPlayer;
        const handSize = game.hands[opponent].length;
        const unseen = unseenCards(game, game.currentPlayer);
        const count = Math.min(this.numDeals, this.numSimulations);
        const roots = Array.from({length: count}, () => {
            const position = cloneGame(game);
            const hidden = shuffle(unseen);
            position.hands[opponent] = hidden.slice(0, handSize);
            position.deck = hidden.slice(handSize);
            return new MCTSNode(position, null, null, opponent);
        });
        for (let i = 0; i < this.numSimulations; i++) {
            await this.simulate(roots[i % roots.length]);
            if (progressCallback) {
                const summary = aggregateRoots(roots);
                const visits = roots.reduce((n, root) => n + root.visits, 0);
                const value = -roots.reduce((n, root) => n + root.value, 0) / Math.max(1, visits);
                progressCallback(i + 1, this.numSimulations, summary.policy, value);
            }
            if (i % 5 === 0) await new Promise(resolve => setTimeout(resolve, 0));
        }
        return {...aggregateRoots(roots), deals: roots.length};
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

        if (node.game.gameOver) {
            let value = 0;
            if (node.game.winner === node.player) {
                value = 1;
            } else if (node.game.winner) {
                value = -1;
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

        // Shuffle moves for exploration
        for (let i = moves.length - 1; i > 0; i--) {
            const j = Math.floor(Math.random() * (i + 1));
            [moves[i], moves[j]] = [moves[j], moves[i]];
        }

        let priors, value;
        if (this.session) {
            const result = await this.evaluate(node.game);
            priors = result.policy;
            value = -result.value;
        } else {
            // Fallback: uniform priors, random value
            priors = new Float32Array(100).fill(1 / 100);
            value = 0;
        }

        // Create child nodes
        for (const move of moves) {
            const childGame = cloneGame(node.game);
            childGame.makeMove(move);

            const child = new MCTSNode(childGame, node, move, node.game.currentPlayer);

            child.prior = priors[move.row * 10 + move.col];

            node.children.push(child);
        }

        const counts = new Map();
        for (const child of node.children) {
            const idx = child.move.row * 10 + child.move.col;
            counts.set(idx, (counts.get(idx) || 0) + 1);
        }
        let total = 0;
        for (const child of node.children) {
            child.prior /= counts.get(child.move.row * 10 + child.move.col);
            total += child.prior;
        }
        for (const child of node.children) {
            child.prior = total > 0 ? child.prior / total : 1 / node.children.length;
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
        const isP2 = false; // Training uses player-relative channels in fixed board coordinates.

        // Get state tensor (rotate if P2)
        const stateTensor = game.getStateTensor(game.currentPlayer, isP2);

        // Create ONNX tensor - shape [1, 11, 10, 10]
        const inputTensor = new ort.Tensor('float32', stateTensor, [1, STATE_CHANNELS, 10, 10]);

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

    backpropagate(node, value) {
        while (node !== null) {
            node.visits += 1;
            node.value += value;
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
    module.exports = { MCTS, MCTSNode, cloneGame, unseenCards };
}
