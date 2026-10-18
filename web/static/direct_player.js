/** Direct neural policy. One network call; no hidden-hand sampling. */
class DirectPlayer {
    constructor(session, runtime = globalThis.ort, config = {}) {
        this.session = session;
        this.runtime = runtime;
        this.exactActions = config.policyKind === 'exact_sequence_policy';
        this.valueMode = config.valueMode || 'bounded';
        // Only an explicitly exported board-only CNN may analyze edited
        // positions. Missing capability metadata retains the history guard.
        this.boardHandOnly = this.exactActions && config.observationMode === 'board_hand';
        if (this.boardHandOnly && (config.variant !== 'cnn' || config.historyArchitecture !== 'board')) {
            throw new Error('Board-hand observation mode requires a board-only CNN');
        }
        if (config.observationMode && !['board_hand', 'public_history'].includes(config.observationMode)) {
            throw new Error('Unsupported observation mode');
        }
        if (config.policyKind && !this.exactActions) throw new Error('Unsupported direct policy kind');
        if (this.exactActions && config.actionEncoding !== 'card_id * 100 + row * 10 + col') {
            throw new Error('Unsupported exact action encoding');
        }
        const exactInputs = session.inputNames.length === 4 &&
            ['board', 'context', 'history', 'actions'].every(name => session.inputNames.includes(name));
        if (this.exactActions && !exactInputs) throw new Error('Exact policy requires board, context, history, and actions inputs');
        this.usesHistory = session.inputNames.length === 3 &&
            ['board', 'context', 'history'].every(name => session.inputNames.includes(name));
        this.boardInput = session.inputNames.length === 1 &&
            ['input', 'board'].includes(session.inputNames[0]) ? session.inputNames[0] : null;
        if (!this.exactActions && !this.usesHistory && !this.boardInput) throw new Error('Unsupported direct policy inputs');
    }

    async chooseMove(game) {
        const moves = game.getLegalMoves();
        if (game.gameOver || !moves.length) {
            return {move: null, policy: new Float32Array(100), value: 0};
        }
        const Tensor = this.runtime.Tensor;
        let feeds, exactMoves;
        if (this.usesHistory || this.exactActions) {
            const inputs = this.exactActions
                ? game.getExactInputs({allowIncompleteHistory: this.boardHandOnly}) : game.getHistoryInputs();
            feeds = {
                board: new Tensor('float32', inputs.board, [1, 11, 10, 10]),
                context: new Tensor('float32', inputs.context, [1, 157]),
                history: new Tensor('int64', inputs.history, [1, 128, 4])
            };
            if (this.exactActions) {
                exactMoves = inputs.moves;
                feeds.actions = new Tensor('int64', inputs.actions, [1, inputs.actions.length]);
            }
        } else {
            feeds = {[this.boardInput]: new Tensor('float32',
                game.getStateTensor(game.currentPlayer), [1, 11, 10, 10])};
        }
        const outputs = await this.session.run(feeds);
        const logits = (this.exactActions ? outputs.logits : outputs.policy)?.data;
        const value = outputs.value?.data[0];
        if (!logits || logits.length !== (this.exactActions ? exactMoves.length : 100) ||
            !Array.from(logits).every(Number.isFinite) || !Number.isFinite(value)) {
            throw new Error('Direct model returned invalid predictions');
        }
        if (this.exactActions) {
            let best = 0;
            for (let index = 1; index < logits.length; index++) {
                if (logits[index] > logits[best]) best = index;
            }
            const peak = logits[best];
            const weights = Array.from(logits, score => Math.exp(score - peak));
            const total = weights.reduce((sum, weight) => sum + weight, 0);
            const policy = new Float32Array(100);
            exactMoves.forEach((move, index) => {
                policy[move.row * 10 + move.col] += weights[index] / total;
            });
            // The network chooses the card identity as well as its square.
            // Aggregate only the displayed heatmap; retain the chosen exact move.
            return {move: exactMoves[best], policy, value};
        }
        // Match Python direct_moves: maximize legal square score, preferring a
        // normal card when two cards play the same equally scored square.
        let move = moves[0];
        for (const candidate of moves.slice(1)) {
            const score = logits[candidate.row * 10 + candidate.col];
            const best = logits[move.row * 10 + move.col];
            if (score > best || (score === best && isJack(move.card) && !isJack(candidate.card))) {
                move = candidate;
            }
        }
        const positions = new Set(moves.map(m => m.row * 10 + m.col));
        const peak = Math.max(...Array.from(positions, p => logits[p]));
        const policy = new Float32Array(100);
        let total = 0;
        for (const p of positions) total += policy[p] = Math.exp(logits[p] - peak);
        for (const p of positions) policy[p] /= total;
        return {move, policy, value};
    }
}

if (typeof module !== 'undefined' && module.exports) module.exports = {DirectPlayer};
