/**
 * Sequence Game Engine - Pure JavaScript Implementation
 * Runs entirely in the browser, no server needed.
 */

// Board layout from BoardDesign.csv - hardcoded for static hosting
const BOARD_LAYOUT = [
    ['FREE', 'AC', 'KC', 'QC', '10C', '9C', '8C', '7C', '6C', 'FREE'],
    ['AD', '7S', '8S', '9S', '10S', 'QS', 'KS', 'AS', '5C', '2S'],
    ['KD', '6S', '10C', '9C', '8C', '7C', '6C', '2D', '4C', '3S'],
    ['QD', '5S', 'QC', '8H', '7H', '6H', '5C', '3D', '3C', '4S'],
    ['10D', '4S', 'KC', '9H', '2H', '5H', '4C', '4D', '2C', '5S'],
    ['9D', '3S', 'AC', '10H', '3H', '4H', '3C', '5D', 'AH', '6S'],
    ['8D', '2S', 'AD', 'QH', 'KH', 'AH', '2C', '6D', 'KH', '7S'],
    ['7D', '2H', 'KD', 'QD', '10D', '9D', '8D', '7D', 'QH', '8S'],
    ['6D', '3H', '4H', '5H', '6H', '7H', '8H', '9H', '10H', '9S'],
    ['FREE', '5D', '4D', '3D', '2D', 'AS', 'KS', 'QS', '10S', 'FREE']
];

const BOARD_SIZE = 10;
const SEQUENCE_LENGTH = 5;
const MAX_HAND_SIZE = 7;
const STATE_CHANNELS = 4 + MAX_HAND_SIZE;

// Chip states
const ChipState = {
    EMPTY: 0,
    PLAYER1: 1,
    PLAYER2: 2,
    FREE: -1
};

// Card utilities
const SUITS = ['C', 'D', 'H', 'S'];
const RANKS = ['A', '2', '3', '4', '5', '6', '7', '8', '9', '10', 'Q', 'K'];

function isRedJack(card) {
    return card === 'JD' || card === 'JC';
}

function isBlackJack(card) {
    return card === 'JH' || card === 'JS';
}

function isJack(card) {
    return card.startsWith('J');
}

function cardToIndex(card) {
    const suit = SUITS.indexOf(card.slice(-1));
    const rank = ['A', '2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K']
        .indexOf(card.slice(0, -1));
    if (suit < 0 || rank < 0) throw new Error('Invalid card: ' + card);
    return suit * 13 + rank;
}

// Build card-to-position mapping
function buildCardPositionMap() {
    const mapping = {};
    for (let r = 0; r < BOARD_SIZE; r++) {
        for (let c = 0; c < BOARD_SIZE; c++) {
            const card = BOARD_LAYOUT[r][c];
            if (card !== 'FREE') {
                if (!mapping[card]) mapping[card] = [];
                mapping[card].push([r, c]);
            }
        }
    }
    return mapping;
}

const CARD_POSITIONS = buildCardPositionMap();

// Create standard 104-card deck (2 decks)
function createDeck() {
    const deck = [];
    for (let i = 0; i < 2; i++) { // 2 decks
        for (const suit of SUITS) {
            for (const rank of RANKS) {
                deck.push(rank + suit);
            }
            // Add Jacks
            deck.push('J' + suit);
        }
    }
    return shuffle(deck);
}

function shuffle(array) {
    const arr = [...array];
    for (let i = arr.length - 1; i > 0; i--) {
        const j = Math.floor(Math.random() * (i + 1));
        [arr[i], arr[j]] = [arr[j], arr[i]];
    }
    return arr;
}

/**
 * Main Game State Class
 */
class SequenceGame {
    constructor({autoReplaceDeadCards = false} = {}) {
        // Reference/training replays keep their original draw rules.
        // The interactive web game enables automatic hand exchanges.
        this.autoReplaceDeadCards = autoReplaceDeadCards;
        this.reset();
    }

    reset() {
        // 10x10 board
        this.board = Array(BOARD_SIZE).fill(null).map(() => Array(BOARD_SIZE).fill(ChipState.EMPTY));

        // Mark corners as FREE
        this.board[0][0] = ChipState.FREE;
        this.board[0][9] = ChipState.FREE;
        this.board[9][0] = ChipState.FREE;
        this.board[9][9] = ChipState.FREE;

        this.currentPlayer = 1;
        this.deck = createDeck();
        this.hands = { 1: [], 2: [] };
        this.discardPiles = { 1: [], 2: [] };
        this.publicEvents = [];
        this.lastCardExchanges = [];
        this.publicHistoryComplete = true;
        this.completedSequences = { 1: [], 2: [] };
        this.gameOver = false;
        this.winner = null;

        // Deal 7 cards each (for 2 players)
        for (let i = 0; i < 7; i++) {
            const card1 = this.drawPlayableCard(1);
            if (card1) this.hands[1].push(card1);

            const card2 = this.drawPlayableCard(2);
            if (card2) this.hands[2].push(card2);
        }
    }

    /**
     * Get all legal moves for current player
     */
    getLegalMoves(player = null) {
        if (player === null) player = this.currentPlayer;
        if (this.gameOver) return [];

        const moves = [];
        const hand = this.hands[player];

        for (const card of new Set(hand)) {
            if (isRedJack(card)) {
                // Two-eyed Jack (Wild): place on any empty space
                for (let r = 0; r < BOARD_SIZE; r++) {
                    for (let c = 0; c < BOARD_SIZE; c++) {
                        if (this.board[r][c] === ChipState.EMPTY) {
                            moves.push({ card, row: r, col: c, is_removal: false });
                        }
                    }
                }
            } else if (isBlackJack(card)) {
                // One-eyed Jack (Remove): remove opponent chip not in sequence
                const opponent = 3 - player;
                const protected_ = this.getProtectedPositions(opponent);
                for (let r = 0; r < BOARD_SIZE; r++) {
                    for (let c = 0; c < BOARD_SIZE; c++) {
                        if (this.board[r][c] === opponent && !protected_.has(`${r},${c}`)) {
                            moves.push({ card, row: r, col: c, is_removal: true });
                        }
                    }
                }
            } else {
                // Normal card: place on matching empty positions
                const positions = CARD_POSITIONS[card] || [];
                for (const [r, c] of positions) {
                    if (this.board[r][c] === ChipState.EMPTY) {
                        moves.push({ card, row: r, col: c, is_removal: false });
                    }
                }
            }
        }

        return moves;
    }

    getProtectedPositions(player) {
        const protected_ = new Set();
        for (const seq of this.completedSequences[player]) {
            for (const pos of seq) {
                protected_.add(pos);
            }
        }
        return protected_;
    }

    /**
     * Check if a card can be played by the given player on current board
     */
    isCardPlayable(card, player) {
        // Two-eyed Jack: JD, JC - Wild
        if (isRedJack(card)) {
            // Playable if there is ANY empty space
            return this.board.some(row => row.some(cell => cell === ChipState.EMPTY));
        }

        // One-eyed Jack: JH, JS - Remove
        if (isBlackJack(card)) {
            // Playable if there is ANY opponent chip that is not protected
            const opponent = 3 - player;
            const protected_ = this.getProtectedPositions(opponent);
            for (let r = 0; r < BOARD_SIZE; r++) {
                for (let c = 0; c < BOARD_SIZE; c++) {
                    if (this.board[r][c] === opponent && !protected_.has(`${r},${c}`)) {
                        return true;
                    }
                }
            }
            return false;
        }

        // Normal Card
        const positions = CARD_POSITIONS[card] || [];
        for (const [r, c] of positions) {
            if (this.board[r][c] === ChipState.EMPTY) {
                return true;
            }
        }
        return false;
    }

    /**
     * Draw a card that is playable. Discards dead cards.
     */
    drawPlayableCard(player) {
        // Safety count to prevent infinite loops (though deck size is finite)

        while (this.deck.length > 0) {
            const card = this.deck.pop();

            if (isJack(card) || this.isCardPlayable(card, player)) {
                return card;
            }
            this.discardPiles[player].push(card);
            this.publicEvents.push([cardToIndex(card), 100, player, 2]);
        }
        return null; // Deck empty
    }

    // Only normal cards with both squares occupied are dead. Keep Jacks,
    // including removal Jacks with no current target.
    replaceDeadCards(player) {
        const exchanges = [];
        const hand = this.hands[player];
        for (let index = 0; index < hand.length;) {
            const card = hand[index];
            if (isJack(card) || this.isCardPlayable(card, player)) { index++; continue; }
            this.discardPiles[player].push(card);
            this.publicEvents.push([cardToIndex(card), 100, player, 2]);
            const replacement = this.drawPlayableCard(player);
            exchanges.push({player, index, card, replacement});
            if (replacement) { hand[index] = replacement; index++; }
            else hand.splice(index, 1);
        }
        return exchanges;
    }

    /**
     * Execute a move
     */
    makeMove(move) {
        if (this.gameOver) return false;

        const player = this.currentPlayer;
        const { card, row, col, is_removal } = move;

        // Validate card in hand
        const cardIndex = this.hands[player].indexOf(card);
        if (cardIndex === -1) return false;

        if (!Number.isInteger(row) || !Number.isInteger(col) ||
            !this.getLegalMoves(player).some(m => m.card === card && m.row === row &&
                m.col === col && m.is_removal === Boolean(is_removal))) return false;

        this.lastCardExchanges = [];

        // Remove card from hand
        this.hands[player].splice(cardIndex, 1);
        this.discardPiles[player].push(card);
        this.publicEvents.push([cardToIndex(card), row * 10 + col, player, Number(Boolean(is_removal))]);

        // Apply move
        if (is_removal) {
            this.board[row][col] = ChipState.EMPTY;
        } else {
            this.board[row][col] = player;

            // Check for new sequences
            const newSeqs = this.findNewSequences(player, row, col);
            this.completedSequences[player].push(...newSeqs);

            // Win condition: 2 sequences
            if (this.completedSequences[player].length >= 2) {
                this.gameOver = true;
                this.winner = player;
            }
        }

        // Draw new card
        if (this.deck.length > 0) {
            const newCard = this.drawPlayableCard(player);
            if (newCard) {
                this.hands[player].push(newCard);
            }
        }

        // Switch player
        this.currentPlayer = 3 - player;
        if (this.autoReplaceDeadCards && !this.gameOver) {
            // The next player draws first if both hands need replacements.
            this.lastCardExchanges = [
                ...this.replaceDeadCards(this.currentPlayer),
                ...this.replaceDeadCards(player)
            ];
        }
        this.finishIfNoLegalMoves();

        return true;
    }

    finishIfNoLegalMoves() {
        if (this.gameOver || this.getLegalMoves().length) return false;
        this.gameOver = true;
        this.winner = null;
        return true;
    }

    findNewSequences(player, row, col) {
        const found = [];
        for (const [dr, dc] of [[0, 1], [1, 0], [1, 1], [1, -1]]) {
            for (let offset = SEQUENCE_LENGTH - 1; offset >= 0; offset--) {
                const cells = Array.from({length: SEQUENCE_LENGTH}, (_, k) =>
                    [row + (k - offset) * dr, col + (k - offset) * dc]);
                if (!cells.every(([r, c]) => r >= 0 && r < BOARD_SIZE && c >= 0 &&
                    c < BOARD_SIZE && this.isPlayerChip(player, r, c))) continue;
                const sequence = cells.map(([r, c]) => `${r},${c}`);
                if ([...this.completedSequences[player], ...found].every(old =>
                    sequence.filter(pos => old.includes(pos)).length <= 1)) found.push(sequence);
            }
        }
        return found;
    }

    isPlayerChip(player, row, col) {
        const state = this.board[row][col];
        return state === player || state === ChipState.FREE;
    }

    /**
     * Convert game state to neural network input tensor
     * Shape: [1, 11, 10, 10]
     */
    getHistoryInputs(player = this.currentPlayer) {
        if (!this.publicHistoryComplete) {
            throw new Error('This position has no complete move history. Start a new game to use the history model.');
        }
        if (player !== 1 && player !== 2) throw new Error('Invalid player');
        if (this.publicEvents.length >= 128) throw new Error('Public history exceeds model capacity');
        const context = new Float32Array(157);
        for (const [offset, cards] of [[0, this.hands[player]],
            [52, this.discardPiles[player]], [104, this.discardPiles[3 - player]]]) {
            for (const card of cards) context[offset + cardToIndex(card)] += 0.5;
        }
        // Opponent hand size is public; its card identities and deck order are not inputs.
        context[156] = this.hands[3 - player].length / 7;
        const history = new BigInt64Array(128 * 4);
        history.set([53n, 100n, 0n, 3n]);
        this.publicEvents.forEach(([card, position, actor, kind], index) => {
            history.set([BigInt(card + 1), BigInt(position), BigInt(Number(actor !== player)),
                BigInt(kind)], (index + 1) * 4);
        });
        return {board: this.getStateTensor(player), context, history};
    }

    getExactInputs({allowIncompleteHistory = false} = {}) {
        const player = this.currentPlayer;
        if (this.hands[player].length > MAX_HAND_SIZE) {
            throw new Error('Exact policy supports at most seven cards per hand');
        }
        let inputs;
        if (!this.publicHistoryComplete && allowIncompleteHistory) {
            // Explicit board-hand-only exports ignore all history and context
            // beyond these own-card counts. Zero buffers fill unused ONNX
            // inputs; they do not represent a reconstructed public history.
            const context = new Float32Array(157);
            for (const card of this.hands[player]) context[cardToIndex(card)] += 0.5;
            inputs = {context, history: new BigInt64Array(128 * 4)};
        } else {
            inputs = this.getHistoryInputs(player);
        }
        inputs.board = this.getStateTensor(player, false, true);
        const byId = new Map();
        for (const move of this.getLegalMoves()) {
            const id = cardToIndex(move.card) * 100 + move.row * 10 + move.col;
            if (!byId.has(id)) byId.set(id, move);
        }
        const ids = Array.from(byId.keys()).sort((a, b) => a - b);
        return {...inputs, actions: BigInt64Array.from(ids, BigInt),
            moves: ids.map(id => byId.get(id))};
    }

    getStateTensor(player, rotate = false, canonicalHand = false) {
        const opponent = 3 - player;
        const tensor = new Float32Array(STATE_CHANNELS * 10 * 10);

        for (let r = 0; r < BOARD_SIZE; r++) {
            for (let c = 0; c < BOARD_SIZE; c++) {
                const rawIdx = c + r * 10;
                // Optional display rotation; model inference uses fixed coordinates.
                const destIdx = rotate ? (99 - rawIdx) : rawIdx;
                const state = this.board[r][c];

                // Channel 0: Current player chips
                tensor[0 * 100 + destIdx] = state === player ? 1 : 0;

                // Channel 1: Opponent chips
                tensor[1 * 100 + destIdx] = state === opponent ? 1 : 0;

                // Channel 2: Free corners
                tensor[2 * 100 + destIdx] = state === ChipState.FREE ? 1 : 0;

                // Channel 3: Empty spaces
                tensor[3 * 100 + destIdx] = state === ChipState.EMPTY ? 1 : 0;
            }
        }

        // Channels 4-10: Playable positions for each hand card
        const hand = canonicalHand
            ? [...this.hands[player]].sort((a, b) => cardToIndex(a) - cardToIndex(b))
            : this.hands[player];
        for (let i = 0; i < Math.min(MAX_HAND_SIZE, hand.length); i++) {
            const card = hand[i];
            const channelOffset = (4 + i) * 100;
            const positions = CARD_POSITIONS[card] || [];
            for (const [r, c] of positions) {
                if (this.board[r][c] === ChipState.EMPTY) {
                    const rawIdx = c + r * 10;
                    const destIdx = rotate ? (99 - rawIdx) : rawIdx;
                    tensor[channelOffset + destIdx] = 1;
                }
            }
            // Handle Jacks
            if (isRedJack(card)) {
                for (let r = 0; r < BOARD_SIZE; r++) {
                    for (let c = 0; c < BOARD_SIZE; c++) {
                        if (this.board[r][c] === ChipState.EMPTY) {
                            const rawIdx = c + r * 10;
                            const destIdx = rotate ? (99 - rawIdx) : rawIdx;
                            tensor[channelOffset + destIdx] = 1;
                        }
                    }
                }
            } else if (isBlackJack(card)) {
                const protected_ = this.getProtectedPositions(opponent);
                for (let r = 0; r < BOARD_SIZE; r++) {
                    for (let c = 0; c < BOARD_SIZE; c++) {
                        if (this.board[r][c] === opponent && !protected_.has(`${r},${c}`)) {
                            const rawIdx = c + r * 10;
                            const destIdx = rotate ? (99 - rawIdx) : rawIdx;
                            tensor[channelOffset + destIdx] = 1;
                        }
                    }
                }
            }
        }

        return tensor;
    }
}

// Export for use in other modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = { SequenceGame, BOARD_LAYOUT, ChipState, CARD_POSITIONS, STATE_CHANNELS };
}
