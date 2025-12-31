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
    constructor() {
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

        for (const card of hand) {
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
        // Red Jack (Two-eyed): JD, JC - Wild
        if (isRedJack(card)) {
            // Playable if there is ANY empty space
            return this.board.some(row => row.some(cell => cell === ChipState.EMPTY));
        }

        // Black Jack (One-eyed): JH, JS - Remove
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
        let checks = 0;
        const maxChecks = this.deck.length + 1;

        while (this.deck.length > 0) {
            const card = this.deck.pop();
            checks++;

            if (this.isCardPlayable(card, player)) {
                return card;
            }
            // Card is dead, discard it (loop continues)
            // Ideally we might want to log this or animate it, but for logic fixes this is sufficient.
        }
        return null; // Deck empty
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

        // Remove card from hand
        this.hands[player].splice(cardIndex, 1);

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

        return true;
    }

    findNewSequences(player, row, col) {
        const directions = [[0, 1], [1, 0], [1, 1], [1, -1]];
        const newSequences = [];

        for (const [dr, dc] of directions) {
            const seq = this.findSequenceInDirection(player, row, col, dr, dc);
            if (seq && seq.length >= SEQUENCE_LENGTH) {
                // Check not duplicate
                const seqKey = [...seq].sort().join('|');
                let isDuplicate = false;
                for (const existing of this.completedSequences[player]) {
                    const overlap = seq.filter(pos => existing.includes(pos));
                    if (overlap.length > 1) {
                        isDuplicate = true;
                        break;
                    }
                }
                if (!isDuplicate) {
                    newSequences.push(seq.slice(0, SEQUENCE_LENGTH));
                }
            }
        }

        return newSequences;
    }

    findSequenceInDirection(player, row, col, dr, dc) {
        const positions = [];

        // Scan backward
        let r = row - dr, c = col - dc;
        while (r >= 0 && r < BOARD_SIZE && c >= 0 && c < BOARD_SIZE) {
            if (this.isPlayerChip(player, r, c)) {
                positions.unshift(`${r},${c}`);
                r -= dr;
                c -= dc;
            } else break;
        }

        positions.push(`${row},${col}`);

        // Scan forward
        r = row + dr;
        c = col + dc;
        while (r >= 0 && r < BOARD_SIZE && c >= 0 && c < BOARD_SIZE) {
            if (this.isPlayerChip(player, r, c)) {
                positions.push(`${r},${c}`);
                r += dr;
                c += dc;
            } else break;
        }

        return positions.length >= SEQUENCE_LENGTH ? positions : null;
    }

    isPlayerChip(player, row, col) {
        const state = this.board[row][col];
        return state === player || state === ChipState.FREE;
    }

    /**
     * Convert game state to neural network input tensor
     * Shape: [1, 8, 10, 10]
     */
    getStateTensor(player, rotate = false) {
        const opponent = 3 - player;
        const tensor = new Float32Array(8 * 10 * 10);

        for (let r = 0; r < BOARD_SIZE; r++) {
            for (let c = 0; c < BOARD_SIZE; c++) {
                const rawIdx = c + r * 10;
                // If rotating (P2), map current (r,c) to the rotated position
                // We fill the tensor at 'destIdx' with the value from 'rawIdx'
                // Wait. We iterate board r,c. 
                // If board[0][0] has a chip.
                // In rotated view, that chip is at [9][9].
                // So tensor index for [9][9] should get the value.
                // Index 99.
                // 99 = 99 - 0. Correct.
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

        // Channels 4-7: Playable positions for hand cards
        const hand = this.hands[player];
        for (let i = 0; i < Math.min(4, hand.length); i++) {
            const card = hand[i];
            const positions = CARD_POSITIONS[card] || [];
            for (const [r, c] of positions) {
                if (this.board[r][c] === ChipState.EMPTY) {
                    const rawIdx = c + r * 10;
                    const destIdx = rotate ? (99 - rawIdx) : rawIdx;
                    tensor[(4 + i) * 100 + destIdx] = 1;
                }
            }
            // Handle Jacks
            if (isRedJack(card)) {
                for (let r = 0; r < BOARD_SIZE; r++) {
                    for (let c = 0; c < BOARD_SIZE; c++) {
                        if (this.board[r][c] === ChipState.EMPTY) {
                            const rawIdx = c + r * 10;
                            const destIdx = rotate ? (99 - rawIdx) : rawIdx;
                            tensor[(4 + i) * 100 + destIdx] = 1;
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
    module.exports = { SequenceGame, BOARD_LAYOUT, ChipState, CARD_POSITIONS };
}
