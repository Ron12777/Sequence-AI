/**
 * Sequence AI - Client-Side Game UI
 * Fully runs in the browser with ONNX.js neural network inference.
 */

// Game state
let game = null;
let mcts = null;
let onnxSession = null;
let selectedCard = null;
let hoveredCard = null;
let watchMode = false;
let isThinking = false;

// DOM Elements
const boardEl = document.getElementById('board');
const handP1El = document.getElementById('handP1');
const handP2El = document.getElementById('handP2');
const turnIndicatorEl = document.getElementById('turnIndicator');
const playerSequences = document.getElementById('playerSequences');
const aiSequences = document.getElementById('aiSequences');
const gameOverOverlay = document.getElementById('gameOverOverlay');
const gameOverText = document.getElementById('gameOverText');
const newGameBtn = document.getElementById('newGameBtn');
const watchAiBtn = document.getElementById('watchAiBtn');
const hintBtn = document.getElementById('hintBtn');
const aiProgressContainer = document.getElementById('aiProgressContainer');
const aiProgressBar = document.getElementById('aiProgressBar');
const aiProgressText = document.getElementById('aiProgressText');

// Suit symbols
const SUIT_SYMBOLS = {
    'C': '♣',
    'D': '♦',
    'H': '♥',
    'S': '♠'
};

// Initialize
document.addEventListener('DOMContentLoaded', async () => {
    newGameBtn.addEventListener('click', () => {
        watchMode = false;
        newGame();
    });
    watchAiBtn.addEventListener('click', watchAiVsAi);
    hintBtn.addEventListener('click', getHint);

    // Depth slider
    const depthSlider = document.getElementById('depth');
    const depthValue = document.getElementById('depthValue');
    const difficultyLabel = document.getElementById('difficultyLabel');

    function getDifficultyLabel(depth) {
        if (depth <= 5) return 'Instant';
        if (depth <= 15) return 'Super Easy';
        if (depth <= 30) return 'Easy';
        if (depth <= 75) return 'Medium';
        if (depth <= 150) return 'Hard';
        if (depth <= 300) return 'Very Hard';
        return 'Extreme';
    }

    function updateDepthDisplay() {
        const val = parseInt(depthSlider.value);
        depthValue.textContent = val;
        difficultyLabel.textContent = getDifficultyLabel(val);
    }

    if (depthSlider && depthValue && difficultyLabel) {
        depthSlider.addEventListener('input', updateDepthDisplay);
        updateDepthDisplay();
    }

    // Setup Play Again button
    const playAgainBtn = document.querySelector('.overlay-content .btn');
    if (playAgainBtn) {
        playAgainBtn.addEventListener('click', newGame);
    }

    // Load ONNX model
    await loadModel();

    // Start game
    newGame();
});

/**
 * Load ONNX neural network model
 */
async function loadModel() {
    try {
        console.log('Loading ONNX model...');
        onnxSession = await ort.InferenceSession.create('model.onnx');
        console.log('✓ ONNX model loaded');
        mcts = new MCTS(onnxSession, 50);
    } catch (e) {
        console.warn('Could not load ONNX model, using random policy:', e);
        mcts = new MCTS(null, 50); // Fallback to random
    }
}

/**
 * Start a new game
 */
function newGame() {
    game = new SequenceGame();
    game.reset();
    selectedCard = null;
    hoveredCard = null;
    isThinking = false;

    if (gameOverOverlay) {
        gameOverOverlay.classList.remove('visible');
    }

    renderBoard();
    renderP1Hand();
    renderP2Hand();
    updateStatus();
    renderScores();

    // Hide progress bar
    if (aiProgressContainer) {
        aiProgressContainer.classList.remove('visible');
    }
}

/**
 * Watch AI vs AI
 */
async function watchAiVsAi() {
    watchMode = true;
    newGame();

    while (!game.gameOver) {
        await triggerAiTurn();
        await sleep(500); // Pause between moves
    }
}

function sleep(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
}

/**
 * Trigger AI turn
 */
async function triggerAiTurn() {
    if (game.gameOver || isThinking) return;

    isThinking = true;
    updateStatus();

    // Show progress
    if (aiProgressContainer) {
        aiProgressContainer.classList.add('visible');
        aiProgressBar.style.width = '0%';
        aiProgressText.textContent = 'Thinking...';
    }

    const depthSlider = document.getElementById('depth');
    const simulations = depthSlider ? parseInt(depthSlider.value) : 50;
    mcts.numSimulations = simulations;

    try {
        const result = await mcts.search(game, (current, total) => {
            const pct = Math.floor((current / total) * 100);
            if (aiProgressBar) aiProgressBar.style.width = pct + '%';
            if (aiProgressText) aiProgressText.textContent = `${current}/${total} simulations`;
        });

        if (result.move) {
            game.makeMove(result.move);
        }
    } catch (e) {
        console.error('MCTS error:', e);
    }

    isThinking = false;

    // Hide progress
    if (aiProgressContainer) {
        aiProgressContainer.classList.remove('visible');
    }

    renderBoard();
    renderP1Hand();
    renderP2Hand();
    updateStatus();
    renderScores();

    if (game.gameOver) {
        showGameOver();
    }
}

/**
 * Render the game board
 */
function renderBoard() {
    boardEl.innerHTML = '';

    for (let row = 0; row < 10; row++) {
        for (let col = 0; col < 10; col++) {
            const cell = document.createElement('div');
            cell.className = 'cell';
            cell.dataset.row = row;
            cell.dataset.col = col;

            const cardLabel = BOARD_LAYOUT[row][col];

            if (cardLabel === 'FREE') {
                cell.classList.add('free-space');
                cell.innerHTML = `<span style="font-size:2rem">★</span>`;
            } else {
                const suitChar = cardLabel.slice(-1);
                const rankChar = cardLabel.slice(0, -1);
                const suitSymbol = SUIT_SYMBOLS[suitChar];
                const isRed = (suitChar === 'H' || suitChar === 'D');

                cell.classList.add(isRed ? 'suit-red' : 'suit-black');
                cell.innerHTML = `
                    <div class="cell-content">
                        <span class="cell-rank">${rankChar}</span>
                        <span class="cell-suit">${suitSymbol}</span>
                    </div>
                `;
            }

            // Chip state
            const chipState = game.board[row][col];
            if (chipState === 1) {
                cell.classList.add('has-chip', 'player-chip');
            } else if (chipState === 2) {
                cell.classList.add('has-chip', 'ai-chip');
            } else if (chipState === -1) {
                cell.classList.add('has-chip', 'free-chip');
            }

            // Highlight valid moves
            const activeCard = hoveredCard || selectedCard;
            if (activeCard && !game.gameOver && game.currentPlayer === 1) {
                const moves = game.getLegalMoves(1);
                const validMove = moves.find(m => m.card === activeCard && m.row === row && m.col === col);
                if (validMove) {
                    cell.classList.add(validMove.is_removal ? 'removal-target' : 'valid-move');
                }
            }

            // Highlight sequences
            highlightSequences(cell, row, col);

            // Click handler
            cell.addEventListener('click', () => handleCellClick(row, col));

            boardEl.appendChild(cell);
        }
    }
}

function highlightSequences(cell, row, col) {
    const checkSequence = (sequences) => {
        for (const seq of sequences) {
            if (seq.includes(`${row},${col}`)) return true;
        }
        return false;
    };

    if (checkSequence(game.completedSequences[1]) || checkSequence(game.completedSequences[2])) {
        cell.classList.add('in-sequence');
        const mark = document.createElement('div');
        mark.className = 'sequence-mark';
        mark.textContent = '✕';
        cell.appendChild(mark);
    }
}

/**
 * Render Player 1 Hand
 */
function renderP1Hand() {
    if (!handP1El) return;
    handP1El.innerHTML = '';

    const hand = game.hands[1] || [];

    for (const cardStr of hand) {
        const card = createCardElement(cardStr);

        if (cardStr === selectedCard) {
            card.classList.add('selected');
        }

        if (game.currentPlayer === 1 && !watchMode && !game.gameOver) {
            card.addEventListener('click', () => handleCardClick(cardStr));
            card.addEventListener('mouseenter', () => {
                hoveredCard = cardStr;
                renderBoard();
            });
            card.addEventListener('mouseleave', () => {
                hoveredCard = null;
                renderBoard();
            });
        }

        handP1El.appendChild(card);
    }
}

/**
 * Render Player 2 Hand
 */
function renderP2Hand() {
    if (!handP2El) return;
    handP2El.innerHTML = '';

    const hand = game.hands[2] || [];

    if (!watchMode && !game.gameOver) {
        // Show backs
        for (let i = 0; i < hand.length; i++) {
            const card = document.createElement('div');
            card.className = 'card back';
            card.style.background = '#4a0e0e';
            card.innerHTML = '<div style="color:rgba(255,255,255,0.1); font-size:3rem; display:flex; justify-content:center; align-items:center; height:100%">S</div>';
            handP2El.appendChild(card);
        }
    } else {
        for (const cardStr of hand) {
            const card = createCardElement(cardStr);
            handP2El.appendChild(card);
        }
    }
}

function createCardElement(cardStr) {
    const card = document.createElement('div');
    card.className = 'card';
    card.dataset.card = cardStr;

    const suitChar = cardStr.slice(-1);
    const rankChar = cardStr.slice(0, -1);
    const suitSymbol = SUIT_SYMBOLS[suitChar];
    const isRed = (suitChar === 'H' || suitChar === 'D');

    card.classList.add(isRed ? 'suit-red' : 'suit-black');

    card.innerHTML = `
        <div class="card-top">${rankChar}<br>${suitSymbol}</div>
        <div class="card-center">${suitSymbol}</div>
        <div class="card-bottom">${rankChar}<br>${suitSymbol}</div>
    `;
    return card;
}

/**
 * Handle card click
 */
function handleCardClick(cardStr) {
    if (game.gameOver) return;
    if (game.currentPlayer !== 1) return;
    if (watchMode) return;

    selectedCard = (selectedCard === cardStr) ? null : cardStr;

    renderP1Hand();
    renderBoard();
}

/**
 * Handle cell click
 */
async function handleCellClick(row, col) {
    if (game.gameOver) return;
    if (game.currentPlayer !== 1) return;
    if (!selectedCard) return;

    const moves = game.getLegalMoves(1);
    const validMove = moves.find(m => m.card === selectedCard && m.row === row && m.col === col);

    if (!validMove) return;

    game.makeMove(validMove);
    selectedCard = null;
    hoveredCard = null;

    renderBoard();
    renderP1Hand();
    renderP2Hand();
    updateStatus();
    renderScores();

    if (game.gameOver) {
        showGameOver();
        return;
    }

    // AI turn
    if (game.currentPlayer === 2) {
        // Force DOM repaint before AI starts thinking
        await sleep(10);
        await triggerAiTurn();
    }
}

/**
 * Update turn indicator
 */
function updateStatus() {
    if (!turnIndicatorEl) return;

    const textEl = turnIndicatorEl.querySelector('.text');

    if (game.gameOver) {
        textEl.textContent = game.winner === 1 ? 'You Win!' : 'AI Wins!';
        turnIndicatorEl.className = 'turn-indicator';
    } else if (isThinking) {
        textEl.textContent = 'AI Thinking...';
        turnIndicatorEl.className = 'turn-indicator ai-turn';
    } else if (game.currentPlayer === 1) {
        textEl.textContent = watchMode ? 'P1 Turn' : 'Your Turn';
        turnIndicatorEl.className = 'turn-indicator player-turn';
    } else {
        textEl.textContent = 'AI Turn';
        turnIndicatorEl.className = 'turn-indicator ai-turn';
    }
}

/**
 * Render scores
 */
function renderScores() {
    if (playerSequences) playerSequences.textContent = game.completedSequences[1].length;
    if (aiSequences) aiSequences.textContent = game.completedSequences[2].length;
}

/**
 * Show game over overlay
 */
function showGameOver() {
    if (gameOverOverlay) {
        gameOverOverlay.classList.add('visible');
        gameOverText.textContent = game.winner === 1 ? '🎉 You Win!' : '🤖 AI Wins!';
    }
}

function dismissGameOver() {
    if (gameOverOverlay) {
        gameOverOverlay.classList.remove('visible');
    }
}

/**
 * Get hint from AI
 */
async function getHint() {
    if (game.gameOver || game.currentPlayer !== 1) return;

    const depthSlider = document.getElementById('depth');
    mcts.numSimulations = depthSlider ? parseInt(depthSlider.value) : 50;

    if (aiProgressContainer) {
        aiProgressContainer.classList.add('visible');
        aiProgressText.textContent = 'Analyzing...';
    }

    const result = await mcts.search(game, (current, total) => {
        const pct = Math.floor((current / total) * 100);
        if (aiProgressBar) aiProgressBar.style.width = pct + '%';
    });

    if (aiProgressContainer) {
        aiProgressContainer.classList.remove('visible');
    }

    if (result.move) {
        selectedCard = result.move.card;
        renderP1Hand();
        renderBoard();

        // Highlight the suggested cell
        const cell = document.querySelector(`.cell[data-row="${result.move.row}"][data-col="${result.move.col}"]`);
        if (cell) {
            cell.classList.add('best-move');
        }
    }
}
