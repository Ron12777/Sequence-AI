/**
 * Sequence AI - Game Client
 */

// Game state
let gameState = null;
let selectedCard = null;
let watchMode = false;
let isThinking = false;
let currentEpoch = 0;

// DOM elements
const boardEl = document.getElementById('board');
const handP1El = document.getElementById('handP1');
const handP2El = document.getElementById('handP2');
const moveHistoryEl = document.getElementById('moveHistory');
const turnIndicator = document.getElementById('turnIndicator');
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
document.addEventListener('DOMContentLoaded', () => {
    newGameBtn.addEventListener('click', () => {
        watchMode = false;
        newGame();
    });
    watchAiBtn.addEventListener('click', watchAiVsAi);
    hintBtn.addEventListener('click', getHint);

    // Setup Play Again button in overlay
    const playAgainBtn = document.querySelector('.overlay-content .btn');
    if (playAgainBtn) {
        playAgainBtn.onclick = () => newGame(); // Inherits current watchMode
    }

    newGame();
});

/**
 * Start a new game
 */
async function newGame() {
    isThinking = false; // Reset thinking state
    stopAiPolling(); // Stop any background thinking
    renderTopMoves(null); // Clear all highlights immediately
    currentEpoch++; // Invalidate previous requests
    const thisEpoch = currentEpoch;

    try {
        const response = await fetch('/api/new_game', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ ai_player: 2 })
        });

        if (!response.ok) throw new Error('Failed to start game');

        const data = await response.json();

        // Race condition check
        if (thisEpoch !== currentEpoch) return;

        gameState = data;
        selectedCard = null;
        renderTopMoves(null); // Clear highlights for new game

        renderBoard();
        renderP1Hand();
        renderP2Hand();
        renderHistory();
        updateStatus();

        gameOverOverlay.classList.remove('visible');

        // Check if AI should start
        checkAndTriggerAi();

    } catch (error) {
        console.error('Error starting game:', error);
        alert('Failed to start game. Is the server running?');
    }
}

/**
 * Start AI vs AI mode
 */
async function watchAiVsAi() {
    watchMode = true;
    await newGame();
}

/**
 * Central logic to check if AI needs to move and trigger it
 */
function checkAndTriggerAi() {
    if (gameState.game_over || isThinking) return;

    const isAiTurn = watchMode || (gameState.current_player === gameState.ai_player);
    if (isAiTurn) {
        triggerAiTurn();
    }
}

/**
 * Hide the game over overlay
 */
function dismissGameOver() {
    if (gameOverOverlay) {
        gameOverOverlay.classList.remove('visible');
    }
}

/**
 * Trigger AI turn
 */
async function triggerAiTurn() {
    if (gameState.game_over || isThinking) return;
    const thisEpoch = currentEpoch;

    isThinking = true;
    renderTopMoves(null); // Clear highlights immediately
    updateStatus();

    // Reset and show progress bar
    if (aiProgressContainer) {
        aiProgressContainer.classList.add('visible');
        aiProgressBar.style.width = '0%';
        aiProgressText.textContent = 'Starting search...';
    }

    // Get difficulty setting
    const difficultyEl = document.getElementById('difficulty');
    const difficulty = difficultyEl ? difficultyEl.value : 'medium';

    try {
        startAiPolling(); // Start watching progress
        const response = await fetch('/api/ai_move', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                difficulty: difficulty,
                game_id: gameState.game_id
            })
        });

        if (!response.ok) throw new Error('AI move failed');

        // Race condition check
        if (thisEpoch !== currentEpoch) {
            stopAiPolling();
            return;
        }

        const data = await response.json();
        stopAiPolling(); // Stop polling when done

        if (thisEpoch !== currentEpoch) return;

        gameState = data;
        isThinking = false;

        renderBoard();
        // Render highlights if available
        if (data.ai_move && data.ai_move.top_moves) {
            renderTopMoves(data.ai_move.top_moves);
        }

        renderP1Hand();
        renderP2Hand();
        renderHistory();
        updateStatus();
        updateActiveTurn();

        // Loop or check for next turn
        checkAndTriggerAi();

    } catch (error) {
        if (thisEpoch === currentEpoch) {
            console.error('Error triggering AI turn:', error);
            stopAiPolling();
            isThinking = false;
            updateStatus();
        }
    }
}

let aiPollingInterval = null;

function startAiPolling() {
    if (aiPollingInterval) clearInterval(aiPollingInterval);
    const thisEpoch = currentEpoch;

    aiPollingInterval = setInterval(async () => {
        try {
            const response = await fetch('/api/ai_status');
            if (!response.ok) return;

            // Race condition check: If game reset while fetch was in flight, abort
            if (thisEpoch !== currentEpoch) return;

            const data = await response.json();

            // Double check epoch after parsing json
            if (thisEpoch !== currentEpoch) return;

            // Update progress bar
            if (data.thinking && data.target_simulations > 0) {
                const percent = Math.min(100, Math.round((data.simulations / data.target_simulations) * 100));
                aiProgressBar.style.width = `${percent}%`;
                aiProgressText.textContent = `${percent}% Thinking...`;
            }

            // Only update visualization if thinking and valid data
            if (data.thinking && data.top_moves && data.top_moves.length > 0) {
                renderTopMoves(data.top_moves);
            }
        } catch (e) {
            console.error("Polling error", e);
        }
    }, 50); // High-frequency 50ms polling for fast AI searches
}

function stopAiPolling() {
    if (aiPollingInterval) {
        clearInterval(aiPollingInterval);
        aiPollingInterval = null;
    }
    // Hide progress bar
    if (aiProgressContainer) {
        aiProgressContainer.classList.remove('visible');
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

            const cardLabel = gameState.board_layout[row][col];

            // Check for corner (free) spaces
            if (cardLabel === 'FREE') {
                cell.classList.add('free-space');
                cell.innerHTML = `<span style="font-size:2rem">★</span>`;
            } else {
                // Parse card
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
            const chipState = gameState.board[row][col];
            if (chipState === gameState.human_player) {
                cell.classList.add('has-chip', 'player-chip');
            } else if (chipState === gameState.ai_player) {
                cell.classList.add('has-chip', 'ai-chip');
            } else if (chipState === -1) {
                cell.classList.add('has-chip', 'free-chip');
            }

            // Highlight valid moves if card selected
            if (selectedCard && !gameState.game_over) {
                const validMove = gameState.legal_moves.find(m =>
                    m.card === selectedCard && m.row === row && m.col === col
                );
                if (validMove) {
                    if (validMove.is_removal) {
                        cell.classList.add('removal-target');
                    } else {
                        cell.classList.add('valid-move');
                    }
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

// Helper 
function formatCardLabel(cardStr) {
    if (cardStr === 'FREE') return '★';

    // Parse
    const suitChar = cardStr.slice(-1);
    const rankChar = cardStr.slice(0, -1);
    const suitSymbol = SUIT_SYMBOLS[suitChar];
    const isRed = (suitChar === 'H' || suitChar === 'D');

    const colorClass = isRed ? 'suit-red' : 'suit-black';

    return `<span class="card-label-inline ${colorClass}">${rankChar}${suitSymbol}</span>`;
}

/**
 * Highlight cells that are part of completed sequences
 */
function highlightSequences(cell, row, col) {
    const checkSequence = (sequences) => {
        for (const seq of sequences) {
            for (const [r, c] of seq) {
                if (r === row && c === col) {
                    return true;
                }
            }
        }
        return false;
    };

    if (checkSequence(gameState.sequences[gameState.human_player]) ||
        checkSequence(gameState.sequences[gameState.ai_player])) {
        cell.classList.add('in-sequence');

        // Add dedicated 'X' element to avoid pseudo-element collision with highlights
        const mark = document.createElement('div');
        mark.className = 'sequence-mark';
        mark.textContent = '✕';
        cell.appendChild(mark);
    }
}

/**
 * Render Player 1 Hand (Left)
 */
function renderP1Hand() {
    if (!handP1El) return;
    handP1El.innerHTML = '';

    // Human Player (P1)
    const p1Hand = (gameState.hands && gameState.hands[1]) || [];

    for (const cardStr of p1Hand) {
        const card = createCardElement(cardStr);

        // Selected state (only if human turn)
        if (cardStr === selectedCard && gameState.current_player === gameState.human_player) {
            card.classList.add('selected');
        }

        // Click handler (only if human turn)
        if (gameState.current_player === gameState.human_player && !watchMode) {
            card.addEventListener('click', () => handleCardClick(cardStr));
        }

        handP1El.appendChild(card);
    }
}

/**
 * Render Player 2 Hand (Right)
 */
function renderP2Hand() {
    if (!handP2El) return;
    handP2El.innerHTML = '';

    const p2Hand = (gameState.hands && gameState.hands[2]) || [];

    // In WatchMode (AI vs AI), show cards face up.
    // In Human vs AI, show card backs.
    const showFaceUp = watchMode;

    if (!showFaceUp && !gameState.game_over) {
        // Show backs
        for (let i = 0; i < p2Hand.length; i++) {
            const card = document.createElement('div');
            card.className = 'card back';
            card.style.background = '#4a0e0e'; // Red-ish back pattern placeholder
            card.innerHTML = '<div style="color:rgba(255,255,255,0.1); font-size:3rem; display:flex; justify-content:center; align-items:center; height:100%">S</div>';
            handP2El.appendChild(card);
        }
    } else {
        // Show cards
        for (const cardStr of p2Hand) {
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
 * Render Top Moves (Highlights)
 */
function renderTopMoves(moves) {
    // Clear previous highlights first
    document.querySelectorAll('.cell').forEach(cell => {
        cell.classList.remove('top-move', 'best-move', 'p1-move', 'p2-move');
        cell.style.removeProperty('--move-score');
    });

    if (!moves || moves.length === 0) return;

    const isP1Turn = gameState.current_player === 1;

    moves.forEach((m, index) => {
        // Find cell
        const cell = document.querySelector(`.cell[data-row="${m.row}"][data-col="${m.col}"]`);
        if (cell) {
            cell.classList.add('top-move');
            if (index === 0) {
                cell.classList.add('best-move');
            } else {
                // Add p1/p2 specific move class for secondary moves
                cell.classList.add(isP1Turn ? 'p1-move' : 'p2-move');
            }
            cell.style.setProperty('--move-score', m.score); // 0.0 to 1.0
        }
    });
}

/**
 * Render History
 */
function renderHistory() {
    if (!moveHistoryEl) return;
    moveHistoryEl.innerHTML = '';

    const history = gameState.history || [];
    // Show newest at top
    for (let i = history.length - 1; i >= 0; i--) {
        const h = history[i];
        const div = document.createElement('div');
        div.className = 'history-item';
        div.classList.add(h.player === 1 ? 'history-player-1' : 'history-player-2');

        let playerName = h.player === 1 ? 'YOU' : 'AI';
        if (watchMode) playerName = `P${h.player}`;

        div.innerHTML = `
            <strong>${playerName}</strong>
            <span>${formatCardLabel(h.card)}</span>
            <span>➜</span>
            <span>(${h.c + 1}, ${h.r + 1})</span>
        `;
        moveHistoryEl.appendChild(div);
    }
}

/**
 * Update game status display
 */
function updateStatus() {
    // Sequences count
    playerSequences.textContent = gameState.sequences[1].length;
    aiSequences.textContent = gameState.sequences[2].length;

    // Turn indicator
    turnIndicator.classList.remove('ai-turn', 'game-over');

    if (gameState.game_over) {
        turnIndicator.classList.add('game-over');
        turnIndicator.querySelector('.text').textContent = 'Game Over';

        // Show overlay
        if (gameState.winner === 1) {
            gameOverText.textContent = watchMode ? '🤖 AI 1 Wins!' : '🎉 You Win!';
        } else if (gameState.winner === 2) {
            gameOverText.textContent = watchMode ? '🤖 AI 2 Wins!' : '🤖 AI Wins!';
        } else {
            gameOverText.textContent = 'Draw!';
        }
        gameOverOverlay.classList.add('visible');

    } else {
        const currentPlayer = gameState.current_player;
        if (watchMode) {
            if (currentPlayer === 2) {
                turnIndicator.classList.add('ai-turn');
            }
            turnIndicator.querySelector('.text').textContent = `AI ${currentPlayer} Thinking...`;
        } else if (currentPlayer === gameState.human_player) {
            turnIndicator.querySelector('.text').textContent = 'Your Turn';
        } else {
            turnIndicator.classList.add('ai-turn');
            turnIndicator.querySelector('.text').textContent = 'AI Thinking...';
        }
    }

    updateActiveTurn();
}

/**
 * Update Active Turn Highlight
 */
function updateActiveTurn() {
    // Reset
    if (handP1El) handP1El.parentElement.classList.remove('active-turn');
    if (handP2El) handP2El.parentElement.classList.remove('active-turn');

    if (gameState.game_over) return;

    if (gameState.current_player === 1) {
        if (handP1El) handP1El.parentElement.classList.add('active-turn');
    } else {
        if (handP2El) handP2El.parentElement.classList.add('active-turn');
    }
}

/**
 * Handle card click in hand
 */
function handleCardClick(cardStr) {
    if (gameState.game_over) return;
    if (gameState.current_player !== gameState.human_player) return;
    if (watchMode) return;

    if (selectedCard === cardStr) {
        selectedCard = null;
    } else {
        selectedCard = cardStr;
    }

    renderP1Hand();
    renderBoard();
}

/**
 * Handle cell click on board
 */
async function handleCellClick(row, col) {
    if (gameState.game_over) return;
    if (gameState.current_player !== gameState.human_player) return;
    if (!selectedCard) return;

    // Check if this is a valid move
    const validMove = gameState.legal_moves.find(m =>
        m.card === selectedCard && m.row === row && m.col === col
    );

    if (!validMove) return;

    // Make the move
    const thisEpoch = currentEpoch;
    try {
        const response = await fetch('/api/move', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                card: selectedCard,
                row: row,
                col: col,
                is_removal: validMove.is_removal,
                game_id: gameState.game_id // [NEW] Pass game_id
            })
        });

        if (!response.ok) {
            const error = await response.json();
            // Silent abort on mismatch (assume new game started)
            if (error.mismatch) {
                console.log("Move aborted: game instance mismatch");
                return;
            }
            throw new Error(error.error || 'Move failed');
        }

        const data = await response.json();

        // Race condition check
        if (thisEpoch !== currentEpoch) return;

        gameState = data;
        selectedCard = null;
        renderTopMoves(null); // Clear highlights after player move success

        renderBoard();
        renderP1Hand();
        renderP2Hand();
        renderHistory();
        updateStatus();

        // Use central logic to check if AI needs to move
        checkAndTriggerAi();

    } catch (error) {
        console.error('Error making move:', error);
        alert('Failed to make move: ' + error.message);
    }
}

/**
 * Get AI hint for current position
 */
async function getHint() {
    if (gameState.game_over) return;

    // In watch mode, maybe allowed? No, hint is usually for human.
    if (gameState.current_player !== gameState.human_player && !watchMode) return;

    try {
        hintBtn.disabled = true;
        hintBtn.textContent = 'Thinking...';

        const response = await fetch('/api/hint');

        if (!response.ok) {
            throw new Error('Failed to get hint');
        }

        const hint = await response.json();

        // Select the suggested card and highlight the move
        selectedCard = hint.card;

        renderP1Hand();
        renderBoard();

        // Flash the suggested cell
        const cells = document.querySelectorAll('.cell');
        const targetCell = cells[hint.row * 10 + hint.col];
        if (targetCell) {
            targetCell.style.outline = '3px solid gold';
            setTimeout(() => {
                targetCell.style.outline = '';
            }, 2000);
        }

    } catch (error) {
        console.error('Error getting hint:', error);
        alert('Failed to get hint');
    } finally {
        hintBtn.disabled = false;
        hintBtn.textContent = 'Get Hint';
    }
}
