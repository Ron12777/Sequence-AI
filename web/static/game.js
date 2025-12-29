/**
 * Sequence AI - Game Client
 */

// Game state
let gameState = null;
let selectedCard = null;
let watchMode = false;
let isThinking = false;

// DOM elements
const boardEl = document.getElementById('board');
const handEl = document.getElementById('hand');
const turnIndicator = document.getElementById('turnIndicator');
const playerSequences = document.getElementById('playerSequences');
const aiSequences = document.getElementById('aiSequences');
const gameOverOverlay = document.getElementById('gameOverOverlay');
const gameOverText = document.getElementById('gameOverText');
const newGameBtn = document.getElementById('newGameBtn');
const watchAiBtn = document.getElementById('watchAiBtn');
const hintBtn = document.getElementById('hintBtn');

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
    try {
        const response = await fetch('/api/new_game', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ ai_player: 2 })
        });
        
        if (!response.ok) throw new Error('Failed to start game');
        
        gameState = await response.json();
        selectedCard = null;
        
        renderBoard();
        renderHand();
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
 * Trigger AI turn
 */
async function triggerAiTurn() {
    if (gameState.game_over || isThinking) return;
    
    isThinking = true;
    updateStatus(); 
    
    // Add a small delay for visual clarity
    await new Promise(r => setTimeout(r, 600));
    
    // Get difficulty setting
    const difficultyEl = document.getElementById('difficulty');
    const difficulty = difficultyEl ? difficultyEl.value : 'medium';
    
    try {
        const response = await fetch('/api/ai_move', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ difficulty: difficulty })
        });
        
        if (!response.ok) throw new Error('AI move failed');
        
        gameState = await response.json();
        isThinking = false;
        
        renderBoard();
        renderHand();
        updateStatus();
        
        // Loop or check for next turn
        checkAndTriggerAi();
        
    } catch (error) {
        console.error('Error triggering AI turn:', error);
        isThinking = false;
        updateStatus();
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

// Helper (no longer used directly for rendering but good utility)
function formatCardLabel(cardStr) {
    if (cardStr === 'FREE') return '★';
    return cardStr;
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
    }
}

/**
 * Render player's hand
 */
function renderHand() {
    handEl.innerHTML = '';
    
    for (const cardStr of gameState.hand) {
        const card = document.createElement('div');
        card.className = 'card';
        card.dataset.card = cardStr;
        
        const suitChar = cardStr.slice(-1);
        const rankChar = cardStr.slice(0, -1);
        const suitSymbol = SUIT_SYMBOLS[suitChar];
        const isRed = (suitChar === 'H' || suitChar === 'D');
        
        card.classList.add(isRed ? 'suit-red' : 'suit-black');
        
        // Content
        card.innerHTML = `
            <div class="card-top">${rankChar}<br>${suitSymbol}</div>
            <div class="card-center">${suitSymbol}</div>
            <div class="card-bottom">${rankChar}<br>${suitSymbol}</div>
        `;
        
        // Selected state
        if (cardStr === selectedCard) {
            card.classList.add('selected');
        }
        
        // Click handler
        card.addEventListener('click', () => handleCardClick(cardStr));
        
        handEl.appendChild(card);
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
}

/**
 * Handle card click in hand
 */
function handleCardClick(cardStr) {
    if (gameState.game_over) return;
    if (gameState.current_player !== gameState.human_player) return;
    
    if (selectedCard === cardStr) {
        selectedCard = null;
    } else {
        selectedCard = cardStr;
    }
    
    renderHand();
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
    try {
        const response = await fetch('/api/move', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                card: selectedCard,
                row: row,
                col: col,
                is_removal: validMove.is_removal
            })
        });
        
        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.error || 'Move failed');
        }
        
        gameState = await response.json();
        selectedCard = null;
        
        renderBoard();
        renderHand();
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
    if (gameState.current_player !== gameState.human_player) return;
    
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
        
        renderHand();
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
