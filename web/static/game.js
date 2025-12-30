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
const evalBarContainer = document.getElementById('evalBarContainer');
const evalBarFill = document.getElementById('evalBarFill');
const evalText = document.getElementById('evalText');
const showEvalBar = document.getElementById('showEvalBar');
const labelP1 = document.getElementById('labelP1');
const labelP2 = document.getElementById('labelP2');

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

    // Depth slider P2 (Default)
    const depthSlider = document.getElementById('depth');
    const depthValue = document.getElementById('depthValue');
    const difficultyLabel = document.getElementById('difficultyLabel');
    const depthTitle = document.getElementById('depthTitle');

    // Depth Slider P1 (Red AI)
    const depthP1Slider = document.getElementById('depthP1');
    const depthP1Value = document.getElementById('depthP1Value');
    const difficultyLabelP1 = document.getElementById('difficultyLabelP1');
    const depthP1Container = document.getElementById('depthP1Container');
    const depthP1Title = document.getElementById('depthP1Title');

    // Judge Depth Slider
    const judgeDepthSlider = document.getElementById('judgeDepth');
    const judgeDepthValue = document.getElementById('judgeDepthValue');

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
        // Update P2 (Default)
        if (depthSlider && depthValue && difficultyLabel) {
            const val = parseInt(depthSlider.value);
            depthValue.textContent = val;
            difficultyLabel.textContent = getDifficultyLabel(val);
        }
        // Update P1
        if (depthP1Slider && depthP1Value && difficultyLabelP1) {
            const val = parseInt(depthP1Slider.value);
            depthP1Value.textContent = val;
            difficultyLabelP1.textContent = getDifficultyLabel(val);
        }
        // Update Judge
        if (judgeDepthSlider && judgeDepthValue) {
            const val = parseInt(judgeDepthSlider.value);
            judgeDepthValue.textContent = val;
        }
    }

    if (depthSlider) depthSlider.addEventListener('input', updateDepthDisplay);
    if (depthP1Slider) depthP1Slider.addEventListener('input', updateDepthDisplay);
    if (judgeDepthSlider) judgeDepthSlider.addEventListener('input', updateDepthDisplay);
    updateDepthDisplay();

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

let cancelAiLoop = false;

/**
 * Load ONNX neural network model
 */
async function loadModel() {
    try {
        console.log('Loading ONNX model...');
        onnxSession = await ort.InferenceSession.create('static/model.onnx');
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
    cancelAiLoop = true; // Stop any ongoing AI vs AI match
    game = new SequenceGame();
    game.reset();
    selectedCard = null;
    hoveredCard = null;
    isThinking = false;
    watchMode = false; // Reset watch mode

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

    // Hide eval bar
    if (evalBarContainer) {
        evalBarContainer.classList.remove('visible');
    }

    // Clear history
    const historyEl = document.getElementById('moveHistory');
    if (historyEl) historyEl.innerHTML = '';

    // Reset Labels (User vs AI)
    if (labelP1) labelP1.textContent = "YOU";
    if (labelP2) labelP2.textContent = "AI";

    // Hide P1 Slider (User plays P1)
    const depthP1Container = document.getElementById('depthP1Container');
    if (depthP1Container) depthP1Container.style.display = 'none';

    // Rename main slider to just "Depth" or "AI Depth"
    const depthTitle = document.getElementById('depthTitle');
    if (depthTitle) depthTitle.textContent = "Depth";
}


/**
 * Watch AI vs AI
 */
async function watchAiVsAi() {
    newGame(); // This sets cancelAiLoop = true (stopping old loops) and watchMode = false

    // Setup for this new match
    cancelAiLoop = false;
    watchMode = true;

    // Re-render to show hidden info (P2 hand)
    renderP1Hand();
    renderP2Hand();
    renderP1Hand();
    renderP2Hand();
    updateStatus();

    // Set Labels (AI vs AI)
    if (labelP1) labelP1.textContent = "AI 1 (Red)";
    if (labelP2) labelP2.textContent = "AI 2 (Blue)";

    // Show P1 Slider
    const depthP1Container = document.getElementById('depthP1Container');
    if (depthP1Container) depthP1Container.style.display = 'block';

    // Update Slider Titles
    const depthTitle = document.getElementById('depthTitle');
    if (depthTitle) depthTitle.textContent = "Blue Depth";

    while (!game.gameOver && !cancelAiLoop) {
        // AI 1 Turn
        await runJudgeEval();
        if (cancelAiLoop) break;

        await triggerAiTurn();
        if (cancelAiLoop) break;
        if (game.gameOver) break;
        await sleep(500);

        // AI 2 Turn (same function, just called again)
        if (!game.gameOver && !cancelAiLoop) {
            await runJudgeEval();
            if (cancelAiLoop) break;

            await triggerAiTurn();
            if (cancelAiLoop) break;
            await sleep(500);
        }
    }
}

function sleep(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
}

/**
 * Run Judge Evaluation (Pre-Move)
 */
async function runJudgeEval() {
    if (!mcts || !game || game.gameOver) return;

    // Only run if eval bar is enabled
    if (!showEvalBar || !showEvalBar.checked) {
        if (evalBarContainer) evalBarContainer.classList.remove('visible');
        return;
    }

    if (evalBarContainer) evalBarContainer.classList.add('visible');
    // if (evalText) evalText.textContent = "Judging...";

    const judgeSlider = document.getElementById('judgeDepth');
    let judgeSims = judgeSlider ? parseInt(judgeSlider.value) : 50;

    // Apply "Min Judge Depth" Logic:
    // "Make the judge depth only kick in if its higher than the depth of the ai it just came from"
    // I.e. Judge Sim Count = Max(Judge Slider, Previous Player Depth)

    // Determine Previous Player
    const prevPlayer = (game.currentPlayer === 1) ? 2 : 1;
    let prevPlayerDepth = 0;

    const depthSlider = document.getElementById('depth'); // P2 (Blue)
    const depthP1Slider = document.getElementById('depthP1'); // P1 (Red) - Visible in Watch Mode

    if (prevPlayer === 2) {
        // Previous was P2 (Always AI)
        if (depthSlider) prevPlayerDepth = parseInt(depthSlider.value);
    } else {
        // Previous was P1
        if (watchMode && depthP1Slider) {
            // P1 is AI in Watch Mode
            prevPlayerDepth = parseInt(depthP1Slider.value);
        } else {
            // P1 is Human
            prevPlayerDepth = 0;
        }
    }

    if (prevPlayerDepth > judgeSims) {
        judgeSims = prevPlayerDepth;
    }

    // Save old sims
    const oldSims = mcts.numSimulations;
    mcts.numSimulations = judgeSims;

    try {
        await mcts.search(game, (current, total, policy, value) => {
            updateEvalBar(value);
        });
    } catch (e) {
        console.error("Judge error:", e);
    } finally {
        mcts.numSimulations = oldSims;
    }
}

function updateEvalBar(value) {
    if (!evalBarContainer || value === undefined) return;

    let p1Advantage = 0; // -1 to 1 scale where 1 is P1 winning

    if (game.currentPlayer === 1) {
        p1Advantage = value;
    } else {
        p1Advantage = -value;
    }

    // Map [-1, 1] to [0%, 100%] width for Red Overlay
    const redPercent = ((p1Advantage + 1) / 2) * 100;
    const clamped = Math.max(0, Math.min(100, redPercent));

    if (evalBarFill) evalBarFill.style.width = `${clamped}%`;

    // Text: "Red 60%" or "Blue 60%"
    if (evalText) {
        if (clamped > 50) {
            evalText.textContent = `Red ${Math.round(clamped)}%`;
        } else {
            evalText.textContent = `Blue ${Math.round(100 - clamped)}%`;
        }
    }
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

    // Determine simulations based on current player
    // If Watch Mode and P1 turn -> use P1 slider
    // All other cases (P2 or Human vs AI where AI is P2) -> use default slider
    let simulations = 50;

    const depthSlider = document.getElementById('depth');
    const depthP1Slider = document.getElementById('depthP1');

    if (game.currentPlayer === 1 && watchMode && depthP1Slider) {
        simulations = parseInt(depthP1Slider.value);
    } else if (depthSlider) {
        simulations = parseInt(depthSlider.value);
    }

    mcts.numSimulations = simulations;

    // Force paint before starting heavy calculation
    await new Promise(r => setTimeout(r, 50));

    // Show eval bar if checked (regardless of mode)
    // But usually only interesting in Watch Mode or if user wants to see their own winning chance?
    // User requested: "checkable show eval bar you can enable in ai vs ai and human vs ai mode"
    if (showEvalBar && showEvalBar.checked && evalBarContainer) {
        evalBarContainer.classList.add('visible');
    } else if (evalBarContainer) {
        evalBarContainer.classList.remove('visible');
    }

    try {
        const result = await mcts.search(game, (current, total, policy, value) => {
            const pct = Math.floor((current / total) * 100);
            if (aiProgressBar) aiProgressBar.style.width = pct + '%';

            // Debug info in UI
            let debugText = `${current}/${total}`;
            if (aiProgressText) aiProgressText.textContent = debugText;

            // Update Eval Bar (REMOVED - now handled by Judge)
            // if (evalBarContainer && value !== undefined) { ... }

            // Render live highlights
            // Update frequently to ensure visibility
            if (policy) {
                renderTopMoves(policy);
            }
        });

        if (result.move) {
            updateHistory(result.move, game.currentPlayer);
            game.makeMove(result.move);

            // Force Eval Bar to 100% if Game Over
            if (game.gameOver && evalBarContainer && evalBarFill && evalText) {
                if (game.winner === 1) {
                    evalBarFill.style.width = '100%';
                    evalText.textContent = 'Red 100%';
                } else if (game.winner === 2) {
                    evalBarFill.style.width = '0%';
                    evalText.textContent = 'Blue 100%';
                }
            }
        }

        // Show AI thinking highlights
        if (result.policy) {
            renderTopMoves(result.policy);
        }
    } catch (e) {
        console.error('MCTS error:', e);
        if (aiProgressText) aiProgressText.textContent = "Error: " + e.message;
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
    updateHistory(validMove, 1);

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
        await runJudgeEval();
        await triggerAiTurn();
    }
}

/**
 * Update turn indicator
 */
/**
 * Update turn indicator
 */
function updateStatus() {
    if (!turnIndicatorEl) return;

    const textEl = turnIndicatorEl.querySelector('.text');
    const dotEl = turnIndicatorEl.querySelector('.turn-dot');

    // Default classes
    turnIndicatorEl.className = 'turn-indicator';

    if (game.gameOver) {
        textEl.textContent = game.winner === 1 ? (watchMode ? 'AI 1 Wins!' : 'You Win!') : (watchMode ? 'AI 2 Wins!' : 'AI Wins!');
        // Keep the winner's color
        if (game.winner === 1) {
            turnIndicatorEl.classList.add('player-turn');
        } else {
            turnIndicatorEl.classList.add('ai-turn');
        }
    } else {
        const isP1 = game.currentPlayer === 1;

        // Update Dot Color
        // P1 = Red (Player/AI1), P2 = Blue (AI/AI2)
        if (isP1) {
            turnIndicatorEl.classList.add('player-turn');
            turnIndicatorEl.classList.remove('ai-turn');
        } else {
            turnIndicatorEl.classList.add('ai-turn');
            turnIndicatorEl.classList.remove('player-turn');
        }

        // Update Text
        if (isThinking) {
            if (watchMode) {
                textEl.textContent = isP1 ? 'AI 1 Thinking...' : 'AI 2 Thinking...';
            } else {
                textEl.textContent = 'AI Thinking...';
            }
        } else {
            if (watchMode) {
                textEl.textContent = isP1 ? 'AI 1 Turn' : 'AI 2 Turn';
            } else {
                textEl.textContent = isP1 ? 'Your Turn' : 'AI Turn';
            }
        }
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
 * Render Top Moves (Highlights)
 * Shows AI thinking process with color-coded orbs (Red/Blue) and Gold for best move.
 */
/**
 * Render Top Moves (Highlights)
 * Shows AI thinking process with color-coded orbs (Red/Blue) and Gold for best move.
 */
function renderTopMoves(policy) {
    try {
        // Clear previous highlights
        document.querySelectorAll('.highlight-orb').forEach(el => el.remove());

        // Debug logging for visibility issues
        if (!watchMode) {
            return;
        }

        if (!policy) return;

        // Convert policy (Float32Array) to array of {row, col, score}
        const moves = [];
        let bestScore = 0;

        for (let i = 0; i < 100; i++) {
            if (policy[i] > bestScore) bestScore = policy[i];
        }

        // Use a tiny epsilon to avoid division by zero, but keep it small
        const maxScore = Math.max(bestScore, 0.0001);

        for (let i = 0; i < 100; i++) {
            const score = policy[i];

            // Reduced Thresholds for specific debugging:
            // Always include if it's the best score (even if low confidence)
            // Or if it's significant (> 1% AND > 20% of best)
            const isBest = (Math.abs(score - bestScore) < 0.0001 && score > 0);

            if (!isBest) {
                if (score < 0.01) continue;
                if (score < maxScore * 0.2) continue;
            }

            moves.push({
                row: Math.floor(i / 10),
                col: i % 10,
                score: score
            });
        }

        // Shuffle moves to prevent 'Top Row' alignment artifact when scores are tied/uniform
        for (let i = moves.length - 1; i > 0; i--) {
            const j = Math.floor(Math.random() * (i + 1));
            [moves[i], moves[j]] = [moves[j], moves[i]];
        }

        // Sort by score descending (Best move first)
        moves.sort((a, b) => b.score - a.score);

        // Limit to max 104 visuals (cover all possible moves including Jacks)
        const topMoves = moves.slice(0, 104);

        const isP1Turn = game.currentPlayer === 1;

        topMoves.forEach((m, index) => {
            const cell = document.querySelector(`.cell[data-row="${m.row}"][data-col="${m.col}"]`);

            if (cell) {
                // Create orb
                const orb = document.createElement('div');

                // Base Class
                orb.className = 'highlight-orb';

                // Determine type
                if (index === 0) {
                    // Best Move -> Gold
                    orb.classList.add('best');
                } else {
                    // Consideration -> Blue (AI) or Red (Player 1)
                    // Note: In AI vs AI, P1 is Red, P2 is Blue.
                    orb.classList.add(isP1Turn ? 'p1' : 'p2');

                    // Opacity based on confidence relative to best
                    // Range 0.4 to 0.9
                    let relativeConfidence = m.score / maxScore;
                    if (relativeConfidence < 0.3) relativeConfidence = 0.3; // Min visibility

                    orb.style.opacity = 0.3 + (0.6 * relativeConfidence);

                    // Size variation based on score
                    const size = 35 + (15 * relativeConfidence); // 35px to 50px
                    orb.style.width = `${size}px`;
                    orb.style.height = `${size}px`;
                }

                cell.appendChild(orb);
            }
        });
    } catch (e) {
        console.error("renderTopMoves failed:", e);
    }
}

/**
 * Show game over overlay
 */
function showGameOver() {
    if (gameOverOverlay) {
        gameOverOverlay.classList.add('visible');

        // Check winner correctly
        let message = '';
        if (watchMode) {
            message = game.winner === 1 ? 'AI 1 Wins!' : 'AI 2 Wins!';
        } else {
            message = game.winner === 1 ? '🎉 You Win!' : '🤖 AI Wins!';
        }

        gameOverText.textContent = message;
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

/**
 * Update Move History
 */
function updateHistory(move, player) {
    const historyEl = document.getElementById('moveHistory');
    if (!historyEl) return;

    const item = document.createElement('div');
    item.className = 'history-item';
    item.classList.add(player === 1 ? 'history-player-1' : 'history-player-2');

    const cardStr = move.card;
    const suitChar = cardStr.slice(-1);
    const rankChar = cardStr.slice(0, -1);
    const suitSymbol = SUIT_SYMBOLS[suitChar];
    const isRed = (suitChar === 'H' || suitChar === 'D');

    let playerName;
    if (watchMode) {
        playerName = player === 1 ? 'AI 1' : 'AI 2';
    } else {
        playerName = player === 1 ? 'YOU' : 'AI';
    }

    // const playerName = player === 1 ? 'YOU' : 'AI';
    const locText = move.is_removal ? `Remove (${move.col + 1}, ${move.row + 1})` : `(${move.col + 1}, ${move.row + 1})`;


    item.innerHTML = `
        <span class="card-label-inline">
            ${playerName}: 
        </span>
        <span style="color:${isRed ? '#ef4444' : '#e0e0e0'}; margin-left: 5px; font-weight:bold;">
            ${rankChar}${suitSymbol}
        </span>
        <span style="margin-left: 5px;">${locText}</span>
    `;

    historyEl.appendChild(item);
    historyEl.scrollTop = historyEl.scrollHeight;
}

