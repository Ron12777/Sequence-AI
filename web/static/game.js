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
let analysisMode = false;
let cardPickerTarget = null; // {type: 'hand', player: 1/2, index: 0}
let isPaused = false;


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

// Suit symbols - using text variation selector to prevent emoji rendering on iOS
const SUIT_SYMBOLS = {
    'C': '♣\uFE0E',
    'D': '♦\uFE0E',
    'H': '♥\uFE0E',
    'S': '♠\uFE0E'
};

// Initialize
document.addEventListener('DOMContentLoaded', async () => {
    newGameBtn.addEventListener('click', () => {
        watchMode = false;
        newGame();
    });
    watchAiBtn.addEventListener('click', watchAiVsAi);
    hintBtn.addEventListener('click', getHint);

    // Analysis Mode
    const analysisBtn = document.getElementById('analysisBtn');
    const toggleTurnBtn = document.getElementById('toggleTurnBtn');
    const bestMoveBtn = document.getElementById('bestMoveBtn');

    if (analysisBtn) analysisBtn.addEventListener('click', toggleAnalysisMode);
    if (toggleTurnBtn) toggleTurnBtn.addEventListener('click', toggleTurn);
    if (bestMoveBtn) bestMoveBtn.addEventListener('click', showBestMove);


    // Win Prob Toggle
    if (showEvalBar) {
        showEvalBar.addEventListener('change', () => {
            if (evalBarContainer) {
                evalBarContainer.style.visibility = showEvalBar.checked ? 'visible' : 'hidden';
                if (showEvalBar.checked) {
                    evalBarContainer.classList.add('visible');
                } else {
                    evalBarContainer.classList.remove('visible');
                }
            }
        });
    }

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
            depthValue.value = val;
            difficultyLabel.textContent = getDifficultyLabel(val);
        }
        // Update P1
        if (depthP1Slider && depthP1Value && difficultyLabelP1) {
            const val = parseInt(depthP1Slider.value);
            depthP1Value.value = val;
            difficultyLabelP1.textContent = getDifficultyLabel(val);
        }
        // Update Judge
        if (judgeDepthSlider && judgeDepthValue) {
            const val = parseInt(judgeDepthSlider.value);
            judgeDepthValue.value = val;
        }
    }

    if (depthSlider) depthSlider.addEventListener('input', updateDepthDisplay);
    if (depthP1Slider) depthP1Slider.addEventListener('input', updateDepthDisplay);
    if (judgeDepthSlider) judgeDepthSlider.addEventListener('input', updateDepthDisplay);

    function handleNumericInput(inputEl, sliderEl, labelEl) {
        if (!inputEl || !sliderEl) return;

        const clamp = (val) => Math.max(1, Math.min(500, val));

        inputEl.addEventListener('input', () => {
            let val = parseInt(inputEl.value);
            if (!isNaN(val)) {
                const clamped = clamp(val);
                sliderEl.value = clamped;
                if (labelEl) labelEl.textContent = getDifficultyLabel(clamped);
            }
        });

        // Strict correction on blur
        inputEl.addEventListener('blur', () => {
            let val = parseInt(inputEl.value);
            if (isNaN(val)) val = 50; // Default fallback
            const clamped = clamp(val);
            inputEl.value = clamped;
            sliderEl.value = clamped;
            if (labelEl) labelEl.textContent = getDifficultyLabel(clamped);
        });
    }

    handleNumericInput(depthValue, depthSlider, difficultyLabel);
    handleNumericInput(depthP1Value, depthP1Slider, difficultyLabelP1);
    handleNumericInput(judgeDepthValue, judgeDepthSlider);

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
    cancelAiLoop = true; // Stop any ongoing AI vs AI match
    game = new SequenceGame();
    // Keep Analysis Mode if active
    if (!analysisMode) {
        game.reset();
    } else {
        // If in analysis mode, we might want to just reset the board but keep mode?
        // Usually New Game means New Game. Let's fully reset.
        toggleAnalysisMode(); // Exit analysis mode
        game.reset();
    }

    selectedCard = null;
    hoveredCard = null;
    isThinking = false;
    watchMode = false; // Reset watch mode
    isPaused = false;

    if (gameOverOverlay) {
        gameOverOverlay.classList.remove('visible');
    }

    // Ensure "Reset Board" button is reset to default state
    updateResetButtonState(false);

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
    const lblP1 = document.getElementById('labelP1');
    const lblP2 = document.getElementById('labelP2');
    const headerP1 = document.getElementById('handHeaderP1');
    const headerP2 = document.getElementById('handHeaderP2');

    if (lblP1) lblP1.textContent = "YOU";
    if (lblP2) lblP2.textContent = "AI";
    if (headerP1) headerP1.textContent = "YOU";
    if (headerP2) headerP2.textContent = "AI";

    // Hide P1 Slider (User plays P1)
    const depthP1Container = document.getElementById('depthP1Container');
    if (depthP1Container) depthP1Container.style.display = 'none';

    // Rename main slider to just "Depth" or "AI Depth"
    const depthTitle = document.getElementById('depthTitle');
    if (depthTitle) depthTitle.textContent = "Depth";

    // Show Watch AI button again
    const watchAiBtn = document.getElementById('watchAiBtn');
    if (watchAiBtn) watchAiBtn.style.display = 'block';
}


/**
 * Watch AI vs AI
 */
async function watchAiVsAi() {
    // Auto-Exit Analysis Mode
    if (analysisMode) {
        toggleAnalysisMode();
    }

    // Resume if Paused
    if (isPaused) {
        isPaused = false;
    }

    if (!game || game.gameOver) {
        newGame();
    } else {
        cancelAiLoop = true;
        await sleep(100);
    }

    // Setup for this new match
    cancelAiLoop = false;
    watchMode = true;

    // Update Button to "Pause"
    updateResetButtonState(true);

    // Hide the separate Watch AI button to avoid confusion
    const watchAiBtn = document.getElementById('watchAiBtn');
    if (watchAiBtn) watchAiBtn.style.display = 'none';

    // Re-render to show hidden info (P2 hand)
    renderP1Hand();
    renderP2Hand();
    updateStatus();

    // Set Labels (AI vs AI)
    const lblP1 = document.getElementById('labelP1');
    const lblP2 = document.getElementById('labelP2');
    const headerP1 = document.getElementById('handHeaderP1');
    const headerP2 = document.getElementById('handHeaderP2');

    if (lblP1) lblP1.textContent = "AI 1 (Red)";
    if (lblP2) lblP2.textContent = "AI 2 (Blue)";
    if (headerP1) headerP1.textContent = "AI 1 (Red)";
    if (headerP2) headerP2.textContent = "AI 2 (Blue)";

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

    // Set both width (desktop) and height (mobile vertical bar)
    if (evalBarFill) {
        evalBarFill.style.width = `${clamped}%`;
        evalBarFill.style.height = `${clamped}%`;
    }

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
            if (aiProgressBar) aiProgressBar.style.width = Math.floor((current / total) * 100) + '%';
            if (aiProgressText) aiProgressText.textContent = `${current}/${total}`;

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
                    evalBarFill.style.height = '100%';
                    evalText.textContent = 'Red 100%';
                } else if (game.winner === 2) {
                    evalBarFill.style.width = '0%';
                    evalBarFill.style.height = '0%';
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
                cell.innerHTML = `<span style="font-size:2rem;color:white">★</span>`;
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

    hand.forEach((cardStr, index) => {
        const card = createCardElement(cardStr);

        if (cardStr === selectedCard) {
            card.classList.add('selected');
        }

        // Gameplay or Analysis
        const isMyTurn = (game.currentPlayer === 1 && !watchMode && !game.gameOver);

        if (isMyTurn || analysisMode) {
            card.addEventListener('click', () => handleCardClick(cardStr, index));
        }

        if (isMyTurn) {
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
    });
}


/**
 * Render Player 2 Hand
 */
function renderP2Hand() {
    if (!handP2El) return;
    handP2El.innerHTML = '';

    const hand = game.hands[2] || [];

    // Show backs only if NOT watch mode AND NOT analysis mode AND NOT game over
    const showBacks = (!watchMode && !analysisMode && !game.gameOver);

    if (showBacks) {
        // Show backs
        for (let i = 0; i < hand.length; i++) {
            const card = document.createElement('div');
            card.className = 'card back';
            card.style.background = '#4a0e0e';
            card.innerHTML = '<div style="color:rgba(255,255,255,0.1); font-size:3rem; display:flex; justify-content:center; align-items:center; height:100%">S</div>';

            // Allow editing even if backs are shown? No, that would be weird.
            // But if analysisMode is true, we fall to 'else' block below.

            handP2El.appendChild(card);
        }
    } else {
        hand.forEach((cardStr, index) => {
            const card = createCardElement(cardStr);

            // Allow editing in Analysis Mode
            if (analysisMode) {
                card.addEventListener('click', () => {
                    openCardPicker(2, index);
                });
                // Add valid-move style cues? Not needed for P2 usually.
            }

            handP2El.appendChild(card);
        });
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
function handleCardClick(cardStr, index = -1) {
    if (game.gameOver) return;

    // Analysis Mode: Open Card Picker
    // Allow even if watchMode is true
    if (analysisMode) {
        if (index === -1) {
            index = game.hands['1'].indexOf(cardStr);
        }
        openCardPicker(1, index);
        return;
    }

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

    // Analysis Mode: Cycle Cell State
    if (analysisMode) {
        handleAnalysisCellClick(row, col);
        return;
    }

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
        if (!watchMode && !analysisMode) {
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
    // Ensure "Reset Board" button is restored when game ends
    updateResetButtonState(false);
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


/**
 * Analysis Mode Functions
 */
function toggleAnalysisMode() {
    analysisMode = !analysisMode;
    const btn = document.getElementById('analysisBtn');
    const controls = document.getElementById('analysisControls');

    if (analysisMode) {
        if (watchMode && !isPaused) {
            pauseGame();
        }

        if (btn) {
            btn.textContent = "Exit Analysis";
            btn.classList.add('primary');
            btn.classList.remove('secondary');
        }
        if (controls) controls.style.display = 'block';

        // Hide global Hint button in Analysis Mode
        if (hintBtn) hintBtn.style.display = 'none';

        document.body.classList.add('analysis-mode');

        // Stop any AI thinking
        cancelAiLoop = true;
        isThinking = false;

    } else {
        if (btn) {
            btn.textContent = "Analysis Board";
            btn.classList.remove('primary');
            btn.classList.add('secondary');
        }
        if (controls) controls.style.display = 'none';

        // Restore global Hint button
        if (hintBtn) hintBtn.style.display = 'block';

        document.body.classList.remove('analysis-mode');
    }

    updateStatus();
    renderBoard();
    renderP1Hand();
    renderP2Hand(); // Re-render to show/hide backs if needed (though P2 hand click needs handler)
}

function toggleTurn() {
    if (!game) return;
    game.currentPlayer = (game.currentPlayer === 1) ? 2 : 1;
    updateStatus();
    renderBoard(); // Update valid moves highlight
    renderP1Hand();
}

function handleAnalysisCellClick(row, col) {
    // Cycle: EMPTY -> P1 -> P2 -> FREE -> EMPTY
    const current = game.board[row][col];
    let nextState;

    // ChipState: 0=EMPTY, 1=P1, 2=P2
    if (current === 0) nextState = 1;
    else if (current === 1) nextState = 2;
    // else if (current === 2) nextState = -1; // Removed FREE per user request
    else nextState = 0;


    game.board[row][col] = nextState;

    // Re-check sequences (expensive but necessary to be correct)
    // Actually, we should probably allow manual sequence setting?
    // For now, let's just update the board array. 
    // The game engine 'completedSequences' might get out of sync.
    // Ideally we should re-scan the board for sequences.

    // A full re-scan is hard without a helper. 
    // Let's at least update the visual.
    renderBoard();
}

/**
 * Card Picker
 */
function openCardPicker(player, index) {
    cardPickerTarget = { player, index };
    const overlay = document.getElementById('cardPickerOverlay');
    const grid = document.getElementById('cardPickerGrid');
    if (!overlay || !grid) return;

    grid.innerHTML = '';

    // Generate all cards
    const suits = ['C', 'D', 'H', 'S'];
    const ranks = ['A', '2', '3', '4', '5', '6', '7', '8', '9', '10', 'Q', 'K', 'J']; // J at end for clarity? Or sorted?

    // Helper to add card
    const addCard = (rank, suit) => {
        const cardStr = rank + suit;
        const el = createCardElement(cardStr);
        el.addEventListener('click', () => selectCard(cardStr));
        grid.appendChild(el);
    };

    suits.forEach(suit => {
        ranks.forEach(rank => {
            addCard(rank, suit);
        });
    });

    overlay.classList.add('visible');
}

function closeCardPicker() {
    const overlay = document.getElementById('cardPickerOverlay');
    if (overlay) overlay.classList.remove('visible');
    cardPickerTarget = null;
}

function selectCard(cardStr) {
    if (!cardPickerTarget || !game) return;

    const { player, index } = cardPickerTarget;
    if (game.hands[player]) {
        game.hands[player][index] = cardStr;
    }

    closeCardPicker();
    renderP1Hand();
    renderP2Hand();
    renderBoard(); // Valid moves updates
}

// Global scope for HTML onclick
window.closeCardPicker = closeCardPicker;
window.toggleAnalysisMode = toggleAnalysisMode; // expose if needed
window.newGame = newGame; // was likely already exposed
window.dismissGameOver = () => {
    if (gameOverOverlay) gameOverOverlay.classList.remove('visible');
};

// Pause Menu Functions
// Pause Menu Functions
window.pauseGame = () => {
    if (!watchMode) return;

    isPaused = true;
    cancelAiLoop = true; // Stop the loop

    // Show inline controls
    const newGameBtn = document.getElementById('newGameBtn');
    const pauseControls = document.getElementById('pauseControls');

    if (newGameBtn) newGameBtn.style.display = 'none';
    if (pauseControls) pauseControls.style.display = 'flex';
};

window.resumeGame = () => {
    const newGameBtn = document.getElementById('newGameBtn');
    const pauseControls = document.getElementById('pauseControls');

    if (newGameBtn) newGameBtn.style.display = 'block';
    if (pauseControls) pauseControls.style.display = 'none';

    isPaused = false;
    // Resume AI loop
    watchAiVsAi();
};

window.resetFromPause = () => {
    const newGameBtn = document.getElementById('newGameBtn');
    const pauseControls = document.getElementById('pauseControls');

    if (newGameBtn) newGameBtn.style.display = 'block';
    if (pauseControls) pauseControls.style.display = 'none';

    isPaused = false;
    newGame();
};

function updateResetButtonState(isWatchMode) {
    const btn = document.getElementById('newGameBtn');
    if (!btn) return;

    // Ensure button is visible when state changes
    btn.style.display = 'block';
    const pauseControls = document.getElementById('pauseControls');
    if (pauseControls) pauseControls.style.display = 'none';

    if (isWatchMode && !game.gameOver) {
        btn.textContent = "Pause";
        btn.classList.add('secondary');
        btn.classList.remove('primary');
        btn.style.background = '#e67e22'; // Orange
    } else {
        btn.textContent = "Reset Board";
        btn.classList.add('primary');
        btn.classList.remove('secondary');
        btn.style.background = ''; // Default
    }
}

// Update the main NewGame listener to handle Pause
document.addEventListener('DOMContentLoaded', () => {
    const btn = document.getElementById('newGameBtn');
    if (btn) {
        const newBtn = btn.cloneNode(true);
        btn.parentNode.replaceChild(newBtn, btn);

        newBtn.addEventListener('click', () => {
            if (watchMode && !game.gameOver) {
                // Pause action
                window.pauseGame();
            } else {
                // Reset action
                watchMode = false;
                newGame();
            }
        });
    }
});

/**
 * Show Best Move (Analysis Mode)
 */
async function showBestMove() {
    if (game.gameOver || isThinking) return;

    // Use current settings
    const depthSlider = document.getElementById('depth');
    mcts.numSimulations = depthSlider ? parseInt(depthSlider.value) : 50;

    isThinking = true;
    updateStatus(); // Update UI to show "Thinking..."

    if (aiProgressContainer) {
        aiProgressContainer.classList.add('visible');
        aiProgressText.textContent = 'Analyzing Best Move...';
    }

    try {
        const result = await mcts.search(game, (current, total, policy, value) => {
            const pct = Math.floor((current / total) * 100);
            if (aiProgressBar) aiProgressBar.style.width = pct + '%';

            // Show Live Highlights
            if (policy) renderTopMoves(policy);

            // Show Win Prob if enabled
            if (value !== undefined) updateEvalBar(value);
        });

        if (result.move && result.policy) {
            renderTopMoves(result.policy);
        }
    } catch (e) {
        console.error("Analysis Error:", e);
    }

    isThinking = false;
    updateStatus();

    if (aiProgressContainer) {
        aiProgressContainer.classList.remove('visible');
    }
}

/**
 * Mobile Drawer Toggle
 * Controls the visibility of the side panel on mobile devices
 */
function toggleMobileDrawer() {
    const sidePanel = document.querySelector('.side-panel');
    const toggleBtn = document.getElementById('mobileDrawerToggle');

    if (!sidePanel) return;

    const isOpen = sidePanel.classList.toggle('drawer-open');

    if (toggleBtn) {
        toggleBtn.classList.toggle('open', isOpen);
        toggleBtn.textContent = isOpen ? '✕' : '⚙️';
    }
}

/**
 * Close the mobile drawer
 */
function closeMobileDrawer() {
    const sidePanel = document.querySelector('.side-panel');
    const toggleBtn = document.getElementById('mobileDrawerToggle');

    if (sidePanel) {
        sidePanel.classList.remove('drawer-open');
    }
    if (toggleBtn) {
        toggleBtn.classList.remove('open');
        toggleBtn.textContent = '⚙️';
    }
}

// Close drawer when clicking outside on mobile
document.addEventListener('click', (e) => {
    const sidePanel = document.querySelector('.side-panel');
    const toggleBtn = document.getElementById('mobileDrawerToggle');

    if (!sidePanel || !toggleBtn) return;

    // Only on mobile when drawer is open
    if (window.innerWidth > 767) return;
    if (!sidePanel.classList.contains('drawer-open')) return;

    // Check if click was outside panel and toggle button
    if (!sidePanel.contains(e.target) && !toggleBtn.contains(e.target)) {
        closeMobileDrawer();
    }
});

// Close drawer after making a game action (like Reset Board)
const originalNewGame = newGame;
newGame = function () {
    closeMobileDrawer();
    originalNewGame();
};
