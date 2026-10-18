/**
 * Sequence AI - Client-Side Game UI
 * Fully runs in the browser with ONNX.js neural network inference.
 */

// Game state
let game = null;
let aiPlayer = null;
let modelError = null;
let onnxSession = null;
let selectedCard = null;
let selectedCardIndex = null;
let hintMove = null;
const handLayout = new HandLayout();
let hoveredCard = null;
let watchMode = false;
let isThinking = false;
let analysisMode = false;
let cardPickerTarget = null; // {type: 'hand', player: 1/2, index: 0}
let isPaused = false;
let uiGeneration = 0;
let watchRun = 0;
let dialogReturnFocus = null;


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
const evalBarContainer = document.getElementById('evalBarContainer');
const evalBarFill = document.getElementById('evalBarFill');
const evalText = document.getElementById('evalText');
const showEvalBar = document.getElementById('showEvalBar');
let hasEvalValue = false;
let evaluationRequest = 0;

function setEvalBarVisible(visible) {
    if (!evalBarContainer) return;
    evalBarContainer.classList.toggle('visible', visible);
    evalBarContainer.style.display = visible ? 'block' : 'none';
    evalBarContainer.style.visibility = visible ? 'visible' : 'hidden';
}

function resetEvalBar() {
    evaluationRequest++;
    hasEvalValue = false;
    if (evalBarFill) {
        evalBarFill.style.width = '50%';
        evalBarFill.style.height = '100%';
    }
    if (evalText) {
        evalText.textContent = 'Model estimate';
    }
}

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
        if (watchMode && !isPaused && !analysisMode && !game.gameOver) pauseGame();
        else newGame();
    });
    watchAiBtn.addEventListener('click', watchAiVsAi);
    hintBtn.addEventListener('click', getHint);
    const historyToggle = document.getElementById('toggleHistoryBtn');
    if (historyToggle) historyToggle.addEventListener('click', () => {
        const expanded = historyToggle.getAttribute('aria-expanded') !== 'true';
        historyToggle.setAttribute('aria-expanded', String(expanded));
        historyToggle.closest('.history-panel').classList.toggle('history-expanded', expanded);
        if (expanded) {
            const history = document.getElementById('moveHistory');
            history.scrollTop = history.scrollHeight;
        }
    });

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
            if (showEvalBar.checked) void runJudgeEval();
            else { evaluationRequest++; setEvalBarVisible(false); }
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

    function getDifficultyLabel(simulations) {
        if (simulations <= 15) return 'Light search';
        if (simulations <= 75) return 'Standard search';
        if (simulations <= 300) return 'Extended search';
        return 'Maximum search';
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

    newGame();
    await loadModel();
    updateStatus();
    renderP1Hand();
    void runJudgeEval();
});

function cancelPendingWork() {
    uiGeneration++;
    watchRun++;
    evaluationRequest++;
    isThinking = false;
    selectedCard = null;
    selectedCardIndex = null;
    hoveredCard = null;
    hintMove = null;
}

function syncControls() {
    const ended = !game || game.gameOver;
    hintBtn.hidden = watchMode || analysisMode || ended;
    hintBtn.disabled = !aiPlayer || isThinking || game?.currentPlayer !== 1;
    watchAiBtn.hidden = (watchMode && !ended) || analysisMode;
    watchAiBtn.disabled = !aiPlayer || isThinking;
    document.getElementById('watchSpeedControl').hidden = analysisMode;
    watchAiBtn.textContent = ended ? 'Watch AI again' : 'Watch AI';
    newGameBtn.hidden = watchMode && isPaused && !analysisMode && !ended;
    newGameBtn.textContent = watchMode && !isPaused && !analysisMode && !ended ? 'Pause' : 'New game';
    document.getElementById('pauseControls').hidden = !(watchMode && isPaused && !analysisMode && !ended);
    const editor = document.getElementById('analysisBtn');
    editor.textContent = analysisMode ? 'Done editing' : 'Edit board';
    editor.classList.toggle('primary', analysisMode);
    document.getElementById('analysisControls').style.display = analysisMode ? 'block' : 'none';
    document.getElementById('bestMoveBtn').disabled = !aiPlayer || isThinking || ended;
    document.body.classList.toggle('analysis-mode', analysisMode);
    for (const id of ['labelP1', 'handHeaderP1']) document.getElementById(id).textContent = watchMode || analysisMode ? 'Red' : 'You';
    for (const id of ['labelP2', 'handHeaderP2']) document.getElementById(id).textContent = watchMode || analysisMode ? 'Blue' : 'AI';
    document.getElementById('depthP1Container').style.display = watchMode ? 'block' : 'none';
    document.getElementById('depthTitle').textContent = watchMode ? 'Blue simulations' : 'Search simulations';
}

function renderPosition() {
    const newDraw = game && !analysisMode && game.finishIfNoLegalMoves();
    renderBoard(); renderP1Hand(); renderP2Hand(); renderScores(); updateStatus();
    if (newDraw) {
        showAiError(null);
        showGameOver();
        void runJudgeEval();
    }
}

/**
 * Load ONNX neural network model
 */
async function loadModel() {
    try {
        console.log('Loading ONNX model...');
        // Static hosting cannot use WASM threads without cross-origin isolation.
        ort.env.wasm.numThreads = globalThis.crossOriginIsolated
            ? Math.min(4, navigator.hardwareConcurrency || 1) : 1;
        const response = await fetch('model.json', {cache: 'no-store'});
        if (!response.ok) throw new Error('Could not load model configuration');
        const config = await response.json();
        if (!['direct', 'search'].includes(config.playMode)) throw new Error('Invalid model play mode');
        onnxSession = await ort.InferenceSession.create(config.model);
        if (config.playMode === 'search' &&
            (onnxSession.inputNames.length !== 1 || onnxSession.inputNames[0] !== 'input')) {
            throw new Error('Search requires a board model with one input');
        }
        console.log('✓ ONNX model loaded');

        aiPlayer = config.playMode === 'direct'
            ? new DirectPlayer(onnxSession, ort, config) : new MCTS(onnxSession, 128);
        document.body.classList.toggle('direct-policy', aiPlayer instanceof DirectPlayer);
    } catch (e) {
        modelError = new Error('AI unavailable: ' + e.message);
        aiPlayer = null;
        showAiError(modelError);
    }
}

function showAiError(error) {
    const message = document.getElementById('aiError');
    if (message) {
        message.textContent = error ? error.message : '';
        message.hidden = !error;
    }
}

async function chooseAiMove(position, onProgress = null, depthId = 'depth') {
    if (!aiPlayer) throw modelError || new Error('AI is not loaded');
    showAiError(null);
    if (aiPlayer instanceof DirectPlayer) {
        const result = await aiPlayer.chooseMove(position);
        if (onProgress) onProgress(1, 1, result.policy, result.value);
        return result;
    }
    const slider = document.getElementById(depthId);
    const search = new MCTS(aiPlayer.session, slider ? Number(slider.value) : 128, aiPlayer.numDeals);
    return search.search(position, onProgress);
}

/**
 * Start a new game
 */
function newGame() {
    cancelPendingWork();
    game = new SequenceGame();
    watchMode = false;
    analysisMode = false;
    isPaused = false;
    closeCardPicker();
    dismissGameOver();
    showAiError(modelError);
    resetEvalBar();
    setEvalBarVisible(false);
    const history = document.getElementById('moveHistory');
    if (history) history.innerHTML = '';
    renderPosition();
    void runJudgeEval();
}


/**
 * Watch AI vs AI
 */
async function watchAiVsAi() {
    if (!aiPlayer || analysisMode || (watchMode && !isPaused && !game.gameOver)) return;
    if (!game || game.gameOver) newGame();
    cancelPendingWork();
    watchMode = true;
    isPaused = false;
    const run = watchRun;
    renderPosition();
    while (run === watchRun && !game.gameOver && !isPaused && !analysisMode) {
        if (!await triggerAiTurn()) break;
        // Yield even at Max so Pause and other controls stay responsive.
        const delay = Number(document.getElementById('watchSpeed').value ?? 500);
        await sleep([0, 100, 250, 500, 1000].includes(delay) ? delay : 500);
    }
}

function sleep(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
}

/** Evaluate a snapshot, never reinterpret a pending result using a later turn. */
function capturePosition() {
    const original = game;
    const fingerprint = JSON.stringify(original);
    const generation = uiGeneration;
    return {
        position: cloneGame(original),
        isCurrent: () => generation === uiGeneration && game === original && JSON.stringify(game) === fingerprint
    };
}

async function runJudgeEval() {
    const request = ++evaluationRequest;
    if (!game || !showEvalBar || !showEvalBar.checked) {
        setEvalBarVisible(false);
        return;
    }
    if (game.gameOver) {
        updateEvalBar(game.winner === 1 ? 1 : game.winner === 2 ? -1 : 0, 1, true);
        return;
    }
    if (!aiPlayer) return;
    const {position, isCurrent} = capturePosition();
    const actor = aiPlayer;
    const current = () => request === evaluationRequest && isCurrent() && aiPlayer === actor;
    if (evalText) evalText.textContent = 'Evaluating…';
    try {
        await chooseAiMove(position, (step, total, policy, value) => {
            if (current()) updateEvalBar(value, position.currentPlayer);
        }, 'judgeDepth');
    } catch (error) {
        if (current()) { resetEvalBar(); setEvalBarVisible(false); showAiError(error); }
    }
}

function updateEvalBar(value, evaluatedPlayer, terminal = false) {
    if (!evalBarContainer || !Number.isFinite(value) || ![1, 2].includes(evaluatedPlayer)) return;
    hasEvalValue = true;
    if (showEvalBar && showEvalBar.checked) setEvalBarVisible(true);

    // The critic predicts the outcome for the player whose observation was
    // encoded. Red always occupies the left side, independently of whose turn
    // it is now. This does not assume that the two private-hand estimates agree.
    const redValue = evaluatedPlayer === 1 ? value : -value;
    const unbounded = !terminal && aiPlayer instanceof DirectPlayer && aiPlayer.valueMode === 'unbounded';
    const advantage = unbounded ? Math.tanh(redValue) : redValue;
    const redPercent = Math.max(0, Math.min(100, (advantage + 1) * 50));
    if (evalBarFill) {
        evalBarFill.style.width = `${redPercent}%`;
        evalBarFill.style.height = '100%';
    }
    if (evalText) {
        evalText.textContent = terminal
            ? (redValue > 0 ? 'Red wins' : redValue < 0 ? 'Blue wins' : 'Draw')
            : unbounded
                ? `${redValue >= 0 ? 'Red' : 'Blue'} +${Math.abs(redValue).toFixed(2)} value`
                : `${redPercent > 50 ? 'Red' : 'Blue'} ${Math.round(Math.max(redPercent, 100 - redPercent))}%`;
        evalText.title = terminal ? 'Final result' : 'Model estimate for the current position and side to move; not a calibrated win probability.';
    }
}

/**
 * Trigger AI turn
 */
async function triggerAiTurn() {
    if (!game || game.gameOver || isThinking || analysisMode || isPaused || !aiPlayer) return false;
    if (game.finishIfNoLegalMoves()) {
        showAiError(null);
        renderPosition();
        showGameOver();
        await runJudgeEval();
        return false;
    }
    const {position, isCurrent} = capturePosition();
    const generation = uiGeneration;
    isThinking = true;
    updateStatus();
    try {
        const result = await chooseAiMove(position, null, watchMode && position.currentPlayer === 1 ? 'depthP1' : 'depth');
        if (!isCurrent()) return false;
        if (!result.move) throw new Error('No legal move in this position. Edit the board or start a new game.');
        const player = game.currentPlayer;
        if (!handLayout.makeMove(game, result.move)) throw new Error('The AI returned an invalid move.');
        updateHistory(result.move, player);
    } catch (error) {
        if (generation === uiGeneration) {
            if (watchMode) isPaused = true;
            showAiError(error);
        }
        return false;
    } finally {
        if (generation === uiGeneration) {
            isThinking = false;
            renderPosition();
        }
    }
    await runJudgeEval();
    if (generation !== uiGeneration) return false;
    if (game.gameOver) showGameOver();
    return true;
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

            // Highlight sequences
            highlightSequences(cell, row, col);

            // Click handler
            if (cardLabel !== 'FREE' && (analysisMode || (aiPlayer && !watchMode && !isThinking && !game.gameOver && game.currentPlayer === 1))) {
                makeInteractive(cell, `${cardLabel}, row ${row + 1}, column ${col + 1}`, () => handleCellClick(row, col));
            }

            boardEl.appendChild(cell);
        }
    }
    updateBoardHighlights();
}

// Keep hit targets in place while the pointer moves between hand and board.
function updateBoardHighlights() {
    const activeCard = hoveredCard || selectedCard;
    const moves = !hintMove && activeCard && !watchMode && !analysisMode && !isThinking && !game.gameOver && game.currentPlayer === 1
        ? game.getLegalMoves(1).filter(move => move.card === activeCard) : [];
    for (const cell of boardEl.querySelectorAll('.cell')) {
        const row = Number(cell.dataset.row), col = Number(cell.dataset.col);
        cell.classList.remove('best-move', 'valid-move', 'removal-target');
        if (hintMove && hintMove.row === row && hintMove.col === col) cell.classList.add('best-move');
        const move = moves.find(move => move.row === row && move.col === col);
        if (move) cell.classList.add(move.is_removal ? 'removal-target' : 'valid-move');
    }
}

function highlightSequences(cell, row, col) {
    if (BOARD_LAYOUT[row][col] === 'FREE') return;
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
        cell.appendChild(mark);
    }
}

/**
 * Render Player 1 Hand
 */
function renderP1Hand() {
    if (!handP1El) return;
    handP1El.innerHTML = '';

    const hand = handLayout.cards(game, 1);

    hand.forEach(({card: cardStr, index, slot}) => {
        const card = createCardElement(cardStr);
        card.style.gridColumn = String(slot + 1);
        card.dataset.slot = String(slot);

        if (cardStr === selectedCard && index === selectedCardIndex) {
            card.classList.add('selected');
            if (hintMove) card.classList.add('hint-card');
        }

        // Gameplay or Analysis
        const isMyTurn = (aiPlayer && game.currentPlayer === 1 && !watchMode && !analysisMode && !isThinking && !game.gameOver);

        if (isMyTurn || analysisMode) {
            makeInteractive(card, `${analysisMode ? 'Edit' : 'Select'} ${cardStr}`, () => handleCardClick(cardStr, index));
        }

        if (isMyTurn) {
            card.addEventListener('pointerenter', event => {
                if (event.pointerType !== 'mouse') return;
                hoveredCard = cardStr;
                updateBoardHighlights();
            });
            card.addEventListener('pointerleave', event => {
                if (event.pointerType !== 'mouse') return;
                hoveredCard = null;
                updateBoardHighlights();
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

    const hand = handLayout.cards(game, 2);

    // Show backs only if NOT watch mode AND NOT analysis mode AND NOT game over
    const showBacks = (!watchMode && !analysisMode && !game.gameOver);

    if (showBacks) {
        // Show backs
        for (let i = 0; i < hand.length; i++) {
            const card = document.createElement('div');
            card.className = 'card back';
            card.style.gridColumn = String(hand[i].slot + 1);
            card.dataset.slot = String(hand[i].slot);
            card.style.background = '#4a0e0e';
            card.innerHTML = '<div style="color:rgba(255,255,255,0.1); font-size:3rem; display:flex; justify-content:center; align-items:center; height:100%">S</div>';

            // Allow editing even if backs are shown? No, that would be weird.
            // But if analysisMode is true, we fall to 'else' block below.

            handP2El.appendChild(card);
        }
    } else {
        hand.forEach(({card: cardStr, index, slot}) => {
            const card = createCardElement(cardStr);
            card.style.gridColumn = String(slot + 1);
            card.dataset.slot = String(slot);

            // Allow editing in Analysis Mode
            if (analysisMode) {
                makeInteractive(card, `Edit ${cardStr}`, () => openCardPicker(2, index));
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
    if (!game) return;

    // Analysis Mode: Open Card Picker
    // Allow even if watchMode is true
    if (analysisMode) {
        if (index === -1) {
            index = game.hands['1'].indexOf(cardStr);
        }
        openCardPicker(1, index);
        return;
    }

    if (game.gameOver || watchMode || isThinking || !aiPlayer || game.currentPlayer !== 1) return;
    if (watchMode) return;



    const deselect = selectedCard === cardStr && selectedCardIndex === index;
    hintMove = null;
    selectedCard = deselect ? null : cardStr;
    selectedCardIndex = deselect ? null : index;

    renderP1Hand();
    renderBoard();
}

/**
 * Handle cell click
 */
async function handleCellClick(row, col) {
    if (!game) return;

    // Analysis Mode: Cycle Cell State
    if (analysisMode) {
        handleAnalysisCellClick(row, col);
        return;
    }

    if (game.gameOver || watchMode || isThinking || !aiPlayer || game.currentPlayer !== 1) return;

    if (!selectedCard) return;

    const moves = game.getLegalMoves(1);
    const validMove = moves.find(m => m.card === selectedCard && m.row === row && m.col === col);

    if (!validMove) return;

    if (!handLayout.makeMove(game, validMove, selectedCardIndex)) return;
    updateHistory(validMove, 1);

    selectedCard = null;
    selectedCardIndex = null;
    hoveredCard = null;
    hintMove = null;

    renderBoard();
    renderP1Hand();
    renderP2Hand();
    updateStatus();
    renderScores();

    if (game.gameOver) {
        await runJudgeEval();
        showGameOver();
        return;
    }

    // AI turn
    if (game.currentPlayer === 2) {
        // Force DOM repaint before AI starts thinking
        const generation = uiGeneration;
        await runJudgeEval();
        if (generation === uiGeneration) await triggerAiTurn();
    }
}

/**
 * Update turn indicator
 */
/**
 * Update turn indicator
 */
function updateStatus() {
    syncControls();
    if (!game || !turnIndicatorEl) return;
    const text = turnIndicatorEl.querySelector('.text');
    const player = game.gameOver && game.winner ? game.winner : game.currentPlayer;
    turnIndicatorEl.className = 'turn-indicator ' + (player === 1 ? 'player-turn' : 'ai-turn');
    const color = game.currentPlayer === 1 ? 'Red' : 'Blue';
    if (analysisMode) text.textContent = `Editing · ${color} to move`;
    else if (game.gameOver) text.textContent = resultMessage();
    else if (watchMode) text.textContent = isPaused ? `Paused · ${color} to move` : `${color} to move`;
    else text.textContent = !aiPlayer ? (modelError ? 'AI unavailable' : 'Loading AI…') : game.currentPlayer === 1 ? 'Your turn' : 'AI turn';
}

function resultMessage() {
    if (![1, 2].includes(game.winner)) return 'Draw';
    return watchMode ? (game.winner === 1 ? 'Red wins' : 'Blue wins') : (game.winner === 1 ? 'You win' : 'AI wins');
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
    gameOverText.textContent = resultMessage();
    gameOverOverlay.classList.add('visible');
    updateStatus();
    focusDialog(gameOverOverlay);
}


function dismissGameOver() {
    if (gameOverOverlay?.classList.contains('visible')) {
        gameOverOverlay.classList.remove('visible');
        restoreDialogFocus();
    }
}

/**
 * Get hint from AI
 */
async function getHint() {
    if (!game || game.gameOver || game.currentPlayer !== 1 || watchMode || analysisMode || isThinking || !aiPlayer) return;
    const {position, isCurrent} = capturePosition();
    const generation = uiGeneration;
    isThinking = true;
    updateStatus();
    try {
        const result = await chooseAiMove(position);
        if (!isCurrent()) return;
        if (!result.move) throw new Error('No legal move in this position. Edit the board or start a new game.');
        if (result.move) {
            selectedCard = result.move.card;
            selectedCardIndex = game.hands[1].indexOf(selectedCard);
            hintMove = result.move;
            hoveredCard = null;
            isThinking = false;
            renderP1Hand();
            renderBoard();
        }
    } catch (error) {
        if (isCurrent()) showAiError(error);
    } finally {
        if (generation === uiGeneration) { isThinking = false; updateStatus(); }
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

    const playerName = watchMode ? (player === 1 ? 'Red' : 'Blue') : (player === 1 ? 'You' : 'AI');
    const number = historyEl.children.length + 1;
    const location = `${move.col + 1}, ${move.row + 1}`;
    const action = move.is_removal ? 'Removed chip at' : 'Placed chip at';
    item.setAttribute('aria-label', `Move ${number}, ${playerName}, ${cardStr}, ${action.toLowerCase()} column ${move.col + 1}, row ${move.row + 1}`);
    item.innerHTML = `
        <span class="history-number">${number}</span>
        <span class="history-player">${playerName}</span>
        <span class="history-card ${isRed ? 'suit-red' : 'suit-black'}">${rankChar}${suitSymbol}</span>
        <span class="history-location" title="${action} column ${move.col + 1}, row ${move.row + 1}">${move.is_removal ? '− ' : ''}${location}</span>
    `;

    historyEl.appendChild(item);
    historyEl.scrollTop = historyEl.scrollHeight;
}


/**
 * Analysis Mode Functions
 */
function toggleAnalysisMode() {
    if (!game) return;
    cancelPendingWork();
    analysisMode = !analysisMode;
    if (watchMode) isPaused = true;
    closeCardPicker();
    dismissGameOver();
    renderPosition();
    void runJudgeEval();
    if (!analysisMode && !watchMode && !game.gameOver && game.currentPlayer === 2) void triggerAiTurn();
}

function toggleTurn() {
    if (!game || !analysisMode) return;
    cancelPendingWork();
    game.publicHistoryComplete = false;
    game.currentPlayer = (game.currentPlayer === 1) ? 2 : 1;
    updateStatus();
    renderBoard(); // Update valid moves highlight
    renderP1Hand();
    void runJudgeEval();
}

function handleAnalysisCellClick(row, col) {
    if (!analysisMode || BOARD_LAYOUT[row][col] === 'FREE') return;
    cancelPendingWork();
    game.publicHistoryComplete = false;
    game.board[row][col] = (game.board[row][col] + 1) % 3;
    // Keep valid existing sequences, then discover lines introduced by edits.
    for (const player of [1, 2]) {
        game.completedSequences[player] = game.completedSequences[player].filter(sequence =>
            sequence.every(pos => { const [r, c] = pos.split(',').map(Number); return game.isPlayerChip(player, r, c); }));
        for (let r = 0; r < 10; r++) for (let c = 0; c < 10; c++) {
            game.completedSequences[player].push(...game.findNewSequences(player, r, c));
        }
    }
    const winners = [1, 2].filter(player => game.completedSequences[player].length >= 2);
    game.gameOver = winners.length > 0;
    game.winner = winners.length === 1 ? winners[0] : null;
    renderPosition();
    void runJudgeEval();
}

/**
 * Card Picker
 */
function openCardPicker(player, index) {
    if (!analysisMode || !game.hands[player]?.[index]) return;
    cardPickerTarget = { player, index };
    const overlay = document.getElementById('cardPickerOverlay');
    const grid = document.getElementById('cardPickerGrid');
    const current = game.hands[player][index];
    const suitNames = { C: 'Clubs', D: 'Diamonds', H: 'Hearts', S: 'Spades' };
    const ranks = ['A', '2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K'];
    grid.replaceChildren();
    for (const suit of ['C', 'D', 'H', 'S']) {
        const group = document.createElement('div');
        group.className = 'picker-suit-group';
        group.setAttribute('role', 'group');
        group.setAttribute('aria-label', suitNames[suit]);
        const cards = document.createElement('div');
        cards.className = 'picker-ranks';
        for (const rank of ranks) {
            const card = rank + suit;
            const button = document.createElement('button');
            button.type = 'button';
            button.className = 'picker-card ' + (suit === 'D' || suit === 'H' ? 'suit-red' : 'suit-black');
            button.classList.toggle('current-card', card === current);
            button.setAttribute('aria-label', `${rank} of ${suitNames[suit].toLowerCase()}${card === current ? ', current card' : ''}`);
            button.innerHTML = `<span class="picker-rank">${rank}</span><span class="picker-symbol" aria-hidden="true">${SUIT_SYMBOLS[suit]}</span>`;
            button.addEventListener('click', () => selectCard(card));
            cards.appendChild(button);
        }
        group.appendChild(cards);
        grid.appendChild(group);
    }
    overlay.classList.add('visible');
    focusDialog(overlay);
}

function closeCardPicker() {
    const overlay = document.getElementById('cardPickerOverlay');
    if (overlay?.classList.contains('visible')) {
        overlay.classList.remove('visible');
        restoreDialogFocus();
    }
    cardPickerTarget = null;
}

function selectCard(cardStr) {
    if (!cardPickerTarget || !game || !analysisMode) return;
    cancelPendingWork();
    game.publicHistoryComplete = false;

    const { player, index } = cardPickerTarget;
    if (game.hands[player]) {
        game.hands[player][index] = cardStr;
    }

    closeCardPicker();
    renderP1Hand();
    renderP2Hand();
    renderBoard(); // Valid moves updates
    updateStatus();
    document.querySelector(`#handP${player} .card[data-card="${cardStr}"]`)?.focus();
    void runJudgeEval();
}

function pauseGame() {
    if (!watchMode || game.gameOver) return;
    cancelPendingWork();
    isPaused = true;
    renderPosition();
}
function resumeGame() {
    if (watchMode && isPaused && !analysisMode) void watchAiVsAi();
}
function resetFromPause() { newGame(); }

/**
 * Show Best Move (Analysis Mode)
 */
async function showBestMove() {
    if (!analysisMode || !game || game.gameOver || isThinking || !aiPlayer) return;
    const {position, isCurrent} = capturePosition();
    const generation = uiGeneration;
    isThinking = true;
    updateStatus();
    try {
        const result = await chooseAiMove(position);
        if (isCurrent()) {
            if (!result.move) throw new Error('No legal move in this position. Edit the board or start a new game.');
            if (result.policy) renderTopMoves(result.policy);
            updateEvalBar(result.value, position.currentPlayer);
        }
    } catch (error) {
        if (isCurrent()) showAiError(error);
    } finally {
        if (generation === uiGeneration) { isThinking = false; updateStatus(); }
    }
}

// Keyboard activation mirrors the same guarded mouse handlers.
function makeInteractive(element, label, action) {
    element.setAttribute('role', 'button');
    element.setAttribute('tabindex', '0');
    element.setAttribute('aria-label', label);
    element.addEventListener('click', action);
    element.addEventListener('keydown', event => {
        if (event.key === 'Enter' || event.key === ' ') { event.preventDefault(); action(); }
    });
}

function focusDialog(overlay) {
    dialogReturnFocus = document.activeElement;
    overlay.querySelector('button, [tabindex="0"]')?.focus();
}
document.addEventListener('keydown', event => {
    const overlay = document.querySelector('.overlay.visible');
    if (!overlay) return;
    if (event.key === 'Escape') {
        event.preventDefault();
        if (overlay.id === 'cardPickerOverlay') closeCardPicker();
        else dismissGameOver();
    } else if (event.key === 'Tab') {
        const items = Array.from(overlay.querySelectorAll('button, [tabindex="0"]'));
        const first = items[0], last = items[items.length - 1];
        if (event.shiftKey && document.activeElement === first) { event.preventDefault(); last.focus(); }
        else if (!event.shiftKey && document.activeElement === last) { event.preventDefault(); first.focus(); }
    }
});

function restoreDialogFocus() {
    const target = dialogReturnFocus?.isConnected ? dialogReturnFocus : document.getElementById('analysisBtn');
    target?.focus();
    dialogReturnFocus = null;
}
