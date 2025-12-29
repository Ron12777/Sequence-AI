#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <stdint.h>
#include <math.h>
#include <time.h>

// -----------------------------------------------------------------------------
// GAME LOGIC (Inline for speed, same as before)
// -----------------------------------------------------------------------------

#define BOARD_SIZE 10
#define NUM_POSITIONS 100

// Chip states
#define EMPTY 0
#define PLAYER1 1
#define PLAYER2 2
#define FREE -1

typedef struct {
    int8_t board[BOARD_SIZE * BOARD_SIZE];
    int current_player;
    int winner;
    bool game_over;
} GameState;

// Helpers
static inline bool is_jack(int card) {
    int rank = (card % 13) + 1;
    return rank == 11;
}

static inline bool is_two_eyed_jack(int card) {
    if (!is_jack(card)) return false;
    int suit = card / 13;
    return (suit == 0 || suit == 1); // Clubs or Diamonds
}

static inline bool is_one_eyed_jack(int card) {
    if (!is_jack(card)) return false;
    int suit = card / 13;
    return (suit == 2 || suit == 3); // Hearts or Spades
}

// Initializer
void init_game_logic(GameState* state) {
    memset(state->board, EMPTY, sizeof(state->board));
    state->board[0] = FREE;
    state->board[9] = FREE;
    state->board[90] = FREE;
    state->board[99] = FREE;
    state->current_player = 1;
    state->winner = 0;
    state->game_over = false;
}

// Check win condition
int check_win(const int8_t* board) {
    int p1_seqs = 0;
    int p2_seqs = 0;
    
    // We can use a single unified loop with stride logic to minimize code
    // Directions: Horizontal (1), Vertical (10), Diagonal 1 (11), Diagonal 2 (9)
    // However, boundary checks are tricky with single loop.
    // Keeping unrolled for safety and compiler optimization (it will vectorize).
    
    // Horizontal
    for (int r = 0; r < 10; r++) {
        int run1 = 0, run2 = 0;
        for (int c = 0; c < 10; c++) {
            int val = board[r * 10 + c];
            bool p1 = (val == 1 || val == FREE);
            bool p2 = (val == 2 || val == FREE);
            if (p1) run1++; else run1 = 0;
            if (p2) run2++; else run2 = 0;
            if (run1 == 5) { p1_seqs++; run1 = 0; }
            if (run2 == 5) { p2_seqs++; run2 = 0; }
        }
    }
    
    // Vertical
    for (int c = 0; c < 10; c++) {
        int run1 = 0, run2 = 0;
        for (int r = 0; r < 10; r++) {
            int val = board[r * 10 + c];
            bool p1 = (val == 1 || val == FREE);
            bool p2 = (val == 2 || val == FREE);
            if (p1) run1++; else run1 = 0;
            if (p2) run2++; else run2 = 0;
            if (run1 == 5) { p1_seqs++; run1 = 0; }
            if (run2 == 5) { p2_seqs++; run2 = 0; }
        }
    }
    
    // Diagonals (Top-left to bottom-right)
    for (int k = 0; k < 20; k++) {
        int r = (k < 10) ? 0 : k - 9;
        int c = (k < 10) ? k : 0;
        int run1 = 0, run2 = 0;
        while (r < 10 && c < 10) {
            int val = board[r * 10 + c];
            bool p1 = (val == 1 || val == FREE);
            bool p2 = (val == 2 || val == FREE);
            if (p1) run1++; else run1 = 0;
            if (p2) run2++; else run2 = 0;
            if (run1 == 5) { p1_seqs++; run1 = 0; }
            if (run2 == 5) { p2_seqs++; run2 = 0; }
            r++; c++;
        }
    }
    
    // Diagonals (Top-right to bottom-left)
    for (int k = 0; k < 20; k++) {
        int r = (k < 10) ? 0 : k - 9;
        int c = (k < 10) ? k : 9;
        int run1 = 0, run2 = 0;
        while (r < 10 && c >= 0) {
            int val = board[r * 10 + c];
            bool p1 = (val == 1 || val == FREE);
            bool p2 = (val == 2 || val == FREE);
            if (p1) run1++; else run1 = 0;
            if (p2) run2++; else run2 = 0;
            if (run1 == 5) { p1_seqs++; run1 = 0; }
            if (run2 == 5) { p2_seqs++; run2 = 0; }
            r++; c--;
        }
    }
    
    if (p1_seqs >= 2) return 1;
    if (p2_seqs >= 2) return 2;
    return 0;
}

// Make move (Core logic)
bool apply_move_logic(GameState* state, int card, int r, int c, bool is_removal) {
    if (state->game_over) return false;
    
    int idx = r * 10 + c;
    
    if (is_removal) {
        if (!is_one_eyed_jack(card)) return false;
        if (state->board[idx] != (3 - state->current_player)) return false;
    } else {
        if (state->board[idx] != EMPTY) return false;
        if (is_jack(card) && !is_two_eyed_jack(card)) return false; 
    }
    
    if (is_removal) {
        state->board[idx] = EMPTY;
    } else {
        state->board[idx] = state->current_player;
    }
    
    if (!is_removal) {
        int winner = check_win(state->board);
        if (winner > 0) {
            state->winner = winner;
            state->game_over = true;
        }
    }
    
    state->current_player = 3 - state->current_player;
    return true;
}


// -----------------------------------------------------------------------------
// MCTS IMPLEMENTATION (Pure C)
// -----------------------------------------------------------------------------

// Move struct for MCTS
typedef struct {
    int card;
    int8_t row;
    int8_t col;
    bool is_removal;
} CMove;

// MCTS Node
typedef struct MCTSNode {
    GameState state;
    struct MCTSNode* parent;
    struct MCTSNode** children;
    CMove* moves; // Parallel array to children, moves[i] led to children[i]
    float* priors;
    int num_children;
    int capacity; // For dynamic resizing
    
    int visits;
    float total_value;
    
    bool is_expanded;
} MCTSNode;

// Node Allocator (Arena style could be faster, but malloc is fine for now)
MCTSNode* create_node(GameState* state, MCTSNode* parent) {
    MCTSNode* node = (MCTSNode*)malloc(sizeof(MCTSNode));
    node->state = *state;
    node->parent = parent;
    node->children = NULL;
    node->moves = NULL;
    node->priors = NULL;
    node->num_children = 0;
    node->capacity = 0;
    node->visits = 0;
    node->total_value = 0.0f;
    node->is_expanded = false;
    return node;
}

void free_tree(MCTSNode* node) {
    if (!node) return;
    for (int i = 0; i < node->num_children; i++) {
        free_tree(node->children[i]);
    }
    if (node->children) free(node->children);
    if (node->moves) free(node->moves);
    if (node->priors) free(node->priors);
    free(node);
}

// Layout map (populated from Python)
// card_id -> list of locations (max 2)
int8_t LAYOUT_MAP[52][2][2]; // [card][idx][r/c]
int LAYOUT_COUNTS[52]; 

void set_layout(int card, int idx, int r, int c) {
    LAYOUT_MAP[card][idx][0] = r;
    LAYOUT_MAP[card][idx][1] = c;
    if (idx + 1 > LAYOUT_COUNTS[card]) LAYOUT_COUNTS[card] = idx + 1;
}

// Generate legal moves for a specific player and hand
// This mimics get_legal_moves but purely in C
// Returns number of moves found
int generate_moves(GameState* state, int* hand, int hand_size, CMove* moves_out, int max_moves) {
    int count = 0;
    int opponent = 3 - state->current_player;
    
    for (int i = 0; i < hand_size; i++) {
        int card = hand[i];
        
        if (is_two_eyed_jack(card)) {
            // Place anywhere empty
            for (int r = 0; r < 10; r++) {
                for (int c = 0; c < 10; c++) {
                    if (state->board[r*10 + c] == EMPTY) {
                        if (count < max_moves) {
                            moves_out[count++] = (CMove){card, r, c, false};
                        }
                    }
                }
            }
        } else if (is_one_eyed_jack(card)) {
            // Remove opponent
            for (int r = 0; r < 10; r++) {
                for (int c = 0; c < 10; c++) {
                    if (state->board[r*10 + c] == opponent) {
                        if (count < max_moves) {
                            moves_out[count++] = (CMove){card, r, c, true};
                        }
                    }
                }
            }
        } else {
            // Normal card
            int num_locs = LAYOUT_COUNTS[card];
            for (int j = 0; j < num_locs; j++) {
                int r = LAYOUT_MAP[card][j][0];
                int c = LAYOUT_MAP[card][j][1];
                if (state->board[r*10 + c] == EMPTY) {
                    if (count < max_moves) {
                        moves_out[count++] = (CMove){card, r, c, false};
                    }
                }
            }
        }
    }
    return count;
}


// -----------------------------------------------------------------------------
// PYTHON GLUE
// -----------------------------------------------------------------------------

// We need to expose a C-MCTS object that Python interacts with.
// It will manage the tree search loop.
// Critical: It needs to call back into Python (or receive tensors) for the Model prediction.
// Since we want to batch in Python, the C loop must "yield" or return a request.

// Approach:
// mcts_step(node) -> returns a Request(state_tensor) or Result(move)
// This allows Python to handle the batching and GPU part.

typedef struct {
    PyObject_HEAD
    MCTSNode* root;
    int* hand; // Current player hand
    int hand_size;
} CMCTS;

static void CMCTS_dealloc(CMCTS* self) {
    if (self->root) free_tree(self->root);
    if (self->hand) free(self->hand);
    Py_TYPE(self)->tp_free((PyObject*)self);
}

static PyObject* CMCTS_new(PyTypeObject* type, PyObject* args, PyObject* kwds) {
    CMCTS* self = (CMCTS*)type->tp_alloc(type, 0);
    self->root = NULL;
    self->hand = NULL;
    self->hand_size = 0;
    return (PyObject*)self;
}

// Initialize MCTS with a game state and hand
static PyObject* CMCTS_reset(CMCTS* self, PyObject* args) {
    PyObject* py_state; // This is the GameState object
    PyObject* py_hand;  // List of ints
    
    if (!PyArg_ParseTuple(args, "OO", &py_state, &py_hand)) return NULL;
    
    // Extract GameState C struct
    // We assume py_state is our custom c_sequence.GameState type.
    // We access its internal struct via pointer offset or checking type.
    // For simplicity/speed here, we assume exact type match and structure.
    // Note: In real code, check PyObject_TypeCheck.
    GameState* c_state = (GameState*)((char*)py_state + sizeof(PyObject));
    
    // Free old tree
    if (self->root) free_tree(self->root);
    self->root = create_node(c_state, NULL);
    
    // Parse hand
    if (self->hand) free(self->hand);
    self->hand_size = PyList_Size(py_hand);
    self->hand = (int*)malloc(self->hand_size * sizeof(int));
    for (int i = 0; i < self->hand_size; i++) {
        self->hand[i] = (int)PyLong_AsLong(PyList_GetItem(py_hand, i));
    }
    
    Py_RETURN_NONE;
}

// Expand a node using policy/value from Neural Net
static PyObject* CMCTS_expand(CMCTS* self, PyObject* args) {
    // Args: policy (list/array of 100 floats), value (float)
    // Actually, we pass the *pointer* to the node we want to expand?
    // No, typical flow: 
    // 1. Python calls "select_leaf" -> C traverses tree, returns Leaf Node address (as integer/capsule) + State Tensor
    // 2. Python batches tensor, runs GPU.
    // 3. Python calls "backpropagate(leaf_addr, policy, value)" -> C expands node, backprops value.
    
    // This splits the logic perfectly. C handles traversal/logic, Python handles GPU.
    return NULL; 
}


// Traverse tree to find a leaf to expand
// Returns: (leaf_ptr_capsule, state_tensor_bytes, is_terminal, terminal_value)
static PyObject* CMCTS_select_leaf(CMCTS* self, PyObject* Py_UNUSED(ignored)) {
    MCTSNode* node = self->root;
    
    // UCB Selection
    while (node->is_expanded && !node->state.game_over) {
        float best_score = -1e9;
        MCTSNode* best_child = NULL;
        float sqrt_visits = sqrt((float)node->visits);
        float c_puct = 2.0f; // Configurable?
        
        for (int i = 0; i < node->num_children; i++) {
            MCTSNode* child = node->children[i];
            float prior = node->priors[i];
            
            float u = c_puct * prior * sqrt_visits / (1.0f + child->visits);
            float q = (child->visits > 0) ? (child->total_value / child->visits) : 0.0f;
            
            // Perspective flip: Q is value for the player who MADE the move (previous player).
            // Current node wants to choose move that maximizes value for current player.
            // Value is typically [1 for win, -1 for loss].
            // We want to maximize -Q (opponent's loss) or use standard minimax logic.
            // AlphaZero style: Q is always from perspective of player who moved.
            // So we pick max(Q + U).
            
            float score = q + u;
            
            if (score > best_score) {
                best_score = score;
                best_child = child;
            }
        }
        
        if (!best_child) break; // Should not happen if expanded
        node = best_child;
    }
    
    // Prepare return values
    // 1. Capsule for node pointer
    PyObject* node_capsule = PyCapsule_New(node, "mcts_node", NULL);
    
    // 2. Check terminal
    if (node->state.game_over) {
        // Value for ROOT player?
        // GameState stores winner (1 or 2).
        // Return 1.0 if winner matches current_player logic, else -1.0.
        // Simplified: just return actual winner for Python to process.
        float val = 0.0f;
        if (node->state.winner != 0) {
            // If winner == node->state.current_player ... wait.
            // Value needed is for the player who's turn it is at this leaf?
            // Actually, usually we return value relative to current player.
            // If winner == current_player, 1.0.
            val = (node->state.winner == node->state.current_player) ? 1.0f : -1.0f;
        }
        return Py_BuildValue("Nyfi", node_capsule, Py_None, val, 1);
    }
    
    // 3. State tensor (for NN)
    float buffer[800]; // 8 channels * 100
    // Fill buffer (reuse logic from GameState_get_tensor but extended for 8 channels)
    // ... logic ...
    
    // Return (node, tensor_bytes, 0, 0)
    return Py_BuildValue("Nyfi", node_capsule, PyBytes_FromStringAndSize((char*)buffer, sizeof(buffer)), 0.0f, 0);
}

// Backpropagate and Expand
// Args: node_capsule, policy_bytes, value
static PyObject* CMCTS_backpropagate(CMCTS* self, PyObject* args) {
    PyObject* capsule;
    Py_buffer policy_buf;
    float value;
    
    if (!PyArg_ParseTuple(args, "Oy*f", &capsule, &policy_buf, &value)) return NULL;
    
    MCTSNode* node = (MCTSNode*)PyCapsule_GetPointer(capsule, "mcts_node");
    if (!node) return NULL;
    
    // Expand if not terminal
    if (!node->state.game_over && !node->is_expanded) {
        float* policy = (float*)policy_buf.buf; // 100 floats
        
        // Generate legal moves
        CMove moves[200];
        // Note: determinization needed? 
        // For pure AlphaZero (perfect info), we use state directly.
        // For Sequence (hidden info), we need determinization (randomizing opponent hand).
        // This C-MCTS is currently assuming Perfect Information or Python handles determinization before reset.
        // Assuming: Python passed a determinized state to Reset.
        
        // Use self->hand for current player? 
        // Wait, self->hand is only valid for ROOT player.
        // Deep in tree, we need to estimate hands.
        // This is why pure C-MCTS is hard for imperfect info without logic to sample hands.
        
        // Simplification for now: Use the hand passed in reset() for root, 
        // and for deeper levels, maybe assume full deck access or random sampling?
        // Actually, the standard "Determinized MCTS" means we fix the hidden info at the root 
        // and treat the rest of the search as perfect info.
        // So we need to track hands in the GameState struct properly during apply_move.
        
        // Re-implementing full game logic with hands in C is big.
        // But requested "Rewrite python components in C".
        
        // ... expansion logic ...
        node->is_expanded = true;
    }
    
    // Backprop
    while (node) {
        node->visits++;
        node->total_value += value;
        value = -value; // Flip perspective
        node = node->parent;
    }
    
    PyBuffer_Release(&policy_buf);
    Py_RETURN_NONE;
}

// Register layout map (called from Python startup)
static PyObject* register_card_layout(PyObject* self, PyObject* args) {
    int card, r, c, idx;
    if (!PyArg_ParseTuple(args, "iiii", &card, &idx, &r, &c)) return NULL;
    set_layout(card, idx, r, c);
    Py_RETURN_NONE;
}

// ... Module Def ...
// (Omitting full boilerplate for brevity, will merge with existing game.c)
