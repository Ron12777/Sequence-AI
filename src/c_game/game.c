#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <stdint.h>
#include <math.h>

// -----------------------------------------------------------------------------
// CONSTANTS
// -----------------------------------------------------------------------------
#define BOARD_SIZE 10
#define NUM_POSITIONS 100
#define MAX_MOVES 150
#define FREE -1
#define EMPTY 0
#define PLAYER1 1
#define PLAYER2 2

// Arena allocator settings
#define ARENA_NODE_CAPACITY 50000    // Max nodes per arena
#define ARENA_CHILDREN_CAPACITY 500000  // Max child pointers
#define ARENA_PRIORS_CAPACITY 500000    // Max prior floats

// -----------------------------------------------------------------------------
// RNG (Thread-safe, Lock-free)
// -----------------------------------------------------------------------------
typedef struct {
    uint32_t state;
} RngState;

static inline uint32_t fast_rand(RngState* rng) {
    uint32_t x = rng->state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    rng->state = x;
    return x;
}

// -----------------------------------------------------------------------------
// STRUCTS
// -----------------------------------------------------------------------------

typedef struct {
    int8_t board[NUM_POSITIONS];
    int current_player;
    int winner;
    bool game_over;
} GameState;

typedef struct {
    int card;
    int8_t row;
    int8_t col;
    bool is_removal;
} CMove;

// Forward declaration for parent pointer
struct MCTSNode;

typedef struct MCTSNode {
    GameState state;
    struct MCTSNode* parent;
    struct MCTSNode** children; // Points into arena children pool
    float* priors;              // Points into arena priors pool
    int num_children;
    int children_offset;        // Offset in arena children pool
    int priors_offset;          // Offset in arena priors pool
    
    int visits;
    float total_value;
    bool is_expanded;
    
    // Embed move inline to avoid separate allocation
    CMove move_from_parent;
    bool has_move;
} MCTSNode;

// Arena allocator for MCTS nodes - eliminates malloc overhead
typedef struct {
    MCTSNode* nodes;           // Pre-allocated node pool
    int node_count;            // Current allocation count
    
    MCTSNode** children_pool;  // Pre-allocated children pointer pool  
    int children_count;
    
    float* priors_pool;        // Pre-allocated priors pool
    int priors_count;
} NodeArena;

// Initialize arena
static void arena_init(NodeArena* arena) {
    arena->nodes = (MCTSNode*)malloc(ARENA_NODE_CAPACITY * sizeof(MCTSNode));
    arena->node_count = 0;
    
    arena->children_pool = (MCTSNode**)malloc(ARENA_CHILDREN_CAPACITY * sizeof(MCTSNode*));
    arena->children_count = 0;
    
    arena->priors_pool = (float*)malloc(ARENA_PRIORS_CAPACITY * sizeof(float));
    arena->priors_count = 0;
}

// Reset arena (reuse memory)
static void arena_reset(NodeArena* arena) {
    arena->node_count = 0;
    arena->children_count = 0;
    arena->priors_count = 0;
}

// Free arena
static void arena_free(NodeArena* arena) {
    if (arena->nodes) free(arena->nodes);
    if (arena->children_pool) free(arena->children_pool);
    if (arena->priors_pool) free(arena->priors_pool);
    arena->nodes = NULL;
    arena->children_pool = NULL;
    arena->priors_pool = NULL;
}

// Allocate node from arena
static MCTSNode* arena_alloc_node(NodeArena* arena) {
    if (arena->node_count >= ARENA_NODE_CAPACITY) return NULL;
    return &arena->nodes[arena->node_count++];
}

// Allocate children array from arena
static MCTSNode** arena_alloc_children(NodeArena* arena, int count) {
    if (arena->children_count + count > ARENA_CHILDREN_CAPACITY) return NULL;
    MCTSNode** ptr = &arena->children_pool[arena->children_count];
    arena->children_count += count;
    return ptr;
}

// Allocate priors array from arena
static float* arena_alloc_priors(NodeArena* arena, int count) {
    if (arena->priors_count + count > ARENA_PRIORS_CAPACITY) return NULL;
    float* ptr = &arena->priors_pool[arena->priors_count];
    arena->priors_count += count;
    return ptr;
}

// -----------------------------------------------------------------------------
// GLOBALS (Static Game Data)
// -----------------------------------------------------------------------------

// Layout: [card_id][index][0=row, 1=col]
int8_t LAYOUT_MAP[52][2][2]; 
int LAYOUT_COUNTS[52];
bool LAYOUT_INITIALIZED = false;

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

// -----------------------------------------------------------------------------
// HELPER FUNCTIONS
// -----------------------------------------------------------------------------

static inline bool is_jack(int card) {
    return ((card % 13) + 1) == 11;
}

static inline bool is_red_jack(int card) {
    if (!is_jack(card)) return false;
    int suit = card / 13;
    return (suit == 1 || suit == 2); // Diamonds, Hearts
}

static inline bool is_black_jack(int card) {
    if (!is_jack(card)) return false;
    int suit = card / 13;
    return (suit == 0 || suit == 3); // Clubs, Spades
}

int check_win(const int8_t* board) {
    int p1_seqs = 0;
    int p2_seqs = 0;
    
    // Check all lines. Simplified run counting.
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
    // Diagonal 1
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
    // Diagonal 2
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

// Incremental win check - only checks lines through (row, col)
// Returns player who completed 2 sequences, or 0 if no winner yet
// Much faster than full board scan: O(36) vs O(400)
static int check_win_at(const int8_t* board, int row, int col, int player) {
    int seqs = 0;
    
    // Check 4 directions: horizontal, vertical, diag1 (\), diag2 (/)
    // For each direction, count consecutive chips including (row,col)
    
    // Direction deltas: (dr, dc)
    static const int dirs[4][2] = {{0, 1}, {1, 0}, {1, 1}, {1, -1}};
    
    for (int d = 0; d < 4; d++) {
        int dr = dirs[d][0];
        int dc = dirs[d][1];
        int count = 1;  // Count the placed chip itself
        
        // Count forward
        int r = row + dr, c = col + dc;
        while (r >= 0 && r < 10 && c >= 0 && c < 10) {
            int8_t v = board[r * 10 + c];
            if (v == player || v == FREE) {
                count++;
                r += dr; c += dc;
            } else break;
        }
        
        // Count backward
        r = row - dr; c = col - dc;
        while (r >= 0 && r < 10 && c >= 0 && c < 10) {
            int8_t v = board[r * 10 + c];
            if (v == player || v == FREE) {
                count++;
                r -= dr; c -= dc;
            } else break;
        }
        
        // A run of 5+ means at least one sequence
        if (count >= 5) seqs++;
        // Could have 2 sequences in one long run (10 = 2 sequences)
        if (count >= 10) seqs++;
    }
    
    // For 2-player game, need 2 sequences to win
    return (seqs >= 2) ? player : 0;
}

// Generate moves
// If hand is NULL, generates moves for ALL cards (simulation)
int generate_moves(const GameState* state, const int* hand, int hand_size, CMove* moves_out) {
    int count = 0;
    int opponent = 3 - state->current_player;
    
    // If hand is provided, iterate cards. 
    // If hand is NULL (opponent simulation), we iterate all 52 card types?
    // Optimization: For opponent simulation, we pick 5 random cards logic?
    // Let's assume hand is always provided (even if random).
    
    for (int i = 0; i < hand_size; i++) {
        int card = hand[i];
        
        if (is_red_jack(card)) {
            for (int j = 0; j < NUM_POSITIONS; j++) {
                if (state->board[j] == EMPTY) {
                    moves_out[count++] = (CMove){card, (int8_t)(j/10), (int8_t)(j%10), false};
                    if (count >= MAX_MOVES) return count;
                }
            }
        } else if (is_black_jack(card)) {
            for (int j = 0; j < NUM_POSITIONS; j++) {
                if (state->board[j] == opponent) {
                    moves_out[count++] = (CMove){card, (int8_t)(j/10), (int8_t)(j%10), true};
                    if (count >= MAX_MOVES) return count;
                }
            }
        } else {
            int n = LAYOUT_COUNTS[card];
            for (int k = 0; k < n; k++) {
                int r = LAYOUT_MAP[card][k][0];
                int c = LAYOUT_MAP[card][k][1];
                if (state->board[r*10 + c] == EMPTY) {
                    moves_out[count++] = (CMove){card, (int8_t)r, (int8_t)c, false};
                    if (count >= MAX_MOVES) return count;
                }
            }
        }
    }
    return count;
}

void apply_move_inplace(GameState* state, const CMove* move) {
    int idx = move->row * 10 + move->col;
    int player = state->current_player;
    
    if (move->is_removal) {
        state->board[idx] = EMPTY;
    } else {
        state->board[idx] = player;
        // Use incremental win check - only check lines through this position
        int w = check_win_at(state->board, move->row, move->col, player);
        if (w > 0) {
            state->winner = w;
            state->game_over = true;
        }
    }
    state->current_player = 3 - state->current_player;
}

// -----------------------------------------------------------------------------
// TREE MANAGEMENT (Arena-based for performance)
// -----------------------------------------------------------------------------

// Create node using arena allocator (fast path)
MCTSNode* create_node_arena(NodeArena* arena, const GameState* state, MCTSNode* parent, CMove* move) {
    MCTSNode* node = arena_alloc_node(arena);
    if (!node) return NULL;  // Arena full
    
    node->state = *state;
    node->parent = parent;
    node->children = NULL;
    node->priors = NULL;
    node->visits = 0;
    node->total_value = 0.0f;
    node->is_expanded = false;
    node->num_children = 0;
    
    if (move) {
        node->move_from_parent = *move;  // Copy inline, no malloc
        node->has_move = true;
    } else {
        node->has_move = false;
    }
    
    return node;
}

// Legacy create_node for backwards compatibility (used by PyGameState)
MCTSNode* create_node(const GameState* state, MCTSNode* parent, CMove* move) {
    MCTSNode* node = (MCTSNode*)malloc(sizeof(MCTSNode));
    if (!node) return NULL;
    
    node->state = *state;
    node->parent = parent;
    node->children = NULL;
    node->priors = NULL;
    node->visits = 0;
    node->total_value = 0.0f;
    node->is_expanded = false;
    node->num_children = 0;
    
    if (move) {
        node->move_from_parent = *move;
        node->has_move = true;
    } else {
        node->has_move = false;
    }
    
    return node;
}

// Free tree - only needed for non-arena nodes
void free_tree(MCTSNode* node) {
    if (!node) return;
    for (int i = 0; i < node->num_children; i++) {
        free_tree(node->children[i]);
    }
    if (node->children) free(node->children);
    if (node->priors) free(node->priors);
    free(node);
}

// -----------------------------------------------------------------------------
// PYTHON CLASS: GAMESTATE (Lightweight wrapper for FastSequenceGame)
// -----------------------------------------------------------------------------

typedef struct {
    PyObject_HEAD
    GameState state;
} PyGameState;

static void GameState_dealloc(PyGameState* self) {
    Py_TYPE(self)->tp_free((PyObject*)self);
}

static PyObject* GameState_new(PyTypeObject* type, PyObject* args, PyObject* kwds) {
    PyGameState* self = (PyGameState*)type->tp_alloc(type, 0);
    if (self != NULL) {
        init_game_logic(&self->state);
    }
    return (PyObject*)self;
}

static PyObject* GameState_apply_move(PyGameState* self, PyObject* args) {
    int card, r, c, is_removal_int;
    if (!PyArg_ParseTuple(args, "iiii", &card, &r, &c, &is_removal_int)) {
        return NULL;
    }
    
    // Convert to CMove
    CMove m = {card, (int8_t)r, (int8_t)c, is_removal_int != 0};
    
    // Validation (simplified reuse of logic or direct call)
    // We reuse apply_move_logic (which I need to rename or expose better)
    // Actually apply_move_logic was in previous version.
    // I need to implement a simple validator here or inline it.
    
    // Simple inline validation/application for wrapper speed
    int idx = r * 10 + c;
    bool valid = false;
    
    // Basic checks
    if (!self->state.game_over) {
        if (m.is_removal) {
            if (is_black_jack(card) && self->state.board[idx] == (3 - self->state.current_player)) valid = true;
        } else {
            if (self->state.board[idx] == EMPTY) {
                if (is_red_jack(card)) valid = true;
                else if (!is_jack(card)) {
                    // Assume layout check passed from Python
                    valid = true;
                }
            }
        }
    }
    
    if (valid) {
        apply_move_inplace(&self->state, &m);
        return PyBool_FromLong(1);
    }
    return PyBool_FromLong(0);
}

static PyObject* GameState_get_tensor(PyGameState* self, PyObject* Py_UNUSED(ignored)) {
    return PyBytes_FromStringAndSize((const char*)self->state.board, 100);
}

static PyObject* GameState_get_info(PyGameState* self, PyObject* Py_UNUSED(ignored)) {
    return Py_BuildValue("iii", self->state.current_player, self->state.winner, self->state.game_over ? 1 : 0);
}

static PyObject* GameState_copy(PyGameState* self, PyObject* Py_UNUSED(ignored)) {
    PyTypeObject* type = Py_TYPE(self);
    PyGameState* new_state = (PyGameState*)type->tp_alloc(type, 0);
    if (new_state != NULL) {
        memcpy(&new_state->state, &self->state, sizeof(GameState));
    }
    return (PyObject*)new_state;
}

static PyMethodDef GameState_methods[] = {
    {"apply_move", (PyCFunction)GameState_apply_move, METH_VARARGS, ""},
    {"get_tensor", (PyCFunction)GameState_get_tensor, METH_NOARGS, ""},
    {"get_info", (PyCFunction)GameState_get_info, METH_NOARGS, ""},
    {"copy", (PyCFunction)GameState_copy, METH_NOARGS, ""},
    {NULL}
};

static PyTypeObject PyGameStateType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "c_sequence.GameState",
    .tp_basicsize = sizeof(PyGameState),
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_new = GameState_new,
    .tp_dealloc = (destructor)GameState_dealloc,
    .tp_methods = GameState_methods,
};

// -----------------------------------------------------------------------------
// PYTHON CLASS: CMCTS
// -----------------------------------------------------------------------------

typedef struct {
    PyObject_HEAD
    MCTSNode* root;
    NodeArena arena;           // Arena allocator for fast node allocation
    int root_hand[7];
    int root_hand_size;
    RngState rng_state;
} CMCTS;

static void CMCTS_dealloc(CMCTS* self) {
    // No need to free_tree - arena handles all memory
    arena_free(&self->arena);
    Py_TYPE(self)->tp_free((PyObject*)self);
}

static PyObject* CMCTS_new(PyTypeObject* type, PyObject* args, PyObject* kwds) {
    CMCTS* self = (CMCTS*)type->tp_alloc(type, 0);
    self->root = NULL;
    self->root_hand_size = 0;
    
    // Initialize arena allocator
    arena_init(&self->arena);
    
    // Seed RNG mixed with address to ensure uniqueness across threads
    uint64_t ptr = (uint64_t)self;
    self->rng_state.state = (uint32_t)(ptr ^ (ptr >> 32)) ^ 0x9e3779b9; 
    return (PyObject*)self;
}

// Reset tree with new state
static PyObject* CMCTS_reset(CMCTS* self, PyObject* args) {
    Py_buffer board_buf;
    PyObject* py_hand;
    int player;
    
    if (!PyArg_ParseTuple(args, "y*Oi", &board_buf, &py_hand, &player)) return NULL;
    
    // Reset arena instead of freeing tree - much faster!
    arena_reset(&self->arena);
    
    GameState state;
    memcpy(state.board, board_buf.buf, 100);
    state.current_player = player;
    state.winner = 0;
    state.game_over = false;
    int w = check_win(state.board);
    if (w > 0) { state.winner = w; state.game_over = true; }
    
    // Use arena-based node creation
    self->root = create_node_arena(&self->arena, &state, NULL, NULL);
    
    // Store root hand
    self->root_hand_size = (int)PyList_Size(py_hand);
    if (self->root_hand_size > 7) self->root_hand_size = 7;
    for(int i=0; i<self->root_hand_size; i++) {
        self->root_hand[i] = (int)PyLong_AsLong(PyList_GetItem(py_hand, i));
    }
    
    PyBuffer_Release(&board_buf);
    Py_RETURN_NONE;
}

// Select Leaf
// Returns (node_capsule, tensor_bytes, is_terminal, value, player_at_leaf)
static PyObject* CMCTS_select_leaf(CMCTS* self, PyObject* Py_UNUSED(ignored)) {
    MCTSNode* node = self->root;
    
    Py_BEGIN_ALLOW_THREADS
    
    // UCB Traversal
    while (node->is_expanded && !node->state.game_over) {
        float best_score = -1e9;
        MCTSNode* best_child = NULL;
        float sqrt_visits = sqrtf((float)node->visits + 1e-8); // Avoid 0
        
        for (int i = 0; i < node->num_children; i++) {
            MCTSNode* child = node->children[i];
            float prior = node->priors[i];
            
            // Exploration
            float u = 2.0f * prior * sqrt_visits / (1.0f + child->visits);
            
            // Exploitation
            float q = 0.0f;
            if (child->visits > 0) {
                q = -1.0f * (child->total_value / child->visits);
            }
            
            if (q + u > best_score) {
                best_score = q + u;
                best_child = child;
            }
        }
        
        if (!best_child) break; // Should not happen
        node = best_child;
    }
    
    Py_END_ALLOW_THREADS
    
    // Prepare return
    PyObject* capsule = PyCapsule_New(node, "mcts_node", NULL);
    
    if (node->state.game_over) {
        float val = 0.0f;
        if (node->state.winner != 0) {
            val = (node->state.winner == node->state.current_player) ? 1.0f : -1.0f;
        }
        return Py_BuildValue("NOfi", capsule, Py_None, val, 1);
    }
    
    // Create Tensor (8x10x10 = 800 floats)
    float buffer[800];
    memset(buffer, 0, sizeof(buffer));
    
    int p = node->state.current_player;
    int opp = 3 - p;
    
    // This loop is fast, but technically could be in ALLOW_THREADS too.
    // Keeping it here for safety with accessing node->state which is C-only (safe),
    // but building the Bytes object must hold GIL.
    for (int i = 0; i < 100; i++) {
        int8_t v = node->state.board[i];
        if (v == p) buffer[i] = 1.0f;
        if (v == opp) buffer[100 + i] = 1.0f;
        if (v == FREE) buffer[200 + i] = 1.0f;
        if (v == EMPTY) buffer[300 + i] = 1.0f;
    }
    
    return Py_BuildValue("NOfi", capsule, PyBytes_FromStringAndSize((char*)buffer, sizeof(buffer)), 0.0f, 0);
}

static PyObject* CMCTS_backpropagate(CMCTS* self, PyObject* args) {
    PyObject* capsule;
    Py_buffer policy_buf; // floats
    float value;
    
    if (!PyArg_ParseTuple(args, "Oy*f", &capsule, &policy_buf, &value)) return NULL;
    
    MCTSNode* node = (MCTSNode*)PyCapsule_GetPointer(capsule, "mcts_node");
    float* policy = (float*)policy_buf.buf;
    
    // Copy necessary data to local vars/buffers to release GIL safely
    // policy buffer is locked by PyArg_ParseTuple "y*", safe to read? 
    // Yes, buffer view holds reference.
    
    Py_BEGIN_ALLOW_THREADS
    
    if (!node->is_expanded && !node->state.game_over) {
        // ... (Hand logic omitted for brevity, reusing existing structure) ...
        int hand[7];
        int sz = 0;
        
        if (node == self->root) {
            sz = self->root_hand_size;
            memcpy(hand, self->root_hand, sz * sizeof(int));
        } else {
            sz = 5;
            for(int i=0; i<5; i++) hand[i] = fast_rand(&self->rng_state) % 52; 
        }
        
        CMove moves[MAX_MOVES];
        int n_moves = generate_moves(&node->state, hand, sz, moves);
        
        if (n_moves > 0) {
            // Use arena allocation instead of malloc
            node->children = arena_alloc_children(&self->arena, n_moves);
            node->priors = arena_alloc_priors(&self->arena, n_moves);
            
            if (node->children && node->priors) {
                node->num_children = n_moves;
                
                float sum_p = 1e-6f;
                for(int i=0; i<n_moves; i++) {
                    int idx = moves[i].row * 10 + moves[i].col;
                    float p = expf(policy[idx]); 
                    node->priors[i] = p;
                    sum_p += p;
                }
                
                for(int i=0; i<n_moves; i++) {
                    node->priors[i] /= sum_p;
                    
                    GameState next = node->state;
                    apply_move_inplace(&next, &moves[i]);
                    // Use arena-based node creation
                    node->children[i] = create_node_arena(&self->arena, &next, node, &moves[i]);
                }
            }
        }
        node->is_expanded = true;
    }
    
    // Backprop
    while (node) {
        node->visits++;
        node->total_value += value;
        value = -value;
        node = node->parent;
    }
    
    Py_END_ALLOW_THREADS
    PyBuffer_Release(&policy_buf);
    Py_RETURN_NONE;
}

static PyObject* CMCTS_get_action(CMCTS* self, PyObject* args) {
    float temp;
    if (!PyArg_ParseTuple(args, "f", &temp)) return NULL;
    
    MCTSNode* root = self->root;
    if (!root || root->num_children == 0) Py_RETURN_NONE;
    
    // Temperature logic
    // Simplified: argmax if temp < 1e-3, else sample
    int chosen = 0;
    
    if (temp < 1e-3) {
        int max_v = -1;
        for(int i=0; i<root->num_children; i++) {
            if (root->children[i]->visits > max_v) {
                max_v = root->children[i]->visits;
                chosen = i;
            }
        }
    } else {
        // Sample based on visits
        float sum = 0;
        float weights[MAX_MOVES];
        for(int i=0; i<root->num_children; i++) {
            weights[i] = powf((float)root->children[i]->visits, 1.0f/temp);
            sum += weights[i];
        }
        
        float r = ((float)fast_rand(&self->rng_state) / (float)4294967295U) * sum;
        float acc = 0;
        for(int i=0; i<root->num_children; i++) {
            acc += weights[i];
            if (acc >= r) { chosen = i; break; }
        }
    }
    
    CMove* m = &root->children[chosen]->move_from_parent;
    
    // Build policy vector for training
    float policy[100];
    memset(policy, 0, sizeof(policy));
    int total_visits = root->visits - 1;
    if (total_visits < 1) total_visits = 1;
    
    for(int i=0; i<root->num_children; i++) {
        CMove* cm = &root->children[i]->move_from_parent;
        int idx = cm->row * 10 + cm->col;
        policy[idx] = (float)root->children[i]->visits / total_visits;
    }
    
    // Return: (card, row, col, is_removal, policy_bytes)
    return Py_BuildValue("iiiiy#", m->card, m->row, m->col, m->is_removal ? 1 : 0, 
                         (char*)policy, (Py_ssize_t)sizeof(policy));
}

// Layout setup
static PyObject* setup_layout(PyObject* self, PyObject* args) {
    int card, idx, r, c;
    if (!PyArg_ParseTuple(args, "iiii", &card, &idx, &r, &c)) return NULL;
    LAYOUT_MAP[card][idx][0] = (int8_t)r;
    LAYOUT_MAP[card][idx][1] = (int8_t)c;
    if (idx + 1 > LAYOUT_COUNTS[card]) LAYOUT_COUNTS[card] = idx + 1;
    Py_RETURN_NONE;
}

// -----------------------------------------------------------------------------
// MODULE INIT
// -----------------------------------------------------------------------------

static PyMethodDef CMCTS_methods[] = {
    {"reset", (PyCFunction)CMCTS_reset, METH_VARARGS, ""},
    {"select_leaf", (PyCFunction)CMCTS_select_leaf, METH_NOARGS, ""},
    {"backpropagate", (PyCFunction)CMCTS_backpropagate, METH_VARARGS, ""},
    {"get_action", (PyCFunction)CMCTS_get_action, METH_VARARGS, ""},
    {NULL}
};

static PyTypeObject CMCTSType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "c_sequence.CMCTS",
    .tp_basicsize = sizeof(CMCTS),
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_new = CMCTS_new,
    .tp_dealloc = (destructor)CMCTS_dealloc,
    .tp_methods = CMCTS_methods,
};

static PyMethodDef module_methods[] = {
    {"setup_layout", setup_layout, METH_VARARGS, ""},
    {NULL}
};

static PyModuleDef c_sequencemodule = {
    PyModuleDef_HEAD_INIT,
    .m_name = "c_sequence",
    .m_doc = "C Sequence Engine",
    .m_size = -1,
    .m_methods = module_methods,
};

PyMODINIT_FUNC PyInit_c_sequence(void) {

    PyObject* m;

    if (PyType_Ready(&CMCTSType) < 0) return NULL;

    if (PyType_Ready(&PyGameStateType) < 0) return NULL;

    

    m = PyModule_Create(&c_sequencemodule);

    if (!m) return NULL;

    

    Py_INCREF(&CMCTSType);

    PyModule_AddObject(m, "CMCTS", (PyObject*)&CMCTSType);

    

    Py_INCREF(&PyGameStateType);

    PyModule_AddObject(m, "GameState", (PyObject*)&PyGameStateType);

    

    return m;

}
