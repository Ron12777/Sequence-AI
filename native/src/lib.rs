//! Sequence rules and bounded, neural-guided search. No raw pointers cross Python's boundary.
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyModule};
use rand::{rngs::StdRng, seq::SliceRandom, Rng, SeedableRng};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::OnceLock;

mod batch_env;
mod ensemble;

const POSITIONS: usize = 100;
const CHANNELS: usize = 11;
const MAX_NODES: usize = 100_000;
type Sequence = [u8; 5];
type Histories = [Vec<Sequence>; 2];
type LeafResult<'py> = ((u64, usize), Option<Bound<'py, PyBytes>>, f32, bool);
type ActionResult<'py> = Option<(u8, usize, usize, bool, Bound<'py, PyBytes>)>;
static GENERATION: AtomicU64 = AtomicU64::new(1);

fn layout() -> &'static [Vec<usize>; 52] {
    static MAP: OnceLock<[Vec<usize>; 52]> = OnceLock::new();
    MAP.get_or_init(|| {
        let mut result: [Vec<usize>; 52] = std::array::from_fn(|_| Vec::new());
        for (idx, label) in include_str!("../../BoardDesign.csv")
            .split([',', '\r', '\n'])
            .filter(|s| !s.trim().is_empty())
            .enumerate()
        {
            let label = label.trim();
            if label == "FREE" {
                continue;
            }
            let (rank, suit) = label.split_at(label.len() - 1);
            let rank: usize = match rank {
                "A" => 1,
                "J" => 11,
                "Q" => 12,
                "K" => 13,
                n => n.parse().expect("board rank"),
            };
            let suit = match suit {
                "C" => 0,
                "D" => 1,
                "H" => 2,
                "S" => 3,
                _ => panic!("board suit"),
            };
            result[suit * 13 + rank - 1].push(idx);
        }
        result
    })
}
fn wild(card: u8) -> bool {
    card % 13 == 10 && card / 13 < 2
}
fn removal(card: u8) -> bool {
    card % 13 == 10 && card / 13 >= 2
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct Move {
    card: u8,
    pos: usize,
    removal: bool,
}

#[derive(Clone)]
struct State {
    board: [i8; POSITIONS],
    player: u8,
    winner: u8,
    over: bool,
    sequences: Histories,
}
impl Default for State {
    fn default() -> Self {
        let mut board = [0; POSITIONS];
        for i in [0, 9, 90, 99] {
            board[i] = -1;
        }
        Self {
            board,
            player: 1,
            winner: 0,
            over: false,
            sequences: [vec![], vec![]],
        }
    }
}
impl State {
    fn protected(&self, player: u8, pos: usize) -> bool {
        self.sequences[(player - 1) as usize]
            .iter()
            .any(|s| s.contains(&(pos as u8)))
    }
    fn legal(&self, m: Move) -> bool {
        if self.over || m.card >= 52 || m.pos >= POSITIONS {
            return false;
        }
        if m.removal {
            removal(m.card)
                && self.board[m.pos] == (3 - self.player) as i8
                && !self.protected(3 - self.player, m.pos)
        } else {
            self.board[m.pos] == 0 && (wild(m.card) || layout()[m.card as usize].contains(&m.pos))
        }
    }
    fn moves(&self, hand: &[u8]) -> Vec<Move> {
        let mut result = Vec::new();
        let mut seen = [false; 52];
        for &card in hand {
            if seen[card as usize] {
                continue;
            }
            seen[card as usize] = true;
            if wild(card) || removal(card) {
                for pos in 0..POSITIONS {
                    let m = Move {
                        card,
                        pos,
                        removal: removal(card),
                    };
                    if self.legal(m) {
                        result.push(m);
                    }
                }
            } else {
                for &pos in &layout()[card as usize] {
                    let m = Move {
                        card,
                        pos,
                        removal: false,
                    };
                    if self.legal(m) {
                        result.push(m);
                    }
                }
            }
        }
        result
    }
    fn add_sequences(&mut self, pos: usize) {
        let p = (self.player - 1) as usize;
        let (row, col) = ((pos / 10) as i32, (pos % 10) as i32);
        for (dr, dc) in [(0, 1), (1, 0), (1, 1), (1, -1)] {
            // All five-cell windows containing the newly placed chip, in board order.
            for offset in (0..5).rev() {
                let mut candidate = [0u8; 5];
                let valid = (0..5).all(|k| {
                    let r = row + (k - offset) * dr;
                    let c = col + (k - offset) * dc;
                    if !(0..10).contains(&r) || !(0..10).contains(&c) {
                        return false;
                    }
                    let idx = (r * 10 + c) as usize;
                    candidate[k as usize] = idx as u8;
                    self.board[idx] == self.player as i8 || self.board[idx] == -1
                });
                if valid
                    && self.sequences[p]
                        .iter()
                        .all(|s| candidate.iter().filter(|v| s.contains(v)).count() <= 1)
                {
                    self.sequences[p].push(candidate);
                }
            }
        }
        if self.sequences[p].len() >= 2 {
            self.winner = self.player;
            self.over = true;
        }
    }
    fn apply(&mut self, m: Move) {
        self.board[m.pos] = if m.removal { 0 } else { self.player as i8 };
        if !m.removal {
            self.add_sequences(m.pos);
        }
        self.player = 3 - self.player;
    }
    fn tensor(&self, hand: &[u8]) -> Vec<f32> {
        let mut tensor = vec![0.; CHANNELS * POSITIONS];
        for (pos, &v) in self.board.iter().enumerate() {
            let channel = if v == self.player as i8 {
                0
            } else if v == (3 - self.player) as i8 {
                1
            } else if v == -1 {
                2
            } else {
                3
            };
            tensor[channel * POSITIONS + pos] = 1.;
        }
        for (i, &card) in hand.iter().enumerate() {
            for pos in 0..POSITIONS {
                // Tensor reflects legal targets even for a terminal state.
                let playable = if wild(card) {
                    self.board[pos] == 0
                } else if removal(card) {
                    self.board[pos] == (3 - self.player) as i8
                        && !self.protected(3 - self.player, pos)
                } else {
                    self.board[pos] == 0 && layout()[card as usize].contains(&pos)
                };
                if playable {
                    tensor[(4 + i) * POSITIONS + pos] = 1.;
                }
            }
        }
        tensor
    }
}
fn check_hand(hand: &[u8]) -> PyResult<()> {
    if hand.len() > 7 || hand.iter().any(|&c| c >= 52) {
        return Err(PyValueError::new_err(
            "hand must contain at most seven card IDs in 0..52",
        ));
    }
    Ok(())
}
fn floats_bytes(values: &[f32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_ne_bytes()).collect()
}

#[pyclass(name = "GameState")]
#[derive(Clone, Default)]
struct PyGameState {
    state: State,
}
#[pymethods]
impl PyGameState {
    #[new]
    fn new() -> Self {
        Self::default()
    }
    fn copy(&self) -> Self {
        self.clone()
    }
    fn get_tensor<'py>(&self, py: Python<'py>) -> Bound<'py, PyBytes> {
        PyBytes::new(
            py,
            &self
                .state
                .board
                .iter()
                .map(|&v| v as u8)
                .collect::<Vec<_>>(),
        )
    }
    fn get_info(&self) -> (u8, u8, bool) {
        (self.state.player, self.state.winner, self.state.over)
    }
    fn get_sequences(&self) -> Histories {
        self.state.sequences.clone()
    }
    fn get_legal_moves(&self, hand: Vec<u8>) -> PyResult<Vec<(u8, usize, usize, bool)>> {
        check_hand(&hand)?;
        Ok(self
            .state
            .moves(&hand)
            .iter()
            .map(|m| (m.card, m.pos / 10, m.pos % 10, m.removal))
            .collect())
    }
    fn state_tensor<'py>(
        &self,
        py: Python<'py>,
        hand: Vec<u8>,
        player: u8,
    ) -> PyResult<Bound<'py, PyBytes>> {
        check_hand(&hand)?;
        if !(1..=2).contains(&player) {
            return Err(PyValueError::new_err("player must be 1 or 2"));
        }
        let mut state = self.state.clone();
        state.player = player;
        Ok(PyBytes::new(py, &floats_bytes(&state.tensor(&hand))))
    }
    fn apply_move(&mut self, card: i32, row: i32, col: i32, is_removal: i32) -> bool {
        if !(0..52).contains(&card)
            || !(0..10).contains(&row)
            || !(0..10).contains(&col)
            || !(0..=1).contains(&is_removal)
        {
            return false;
        }
        let m = Move {
            card: card as u8,
            pos: (row * 10 + col) as usize,
            removal: is_removal != 0,
        };
        if !self.state.legal(m) {
            return false;
        }
        self.state.apply(m);
        true
    }
}

#[derive(Clone)]
struct Position {
    state: State,
    hands: [Vec<u8>; 2],
    deck: Vec<u8>,
}
impl Position {
    fn play(&mut self, m: Move) {
        let player = (self.state.player - 1) as usize;
        let i = self.hands[player]
            .iter()
            .position(|&c| c == m.card)
            .expect("generated legal card");
        self.hands[player].remove(i);
        self.state.apply(m);
        while let Some(card) = self.deck.pop() {
            if card % 13 == 10
                || layout()[card as usize]
                    .iter()
                    .any(|&p| self.state.board[p] == 0)
            {
                self.hands[player].push(card);
                break;
            }
        }
    }
}
struct Edge {
    mv: Move,
    prior: f32,
    child: Option<usize>,
}
struct Node {
    position: Position,
    parent: Option<usize>,
    edges: Vec<Edge>,
    visits: u32,
    value: f32,
    expanded: bool,
}
impl Node {
    fn new(position: Position, parent: Option<usize>) -> Self {
        Self {
            position,
            parent,
            edges: vec![],
            visits: 0,
            value: 0.,
            expanded: false,
        }
    }
}
#[pyclass(name = "CMCTS")]
struct Search {
    nodes: Vec<Node>,
    generation: u64,
    pending: Option<usize>,
    rng: StdRng,
}
#[pymethods]
impl Search {
    #[new]
    #[pyo3(signature = (seed=None))]
    fn new(seed: Option<u64>) -> Self {
        Self {
            nodes: vec![],
            generation: 0,
            pending: None,
            rng: seed
                .map(StdRng::seed_from_u64)
                .unwrap_or_else(StdRng::from_entropy),
        }
    }
    #[pyo3(signature = (board, hand, player, sequences=None, unseen=None, opponent_size=7))]
    fn reset(
        &mut self,
        board: &[u8],
        hand: Vec<u8>,
        player: u8,
        sequences: Option<Histories>,
        unseen: Option<Vec<u8>>,
        opponent_size: usize,
    ) -> PyResult<()> {
        check_hand(&hand)?;
        if board.len() != POSITIONS || !(1..=2).contains(&player) || opponent_size > 7 {
            return Err(PyValueError::new_err(
                "expected 100 board bytes, player 1 or 2, opponent_size <= 7",
            ));
        }
        let mut state = State {
            player,
            ..State::default()
        };
        for (i, &v) in board.iter().enumerate() {
            let v = v as i8;
            if if [0, 9, 90, 99].contains(&i) {
                v != -1
            } else {
                !(0..=2).contains(&v)
            } {
                return Err(PyValueError::new_err("invalid board cell or corner"));
            }
            state.board[i] = v;
        }
        if let Some(histories) = sequences {
            for (p, seqs) in histories.iter().enumerate() {
                for (number, seq) in seqs.iter().enumerate() {
                    if seq.iter().any(|&i| {
                        i >= 100
                            || (state.board[i as usize] != (p + 1) as i8
                                && state.board[i as usize] != -1)
                    }) {
                        return Err(PyValueError::new_err("sequence does not match board"));
                    }
                    let mut ordered = *seq;
                    ordered.sort_unstable();
                    let row = i32::from(ordered[0] / 10);
                    let col = i32::from(ordered[0] % 10);
                    let dr = i32::from(ordered[1] / 10) - row;
                    let dc = i32::from(ordered[1] % 10) - col;
                    if ![(0, 1), (1, 0), (1, 1), (1, -1)].contains(&(dr, dc))
                        || ordered.iter().enumerate().any(|(k, &pos)| {
                            i32::from(pos / 10) != row + k as i32 * dr
                                || i32::from(pos % 10) != col + k as i32 * dc
                        })
                        || seqs[..number]
                            .iter()
                            .any(|old| seq.iter().filter(|p| old.contains(p)).count() > 1)
                    {
                        return Err(PyValueError::new_err(
                            "invalid or overlapping sequence history",
                        ));
                    }
                }
                if seqs.len() >= 2 {
                    state.over = true;
                    state.winner = (p + 1) as u8;
                }
            }
            state.sequences = histories;
        }
        let mut hidden = unseen.unwrap_or_else(|| {
            let mut cards: Vec<u8> = (0..52).chain(0..52).collect();
            for &card in &hand {
                if let Some(i) = cards.iter().position(|&c| c == card) {
                    cards.remove(i);
                }
            }
            cards
        });
        if hidden.iter().any(|&c| c >= 52) || hidden.len() > 104 || hidden.len() < opponent_size {
            return Err(PyValueError::new_err("invalid unseen card pool"));
        }
        let mut counts = [0u8; 52];
        for &card in hand.iter().chain(hidden.iter()) {
            counts[card as usize] += 1;
            if counts[card as usize] > 2 {
                return Err(PyValueError::new_err(
                    "a two-deck game has at most two copies of each card",
                ));
            }
        }
        hidden.shuffle(&mut self.rng);
        let opponent = hidden.split_off(hidden.len() - opponent_size);
        let hands = if player == 1 {
            [hand, opponent]
        } else {
            [opponent, hand]
        };
        self.nodes.clear();
        self.nodes.push(Node::new(
            Position {
                state,
                hands,
                deck: hidden,
            },
            None,
        ));
        self.generation = GENERATION.fetch_add(1, Ordering::Relaxed);
        self.pending = None;
        Ok(())
    }
    fn select_leaf<'py>(&mut self, py: Python<'py>) -> PyResult<LeafResult<'py>> {
        if self.nodes.is_empty() {
            return Err(PyRuntimeError::new_err("reset must precede search"));
        }
        if self.pending.is_some() {
            return Err(PyRuntimeError::new_err(
                "backpropagate the pending leaf before selecting another",
            ));
        }
        let mut idx = 0;
        loop {
            let node = &self.nodes[idx];
            if !node.expanded || node.position.state.over || node.edges.is_empty() {
                break;
            }
            let sqrt = (node.visits.max(1) as f32).sqrt();
            let (edge_idx, _) = node
                .edges
                .iter()
                .enumerate()
                .map(|(i, e)| {
                    let (q, n) = e
                        .child
                        .map(|c| {
                            let child = &self.nodes[c];
                            (-child.value / child.visits.max(1) as f32, child.visits)
                        })
                        .unwrap_or((0., 0));
                    (i, q + 2. * e.prior * sqrt / (1 + n) as f32)
                })
                .max_by(|a, b| a.1.total_cmp(&b.1).then_with(|| b.0.cmp(&a.0)))
                .unwrap();
            if let Some(child) = self.nodes[idx].edges[edge_idx].child {
                idx = child;
            } else {
                if self.nodes.len() >= MAX_NODES {
                    return Err(PyRuntimeError::new_err("search node budget exhausted"));
                }
                let mut position = self.nodes[idx].position.clone();
                position.play(self.nodes[idx].edges[edge_idx].mv);
                let child = self.nodes.len();
                self.nodes.push(Node::new(position, Some(idx)));
                self.nodes[idx].edges[edge_idx].child = Some(child);
                idx = child;
            }
        }
        let node = &self.nodes[idx];
        let s = &node.position.state;
        let hand = &node.position.hands[(s.player - 1) as usize];
        let terminal = s.over || s.moves(hand).is_empty();
        let value = if s.winner == 0 {
            0.
        } else if s.winner == s.player {
            1.
        } else {
            -1.
        };
        let tensor = if terminal {
            None
        } else {
            Some(PyBytes::new(py, &floats_bytes(&s.tensor(hand))))
        };
        self.generation = GENERATION.fetch_add(1, Ordering::Relaxed);
        self.pending = Some(idx);
        Ok(((self.generation, idx), tensor, value, terminal))
    }
    fn backpropagate(
        &mut self,
        token: (u64, usize),
        policy: &[u8],
        mut value: f32,
    ) -> PyResult<()> {
        let (generation, idx) = token;
        if generation != self.generation || self.pending != Some(idx) {
            return Err(PyValueError::new_err(
                "stale, foreign, or already consumed leaf token",
            ));
        }
        if !value.is_finite() || value.abs() > 1.0001 {
            return Err(PyValueError::new_err("value must be finite and in [-1,1]"));
        }
        let node = &mut self.nodes[idx];
        if !node.expanded && !node.position.state.over {
            let hand = &node.position.hands[(node.position.state.player - 1) as usize];
            let moves = node.position.state.moves(hand);
            if !moves.is_empty() {
                if policy.len() != 400 {
                    return Err(PyValueError::new_err(
                        "policy must contain 100 float32 probabilities",
                    ));
                }
                let probabilities: Vec<f32> = policy
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .map(|b| f32::from_ne_bytes(*b))
                    .collect();
                if probabilities.iter().any(|&p| !p.is_finite() || p < 0.) {
                    return Err(PyValueError::new_err("invalid policy probability"));
                }
                // Divide shared spatial priors between distinct cards targeting the same square.
                let mut counts = [0usize; 100];
                for m in &moves {
                    counts[m.pos] += 1;
                }
                let sum: f64 = moves
                    .iter()
                    .map(|m| probabilities[m.pos] as f64 / counts[m.pos] as f64)
                    .sum();
                let uniform = 1. / moves.len() as f32;
                node.edges = moves
                    .into_iter()
                    .map(|mv| Edge {
                        prior: if sum > 0. {
                            (probabilities[mv.pos] as f64 / counts[mv.pos] as f64 / sum) as f32
                        } else {
                            uniform
                        },
                        mv,
                        child: None,
                    })
                    .collect();
            }
            node.expanded = true;
        }
        self.pending = None;
        let mut cursor = Some(idx);
        while let Some(i) = cursor {
            let n = &mut self.nodes[i];
            n.visits += 1;
            n.value += value;
            value = -value;
            cursor = n.parent;
        }
        Ok(())
    }
    fn get_action<'py>(
        &mut self,
        py: Python<'py>,
        temperature: f64,
    ) -> PyResult<ActionResult<'py>> {
        if !temperature.is_finite() || temperature < 0. {
            return Err(PyValueError::new_err(
                "temperature must be finite and nonnegative",
            ));
        }
        let Some(root) = self.nodes.first() else {
            return Ok(None);
        };
        if root.edges.is_empty() {
            return Ok(None);
        }
        let visits: Vec<f64> = root
            .edges
            .iter()
            .map(|e| e.child.map(|i| self.nodes[i].visits as f64).unwrap_or(0.))
            .collect();
        let total: f64 = visits.iter().sum();
        let base: Vec<f64> = if total > 0. {
            visits.iter().map(|v| v / total).collect()
        } else {
            root.edges.iter().map(|e| e.prior as f64).collect()
        };
        let chosen = if temperature < 1e-3 {
            base.iter()
                .enumerate()
                .max_by(|a, b| a.1.total_cmp(b.1).then_with(|| b.0.cmp(&a.0)))
                .unwrap()
                .0
        } else {
            let max = base.iter().copied().fold(0., f64::max);
            let weights: Vec<f64> = base
                .iter()
                .map(|p| (p / max).powf(1. / temperature))
                .collect();
            let mut target = self.rng.gen::<f64>() * weights.iter().sum::<f64>();
            let mut chosen = weights.len() - 1;
            for (i, w) in weights.iter().enumerate() {
                target -= w;
                if target <= 0. {
                    chosen = i;
                    break;
                }
            }
            chosen
        };
        let mut policy = [0f32; 100];
        for (e, p) in root.edges.iter().zip(base) {
            policy[e.mv.pos] += p as f32;
        }
        let mv = root.edges[chosen].mv;
        Ok(Some((
            mv.card,
            mv.pos / 10,
            mv.pos % 10,
            mv.removal,
            PyBytes::new(py, &floats_bytes(&policy)),
        )))
    }
}

// Compatibility with callers that used to initialize a mutable C global layout.
#[pyfunction]
fn setup_layout(card: usize, index: usize, row: usize, col: usize) -> PyResult<()> {
    if card >= 52 || row >= 10 || col >= 10 || layout()[card].get(index) != Some(&(row * 10 + col))
    {
        return Err(PyValueError::new_err("layout must match BoardDesign.csv"));
    }
    Ok(())
}
#[pymodule]
fn sequence_native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyGameState>()?;
    m.add_class::<batch_env::BatchEnv>()?;
    m.add_class::<Search>()?;
    m.add_class::<ensemble::Ensemble>()?;
    m.add_function(wrap_pyfunction!(setup_layout, m)?)?;
    Ok(())
}
