//! Full-game batched environment. Policies see only own hands and public history.
//! The explicit-deck constructor and debug snapshots exist for parity testing.
use crate::{layout, removal, Move, State, CHANNELS, POSITIONS};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict, PyList};
use rand::{rngs::StdRng, seq::SliceRandom, SeedableRng};

const MAX_EVENTS: usize = 128;
const CONTEXT: usize = 157;
type Event = (u8, u8, u8, u8); // card, position, absolute actor, kind

struct Game {
    state: State,
    hands: [Vec<u8>; 2],
    discards: [Vec<u8>; 2],
    deck: Vec<u8>,
    events: Vec<Event>,
    legal: Vec<i16>,
}
impl Game {
    fn from_deck(mut deck: Vec<u8>) -> Self {
        let hands = std::array::from_fn(|_| (0..7).map(|_| deck.pop().unwrap()).collect());
        let mut game = Self {
            state: State::default(),
            hands,
            discards: [vec![], vec![]],
            deck,
            events: Vec::with_capacity(104),
            legal: Vec::new(),
        };
        game.refresh();
        game
    }
    fn refresh(&mut self) {
        self.legal = self
            .state
            .moves(&self.hands[(self.state.player - 1) as usize])
            .into_iter()
            .map(|m| (usize::from(m.card) * 100 + m.pos) as i16)
            .collect();
        self.legal.sort_unstable();
        self.legal.dedup();
        if !self.state.over && self.legal.is_empty() {
            // Match Python arena termination: an exhausted/dead hand is a draw.
            self.state.over = true;
            self.state.winner = 0;
        }
    }
    fn play(&mut self, action: i16) {
        let actor = self.state.player;
        let p = usize::from(actor - 1);
        let card = (action / 100) as u8;
        let pos = (action % 100) as usize;
        let index = self.hands[p].iter().position(|&c| c == card).unwrap();
        self.hands[p].remove(index);
        self.discards[p].push(card);
        self.events
            .push((card, pos as u8, actor, u8::from(removal(card))));
        self.state.apply(Move {
            card,
            pos,
            removal: removal(card),
        });
        // Drawing still happens on a winning move, matching FastSequenceGame.
        while let Some(drawn) = self.deck.pop() {
            if drawn % 13 == 10
                || layout()[usize::from(drawn)]
                    .iter()
                    .any(|&pos| self.state.board[pos] == 0)
            {
                self.hands[p].push(drawn);
                break;
            }
            self.discards[p].push(drawn);
            self.events.push((drawn, 100, actor, 2));
        }
        self.refresh();
    }
    fn observation(&self, canonical: bool) -> Observation {
        let player = self.state.player;
        let p = usize::from(player - 1);
        let mut hand = self.hands[p].clone();
        if canonical {
            hand.sort_unstable();
        }
        let board = self
            .state
            .tensor(&hand)
            .into_iter()
            .map(|v| v as u8)
            .collect();
        let mut context = [0f32; CONTEXT];
        for &card in &self.hands[p] {
            context[usize::from(card)] += 0.5;
        }
        for &card in &self.discards[p] {
            context[52 + usize::from(card)] += 0.5;
        }
        for &card in &self.discards[1 - p] {
            context[104 + usize::from(card)] += 0.5;
        }
        context[156] = self.hands[1 - p].len() as f32 / 7.;
        let mut history = [0u8; MAX_EVENTS * 4];
        history[..4].copy_from_slice(&[53, 100, 0, 3]);
        for (index, &(card, position, actor, kind)) in self.events.iter().enumerate() {
            history[(index + 1) * 4..(index + 2) * 4].copy_from_slice(&[
                card + 1,
                position,
                u8::from(actor != player),
                kind,
            ]);
        }
        Observation {
            board,
            context,
            history,
            actions: self.legal.clone(),
            player,
        }
    }
}
struct Observation {
    board: Vec<u8>,
    context: [f32; CONTEXT],
    history: [u8; MAX_EVENTS * 4],
    actions: Vec<i16>,
    player: u8,
}

#[pyclass]
pub struct BatchEnv {
    games: Vec<Game>,
    threads: usize,
}
impl BatchEnv {
    fn validate_ids(&self, ids: &[usize], unique: bool) -> PyResult<()> {
        let mut seen = if unique {
            vec![false; self.games.len()]
        } else {
            vec![]
        };
        for &id in ids {
            if id >= self.games.len() {
                return Err(PyValueError::new_err("Environment index out of range"));
            }
            if unique {
                if seen[id] {
                    return Err(PyValueError::new_err("Duplicate environment index"));
                }
                seen[id] = true;
            }
        }
        Ok(())
    }
    fn status<'py>(&self, py: Python<'py>, ids: &[usize]) -> PyResult<Bound<'py, PyDict>> {
        let result = PyDict::new(py);
        result.set_item(
            "players",
            ids.iter()
                .map(|&i| usize::from(self.games[i].state.player))
                .collect::<Vec<_>>(),
        )?;
        result.set_item(
            "winners",
            ids.iter()
                .map(|&i| usize::from(self.games[i].state.winner))
                .collect::<Vec<_>>(),
        )?;
        result.set_item(
            "done",
            ids.iter()
                .map(|&i| self.games[i].state.over)
                .collect::<Vec<_>>(),
        )?;
        result.set_item("active", self.active_ids())?;
        Ok(result)
    }
}
#[pymethods]
impl BatchEnv {
    #[new]
    #[pyo3(signature = (count, seed=0, threads=1))]
    fn new(py: Python<'_>, count: usize, seed: u64, threads: usize) -> PyResult<Self> {
        if count == 0 || threads == 0 || threads > 256 {
            return Err(PyValueError::new_err(
                "Positive count and threads in 1..=256 required",
            ));
        }
        let games = py.allow_threads(|| {
            (0..count)
                .map(|index| {
                    let mut deck: Vec<u8> = (0..2).flat_map(|_| 0..52u8).collect();
                    deck.shuffle(&mut StdRng::seed_from_u64(seed.wrapping_add(index as u64)));
                    Game::from_deck(deck)
                })
                .collect()
        });
        Ok(Self { games, threads })
    }
    #[staticmethod]
    #[pyo3(signature = (decks, threads=1))]
    fn from_decks(py: Python<'_>, decks: Vec<Vec<u8>>, threads: usize) -> PyResult<Self> {
        if decks.is_empty() || threads == 0 || threads > 256 {
            return Err(PyValueError::new_err(
                "Nonempty decks and threads in 1..=256 required",
            ));
        }
        for deck in &decks {
            let mut counts = [0u8; 52];
            if deck.len() != 104 || deck.iter().any(|&card| card >= 52) {
                return Err(PyValueError::new_err(
                    "Each deck must contain 104 IDs in 0..52",
                ));
            }
            for &card in deck {
                counts[usize::from(card)] += 1;
            }
            if counts.iter().any(|&n| n != 2) {
                return Err(PyValueError::new_err(
                    "Each deck must contain exactly two of every card",
                ));
            }
        }
        let games = py.allow_threads(|| decks.into_iter().map(Game::from_deck).collect());
        Ok(Self { games, threads })
    }
    fn __len__(&self) -> usize {
        self.games.len()
    }
    fn active_ids(&self) -> Vec<usize> {
        self.games
            .iter()
            .enumerate()
            .filter_map(|(i, g)| (!g.state.over).then_some(i))
            .collect()
    }
    fn info<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        self.status(py, &(0..self.games.len()).collect::<Vec<_>>())
    }
    fn results(&self, player: u8) -> PyResult<Vec<f32>> {
        if !(1..=2).contains(&player) {
            return Err(PyValueError::new_err("player must be 1 or 2"));
        }
        Ok(self
            .games
            .iter()
            .map(|g| {
                if g.state.winner == 0 {
                    0.
                } else if g.state.winner == player {
                    1.
                } else {
                    -1.
                }
            })
            .collect())
    }
    /// Byte shapes: board[N,11,10,10]u8, context[N,157]f32,
    /// history[N,128,4]u8, actions[N,width]i16 with -1 padding.
    #[pyo3(signature = (ids, canonical=true))]
    fn observe<'py>(
        &self,
        py: Python<'py>,
        ids: Vec<usize>,
        canonical: bool,
    ) -> PyResult<Bound<'py, PyDict>> {
        self.validate_ids(&ids, false)?;
        let observations = py.allow_threads(|| {
            let workers = self.threads.min(ids.len().div_ceil(32).max(1));
            if workers <= 1 {
                ids.iter()
                    .map(|&i| self.games[i].observation(canonical))
                    .collect::<Vec<_>>()
            } else {
                std::thread::scope(|scope| {
                    let handles = ids
                        .chunks(ids.len().div_ceil(workers))
                        .map(|chunk| {
                            scope.spawn(move || {
                                chunk
                                    .iter()
                                    .map(|&i| self.games[i].observation(canonical))
                                    .collect::<Vec<_>>()
                            })
                        })
                        .collect::<Vec<_>>();
                    handles
                        .into_iter()
                        .flat_map(|handle| handle.join().expect("observation worker"))
                        .collect()
                })
            }
        });
        let width = observations
            .iter()
            .map(|row| row.actions.len())
            .max()
            .unwrap_or(0);
        let n = ids.len();
        let mut board = Vec::with_capacity(n * CHANNELS * POSITIONS);
        let mut context = Vec::with_capacity(n * CONTEXT * 4);
        let mut history = Vec::with_capacity(n * MAX_EVENTS * 4);
        let mut actions = Vec::with_capacity(n * width * 2);
        let mut counts = Vec::with_capacity(n);
        let mut players = Vec::with_capacity(n);
        for row in observations {
            board.extend(row.board);
            context.extend(row.context.into_iter().flat_map(f32::to_le_bytes));
            history.extend(row.history);
            counts.push(row.actions.len());
            players.push(usize::from(row.player));
            for index in 0..width {
                actions.extend(row.actions.get(index).copied().unwrap_or(-1).to_le_bytes());
            }
        }
        let result = PyDict::new(py);
        result.set_item("board", PyBytes::new(py, &board))?;
        result.set_item("context", PyBytes::new(py, &context))?;
        result.set_item("history", PyBytes::new(py, &history))?;
        result.set_item("actions", PyBytes::new(py, &actions))?;
        result.set_item("width", width)?;
        result.set_item("counts", counts)?;
        result.set_item("players", players)?;
        Ok(result)
    }
    /// Validate the entire batch before changing any game. No auto-reset.
    fn step<'py>(
        &mut self,
        py: Python<'py>,
        ids: Vec<usize>,
        actions: Vec<i16>,
    ) -> PyResult<Bound<'py, PyDict>> {
        if ids.len() != actions.len() {
            return Err(PyValueError::new_err("ids/actions length mismatch"));
        }
        self.validate_ids(&ids, true)?;
        for (&id, &action) in ids.iter().zip(&actions) {
            if self.games[id].legal.binary_search(&action).is_err() {
                return Err(PyValueError::new_err(format!(
                    "Illegal exact action {action} in environment {id}"
                )));
            }
        }
        let mut selected = vec![None; self.games.len()];
        for (&id, &action) in ids.iter().zip(&actions) {
            selected[id] = Some(action);
        }
        let threads = self.threads;
        py.allow_threads(|| {
            let workers = threads.min(self.games.len().div_ceil(32).max(1));
            let chunk_size = self.games.len().div_ceil(workers);
            if workers <= 1 {
                for (game, action) in self.games.iter_mut().zip(&selected) {
                    if let Some(action) = action {
                        game.play(*action);
                    }
                }
            } else {
                std::thread::scope(|scope| {
                    for (games, actions) in self
                        .games
                        .chunks_mut(chunk_size)
                        .zip(selected.chunks(chunk_size))
                    {
                        scope.spawn(move || {
                            for (game, action) in games.iter_mut().zip(actions) {
                                if let Some(action) = action {
                                    game.play(*action);
                                }
                            }
                        });
                    }
                });
            }
        });
        self.status(py, &ids)
    }
    /// Original-order own hand and public data for the existing search player.
    /// Opponent identities and the actual deck order are never included.
    fn search_states<'py>(&self, py: Python<'py>, ids: Vec<usize>) -> PyResult<Bound<'py, PyList>> {
        self.validate_ids(&ids, false)?;
        let result = PyList::empty(py);
        for id in ids {
            let game = &self.games[id];
            let player = usize::from(game.state.player - 1);
            let mut unseen = [2i16; 52];
            for &card in game.hands[player]
                .iter()
                .chain(game.discards[0].iter())
                .chain(game.discards[1].iter())
            {
                unseen[usize::from(card)] -= 1;
            }
            let row = PyDict::new(py);
            row.set_item(
                "board",
                PyBytes::new(
                    py,
                    &game
                        .state
                        .board
                        .iter()
                        .map(|&v| v as u8)
                        .collect::<Vec<_>>(),
                ),
            )?;
            row.set_item(
                "hand",
                game.hands[player]
                    .iter()
                    .map(|&v| usize::from(v))
                    .collect::<Vec<_>>(),
            )?;
            row.set_item("player", game.state.player)?;
            row.set_item("sequences", game.state.sequences.clone())?;
            row.set_item(
                "unseen",
                unseen
                    .iter()
                    .enumerate()
                    .flat_map(|(card, &n)| std::iter::repeat_n(card, n as usize))
                    .collect::<Vec<_>>(),
            )?;
            row.set_item("opponent_hand_size", game.hands[1 - player].len())?;
            row.set_item(
                "legal",
                game.state
                    .moves(&game.hands[player])
                    .iter()
                    .map(|m| usize::from(m.card) * 100 + m.pos)
                    .collect::<Vec<_>>(),
            )?;
            result.append(row)?;
        }
        Ok(result)
    }
    /// Testing/debugging only. Hidden hands/deck must never be policy features.
    fn debug_snapshot<'py>(&self, py: Python<'py>, index: usize) -> PyResult<Bound<'py, PyDict>> {
        self.validate_ids(&[index], false)?;
        let g = &self.games[index];
        let out = PyDict::new(py);
        out.set_item(
            "board",
            PyBytes::new(
                py,
                &g.state.board.iter().map(|&v| v as u8).collect::<Vec<_>>(),
            ),
        )?;
        out.set_item(
            "hands",
            g.hands
                .iter()
                .map(|hand| hand.iter().map(|&c| usize::from(c)).collect::<Vec<_>>())
                .collect::<Vec<_>>(),
        )?;
        out.set_item(
            "discards",
            g.discards
                .iter()
                .map(|hand| hand.iter().map(|&c| usize::from(c)).collect::<Vec<_>>())
                .collect::<Vec<_>>(),
        )?;
        out.set_item(
            "deck",
            g.deck.iter().map(|&c| usize::from(c)).collect::<Vec<_>>(),
        )?;
        out.set_item("events", g.events.clone())?;
        out.set_item("sequences", g.state.sequences.clone())?;
        out.set_item("player", g.state.player)?;
        out.set_item("winner", g.state.winner)?;
        out.set_item("done", g.state.over)?;
        out.set_item("legal", g.legal.clone())?;
        Ok(out)
    }
}
