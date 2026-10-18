//! Several independently sampled deals with a fixed total simulation budget.
use super::*;

#[pyclass(name = "EnsembleMCTS")]
pub struct Ensemble {
    trees: Vec<Search>,
    cursor: usize,
    pending: bool,
    rng: StdRng,
}
#[pymethods]
impl Ensemble {
    #[new]
    #[pyo3(signature = (seed=None, deals=4))]
    fn new(seed: Option<u64>, deals: usize) -> PyResult<Self> {
        if deals == 0 || deals > 32 {
            return Err(PyValueError::new_err("deals must be in [1,32]"));
        }
        let mut rng = seed
            .map(StdRng::seed_from_u64)
            .unwrap_or_else(StdRng::from_entropy);
        let trees = (0..deals).map(|_| Search::new(Some(rng.gen()))).collect();
        Ok(Self {
            trees,
            cursor: 0,
            pending: false,
            rng,
        })
    }
    #[pyo3(signature = (board, hand, player, sequences=None, unseen=None, opponent_size=7))]
    fn reset(
        &mut self,
        board: &[u8],
        hand: Vec<u8>,
        player: u8,
        sequences: Option<Histories>,
        mut unseen: Option<Vec<u8>>,
        opponent_size: usize,
    ) -> PyResult<()> {
        if let Some(cards) = unseen.as_mut() {
            cards.sort_unstable();
        }
        for tree in &mut self.trees {
            tree.reset(
                board,
                hand.clone(),
                player,
                sequences.clone(),
                unseen.clone(),
                opponent_size,
            )?;
        }
        self.cursor = 0;
        self.pending = false;
        Ok(())
    }
    fn select_leaf<'py>(&mut self, py: Python<'py>) -> PyResult<LeafResult<'py>> {
        if self.pending {
            return Err(PyRuntimeError::new_err("backpropagate pending leaf first"));
        }
        if self.trees[self.cursor].nodes.len() >= MAX_NODES / self.trees.len() {
            return Err(PyRuntimeError::new_err(
                "ensemble search node budget exhausted",
            ));
        }
        let result = self.trees[self.cursor].select_leaf(py)?;
        self.pending = true;
        Ok(result)
    }
    fn backpropagate(&mut self, token: (u64, usize), policy: &[u8], value: f32) -> PyResult<()> {
        if !self.pending {
            return Err(PyValueError::new_err("no pending leaf"));
        }
        self.trees[self.cursor].backpropagate(token, policy, value)?;
        self.pending = false;
        self.cursor = (self.cursor + 1) % self.trees.len();
        Ok(())
    }
    fn get_action<'py>(
        &mut self,
        py: Python<'py>,
        temperature: f64,
    ) -> PyResult<ActionResult<'py>> {
        // Use the existing temperature/action implementation on summed root counts.
        let Some(root) = self
            .trees
            .iter()
            .find_map(|t| t.nodes.first().filter(|n| n.expanded))
        else {
            return Ok(None);
        };
        let mut combined = Search::new(Some(self.rng.gen()));
        combined.nodes.push(Node::new(root.position.clone(), None));
        for (i, edge) in root.edges.iter().enumerate() {
            let child = combined.nodes.len();
            let mut node = Node::new(root.position.clone(), Some(0));
            for tree in &self.trees {
                if let Some(e) = tree.nodes.first().and_then(|n| n.edges.get(i)) {
                    if e.mv.card != edge.mv.card
                        || e.mv.pos != edge.mv.pos
                        || e.mv.removal != edge.mv.removal
                    {
                        return Err(PyRuntimeError::new_err("ensemble root actions disagree"));
                    }
                    node.visits += e.child.map(|c| tree.nodes[c].visits).unwrap_or(0);
                }
            }
            combined.nodes.push(node);
            combined.nodes[0].edges.push(Edge {
                mv: edge.mv,
                prior: edge.prior,
                child: Some(child),
            });
        }
        combined.get_action(py, temperature)
    }
}
