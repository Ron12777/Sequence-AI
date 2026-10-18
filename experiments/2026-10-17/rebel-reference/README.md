# Official ReBeL reference reproduction

This directory evaluates Facebook Research's released ReBeL implementation and pretrained value network on **Liar's Dice**, with one four-sided die per player. It does not train or evaluate a Sequence player and does not establish that ReBeL improves Sequence.

The source revision, checkpoint URL/hash, compiler and PyTorch versions are in `provenance.json`. The experiment uses the official recursive, public-belief-state CFR solver with learned depth-two leaf values. Its input contains the public betting state, queried player, and both players' ranges; its output is a value for each possible private hand. It does not pick one private deal and solve that deal as if it were known.

`run_reference.py` runs the pretrained value model and a zero-leaf-value ablation with identical CFR iterations, search depth, sampled-policy seeds and repeat count. Each invocation also runs full-tree CFR. The full-tree calculation has access to the game tree and information sets, **not to the actual opponent's hand**. At this tiny game's size, solving the full tree is both feasible and much cheaper than evaluating a learned recursive solver across many sampled policies. This is a correctness/reference experiment, not a claim of computational advantage on small games.

## Completed result (October 17, 2026)

All methods used 1,024 CFR iterations per solve. Recursive methods used depth two and 1,024 sampled-policy repeats (official seeds 0 through 1,023), with four worker threads and one PyTorch intra-op thread each.

| Method | Exploitability (lower is better) | Wall seconds | CPU seconds |
|---|---:|---:|---:|
| Learned public-belief value + recursive CFR | 0.012512 | 275.875 | 1,098.323 |
| Same recursive CFR, zero leaf values | 0.609375 | 24.925 | 95.368 |
| Full-tree CFR, evaluated once | 0.000577 | 0.274 | 0.268 |

The pretrained belief-value network sharply improves the depth-limited solver over the zero-value ablation. Full-tree CFR remains best and cheapest on this tiny game. These timings cover construction/evaluation of entire strategy profiles, **not per-move latency**. The learned and zero variants have equal iteration/repeat settings, not equal runtime; the learned network also has external pretraining that the zero ablation lacks. No claim of matched training compute is made.

Raw results and all prefix measurements are in `results-i1024-r1024.json`; the separately timed full-tree control is in `full-tree-i1024/result.json`. The program calls the full-tree result "Full FP" in a log label even when `--cfr` is enabled; the recorded commands confirm CFR was used.

## Interpretation

The reported scalar is half the sum of the two exact best-response payoffs against the evaluated strategy profile, on a terminal utility scale of -1/+1. Lower is better; zero indicates an unexploitable equilibrium. Individual player best-response payoffs may be negative because the game need not be symmetric in player order; the half-sum is the exploitability measure. It is not a win rate, loss, or Elo.

For recursive evaluation the official program averages independently seeded sampled policies, weighted by each player's information-set reach probability. It computes the exact best response against that finite aggregate. The official README describes this sampled-policy procedure as an upper-bound estimate for the underlying policy's exploitability. The prefix curve is correlated, and it is not an independent-seed confidence interval.

Each sampled policy can solve many public subgames. Equal `subgame_iters` is an equal solver-iteration setting, not equal total compute. End-to-end CPU and wall times are recorded separately; they include loading, strategy construction, exact best-response evaluation and output. No GPU was rented for this reproduction. The released network's original training cost is unknown here and is excluded; there was no new training of this value network.

## Reproduction

1. Clone `https://github.com/facebookresearch/rebel` at the revision in `provenance.json` into `/tmp/sequence-rebel-reference`.
2. Apply `compatibility.patch` with `git apply` in that checkout.
3. Download the checkpoint from the URL in `provenance.json`, and verify its SHA-256.
4. Configure this directory's `CMakeLists.txt` in a separate build directory, pointing `CMAKE_PREFIX_PATH` to the installed PyTorch CMake package. The minimal CMake file intentionally builds only the official solver/library, without old Python/pybind/Conda requirements. It uses C++20 for modern PyTorch and discovers PyTorch's ABI through its CMake package.
5. Build and run `run_reference.py --binary PATH_TO_recursive_eval --checkpoint PATH_TO_CHECKPOINT --iterations 1024 --repeats 1024 --threads 4`.

Compatibility changes are limited to including `<fstream>`, explicitly mapping saved tensors to the chosen CPU device, failing immediately if model loading fails, and enabling the existing zero-value network in the repeated-policy evaluator. No CFR, game, belief-update, strategy averaging, or exploitability equations were changed.

References: [official source](https://github.com/facebookresearch/rebel/tree/7960a42750f3407ea9eb2c3333d4c2a7961f6df4), [paper](https://arxiv.org/abs/2007.13544).
