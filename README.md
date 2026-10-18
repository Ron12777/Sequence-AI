# Sequence AI

Play the static browser app at https://ron12777.github.io/Sequence-AI/.

This repository contains a two-player Sequence engine, a PyTorch policy/value network, a Rust search backend, and a browser implementation. Python is the reference rules implementation; regression tests compare complete games against Rust.

## Current local browser model — October 17, 2026

The local static app now uses an exact-action CNN trained with PPO: three residual blocks, 64 channels, and an order-invariant encoder for the player's hand. It chooses both the card to spend and the board square directly, without MCTS or a forced-win override. This board/hand variant does not consume ordered public history or played-card counts. The GitHub Pages workflow publishes this static model when these files are pushed to `main`. The legacy Flask API still uses search.

The final checkpoint passed a fixed independent confirmation: 53.62% game score against the previous validated CNN over 8,192 paired games (95% interval 52.75–54.49%), 64.40% against the original 128-simulation search player, and 80.58% against the held-out 32-simulation baseline. Draws count as half a point. Both held-out comparisons passed the predeclared retention gates; see [final confirmation evidence](experiments/2026-10-17/final-confirmation-summary.json).

The selected model completed 6.32 additional training wall-hours and 10,846,208 games after common initialization distilled from a CNN with 7.48 recorded training hours. Its additional allocated training budget was 9.13 GPU-hours across mixed hardware; this excludes inherited training, distillation, evaluation, and idle rental time. The final experiment does not establish an architecture's ultimate ceiling or equal-FLOP superiority.

The selected checkpoint is `artifacts/checkpoints/exact-final/cnn_ppo-final.pt`. Final CNN, transformer, and R-NaD continuation states, historical opponent pools, and their provenance are versioned under `artifacts/retained-campaign-final/`; build caches and abandoned smoke checkpoints are excluded. The browser manifest selects `web/static/model-exact-0bd3130842892f06.onnx`.

Native ONNX and browser WASM action parity passed, followed by rendered Chrome gameplay and edited-board checks. Previous weights and manifest are retained under `artifacts/rollback/exact-final-2026-10-17/`. All selected continuation states and evidence were retrieved and checksummed before the sole rental was destroyed. Serve the local app with:

```bash
python -m http.server 8765 --bind 127.0.0.1 --directory web/static
```

Large experiment traces are stored losslessly as `.json.gz` or `.jsonl.gz`. To restore the raw files used by analysis scripts, run `python scripts/unpack_evidence.py`. The helper checks existing files and does not overwrite different content.

## Local setup

Install Python 3.10 or newer and Rust 1.88 or newer. Install a PyTorch build appropriate for your hardware, then:

```bash
python -m pip install -r requirements.txt pytest
python -m pip install .
python -m pytest -q
node tests/browser.test.cjs
```

`pip install .` builds `sequence_native` through maturin. The old `setup.py build_ext` route is retired. The pre-existing C source in `src/c_game/` is retained for reference and is not built or imported by the application.

The local Flask service can be launched with `python web/app.py`. It serves a single shared game; state-changing requests are serialized. The static browser app does not require Flask or Python.

## Rules implemented

The engines support two players, seven cards per player, and two sequences to win. Clubs/diamonds Jacks place wild chips; hearts/spades Jacks remove unprotected opposing chips. Completed five-chip sequences are retained, protected against removal, and may share at most one chip. A nine-chip run can therefore contain two sequences.

The current game variant skips ordinary dead cards when drawing, retains Jacks, and does not recycle discards. Self-play scores a position with no legal moves, or a game that reaches 200 moves, as a draw. It does not implement the full official exchange/pass/deck-recycling procedure or multiplayer teams. These choices are shared between training and the playable engines where applicable.

## Legacy search and model

The network receives 11 channels: own chips, opponent chips, corners, empty spaces, and seven legal-target masks for the current hand. It predicts a 100-square spatial policy and a value from the player-to-move's perspective. Coordinates remain fixed for both players. Actions using different cards at the same square share the spatial prior; search can distinguish them, but the policy head cannot directly learn a separate card preference.

Legacy search gameplay and search-based training use four independently sampled hidden deals and aggregate root action visits by full card/move identity. The default gameplay budget is 128 simulations total, split across the four deals. The unseen multiset comes only from two decks minus the observer’s own hand and public discard piles, including skipped dead cards. Search never reads the opponent’s actual hidden card identities to construct that pool. Single-deal search remains an explicit benchmark control. There is no forced-win heuristic override.

This is ensemble determinization, not a full belief-state solver: future decisions within each sampled tree can still exploit that tree’s hidden allocation, and sampling is not conditioned on what opponents’ historical choices imply about their hands. The network itself still receives 11 board/hand channels without discard/history features. A history-aware policy or an imperfect-information learning/search method such as [ReBeL](https://arxiv.org/abs/2007.13544) is a separate architecture change, not something four-deal sampling implements. More simulations are a compute budget, not a fixed depth or a guarantee of stronger play.

Rust owns nodes through vector indices, allocates children lazily, limits the tree to 100,000 nodes, and checks leaf generations and buffer lengths at the Python boundary. A tree may have only one outstanding leaf. Independent games provide inference batching.

## Search-based training

CUDA and a GPU with bfloat16 support (Ampere or newer) are required for the multiprocess trainer. Use one GPU and tune worker count and games per worker to the host; more processes can reduce throughput. The inference server uses bfloat16 and channels-last tensors. Actors share observation and prediction buffers with the server; queues carry only notifications. Each actor owns one slot and waits for its response before reusing it. Use `--no-shared-inference` to benchmark the older queue transport. `--no-shared-inference --compact-observations` transports binary masks as bytes and restores float32 at inference. It cuts payload size fourfold, but is experimental and off by default: its throughput benefit varied with worker count in our tests. `--overlap-training` keeps a frozen actor model and interleaves learner updates with inference so CPU actors can progress while the GPU trains. It batches samples across workers with a short maximum wait. Checkpoints include architecture, optimizer, scheduler, and epoch, and are replaced atomically.

```bash
python -m src.train train --fresh --no-compile \
  --epochs 24 --games-per-epoch 512 --simulations 16 \
  --workers 16 --games-per-worker 128 \
  --num-filters 64 --num-res-blocks 3 \
  --batch-size 2048 --train-steps 96 --buffer-size 100000 \
  --output-dir experiments/small --seed 37
```

The command above is the measured small-model recipe; lower `--workers` for a smaller CPU host. Use `--search-mode single` only to reproduce the earlier single-deal training controls. `--root-noise-fraction` controls the exploration mixture at legal root targets (default 0.25).

Training stores only `latest.pt` for that run, JSONL metrics alongside it, and TensorBoard events under the sibling `runs/` directory. Omit `--fresh` to resume. After its saved cosine schedule reaches the minimum learning rate, resumed training keeps that rate. Old checkpoints with eight input channels load with zero-initialized additional channels; compiled checkpoint key prefixes are normalized. Random-generator state and the replay buffer are not restored, so resume is not bit-for-bit continuation.

Use `--init-from path/to/checkpoint.pt` to initialize from trained weights with a new optimizer and learning-rate schedule. The checkpoint determines the architecture. This is useful for fine-tuning after the original cosine schedule has finished.

## Experiments and validation

```bash
# Short, end-to-end GPU throughput sweep. Reports actual GPU utilization.
python scripts/benchmark.py --seconds 60 --shared-inference
# Omit --shared-inference for the queue transport control.

# Held-out, color-balanced games against the same baseline, with confidence intervals.
python scripts/evaluate.py --checkpoint experiments/small/latest.pt \
  --games 256 --simulations 0,8,32 --opponent greedy

# Bounded small/base runs and evaluations, sequentially on one GPU.
bash scripts/run_experiments.sh
```

Use game score, wall time, and cost together. A lower loss or fuller GPU does not by itself establish better play. Evaluation uses fixed seeds and reports draws separately. The greedy opponent is a basic adjacent-chip heuristic, not a strong reference player.

### Measured experiments — 2026-10-16

The selected candidate uses 64 filters, 3 residual blocks, 16 training simulations, and a 0.25 root-noise fraction. It trained on 12,288 self-play games. On 512 fresh, color-balanced games per evaluation budget against the greedy baseline:

| Evaluation budget | Wins / losses / draws | Score |
| --- | --- | --- |
| Policy only | 367 / 145 / 0 | 71.7% |
| 8 simulations | 387 / 125 / 0 | 75.6% |
| 32 simulations | 400 / 112 / 0 | 78.1% |

The 32-simulation decisive win-rate interval is 74.3–81.5% (95% Wilson). Reducing training search to 8 simulations scored 67.0% at the same evaluation budget; increasing root noise to 0.75 scored 54.0%. These are short, single-training-seed comparisons against a simple heuristic, not a general strength ranking. An earlier 512-game comparison also favored the small model over 128 filters / 6 blocks (78.9% versus 74.0% at 32 simulations).

An RTX 5080 rental at $0.278/hour with 32 vCPUs, 64 GB RAM, and 40 GB disk provided about 1.8 times the measured self-play throughput per dollar of the $0.537/hour RTX 6000 Ada host in repeat controls. Float observation transport measured 95–99k leaf evaluations/second in those controls; compact byte transport measured 56k and remains opt-in. The selected training run averaged 43.7% GPU utilization, peaking at 99% during learning. Self-play remains CPU/IPC-bound; sustained full GPU saturation was not achieved. On the subsequent RTX 4090 host ($0.468/hour), a queue/shared/shared/queue comparison at 24 workers and 64 simulations measured 94k versus 188k leaf evaluations/second. A complete 3,072-game training check took 159 versus 82 seconds, including startup and checkpoint saving. Shared inference is now the default. Its sampled inference GPU utilization was about 23%; full saturation remains unachieved. Raw measurements and limitations are in `experiments/2026-10-17/inference_transport_results.json`. Overlapping training did not materially improve utilization in the earlier controlled comparison and also remains opt-in.

Across all runs, 67,584 self-play games contributed to training and 11,136 evaluation games were played. Both rentals were destroyed after verifying retrieved results. The observed Vast credit decrease was approximately $0.31; billing may settle later. Raw results, confidence intervals, configurations, validation, and limitations are recorded in [the experiment report](experiments/2026-10-16/results.json).

Local candidate artifacts are `artifacts/checkpoints/small-standard.pt` and `artifacts/small-standard.onnx`. They are ignored by Git. The ONNX candidate passed numerical comparison and a browser game smoke test; the original model files were preserved.

### Deadline-bounded campaigns

`scripts/overnight.py` runs sequential training trials on one GPU until an explicit UTC deadline. It preserves a champion, rollback checkpoint, and fixed baseline, and removes rejected trial weights. A file lock prevents duplicate controllers. New candidate comparisons use equal 128-simulation budgets and record each agent’s search method. The editable `campaign/plan.json` controls subsequent trials; `campaign/state.json`, per-round metrics, and evaluation results record progress.

```bash
python scripts/overnight.py --baseline models/latest.pt \
  --stop-at 2026-10-17T15:00:00Z --workers 16 --output-dir campaign
```

`scripts/arena.py` compares models on paired deals with swapped colors. Candidates must pass a 512-game screen, a separate 1,024-game confirmation with a positive paired confidence interval, and regression checks before replacing the champion. These selection checks do not establish general optimality; the controller reserves fresh seeds for final evaluation against the original baseline and greedy opponent.

`scripts/campaign_watchdog.py` can stop its own Vast instance at a separate deadline using the instance-scoped credential already supplied by Vast. It preserves disk contents for retrieval. Stopping compute does not destroy storage; download and checksum-verify the selected checkpoint and evidence, then destroy the rental. Local scheduled reviews require the computer and Codex app to remain running. The campaign's rental details and deadlines are recorded in `experiments/2026-10-17/campaign.json`.

## Moving between machines

`scripts/package_run.py` creates an allowlisted source bundle and SHA-256 manifest. It excludes `.git`, credentials, caches, build directories, ONNX assets, and intermediate checkpoints. Pass `--checkpoint` to include exactly one required checkpoint and `--wheel` to include a compatible prebuilt Rust wheel:

```bash
python scripts/package_run.py /tmp/sequence-run.tar.gz \
  --checkpoint experiments/small/latest.pt
```

Size disk for the container, Python environment, replay/log output, and the few retained checkpoints. Do not fill RAM or disk merely to claim utilization. Keep only one rental active while benchmarking, retrieve selected results before stopping it, and include storage and transfer charges when comparing costs.

## Export

```bash
python -m pip install onnx
python scripts/export_onnx.py
# Export an experimental candidate without replacing the browser model.
python scripts/export_onnx.py --checkpoint experiments/small/latest.pt --output artifacts/candidate.onnx
```

The exporter reads `models/latest.pt` and writes the browser's `web/static/model.onnx`. Existing artifacts are not automatically replaced by experimental checkpoints. Validate a candidate's playing strength before promoting it.

## Layout

- `native/`: Rust rules and MCTS, PyO3 bindings, Cargo lockfile.
- `src/`: reference rules, Rust wrapper, neural network, search interface, training.
- `tests/`: native/Python parity, search, checkpoint, training, Flask and browser regressions.
- `scripts/`: export, packaging, benchmark and evaluation entry points.
- `web/static/`: browser engine and UI.

## Experimental policies with public history

`src/history_policy.py` provides single-network direct policies initialized from a
trained CNN: a board-only control, a public-card-summary model, a GRU history
encoder, an attention history encoder, and joint board/history attention. Inputs
contain the observing player's current hand, board, public discards, opponent hand
size, and chronological public move/dead-card history. The policies never receive
hidden hand identities or deck order. They choose legal moves without search.
The joint variant lets board positions and past public events attend to one
another before producing the policy. The other history variants combine their
history representation with the CNN at the output.

These are experimental architectures, not ReBeL or equilibrium solvers. The
production champion remains selected by independent matches. The checkpoint
format preserves the complete history architecture and its training metadata;
history exports use their own multi-input ONNX interface; the current board/hand exact-action export has explicit capability metadata.

Generate one fixed teacher dataset and fit variants on the same game-level split:

```bash
python scripts/architecture_experiments.py \
  --champion campaign/champion.pt --output-dir campaign/architectures_v1 \
  --games 512 --epochs 3 --evaluation-games 128
```

The teacher uses four-deal search; candidates act directly. Evaluation uses fresh
color-paired deals against the teacher's direct policy, its 128-simulation ensemble
policy, and greedy play. These initial screens are too small to establish
superiority by themselves. The larger joint-attention experiment uses
`--architectures board,joint_attention --games 4096 --teacher-simulations 64`.
The fitter now includes the unchanged initial checkpoint in selection and reports
held-out policy and value losses separately. `--freeze-backbone` trains only the
new history layers; use `--learning-rate` to control that experiment. Board-only
models cannot use this flag because they would have no trainable parameters.

Train a candidate through search-free PPO self-play:

```bash
python scripts/direct_selfplay.py \
  --init-from campaign/architectures_v1/gru.pt \
  --champion campaign/architectures_v1/teacher.pt \
  --output-dir campaign/direct_selfplay_gru \
  --iterations 16 --games-per-iteration 128 --evaluation-games 512
```

PPO samples legal positions and learns from terminal outcomes. BatchNorm
statistics stay fixed during updates to match rollout inference; recurrent layers
stay in training mode for cuDNN backpropagation. There is no privileged critic or
sampled hidden deal during direct self-play. This method does not imply equilibrium
convergence. CPU and CUDA tests cover legal sampling and actual parameter updates.
Evaluation seeds are allocated beyond the entire training seed range, including
long runs. Detailed results and pending research-job metadata are saved in
`experiments/2026-10-17/architecture_results.json`.

For an incremental GPU-hour comparison, `scripts/compute_budget_experiments.py`
starts CNN, GRU, and joint-attention candidates from the same frozen champion.
It tests the same two rollout/update batch configurations for each architecture,
discards pilot weights, and trains each candidate under the same elapsed-time
budget on an otherwise idle GPU. Self-play, updates, initialization, and checkpoint
I/O count toward the budget. A final update crossing the deadline is discarded;
its actual extra cost is still reported. Two independent training seeds use the
same held-out evaluation deals, with architecture order reversed on the second
seed. Evaluation time and pilot costs are separate from training time.

The script temporarily pauses the specified existing campaign process tree and
resumes it on exit. An independent recovery guard also resumes it if the experiment
supervisor dies. It refuses to run alongside another active GPU workload. Use a
window shorter than the main controller's remaining per-trial timeout. Results
include utilization samples, completed games/positions/updates, checkpoint hashes,
and synchronized end-to-end move latency at batch sizes 1 and 128. GPU utilization
is diagnostic; held-out strength per allocated GPU-hour is the objective. This
warm-start comparison does not establish the best architecture from scratch or
at arbitrary training scales, and direct policy versus search scores still have
different inference costs.

`scripts/sustained_selfplay.py` runs the longer comparison. It alternates CNN and
GRU in 30-minute training blocks until an absolute deadline, retaining each
model's Adam moments, RNG state, cumulative training time, and learning-rate
schedule. Both start from the same frozen champion. Half the games use current
self-play; the remainder face the frozen reference or a rotating historical
policy, with both learner seats represented. Opponent actions never enter the
learner's PPO update. The recipe uses smaller updates than the exploratory PPO
runs and anneals learning rate and entropy by cumulative training time.

Atomic `latest.pt` files contain continuation state. Historical opponents and
`best_monitor.pt` are separate; monitoring-selected weights still require fresh
independent confirmation before promotion. Monitoring reports policy-only,
greedy, champion-search, and held-out older-checkpoint opponents. The shared
campaign controller lock prevents overlap with the original tournament. A
`selection.json` can narrow training only after every contender has received at
least one hour. The runner itself never promotes an experimental checkpoint.

Evaluation explicitly freezes both networks, and the shared prediction helper
rejects train-mode modules. Regression tests verify that neither weights nor
BatchNorm buffers change during direct or search evaluation. Some earlier
standalone PPO and compute-budget research scores violated this requirement;
their limitations are recorded in
`experiments/2026-10-17/evaluation_mode_audit.json`. Those results must not be used
to rank architectures or claim a model improvement.
