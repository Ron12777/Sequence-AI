# Sequence AI

[Play Sequence](https://ron12777.github.io/Sequence-AI/)

Play Sequence against a neural network, watch the AI play itself, or edit the board to explore a position. The game and model run entirely in your browser.

- **Hint** highlights a recommended move.
- **Edit board** lets you change chips, hands, and whose turn it is.
- **Watch AI** has speed controls from 0.5× to Max, plus pause and resume.
- The evaluation bar shows the model’s estimate of the position.

## Run locally

```sh
python3 -m http.server 8765 --bind 127.0.0.1 --directory web/static
```

Open [localhost:8765](http://127.0.0.1:8765/).

## How it works

The AI is a CNN trained with PPO through self-play. Three residual blocks with 64 channels process the board, and a hand encoder processes the cards available to the player. The network chooses a card and a square in one inference call.

Training uses PyTorch and a Rust game engine exposed to Python through PyO3. The browser uses a JavaScript game engine and runs the exported network with ONNX Runtime Web. The AI sees the board and its own hand; your cards and the draw pile stay hidden.

The current model was trained on an RTX 5090 and scored **54.31% against the previous CNN across 8,192 games**. The [model manifest](web/static/model.json) identifies the deployed weights. Checkpoints, training details, and evaluation results are saved under [artifacts](artifacts/) and [experiments](experiments/).

## Development

Requires Python 3.10+, Rust 1.88+, and Node.js 22+. Install PyTorch for your CPU or CUDA setup, then:

```sh
python -m pip install -r requirements-test.txt .
OMP_NUM_THREADS=1 python -m pytest -q
npm ci
npm test
npx playwright install --with-deps chromium
npm run test:ui
```

`pip install .` builds the Rust extension. GitHub Actions runs the Python, Rust, and desktop/mobile browser tests before deploying to GitHub Pages. See [tests/README.md](tests/README.md) for coverage and [Rules.md](Rules.md) for the game rules.

## Training

[`train_exact_policy.py`](scripts/train_exact_policy.py) handles PPO training. [`train_exact_multigpu.py`](scripts/train_exact_multigpu.py) schedules workers and resumes checkpoints, including optimizer state and opponent pools.

```sh
python scripts/train_exact_policy.py --help
python scripts/train_exact_multigpu.py --help
```

Export a checkpoint for the browser:

```sh
python scripts/export_exact_onnx.py \
  --checkpoint artifacts/checkpoints/overnight-2026-10-18/final/cnn_ppo-final.pt \
  --output artifacts/candidate.onnx --samples 192
```

## Project layout

| Directory | Contents |
| --- | --- |
| `native/` | Rust game engine and Python bindings |
| `src/` | Python rules, networks, and learning algorithms |
| `scripts/` | Training, evaluation, and export tools |
| `web/static/` | Browser game and ONNX models |
| `tests/` | Engine, model, and UI tests |
| `experiments/` | Training runs and evaluation results |
| `artifacts/` | Model checkpoints |

Large experiment reports are compressed. To unpack them, run `python scripts/unpack_evidence.py`.
