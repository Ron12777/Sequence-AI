# Sequence AI (AlphaZero Implementation)

An AlphaZero-style AI for the board game **Sequence**, optimized for massive throughput on NVIDIA GH200 (Grace Hopper) superchips.

## Architecture

*   **Core Engine:** C-based implementation (`src/c_game/`) for lightning-fast game logic and MCTS simulation.
*   **Neural Network:** ResNet-based policy/value network (PyTorch).
*   **Training:** Multiprocessing architecture with batched GPU inference.
    *   **Workers:** Multiple processes running MCTS simulations in parallel.
    *   **GPU Server:** Batches inference requests from all workers for efficient GPU utilization.
    *   **IPC Batching:** Workers accumulate 256+ states before IPC to minimize overhead.

---

## 🐳 Docker Setup (Recommended for GH200)

Use NVIDIA's NGC container for full `torch.compile()` support.

```bash
# ============================================================
# FROM YOUR LOCAL MACHINE - Upload code
# ============================================================
scp -r C:\Users\rhnm0\Documents\GitHub\Sequence ubuntu@<SERVER_IP>:~/Sequence-Filesystem/

# ============================================================
# ON THE GH200 SERVER after you ssh into it
# ============================================================

# 1. (One-time) Add user to docker group
sudo usermod -aG docker $USER
newgrp docker

# 2. Pull NGC PyTorch container
docker pull nvcr.io/nvidia/pytorch:24.12-py3

# 3. Run container with your code mounted
docker run --gpus all -it --rm \
  -v ~/Sequence-Filesystem/Sequence:/workspace/Sequence \
  -w /workspace/Sequence \
  -p 6006:6006 \
  nvcr.io/nvidia/pytorch:24.12-py3

# ============================================================
# INSIDE THE CONTAINER
# ============================================================

# 4. Install deps and build C extension
pip install flask tensorboard
python setup.py build_ext --inplace

# 5. Verify installation
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()} | Device: {torch.cuda.get_device_name(0)}'); import c_sequence; print('C OK')"
```

---

## 🏋️ Training Commands

### Quick Test (5 minutes)
Verify the pipeline works:
```bash
python -m src.train train \
  --epochs 2 \
  --games-per-epoch 20 \
  --simulations 30 \
  --workers 16 \
  --games-per-worker 16 \
  --verbose
```

### Standard Training (Sweet Spot)
Optimal settings for GH200 (~0.5-0.7 games/s, 70% GPU):
```bash
python -m src.train train \
  --epochs 100 \
  --games-per-epoch 100 \
  --simulations 50 \
  --workers 32 \
  --games-per-worker 16 \
  --batch-size 256 \
  --buffer-size 50000 \
  --verbose
```

### Long Training (Overnight)
Maximum quality:
```bash
python -m src.train train \
  --epochs 1000 \
  --games-per-epoch 200 \
  --simulations 100 \
  --workers 32 \
  --games-per-worker 16 \
  --batch-size 256 \
  --buffer-size 100000 \
  --verbose
```

### Performance Tuning

| Workers | Games/Worker | Total Games | GPU Usage | Games/s |
|---------|--------------|-------------|-----------|---------|
| 8       | 64           | 512         | 40-50%    | ~0.35   |
| 16      | 32           | 512         | 60%       | ~0.49   |
| 32      | 16           | 512         | 70%       | ~0.55   |
| 64      | 8            | 512         | 70%       | ~0.46   |

**Sweet spot: 32 workers × 16 games** - best balance of GPU utilization and throughput.

### Flags

| Flag | Default | Description |
|------|---------|-------------|
| `--workers` | 8 | Number of worker processes |
| `--games-per-worker` | 64 | Games per worker |
| `--simulations` | 50 | MCTS simulations per move |
| `--batch-size` | 256 | Training batch size |
| `--no-compile` | false | Disable torch.compile() |
| `--fresh` | false | Start training from scratch |

---

## 📊 Monitoring

View real-time loss curves:

1.  **In Container:**
    ```bash
    tensorboard --logdir runs --port 6006 --bind_all &
    ```
2.  **On Laptop (SSH Tunnel):**
    ```powershell
    ssh -N -L 6006:localhost:6006 ubuntu@<SERVER_IP>
    ```
3.  **Browser:** Go to `http://localhost:6006`

---

## 🎮 Play Against AI (Web Interface)

The web interface runs **entirely in your browser** - no server required!

### Local Play
```bash
# Serve static files
python -m http.server 8080 --directory web/static
# Open http://localhost:8080
```

### Deploy to GitHub Pages
1. Push the `web/static/` folder to your `gh-pages` branch
2. Enable GitHub Pages in repository settings
3. Play at `https://yourusername.github.io/Sequence/`

### Features
- **Client-Side AI:** Neural network runs in browser via ONNX.js (zero latency, privacy-first).
- **Analysis Mode:** Interactive board editor with "Show Best Move" helper (Gold Highlights).
- **Review Tools:** "Watch AI vs AI" mode to observe self-play strategies.
- **Dynamic UI:** Live win probability bar, extensive move history, and polished dark-mode aesthetics.
- **Adjustable Difficulty:** Depth slider (1-500 MCTS simulations) for both Red and Blue AI.

---

## Project Structure

*   `src/`: Core Python logic for training and MCTS.
*   `src/c_game/`: C engine for deterministic, high-performance game logic.
*   `tests/`: Core game logic unit tests.
*   `web/static/`: **Static web app** (HTML/CSS/JS) - deployable to any static host.
    - `game_engine.js`: Client-side game logic
    - `mcts.js`: JavaScript MCTS with ONNX.js inference
    - `model.onnx`: Trained neural network (exported from PyTorch)
*   `models/`: Storage for trained neural network weights.
*   `scripts/`: Utility scripts (ONNX export, etc.)