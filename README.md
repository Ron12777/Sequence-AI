# Sequence AI (AlphaZero Implementation)

An AlphaZero-style AI for the board game **Sequence**, optimized for massive throughput on NVIDIA GH200 (Grace Hopper) superchips.

## Architecture

*   **Core Engine:** C-based implementation (`src/c_game/`) for lightning-fast game logic and MCTS simulation.
*   **Neural Network:** ResNet-based policy/value network (PyTorch).
*   **Training:** High-throughput "Micro-Batching" architecture.
    *   **Grace CPU (72 cores):** Runs thousands of concurrent lightweight C-threads for self-play.
    *   **Hopper GPU (H100):** Consumes massive batches (up to 16k states) for inference.
    *   **Lock-Free RNG:** Custom Xorshift RNG in C to bypass the Global Interpreter Lock (GIL) and standard C library locks.

## 🚀 Option 1: One-Shot Installation (GH200 / Linux)

Run these commands exactly on your fresh Ubuntu server to get everything running in < 2 minutes.

```bash
# 1. Update & Install System Deps
sudo apt update && sudo apt install -y python3-pip python3-dev build-essential git

# 2. Create Virtual Env
python3 -m venv venv
source venv/bin/activate

# 3. Install PyTorch for GH200 (ARM64 + CUDA 12.4)
pip uninstall -y torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# 4. Install Other Python Deps
pip install numpy flask tensorboard

# 5. Compile C Extension (Crucial for Speed)
python setup.py build_ext --inplace

# 6. Verify Installation (Should show "NVIDIA GH200 480GB")
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()} | Device: {torch.cuda.get_device_name(0)}'); import c_sequence; print('C-Engine: Loaded')"
```

---

## 🐳 Option 2: Docker (NGC Container)

Use NVIDIA's pre-built container for full `torch.compile()` support (Triton included).

```bash
# ============================================================
# FROM YOUR LOCAL MACHINE - Upload code
# ============================================================
scp -r C:\Users\rhnm0\Documents\GitHub\Sequence ubuntu@<SERVER_IP>:~/Sequence-Filesystem/

# ============================================================
# ON THE GH200 SERVER
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

# 4. Install deps and build
pip install flask tensorboard
python setup.py build_ext --inplace

# 5. Verify
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()} | Device: {torch.cuda.get_device_name(0)}'); import c_sequence; print('C OK')"

# 6. Train (can use torch.compile now - no --no-compile needed)
python -m src.train train \
  --epochs 2000 \
  --games-per-epoch 100 \
  --simulations 30 \
  --batch-size 128 \
  --batch-size-inference 512 \
  --buffer-size 50000 \
  --train-steps 20 \
  --lr 0.0005 \
  --workers 14 \
  --games-per-worker 16 \
  --verbose
```

> **Note:** GH200 has 72 physical cores but Docker exposes 64 vCPU. Use `workers=14, games-per-worker=16` = 224 concurrent games.

---

## 🏋️ Training Commands

### 1. The "Baby Model" Test (10 Minutes)
Use this to quickly verify the pipeline works and generate a playable (but weak) model.
```bash
python -m src.train train --epochs 5 --games-per-epoch 100 --train-steps 50 --simulations 50 --batch-size 256 --batch-size-inference 512 --workers 8 --games-per-worker 16 --buffer-size 10000 --verbose --fresh
```

### 2. The "Sweet Spot" Run (5 Hours)
Balanced for maximum improvement in a short session.
```bash
python -m src.train train --epochs 500 --games-per-epoch 200 --train-steps 100 --simulations 60 --batch-size 256 --batch-size-inference 1024 --workers 14 --games-per-worker 32 --buffer-size 100000 --verbose
```

### 3. The "God Mode" Run (24+ Hours)
Maximum depth and quality.
```bash
python -m src.train train --epochs 5000 --games-per-epoch 500 --train-steps 200 --simulations 200 --batch-size 512 --batch-size-inference 2048 --workers 14 --games-per-worker 64 --buffer-size 500000 --verbose
```

---

## 📊 Monitoring

View real-time loss curves on your laptop while the server trains.

1.  **On Server:**
    ```bash
    nohup tensorboard --logdir runs --port 6006 --bind_all > /dev/null 2>&1 &
    ```
2.  **On Laptop (SSH Tunnel):**
    ```powershell
    ssh -N -L 6006:localhost:6006 ubuntu@<SERVER_IP>
    ```
3.  **Browser:** Go to `http://localhost:6006`

---

## 🎮 Play Against AI

1.  **Download Model:**
    ```powershell
    scp ubuntu@<SERVER_IP>:~/Sequence/models/latest.pt ./models/
    ```

2.  **Start Web Server:**
    ```bash
    python web/app.py
    ```
3.  **Open Browser:**
    Go to `http://localhost:5000`.

## Project Structure

*   `src/c_game/`: C source code for game logic and MCTS.
*   `src/model.py`: PyTorch ResNet architecture.
*   `src/train.py`: Main training loop (Multiprocessing + Async Inference).
*   `web/`: Flask web interface.