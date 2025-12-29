# Sequence AI (AlphaZero Implementation)

An AlphaZero-style AI for the board game **Sequence**, optimized for massive throughput on NVIDIA GH200 (Grace Hopper) superchips.

## Architecture

*   **Core Engine:** C-based implementation (`src/c_game/`) for lightning-fast game logic and MCTS simulation.
*   **Neural Network:** ResNet-based policy/value network (PyTorch).
*   **Training:** High-throughput "Micro-Batching" architecture.
    *   **Grace CPU (72 cores):** Runs thousands of concurrent lightweight C-threads for self-play.
    *   **Hopper GPU (H100):** Consumes massive batches (up to 16k states) for inference.
    *   **Lock-Free RNG:** Custom Xorshift RNG in C to bypass the Global Interpreter Lock (GIL) and standard C library locks.

## 🚀 One-Shot Installation (GH200 / Linux)

Run these commands exactly on your fresh Ubuntu server to get everything running in < 2 minutes.

```bash
# 1. Update & Install System Deps
sudo apt update && sudo apt install -y python3-pip python3-dev build-essential git

# 2. Create Virtual Env
python3 -m venv venv
source venv/bin/activate

# 3. Install PyTorch for GH200 (ARM64 + CUDA 12.4)
# IMPORTANT: Uninstall any pre-existing torch first to avoid conflicts
#This is only for the Lambda GPU one. I'm not exactly sure why we need to do this, because Lambda stack is supposed to already have torch installed, but it works I guess lmao. 
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

## 🏋️ Training Commands

### 1. The "Baby Model" Test (10 Minutes)
Use this to quickly verify the pipeline works and generate a playable (but weak) model.
```bash
python -m src.train train --epochs 5 --games-per-epoch 1024 --train-steps 100 --simulations 100 --batch-size 1024 --batch-size-inference 1024 --workers 12 --threads 6 --games-per-thread 14 --buffer-size 10000 --verbose --no-compile --fresh
```

### 2. The "Sweet Spot" Run (5 Hours)
Balanced for maximum improvement in a short session.
```bash
python -m src.train train --epochs 400 --games-per-epoch 16384 --train-steps 1000 --simulations 60 --batch-size 8192 --batch-size-inference 16384 --workers 8 --threads 8 --games-per-thread 256 --buffer-size 1000000 --verbose --no-compile
```

### 3. The "God Mode" Run (24+ Hours)
Maximum depth and quality.
```bash
python -m src.train train --epochs 5000 --games-per-epoch 20000 --train-steps 5000 --simulations 400 --batch-size 8192 --batch-size-inference 16384 --workers 8 --threads 8 --games-per-thread 256 --buffer-size 2000000 --verbose --no-compile
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