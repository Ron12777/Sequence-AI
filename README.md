# Play at https://ron12777.github.io/Sequence-AI/

# Sequence AI 

An AI for the board game Sequence, trained on a NVIDIA GH200

## Training Architecture

*   **Core Engine:** C-based implementation (`src/c_game/`) for for game logic and MCTS simulation. 
*   **Neural Network:** ResNet-based policy/value network (PyTorch).
*   **Training:** Multiprocessing architecture with batched GPU inference.
    *   **Workers:** Multiple processes running MCTS simulations in parallel.
    *   **GPU Server:** Batches inference requests from all workers for efficient GPU utilization.
The game logic and MCTS does not need to be in C. You could very easily use python and do everything through numpy arrays, it is not CPU bound in any sense. 


## Docker Setup 
This was done through Lambda Cloud on a GH200 running Ubuntu Server 24.04. 
You do not need to do this, but it is recommended as for some reason the base image they have the pytorch wasn't working with the CUDA version they had on there, and you would have to update pytorch. Also torch.compile() doesn't work on the base image, so you would have to built from the base image to get that to work, it's just easier to use docker. 

If you want to do training on something more powerful than a GH200 or multiple GPUs, you would also need to switch up the training commands and likely have to change the code. 

The training was not CPU bound in any sense so it will scale up to multiple GPUs nicely. 

```bash
# FROM YOUR LOCAL MACHINE - Upload code
scp -r <YOUR_PATH_TO_SEQUENCE> ubuntu@<SERVER_IP>:~/<Name of your directory on the server>/

# ON THE GH200 SERVER after you ssh into it
# 1. (One-time) Add user to docker group
sudo usermod -aG docker $USER
newgrp docker

# 2. Pull NGC PyTorch container
docker pull nvcr.io/nvidia/pytorch:24.12-py3

# 3. Run container with your code mounted
docker run --gpus all -it --rm \
  -v ~/<Name of your directory on the server>/Sequence:/workspace/Sequence \
  -w /workspace/Sequence \
  -p 6006:6006 \
  nvcr.io/nvidia/pytorch:24.12-py3

# INSIDE THE CONTAINER
# 4. Install deps and build C extension
pip install flask tensorboard
python setup.py build_ext --inplace

# 5. Verify installation
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()} | Device: {torch.cuda.get_device_name(0)}'); import c_sequence; print('C OK')"
```

---

## Training Commands

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

### Standard Training 
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

### Long Training 
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

## Monitoring
View cool loss curves you can post on linkedin:

1.  **In Container:**
    ```bash
    tensorboard --logdir runs --port 6006 --bind_all &
    ```
2.  **On your device (SSH Tunnel):**
    ```powershell
    ssh -N -L 6006:localhost:6006 ubuntu@<SERVER_IP>
    ```
3.  **Browser:** Go to `http://localhost:6006`

## Project Structure

*   `src/`: Core Python logic for training and MCTS.
*   `src/c_game/`: C engine for game logic.
*   `tests/`:  Game logic unit tests.
*   `web/static/`: Static web app (HTML/CSS/JS) - deployable to any static host.
    - `game_engine.js`: Client-side game logic
    - `mcts.js`: JavaScript MCTS with ONNX.js inference
    - `model.onnx`: Trained neural network (exported from PyTorch)
*   `models/`: Storage for trained neural network weights.
*   `scripts/`: Utility scripts (ONNX export, etc.)