"""
Neural network model for Sequence game AI.
Uses convolutional residual blocks with policy and value heads.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


STATE_CHANNELS = 11


class ResidualBlock(nn.Module):
    """Residual block with two convolutional layers."""

    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        out = F.relu(out)
        return out


class SequenceNet(nn.Module):
    """
    Neural network for Sequence game evaluation.

    Input: (batch, 11, 10, 10) tensor representing game state
    Output:
        - policy: (batch, 100) log probabilities for each board position
        - value: (batch, 1) expected game outcome [-1, 1]
    """

    def __init__(
        self,
        input_channels: int = STATE_CHANNELS,
        num_filters: int = 128,
        num_res_blocks: int = 6,
        board_size: int = 10,
    ):
        super().__init__()

        self.board_size = board_size
        self.num_positions = board_size * board_size

        # Input convolution
        self.input_conv = nn.Sequential(
            nn.Conv2d(
                input_channels, num_filters, kernel_size=3, padding=1, bias=False
            ),
            nn.BatchNorm2d(num_filters),
            nn.ReLU(),
        )

        # Residual tower
        self.res_blocks = nn.ModuleList(
            [ResidualBlock(num_filters) for _ in range(num_res_blocks)]
        )

        # Policy head
        self.policy_head = nn.Sequential(
            nn.Conv2d(num_filters, 32, kernel_size=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * board_size * board_size, self.num_positions),
        )

        # Value head
        self.value_head = nn.Sequential(
            nn.Conv2d(num_filters, 8, kernel_size=1, bias=False),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(8 * board_size * board_size, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch, 11, 10, 10)

        Returns:
            policy: Log probabilities for each position (batch, 100)
            value: Expected outcome (batch, 1)
        """
        # Shared body
        out = self.input_conv(x)
        for block in self.res_blocks:
            out = block(out)

        # Policy and value predictions
        policy_logits = self.policy_head(out)
        policy = F.log_softmax(policy_logits, dim=1)
        value = self.value_head(out)

        return policy, value

    def predict(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Inference mode prediction."""
        self.eval()
        with torch.no_grad():
            policy, value = self(x)
        return policy.exp(), value  # Return probabilities, not log-probs


def create_model(
    device: str = "auto",
    compile_model: bool = True,
    num_filters: int = 128,
    num_res_blocks: int = 6,
) -> Tuple[SequenceNet, torch.device]:
    """
    Create model and move to appropriate device.

    Args:
        device: 'auto', 'cuda', 'cpu'
        compile_model: If True, use torch.compile() for speedup (PyTorch 2.0+)
        num_filters: Residual tower width
        num_res_blocks: Residual tower depth

    Returns:
        model, device
    """
    if device == "auto":
        device = (
            "cuda"
            if torch.cuda.is_available() and torch.cuda.device_count() > 0
            else "cpu"
        )
        if device == "cuda":
            print(f"Initializing model on CUDA (GPU: {torch.cuda.get_device_name(0)})")

    device = torch.device(device)
    model = SequenceNet(num_filters=num_filters, num_res_blocks=num_res_blocks)
    model = model.to(device)

    # JIT compile for speedup (PyTorch 2.0+)
    if compile_model and hasattr(torch, "compile"):
        # Check if we are on Windows and Triton is missing
        import sys

        is_windows = sys.platform == "win32"

        try:
            # On Windows, Triton is often missing, so we might need a different backend
            # or just skip compilation if it fails.
            if is_windows:
                # 'cudagraphs' is sometimes a viable alternative on Windows,
                # but 'eager' is safest if Triton is missing.
                print(
                    "Windows detected: attempting torch.compile() with default backend..."
                )

            model = torch.compile(model)
            print("Model compiled with torch.compile()")
        except Exception as e:
            print(f"torch.compile() failed or not supported, using eager mode: {e}")

    return model, device


def _adapt_state_dict_for_model(state_dict: dict, model: SequenceNet) -> dict:
    """Preserve old predictions when extending the hand channels from 8 to 11."""
    result = dict(state_dict)
    key = "input_conv.0.weight"
    source, target = result[key], model.state_dict()[key]
    if source.shape != target.shape:
        if (
            source.shape[0] != target.shape[0]
            or source.shape[2:] != target.shape[2:]
            or source.shape[1] > target.shape[1]
        ):
            raise ValueError("Checkpoint architecture is incompatible with the model")
        weight = torch.zeros_like(target)
        weight[:, : source.shape[1]] = source
        result[key] = weight
    return result


def save_model(model, path, optimizer=None, epoch=0, scheduler=None, model_config=None):
    """Atomically replace a portable checkpoint, keeping the last good file on failure."""
    from pathlib import Path
    import os
    import tempfile

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    model = getattr(model, "_orig_mod", model)
    config = {
        "num_filters": model.input_conv[0].out_channels,
        "num_res_blocks": len(model.res_blocks),
    }
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "epoch": epoch,
        "model_config": config,
    }
    if optimizer is not None:
        checkpoint["optimizer_state_dict"] = optimizer.state_dict()
    if scheduler is not None:
        checkpoint["scheduler_state_dict"] = scheduler.state_dict()
    fd, temporary = tempfile.mkstemp(dir=target.parent, suffix=".tmp")
    os.close(fd)
    try:
        torch.save(checkpoint, temporary)
        os.replace(temporary, target)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def load_model(path, device="auto", compile_model=True):
    """Load compiled or eager checkpoints on any device, including CPU-only hosts."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    state = {
        key.removeprefix("_orig_mod."): value
        for key, value in checkpoint["model_state_dict"].items()
    }
    config = checkpoint.get("model_config") or {
        "num_filters": state["input_conv.0.weight"].shape[0],
        "num_res_blocks": len(
            {key.split(".")[1] for key in state if key.startswith("res_blocks.")}
        ),
    }
    model, device = create_model(device, compile_model=compile_model, **config)
    target = getattr(model, "_orig_mod", model)
    target.load_state_dict(_adapt_state_dict_for_model(state, target))
    checkpoint["model_config"] = config
    return model, device, checkpoint
