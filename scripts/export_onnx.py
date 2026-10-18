"""Export and validate a checkpoint before replacing a browser ONNX artifact."""

import argparse
import os
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from src.model import load_model, STATE_CHANNELS


def export_to_onnx(checkpoint_path=None, output_path=None):
    import onnx

    checkpoint_path = Path(checkpoint_path or ROOT / "models/latest.pt")
    output_path = Path(output_path or ROOT / "web/static/model.onnx")
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"No checkpoint at {checkpoint_path}")
    model, _, _ = load_model(checkpoint_path, device="cpu", compile_model=False)
    model.eval()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=output_path.parent, suffix=".onnx")
    os.close(fd)
    try:
        torch.onnx.export(
            model,
            torch.zeros(1, STATE_CHANNELS, 10, 10),
            temporary,
            export_params=True,
            opset_version=12,
            do_constant_folding=True,
            input_names=["input"],
            output_names=["policy", "value"],
            dynamic_axes={
                "input": {0: "batch_size"},
                "policy": {0: "batch_size"},
                "value": {0: "batch_size"},
            },
            dynamo=False,
        )
        onnx.checker.check_model(onnx.load(temporary))
        os.replace(temporary, output_path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    print(
        f"Validated ONNX model saved to {output_path} ({output_path.stat().st_size:,} bytes)"
    )
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    export_to_onnx(args.checkpoint, args.output)
