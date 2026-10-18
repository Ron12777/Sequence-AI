"""Export a direct CNN or GRU policy and verify real-state ONNX parity."""

import argparse
import hashlib
import os
import random
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import onnx
import onnxruntime as ort
import torch

from scripts.overnight import atomic_json
from src.fast_game import FastSequenceGame
from src.history_policy import apply_public_move, batch_inputs, load_policy, observe


def export_history(checkpoint, output):
    checkpoint, output = Path(checkpoint), Path(output)
    torch.set_num_threads(2)
    model, metadata = load_policy(checkpoint, "cpu")
    if model.architecture not in {"board", "gru"}:
        raise ValueError("This exporter validates direct CNN and GRU policies only")
    model.eval()
    board_only = model.architecture == "board"
    input_names = ["input"] if board_only else ["board", "context", "history"]
    random.seed(731701)
    observations = []
    while len(observations) < 128:
        game, events = FastSequenceGame(), []
        while not game.game_over and len(observations) < 128:
            moves = game.get_legal_moves()
            if not moves:
                break
            observations.append(observe(game, events))
            apply_public_move(game, events, random.choice(moves))
    output.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=output.parent, suffix=".onnx")
    os.close(fd)
    # ONNX has Atanh; the legacy PyTorch exporter lacks its symbolic mapping.
    torch.onnx.register_custom_op_symbolic(
        "aten::atanh", lambda graph, value: graph.op("Atanh", value), 17
    )
    try:
        torch.onnx.export(
            model.base if board_only else model,
            batch_inputs(observations[:1], "cpu")[:1]
            if board_only
            else batch_inputs(observations[:1], "cpu"),
            temporary,
            input_names=input_names,
            output_names=["policy", "value"],
            opset_version=17,
            dynamo=False,
            dynamic_axes={
                name: {0: "batch"} for name in [*input_names, "policy", "value"]
            },
        )
        onnx.checker.check_model(onnx.load(temporary))
        options = ort.SessionOptions()
        options.intra_op_num_threads = 2
        session = ort.InferenceSession(
            temporary, sess_options=options, providers=["CPUExecutionProvider"]
        )
        errors = []
        for size in [1, 4, 32]:
            for offset in [0, 48, 96]:
                inputs = batch_inputs(observations[offset : offset + size], "cpu")
                with torch.inference_mode():
                    expected = [x.numpy() for x in model(*inputs)]
                actual = session.run(
                    None,
                    {name: x.numpy() for name, x in zip(input_names, inputs)},
                )
                actual[1] = actual[1].reshape(-1)
                for x, y in zip(actual, expected):
                    np.testing.assert_allclose(x, y, rtol=1e-5, atol=1e-5)
                errors.append(
                    {
                        "batch": size,
                        "offset": offset,
                        "policy_max_abs": float(
                            np.max(np.abs(actual[0] - expected[0]))
                        ),
                        "value_max_abs": float(np.max(np.abs(actual[1] - expected[1]))),
                    }
                )
        report = {
            "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            "onnx_sha256": hashlib.sha256(Path(temporary).read_bytes()).hexdigest(),
            "architecture": model.architecture,
            "metadata": metadata,
            "inputs": {"input": "float32[B,11,10,10]"}
            if board_only
            else {
                "board": "float32[B,11,10,10]",
                "context": "float32[B,157]",
                "history": "int64[B,128,4]",
            },
            "opset": 17,
            "checker_passed": True,
            "real_state_checks": errors,
            "deployment_changed": False,
        }
        os.replace(temporary, output)
        atomic_json(output.with_suffix(".validation.json"), report)
        return report
    finally:
        torch.onnx.unregister_custom_op_symbolic("aten::atanh", 17)
        if os.path.exists(temporary):
            os.unlink(temporary)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    report = export_history(args.checkpoint, args.output)
    print("Validated direct-policy ONNX:", args.output, report["onnx_sha256"])
