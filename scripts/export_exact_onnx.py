"""Export one selected exact-action checkpoint and verify real-state parity.

Both CNN and transformer variants expose the same four-input browser contract.
This command never selects a model or updates a production deployment.
"""

import argparse
import hashlib
import json
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

from src.exact_policy import (
    ACTION_ENCODING,
    batch_inputs,
    legal_actions,
    load_exact_policy,
    observe_exact,
)
from src.game import SequenceGame
from src.history_policy import apply_public_move

INPUT_NAMES = ("board", "context", "history", "actions")
OUTPUT_NAMES = ("logits", "value")


class ExactExport(torch.nn.Module):
    """Stable signature, including the unused history input in the CNN graph."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, board, context, history, actions):
        logits, value = self.model(board, context, history, actions)
        # Preserve the four-input contract when the CNN has no history branch.
        # Runtime optimizers may remove this arithmetic but retain the input.
        history_zero = history.to(logits.dtype).sum(dim=(1, 2)) * 0.0
        return logits + history_zero[:, None], value


def sample_positions(count=96, seed=731709):
    """Deterministic legal full-game positions without native/GPU dependencies."""
    if count < 4:
        raise ValueError("At least four real positions are required for validation")
    saved_random = random.getstate()
    random.seed(seed)
    rows = []
    try:
        while len(rows) < count:
            game, events = SequenceGame(), []
            game.reset()
            while not game.game_over and len(rows) < count:
                moves, ids = legal_actions(game)
                if not moves:
                    break
                rows.append((observe_exact(game, events), ids.copy()))
                apply_public_move(game, events, random.choice(moves))
    finally:
        random.setstate(saved_random)
    return rows


def position_batch(rows, padding=0, reverse=False):
    board, context, history = batch_inputs([row[0] for row in rows], "cpu")
    width = max(len(row[1]) for row in rows) + padding
    actions = np.full((len(rows), width), -1, dtype=np.int64)
    for i, (_, ids) in enumerate(rows):
        actions[i, :len(ids)] = ids[::-1] if reverse else ids
    return board, context, history, torch.from_numpy(actions)


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_session(model, session, rows, batch_sizes=(1, 4, 16), atol=2e-5, rtol=2e-5):
    checks = []
    for size in batch_sizes:
        if size < 1:
            raise ValueError("Validation batch sizes must be positive")
        for offset in range(0, len(rows), size):
            selected = rows[offset:offset + size]
            # Changing padding width and ordering independently verifies both
            # dynamic action axes and card identity, not only board positions.
            padding = 7 if (offset // size) % 2 else 0
            reverse = (offset // size) % 3 == 1
            inputs = position_batch(selected, padding=padding, reverse=reverse)
            with torch.inference_mode():
                expected = tuple(value.numpy() for value in model(*inputs))
            actual = session.run(list(OUTPUT_NAMES), {
                name: value.numpy() for name, value in zip(INPUT_NAMES, inputs)
            })
            if actual[0].shape != expected[0].shape or actual[1].shape != expected[1].shape:
                raise AssertionError("ONNX outputs violated dynamic shape contract")
            for left, right in zip(actual, expected):
                if not np.isfinite(left).all():
                    raise AssertionError("Non-finite ONNX output")
                np.testing.assert_allclose(left, right, atol=atol, rtol=rtol)
            legal = inputs[-1].numpy() >= 0
            if not (actual[0][~legal] == -1e9).all():
                raise AssertionError("ONNX failed to mask padded actions")
            indices = np.arange(len(selected))
            onnx_choice = actual[0].argmax(-1)
            torch_choice = expected[0].argmax(-1)
            regret = expected[0][indices, torch_choice] - expected[0][indices, onnx_choice]
            # Different argmax indices at numerical ties are harmless; a
            # materially worse chosen logit is an export error.
            np.testing.assert_array_less(regret, atol + rtol * np.abs(expected[0][indices, torch_choice]) + 1e-12)
            checks.append({
                "batch": len(selected), "actions": actual[0].shape[1], "offset": offset,
                "extra_padding": padding, "reversed_actions": reverse,
                "logits_max_abs": float(np.max(np.abs(actual[0][legal] - expected[0][legal]))),
                "value_max_abs": float(np.max(np.abs(actual[1] - expected[1]))),
                "argmax_matches": int(np.sum(onnx_choice == torch_choice)),
                "chosen_logit_regret_max": float(regret.max()),
            })
    return checks


def export_exact(checkpoint, output, samples=96, batch_sizes=(1, 4, 16), atol=2e-5, rtol=2e-5):
    checkpoint, output = Path(checkpoint), Path(output)
    if not batch_sizes or any(not isinstance(size, int) or size < 1 for size in batch_sizes):
        raise ValueError("At least one positive integer validation batch size is required")
    if checkpoint.resolve() == output.resolve():
        raise ValueError("ONNX output must not overwrite its source checkpoint")
    source_hash = _sha256(checkpoint)
    saved_rng = torch.get_rng_state()
    saved_threads = torch.get_num_threads()
    saved_fastpath = torch.backends.mha.get_fastpath_enabled()
    temporary = None
    report_temp = None
    try:
        torch.set_num_threads(2)
        model, metadata = load_exact_policy(checkpoint, "cpu")
        model.eval()
        wrapper = ExactExport(model).eval()
        rows = sample_positions(samples)
        output.parent.mkdir(parents=True, exist_ok=True)
        handle, temporary = tempfile.mkstemp(dir=output.parent, suffix=".onnx.tmp")
        os.close(handle)
        torch.backends.mha.set_fastpath_enabled(False)
        # ONNX provides Atanh; legacy PyTorch export needs its symbolic mapping.
        torch.onnx.register_custom_op_symbolic("aten::atanh", lambda graph, value: graph.op("Atanh", value), 17)
        dynamic = {name: {0: "batch"} for name in (*INPUT_NAMES, *OUTPUT_NAMES)}
        dynamic["actions"][1] = "legal_actions"
        dynamic["logits"][1] = "legal_actions"
        torch.onnx.export(
            wrapper, position_batch(rows[:1]), temporary,
            input_names=list(INPUT_NAMES), output_names=list(OUTPUT_NAMES),
            opset_version=17, dynamo=False, dynamic_axes=dynamic,
        )
        exported = onnx.load(temporary)
        onnx.checker.check_model(exported, full_check=True)
        options = ort.SessionOptions()
        options.intra_op_num_threads = 2
        session = ort.InferenceSession(temporary, sess_options=options, providers=["CPUExecutionProvider"])
        if [item.name for item in session.get_inputs()] != list(INPUT_NAMES):
            raise AssertionError("Export did not retain the four-input contract")
        torch.backends.mha.set_fastpath_enabled(saved_fastpath)
        checks = validate_session(model, session, rows, batch_sizes, atol=atol, rtol=rtol)
        if _sha256(checkpoint) != source_hash:
            raise RuntimeError("Checkpoint changed during export; use an immutable selected checkpoint")
        report = {
            "checkpoint": str(checkpoint.resolve()), "checkpoint_sha256": source_hash,
            "onnx": str(output.resolve()), "onnx_sha256": _sha256(temporary),
            "variant": model.variant, "history_architecture": model.history_architecture,
            "kind": "exact_sequence_policy",
            "value_mode": model.value_mode, "checkpoint_metadata": metadata,
            "browserManifestFields": {
                "policyKind": "exact_sequence_policy", "playMode": "direct",
                "actionEncoding": ACTION_ENCODING, "variant": model.variant,
                "valueMode": model.value_mode,
                "historyArchitecture": model.history_architecture,
                "observationMode": "board_hand" if model.variant == "cnn" and model.history_architecture == "board" else "public_history",
            },
            "action_encoding": ACTION_ENCODING,
            "inputs": {
                "board": "float32[B,11,10,10]; own hand channels sorted by ascending card ID",
                "context": "float32[B,157]; own/own-discard/opponent-discard counts divided by2, opponent hand size divided by7",
                "history": "int64[B,128,4]; exact observe_exact encoding; CNN retains but ignores this input",
                "actions": "int64[B,A]; complete unique legal exact-action IDs; -1 pads shorter rows; at least one legal action per row",
            },
            "outputs": {"logits": "float32[B,A]; padded entries -1e9; unnormalized", "value": "float32[B]; acting-player perspective"},
            "dynamic_axes": {name: {str(axis): label for axis, label in axes.items()} for name, axes in dynamic.items()},
            "opset": 17, "checker_passed": True,
            "validation": {"provider": "CPUExecutionProvider", "states": len(rows), "atol": atol, "rtol": rtol, "checks": checks},
            "torch_version": str(torch.__version__), "onnxruntime_version": ort.__version__,
            "deployment_changed": False,
        }
        handle, report_temp = tempfile.mkstemp(dir=output.parent, suffix=".validation.json.tmp")
        with os.fdopen(handle, "w") as stream:
            json.dump(report, stream, indent=2)
            stream.write("\n")
        os.replace(temporary, output)
        temporary = None
        os.replace(report_temp, output.with_suffix(".validation.json"))
        report_temp = None
        return report
    finally:
        torch.onnx.unregister_custom_op_symbolic("aten::atanh", 17)
        torch.backends.mha.set_fastpath_enabled(saved_fastpath)
        torch.set_num_threads(saved_threads)
        torch.set_rng_state(saved_rng)
        for path in (temporary, report_temp):
            if path is not None and os.path.exists(path):
                os.unlink(path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--samples", type=int, default=96)
    args = parser.parse_args()
    result = export_exact(args.checkpoint, args.output, samples=args.samples)
    print(json.dumps({"output": str(args.output), "sha256": result["onnx_sha256"], "validated": True}))
