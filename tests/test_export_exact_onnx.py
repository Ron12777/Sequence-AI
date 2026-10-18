"""Real-state export parity, dynamic axes, identity and failure isolation."""

import hashlib
import json
import random

import numpy as np
import pytest
import torch

import onnxruntime as ort

from scripts.export_exact_onnx import export_exact, position_batch, sample_positions
from src.exact_policy import ExactPolicy, save_exact_policy
from src.history_policy import HistoryPolicy
from src.model import SequenceNet


def checkpoint(path, variant="cnn", value_mode="bounded"):
    torch.manual_seed(117)
    model = ExactPolicy(
        HistoryPolicy(SequenceNet(num_filters=8, num_res_blocks=1), "board", width=8),
        card_width=8, variant=variant, transformer_width=16, transformer_layers=1,
        value_mode=value_mode,
    ).eval()
    # Exercise learned branches. Zero residual heads would hide an incorrect
    # transformer export by making its contribution numerically irrelevant.
    with torch.no_grad():
        model.spatial_projection.weight.normal_(std=0.05)
        model.spatial_projection.bias.normal_(std=0.05)
        model.set_value.weight.normal_(std=0.05)
        if variant == "transformer":
            for name in ("transformer_position", "transformer_spatial", "transformer_card", "transformer_value"):
                getattr(model, name).weight.normal_(std=0.05)
                getattr(model, name).bias.normal_(std=0.05)
        if value_mode == "unbounded":
            model.rnad_value.bias.fill_(3)
    save_exact_policy(model, path, {"test_checkpoint": True})
    return model


@pytest.mark.parametrize("variant,value_mode", [
    ("cnn", "bounded"), ("cnn", "unbounded"),
    ("transformer", "bounded"), ("transformer", "unbounded"),
])
def test_exact_export_real_states_dynamic_batch_actions_and_value_modes(tmp_path, variant, value_mode):
    source, output = tmp_path / "selected.pt", tmp_path / "export.onnx"
    model = checkpoint(source, variant, value_mode)
    before = source.read_bytes()
    random_state, torch_state = random.getstate(), torch.get_rng_state().clone()
    fastpath = torch.backends.mha.get_fastpath_enabled()
    report = export_exact(source, output, samples=8, batch_sizes=(1, 3, 8))
    assert source.read_bytes() == before
    assert random.getstate() == random_state
    assert torch.equal(torch.get_rng_state(), torch_state)
    assert torch.backends.mha.get_fastpath_enabled() == fastpath
    assert report["variant"] == variant and report["value_mode"] == value_mode
    assert report["deployment_changed"] is False
    assert report["onnx_sha256"] == hashlib.sha256(output.read_bytes()).hexdigest()
    assert json.loads(output.with_suffix(".validation.json").read_text()) == report
    assert {check["batch"] for check in report["validation"]["checks"]} >= {1, 3, 8}
    assert len({check["actions"] for check in report["validation"]["checks"]}) > 1
    assert report["browserManifestFields"]["policyKind"] == "exact_sequence_policy"
    assert report["browserManifestFields"]["historyArchitecture"] == "board"
    assert report["browserManifestFields"]["observationMode"] == ("board_hand" if variant == "cnn" else "public_history")
    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    session = ort.InferenceSession(str(output), sess_options=options, providers=["CPUExecutionProvider"])
    rows = sample_positions(4, seed=9921)
    inputs = position_batch(rows, padding=13, reverse=True)
    output_values = session.run(None, dict(zip(("board", "context", "history", "actions"), [x.numpy() for x in inputs])))
    with torch.inference_mode():
        expected = model(*inputs)
    for actual, wanted in zip(output_values, expected):
        np.testing.assert_allclose(actual, wanted.numpy(), atol=2e-5, rtol=2e-5)
    if variant == "cnn":
        # Edited-position mode supplies real board/own counts and zero buffers
        # for inputs the explicit board-only export does not consume.
        edited_inputs = [value.clone() for value in inputs]
        edited_inputs[1][:, 52:] = 0
        edited_inputs[2].zero_()
        edited_outputs = session.run(None, dict(zip(("board", "context", "history", "actions"), [x.numpy() for x in edited_inputs])))
        for edited, original in zip(edited_outputs, output_values):
            np.testing.assert_array_equal(edited, original)
    if value_mode == "unbounded":
        assert output_values[1].min() > 1


def test_export_failure_does_not_replace_existing_output_or_leave_temporary_files(tmp_path, monkeypatch):
    source, output = tmp_path / "selected.pt", tmp_path / "prior.onnx"
    checkpoint(source)
    output.write_bytes(b"previous validated artifact")
    output.with_suffix(".validation.json").write_text('{"previous":true}')
    import scripts.export_exact_onnx as exporter

    def fail(*args, **kwargs):
        raise AssertionError("deliberate numerical validation failure")

    monkeypatch.setattr(exporter, "validate_session", fail)
    fastpath = torch.backends.mha.get_fastpath_enabled()
    with pytest.raises(AssertionError, match="deliberate numerical"):
        export_exact(source, output, samples=4)
    assert output.read_bytes() == b"previous validated artifact"
    assert output.with_suffix(".validation.json").read_text() == '{"previous":true}'
    assert not list(tmp_path.glob("*.tmp"))
    assert torch.backends.mha.get_fastpath_enabled() == fastpath


def test_export_refuses_to_overwrite_source_checkpoint(tmp_path):
    source = tmp_path / "selected.pt"
    checkpoint(source)
    original = source.read_bytes()
    with pytest.raises(ValueError, match="source checkpoint"):
        export_exact(source, source)
    assert source.read_bytes() == original
