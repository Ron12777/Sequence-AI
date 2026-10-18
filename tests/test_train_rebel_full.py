import copy
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from scripts import train_rebel_full as training
from src.rebel_full_value import (
    FEATURES,
    BeliefResidual,
    FullBeliefValue,
    load_value,
    save_value,
)


class FakeGame:
    def __init__(self):
        self.current_player = 1
        self.game_over = False
        self.winner = None

    def get_legal_moves(self):
        return [] if self.game_over else ["move"]

    def get_result(self, player):
        return 0 if self.winner is None else (1 if self.winner == player else -1)


class FakeAgent:
    def __init__(self, *args, **kwargs):
        self.model = SimpleNamespace(
            residual=torch.nn.Linear(1, 1),
            examples=lambda *args: (np.arange(4, dtype=np.float32)[:, None], np.zeros(4, np.float32)),
            query_example=lambda adapter, action, belief, hand: (np.array([hand[0]], np.float32), np.float32(0)),
        )

    def reset(self, **kwargs):
        pass

    def act(self, game, player, events):
        self.last_search = {
            "adapter": SimpleNamespace(root="root"),
            "root_belief": SimpleNamespace(compatible_mass=(np.array([1, 0]), np.array([1, 0]))),
            "mean_root_conditional_values": (np.array([0.25, 999]), np.array([-0.5, 999])),
            "last_query": ("action", "belief", [player]),
            "metrics": {"public_legality_fallbacks": 0, "unsupported_action_fallbacks": 0, "wall_seconds": 0.01},
        }
        return "move"


@pytest.fixture
def fake_training(monkeypatch, tmp_path):
    def apply(game, events, move):
        assert move == "move"
        events.append((0, 0, game.current_player, 0))
        if game.current_player == 2:
            game.game_over = True
            game.winner = 1
        game.current_player = 3 - game.current_player

    monkeypatch.setattr(training, "_worker_agent", None)
    monkeypatch.setattr(training, "_worker_reference", None)
    monkeypatch.setattr(training, "ReBeLAgent", FakeAgent)
    monkeypatch.setattr(training, "FastSequenceGame", FakeGame)
    monkeypatch.setattr(training, "apply_public_move", apply)
    monkeypatch.setattr(training, "load_policy", lambda *args: ("reference", {}))
    weights = tmp_path / "weights.pt"
    torch.save({"residual": torch.nn.Linear(1, 1).state_dict(), "metadata": {"games": 17}}, weights)
    return {"weights": str(weights), "reference": "reference.pt", "seed": 71,
            "particles": 2, "iterations": 2, "depth": 1}


def test_complete_selfplay_mc_targets_use_each_decision_players_perspective(fake_training):
    roots, mc, metrics = training.generate_game(fake_training, 0)
    assert metrics["winner"] == 1
    assert metrics["moves"] == metrics["decisions"] == 2
    assert metrics["reference_seat"] == 0
    assert metrics["version"] == 17
    assert [row[2] for row in mc] == [1, -1]
    assert [row[0][0] for row in mc] == [1, 2]
    assert [row[2] for row in roots] == [0.25, -0.5, 0.25, -0.5]
    assert all(row[2] != 999 for row in roots)


def test_frozen_opponent_uses_correct_direct_moves_api_and_balanced_learner_seat(fake_training, monkeypatch):
    calls = []

    def direct(model, device, games, events, ids):
        calls.append((model, device, ids, games[0].current_player))
        return ["move"]

    monkeypatch.setattr(training, "direct_moves", direct)
    _, mc, metrics = training.generate_game(fake_training, 2)
    assert calls == [("reference", "cpu", [0], 1)]
    assert metrics["reference_seat"] == 1
    assert metrics["decisions"] == 1
    assert [row[2] for row in mc] == [-1]
    calls.clear()
    _, mc, metrics = training.generate_game(fake_training, 3)
    assert calls == [("reference", "cpu", [0], 2)]
    assert metrics["reference_seat"] == 2
    assert [row[2] for row in mc] == [1]


def test_training_preserves_optimizer_rng_replay_and_frozen_teacher_on_resume(tmp_path, monkeypatch):
    torch.set_num_threads(1)
    torch.manual_seed(34)
    teacher = torch.nn.Linear(3, 1)
    value = FullBeliefValue(teacher, BeliefResidual(width=8))
    teacher_before = copy.deepcopy(teacher.state_dict())
    optimizer = torch.optim.Adam(value.residual.parameters(), lr=1e-4)
    rng = np.random.default_rng(56)
    roots = [(rng.normal(size=FEATURES).astype(np.float32), np.float32(0.1), np.float32(0.25)) for _ in range(9)]
    mc = [(rng.normal(size=FEATURES).astype(np.float32), np.float32(-0.1), np.float32((-1) ** i)) for i in range(7)]
    before = copy.deepcopy(value.residual.state_dict())
    first = training.update(value, optimizer, roots, mc, rng, steps=2, batch_size=4)
    assert all(np.isfinite(v) for v in first.values())
    assert any(not torch.equal(v, before[k]) for k, v in value.residual.state_dict().items())
    assert not value.residual.training
    for k, v in teacher.state_dict().items():
        torch.testing.assert_close(v, teacher_before[k], rtol=0, atol=0)
    path = tmp_path / "latest.pt"
    state = {"optimizer": optimizer.state_dict(), "rng": copy.deepcopy(rng.bit_generator.state),
             "torch_rng": torch.get_rng_state(), "root_replay": roots, "mc_replay": mc}
    save_value(value, path, "teacher.pt", training_state=state)
    expected = training.update(value, optimizer, roots, mc, rng, steps=2, batch_size=4)
    monkeypatch.setattr("src.rebel_full_value.load_policy", lambda *args: (SimpleNamespace(base=copy.deepcopy(teacher)), {}))
    restored = load_value(path)
    saved = torch.load(path, weights_only=False)["training_state"]
    restored_optimizer = torch.optim.Adam(restored.residual.parameters(), lr=1e-4)
    restored_optimizer.load_state_dict(saved["optimizer"])
    restored_rng = np.random.default_rng()
    restored_rng.bit_generator.state = saved["rng"]
    torch.set_rng_state(saved["torch_rng"])
    actual = training.update(restored, restored_optimizer, saved["root_replay"], saved["mc_replay"],
                             restored_rng, steps=2, batch_size=4)
    assert actual == expected
    for k, v in value.residual.state_dict().items():
        torch.testing.assert_close(v, restored.residual.state_dict()[k], rtol=0, atol=0)


def test_nonfinite_update_is_rejected_before_optimizer_step():
    value = FullBeliefValue(torch.nn.Identity(), BeliefResidual(width=8))
    optimizer = torch.optim.Adam(value.residual.parameters(), lr=1e-4)
    row = (np.zeros(FEATURES, np.float32), np.float32(0), np.float32(np.nan))
    before = copy.deepcopy(value.residual.state_dict())
    with pytest.raises(RuntimeError, match="Nonfinite"):
        training.update(value, optimizer, [row], [row], np.random.default_rng(0), steps=1, batch_size=1)
    for k, v in value.residual.state_dict().items():
        torch.testing.assert_close(v, before[k], rtol=0, atol=0)
