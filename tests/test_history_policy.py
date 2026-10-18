import random
import numpy as np
import pytest
import torch
from src.fast_game import FastSequenceGame
from src.model import SequenceNet
from src.history_policy import (
    HistoryPolicy,
    observe,
    apply_public_move,
    batch_inputs,
    save_policy,
    load_policy,
)


def test_observation_cannot_see_hidden_partition():
    random.seed(24)
    game = FastSequenceGame()
    events = []
    for _ in range(14):
        apply_public_move(game, events, game.get_legal_moves()[0])
        if game.game_over:
            break
    expected = observe(game, events)
    opponent = 3 - game.current_player
    # Change the actual hidden partition, preserving only public hand size.
    hidden = game.hands[opponent] + game.deck.cards
    random.shuffle(hidden)
    game.hands[opponent] = hidden[: len(game.hands[opponent])]
    game.deck = None
    actual = observe(game, events)
    for a, b in zip(expected, actual):
        np.testing.assert_array_equal(a, b)
    assert len(events) == sum(map(len, game.discard_piles.values()))


@pytest.mark.parametrize(
    "architecture", ["board", "summary", "gru", "attention", "joint_attention"]
)
def test_initial_policy_transfer_roundtrip_and_history_gradients(
    architecture, tmp_path
):
    torch.set_num_threads(1)
    base = SequenceNet(num_filters=8, num_res_blocks=1).eval()
    model = HistoryPolicy(base, architecture, width=16).eval()
    inputs = batch_inputs([observe(FastSequenceGame(), [])], "cpu")
    with torch.no_grad():
        expected = base(inputs[0])
        actual = model(*inputs)
    torch.testing.assert_close(expected[0], actual[0])
    torch.testing.assert_close(expected[1].flatten(), actual[1], atol=1e-5, rtol=1e-5)
    if architecture != "board":
        with torch.no_grad():
            if architecture == "joint_attention":
                model.policy_correction.weight.fill_(0.01)
                model.value_correction.weight.fill_(0.01)
            else:
                model.condition[-1].weight.fill_(0.01)
    logpi, value = model(*inputs)
    (-logpi[:, 0].mean() + value.square().mean()).backward()
    assert all(
        p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()
    )
    if architecture in ("gru", "attention", "joint_attention"):
        assert (
            sum(
                p.grad.abs().sum()
                for p in model.encoder.parameters()
                if p.grad is not None
            )
            > 0
        )
    path = tmp_path / "policy.pt"
    save_policy(model, path, {"seed": 12})
    restored, metadata = load_policy(path)
    assert metadata["seed"] == 12
    for x, y in zip(model(*inputs), restored(*inputs)):
        torch.testing.assert_close(x, y)


def test_distillation_can_retain_initial_weights_and_freeze_backbone(tmp_path):
    from scripts.architecture_experiments import fit

    base = SequenceNet(num_filters=8, num_res_blocks=1).eval()
    observations = [observe(FastSequenceGame(), []) for _ in range(8)]
    board, context, history = batch_inputs(observations, "cpu")
    with torch.no_grad():
        pi, value = base(board)
    data = {
        "board": board,
        "context": context,
        "history": history,
        "policy": pi.exp(),
        "value": value.flatten(),
        "game": torch.arange(8),
    }
    output = tmp_path / "control.pt"
    fit(
        base,
        "gru",
        data,
        torch.device("cpu"),
        1,
        17,
        output,
        learning_rate=0,
        freeze_backbone=True,
    )
    loaded, metadata = load_policy(output)
    assert metadata["epoch"] == 0
    for name, tensor in base.state_dict().items():
        torch.testing.assert_close(
            tensor, loaded.base.state_dict()[name], rtol=0, atol=0
        )
