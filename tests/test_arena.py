import numpy as np
import pytest
import torch
from scripts.arena import arena, summarize
from src.model import SequenceNet


def test_identical_deterministic_agents_split_paired_deals():
    torch.manual_seed(17)
    model = SequenceNet(num_filters=8, num_res_blocks=1).eval()
    agent = (model, torch.device("cpu"))
    result = arena(agent, agent, games_count=4, simulations=0, seed=123)
    assert result["score"] == 0.5
    assert result["wins"] == result["losses"]
    assert np.array(result["outcomes"]).reshape(-1, 2).sum(axis=1).tolist() == [1, 1]


def test_paired_score_interval_and_invalid_counts():
    result = summarize([1.0, 0.0, 0.5, 0.5])
    assert result["paired_score_95ci"] == [0.5, 0.5]
    with pytest.raises(ValueError):
        summarize([1.0])
