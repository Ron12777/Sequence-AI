"""Experimental information-respecting direct policies, separate from production search."""

import copy

import numpy as np
import torch
from torch import nn

from src.model import SequenceNet

MAX_EVENTS = 128  # At most 104 cards can be publicly discarded in this ruleset.
CONTEXT_SIZE = 157  # Own hand, own/opponent public discards, opponent hand size.


def observe(game, events):
    """Read only this player's hand and public information; never hidden identities."""
    player = game.current_player
    counts = np.zeros(CONTEXT_SIZE, np.float32)
    for offset, cards in (
        (0, game.hands[player]),
        (52, game.discard_piles[player]),
        (104, game.discard_piles[3 - player]),
    ):
        for card in cards:
            counts[offset + card.to_int()] += 0.5
    counts[-1] = len(game.hands[3 - player]) / 7
    if len(events) >= MAX_EVENTS:
        raise ValueError("Public history exceeds supported complete-game length")
    history = np.zeros((MAX_EVENTS, 4), np.int64)
    history[0] = (53, 100, 0, 3)  # Start token: avoids an all-padding attention input.
    for i, (card, position, actor, kind) in enumerate(events, 1):
        history[i] = (card + 1, position, int(actor != player), kind)
    return game.get_state_numpy(player).astype(np.uint8), counts, history


def apply_public_move(game, events, move):
    player = game.current_player
    before = len(game.discard_piles[player])
    if not game.make_move(move):
        raise RuntimeError("Illegal experimental policy move")
    events.append(
        (move.card.to_int(), move.row * 10 + move.col, player, int(move.is_removal))
    )
    for card in game.discard_piles[player][before + 1 :]:
        events.append((card.to_int(), 100, player, 2))


class HistoryPolicy(nn.Module):
    """Common transferred CNN with either summary, GRU, or attention conditioning.

    The zero-initialized context projection preserves the pretrained policy at
    initialization. All variants then train end-to-end on identical examples.
    """

    def __init__(self, base, architecture="gru", width=64):
        super().__init__()
        if architecture not in (
            "board",
            "summary",
            "gru",
            "attention",
            "joint_attention",
        ):
            raise ValueError("Unknown history architecture")
        self.architecture, self.width = architecture, width
        self.base = copy.deepcopy(base)
        if architecture == "board":
            return
        self.summary = nn.Sequential(nn.Linear(CONTEXT_SIZE, width), nn.GELU())
        if architecture in ("gru", "attention", "joint_attention"):
            self.embeddings = nn.ModuleList(
                [nn.Embedding(n, width) for n in (54, 101, 2, 4)]
            )
            if architecture == "gru":
                self.encoder = nn.GRU(width, width, batch_first=True)
            else:
                self.position = nn.Embedding(MAX_EVENTS, width)
                layer = nn.TransformerEncoderLayer(
                    width, 4, width * 2, dropout=0, batch_first=True
                )
                self.encoder = nn.TransformerEncoder(
                    layer, 2, enable_nested_tensor=False
                )
        if architecture == "joint_attention":
            self.board_tokens = nn.Conv2d(11, width, 1)
            self.board_positions = nn.Embedding(100, width)
            self.policy_correction = nn.Linear(width, 1)
            self.value_correction = nn.Linear(width, 1)
            for projection in (self.policy_correction, self.value_correction):
                nn.init.zeros_(projection.weight)
                nn.init.zeros_(projection.bias)
            return
        size = width if architecture == "summary" else 2 * width
        self.condition = nn.Sequential(
            nn.Linear(size, width), nn.GELU(), nn.Linear(width, 101)
        )
        nn.init.zeros_(self.condition[-1].weight)
        nn.init.zeros_(self.condition[-1].bias)

    def forward(self, board, context, history):
        logpi, value = self.base(board)
        if self.architecture == "board":
            return logpi, value.flatten()
        summary = self.summary(context)
        if self.architecture != "summary":
            mask = history[:, :, 0] == 0
            tokens = sum(e(history[:, :, i]) for i, e in enumerate(self.embeddings))
            if self.architecture == "gru":
                encoded, _ = self.encoder(tokens)
                last = (~mask).sum(1) - 1
                encoded = encoded[
                    torch.arange(board.shape[0], device=board.device), last
                ]
            else:
                tokens = tokens + self.position(
                    torch.arange(history.shape[1], device=board.device)
                )
                if self.architecture == "joint_attention":
                    board_tokens = self.board_tokens(board).flatten(2).transpose(1, 2)
                    board_tokens = board_tokens + self.board_positions(
                        torch.arange(100, device=board.device)
                    )
                    tokens = torch.cat((summary[:, None], board_tokens, tokens), dim=1)
                    mask = torch.cat(
                        (
                            torch.zeros(
                                (len(board), 101), dtype=torch.bool, device=board.device
                            ),
                            mask,
                        ),
                        dim=1,
                    )
                    encoded = self.encoder(tokens, src_key_padding_mask=mask)
                    logits = logpi + self.policy_correction(encoded[:, 1:101]).squeeze(
                        -1
                    )
                    value = (
                        torch.atanh(value.flatten().clamp(-0.99999, 0.99999))
                        + self.value_correction(encoded[:, 0]).flatten()
                    ).tanh()
                    return logits.log_softmax(1), value
                encoded = self.encoder(tokens, src_key_padding_mask=mask)
                # All supplied tokens are in the past; bidirectional attention is valid here.
                valid = (~mask).unsqueeze(-1)
                encoded = (encoded * valid).sum(1) / valid.sum(1)
            summary = torch.cat((summary, encoded), dim=1)
        correction = self.condition(summary)
        logits = logpi + correction[:, :100]
        value = (
            torch.atanh(value.flatten().clamp(-0.99999, 0.99999)) + correction[:, 100]
        ).tanh()
        return logits.log_softmax(1), value


def batch_inputs(observations, device):
    return (
        torch.from_numpy(np.stack([x[0] for x in observations])).to(device).float(),
        torch.from_numpy(np.stack([x[1] for x in observations])).to(device),
        torch.from_numpy(np.stack([x[2] for x in observations])).to(device).long(),
    )


def save_policy(model, path, metadata, training_state=None):
    from pathlib import Path

    path = Path(path)
    payload = {
        "state_dict": model.state_dict(),
        "architecture": model.architecture,
        "width": model.width,
        "filters": model.base.input_conv[0].out_channels,
        "blocks": len(model.base.res_blocks),
        "metadata": metadata,
    }
    if training_state is not None:
        payload["training_state"] = training_state
    temporary = path.with_suffix(".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def load_policy(path, device="cpu"):
    data = torch.load(path, map_location="cpu", weights_only=True)
    base = SequenceNet(num_filters=data["filters"], num_res_blocks=data["blocks"])
    model = HistoryPolicy(base, data["architecture"], data["width"])
    model.load_state_dict(data["state_dict"])
    return model.to(device).eval(), data["metadata"]
