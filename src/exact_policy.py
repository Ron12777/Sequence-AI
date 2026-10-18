"""Permutation-invariant observations and exact-card Sequence policy heads.

Actions encode the card that is spent as well as the board square. A canonical
hand ordering preserves the pretrained eleven-channel CNN interface while a
count-weighted card embedding adds a learned set representation. No hidden hand
or deck identities enter observations. This module does not prescribe which
card to spend when multiple cards can reach the same square.
"""

import copy
import math
import os
import tempfile
from pathlib import Path

import numpy as np
import torch
from torch import nn

from src.history_policy import CONTEXT_SIZE, MAX_EVENTS, HistoryPolicy, batch_inputs
from src.model import SequenceNet

ACTION_COUNT = 52 * 100
ACTION_ENCODING = "card_id * 100 + row * 10 + col"


def action_id(move):
    """One-eyed Jack removal is determined by card identity, not another bit."""
    return move.card.to_int() * 100 + move.row * 10 + move.col


def legal_actions(game):
    """Return unique exact moves and their aligned integer action IDs."""
    unique = {}
    for move in game.get_legal_moves():
        unique.setdefault(action_id(move), move)
    ids = np.array(sorted(unique), dtype=np.int64)
    return [unique[int(index)] for index in ids], ids


def canonical_board(game, player):
    hand = game.hands[player]
    if len(hand) > 7:
        raise ValueError("Exact policy supports at most seven cards per hand")
    tensor = game.get_state_tensor(player)
    ordered = sorted(range(len(hand)), key=lambda i: hand[i].to_int())
    channels = [0, 1, 2, 3] + [4 + i for i in ordered] + list(range(4 + len(hand), 11))
    return np.asarray(tensor)[channels].astype(np.uint8)


def observe_exact(game, events):
    """Canonical CNN board, public/own card counts, and complete public events.

    Return shapes are (11,10,10), (157,), and (128,4), compatible with the
    existing batch_inputs helper. Context[0:52]*2 is the own-card count vector.
    """
    player = game.current_player
    context = np.zeros(CONTEXT_SIZE, np.float32)
    for offset, cards in (
        (0, game.hands[player]),
        (52, game.discard_piles[player]),
        (104, game.discard_piles[3 - player]),
    ):
        for card in cards:
            context[offset + card.to_int()] += 0.5
    context[-1] = len(game.hands[3 - player]) / 7
    if len(events) >= MAX_EVENTS:
        raise ValueError("Public history exceeds complete-game capacity")
    history = np.zeros((MAX_EVENTS, 4), np.int64)
    history[0] = (53, 100, 0, 3)
    for i, (card, position, actor, kind) in enumerate(events, 1):
        history[i] = (card + 1, position, int(actor != player), kind)
    return canonical_board(game, player), context, history


class ExactPolicy(nn.Module):
    """Transfer a board/summary/GRU policy and learn exact-card action residuals.

    ``forward(board, context, history, actions, action_mask)`` efficiently scores
    only the supplied legal action IDs. With actions omitted it returns 5,200
    logits; pass a dense legal mask to preserve warm-start position marginals.
    Padded legal lists may use -1 (or any valid ID with mask=False).

    The new spatial/card and set-value heads start at zero. On canonical inputs,
    initial masked position marginals and values match the transferred model.
    Probability at a square is initially shared equally between its legal cards.
    """

    def __init__(
        self,
        history_policy,
        card_width=32,
        variant="cnn",
        transformer_width=64,
        transformer_layers=2,
        value_mode="bounded",
    ):
        super().__init__()
        if history_policy.architecture not in ("board", "summary", "gru"):
            raise ValueError("ExactPolicy transfers board, summary, or GRU policies")
        if not isinstance(card_width, int) or card_width < 1:
            raise ValueError("card_width must be positive")
        if variant not in ("cnn", "transformer"):
            raise ValueError("variant must be cnn or transformer")
        if value_mode not in ("bounded", "unbounded"):
            raise ValueError("value_mode must be bounded or unbounded")
        self.value_mode = value_mode
        if transformer_width % 4 or transformer_width < 4 or transformer_layers < 1:
            raise ValueError(
                "Transformer width must be divisible by four; layers positive"
            )
        self.variant = variant
        self.transformer_width = transformer_width
        self.transformer_layers = transformer_layers
        self.backbone = copy.deepcopy(history_policy)
        self.card_width = card_width
        self.history_architecture = history_policy.architecture
        filters = self.base.input_conv[0].out_channels
        self.card_embedding = nn.Embedding(52, card_width)
        self.hand_set = nn.Sequential(
            nn.Linear(card_width, card_width),
            nn.GELU(),
            nn.Linear(card_width, card_width),
        )
        self.spatial_projection = nn.Conv2d(filters, card_width, 1)
        self.card_bias = nn.Parameter(torch.zeros(52))
        self.set_value = nn.Linear(card_width, 1)
        nn.init.zeros_(self.spatial_projection.weight)
        nn.init.zeros_(self.spatial_projection.bias)
        nn.init.zeros_(self.set_value.weight)
        nn.init.zeros_(self.set_value.bias)
        if value_mode == "unbounded":
            # Added only for R-NaD after common distillation. Keeping all
            # bounded-head weights intact preserves the trained warm start.
            self.rnad_value = nn.Linear(card_width, 1)
            nn.init.zeros_(self.rnad_value.weight)
            nn.init.zeros_(self.rnad_value.bias)
        if variant == "transformer":
            width = transformer_width
            self.token_board = nn.Conv2d(filters, width, 1)
            self.token_board_position = nn.Embedding(100, width)
            self.token_card = nn.Linear(card_width, width)
            self.token_count = nn.Linear(1, width)
            self.token_context = nn.Linear(CONTEXT_SIZE, width)
            self.token_history = nn.ModuleList(
                [nn.Embedding(size, width) for size in (54, 101, 2, 4)]
            )
            self.token_history_position = nn.Embedding(MAX_EVENTS, width)
            layer = nn.TransformerEncoderLayer(
                width,
                4,
                width * 2,
                dropout=0,
                activation="gelu",
                batch_first=True,
            )
            self.transformer = nn.TransformerEncoder(
                layer, transformer_layers, enable_nested_tensor=False
            )
            self.transformer_position = nn.Linear(width, 1)
            self.transformer_spatial = nn.Linear(width, card_width)
            self.transformer_value = nn.Linear(width, 1)
            self.transformer_card = nn.Linear(width, card_width)
            for head in (
                self.transformer_position,
                self.transformer_spatial,
                self.transformer_value,
                self.transformer_card,
            ):
                nn.init.zeros_(head.weight)
                nn.init.zeros_(head.bias)

    @property
    def base(self):
        return self.backbone.base

    @classmethod
    def from_history_policy(cls, teacher, card_width=32, variant="cnn", **kwargs):
        return cls(teacher, card_width=card_width, variant=variant, **kwargs)

    def _inherited(self, board, context, history):
        trunk = self.base.input_conv(board)
        for block in self.base.res_blocks:
            trunk = block(trunk)
        position = self.base.policy_head(trunk)
        value = self.base.value_head(trunk).flatten()
        if self.history_architecture != "board":
            summary = self.backbone.summary(context)
            if self.history_architecture == "gru":
                tokens = sum(
                    embedding(history[:, :, i])
                    for i, embedding in enumerate(self.backbone.embeddings)
                )
                encoded, _ = self.backbone.encoder(tokens)
                last = (history[:, :, 0] != 0).sum(1) - 1
                encoded = encoded[
                    torch.arange(board.shape[0], device=board.device), last
                ]
                summary = torch.cat((summary, encoded), dim=1)
            correction = self.backbone.condition(summary)
            position = position + correction[:, :100]
            value = (
                torch.atanh(value.clamp(-0.99999, 0.99999)) + correction[:, 100]
            ).tanh()
        return trunk, position, value

    def _joint_tokens(self, trunk, context, history, counts):
        batch = trunk.shape[0]
        board = self.token_board(trunk).flatten(2).transpose(1, 2)
        board = board + self.token_board_position(
            torch.arange(100, device=trunk.device)
        )
        cards = self.token_card(self.card_embedding.weight)[None].expand(batch, -1, -1)
        cards = cards + self.token_count(counts[:, :, None] / 2)
        past = sum(
            embedding(history[:, :, i])
            for i, embedding in enumerate(self.token_history)
        )
        past = past + self.token_history_position(
            torch.arange(history.shape[1], device=trunk.device)
        )
        tokens = torch.cat(
            (self.token_context(context)[:, None], board, cards, past), dim=1
        )
        padding = torch.cat(
            (
                torch.zeros((batch, 101), dtype=torch.bool, device=trunk.device),
                counts <= 0,
                history[:, :, 0] == 0,
            ),
            dim=1,
        )
        return self.transformer(tokens, src_key_padding_mask=padding)

    def forward(self, board, context, history, actions=None, action_mask=None):
        trunk, position, value = self._inherited(board, context, history)
        counts = context[:, :52] * 2
        hand = counts @ self.card_embedding.weight
        hand = hand / counts.sum(1, keepdim=True).clamp_min(1)
        set_context = self.hand_set(hand)
        spatial = self.spatial_projection(trunk).flatten(2).transpose(1, 2)
        value_delta = self.set_value(set_context).flatten()
        card_context = None
        if self.variant == "transformer":
            encoded = self._joint_tokens(trunk, context, history, counts)
            board_tokens = encoded[:, 1:101]
            position = position + self.transformer_position(board_tokens).squeeze(-1)
            spatial = spatial + self.transformer_spatial(board_tokens)
            card_context = self.transformer_card(encoded[:, 101:153])
            value_delta = value_delta + self.transformer_value(encoded[:, 0]).flatten()
        value = (torch.atanh(value.clamp(-0.99999, 0.99999)) + value_delta).tanh()
        if self.value_mode == "unbounded":
            # Regularized R-NaD returns need not lie in [-1,1].
            value = value + self.rnad_value(set_context).flatten()
        if actions is None:
            actions = torch.arange(ACTION_COUNT, device=board.device).expand(
                board.shape[0], -1
            )
        if action_mask is None:
            action_mask = actions >= 0
        safe = actions.clamp_min(0)
        cards, squares = safe // 100, safe % 100
        query = self.card_embedding(cards) + set_context[:, None, :]
        if card_context is not None:
            query = query + card_context.gather(
                1, cards[:, :, None].expand(-1, -1, self.card_width)
            )
        local = spatial.gather(1, squares[:, :, None].expand(-1, -1, self.card_width))
        residual = (query * local).sum(-1) / math.sqrt(self.card_width)
        logits = position.gather(1, squares) + residual + self.card_bias[cards]
        # Lifting a position policy to exact actions must not increase a square's
        # probability just because more card identities happen to reach it.
        multiplicity = position.new_zeros((board.shape[0], 100))
        multiplicity.scatter_add_(1, squares, action_mask.to(position.dtype))
        logits = logits - multiplicity.gather(1, squares).clamp_min(1).log()
        return logits.masked_fill(~action_mask, -1e9), value


def save_exact_policy(model, path, metadata, training_state=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    model = getattr(model, "_orig_mod", model)
    payload = {
        "kind": "exact_sequence_policy",
        "version": 1,
        "state_dict": model.state_dict(),
        "architecture": model.history_architecture,
        "width": model.backbone.width,
        "card_width": model.card_width,
        "variant": model.variant,
        "value_mode": model.value_mode,
        "transformer_width": model.transformer_width,
        "transformer_layers": model.transformer_layers,
        "filters": model.base.input_conv[0].out_channels,
        "blocks": len(model.base.res_blocks),
        "action_encoding": ACTION_ENCODING,
        "observation": "ascending card-id hand channels; public history and own counts",
        "metadata": metadata,
    }
    if training_state is not None:
        payload["training_state"] = training_state
    handle, temporary = tempfile.mkstemp(
        prefix=path.name + ".", suffix=".tmp", dir=path.parent
    )
    os.close(handle)
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def load_exact_policy(path, device="cpu"):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("kind") != "exact_sequence_policy" or payload.get("version") != 1:
        raise ValueError("Unsupported exact-policy checkpoint")
    if payload.get("action_encoding") != ACTION_ENCODING:
        raise ValueError("Incompatible exact-action encoding")
    base = SequenceNet(num_filters=payload["filters"], num_res_blocks=payload["blocks"])
    history = HistoryPolicy(base, payload["architecture"], width=payload["width"])
    model = ExactPolicy(
        history,
        card_width=payload["card_width"],
        variant=payload.get("variant", "cnn"),
        transformer_width=payload.get("transformer_width", 64),
        transformer_layers=payload.get("transformer_layers", 2),
        value_mode=payload.get("value_mode", "bounded"),
    )
    model.load_state_dict(payload["state_dict"])
    return model.to(device).eval(), payload["metadata"]


__all__ = [
    "ACTION_COUNT",
    "ACTION_ENCODING",
    "ExactPolicy",
    "action_id",
    "batch_inputs",
    "canonical_board",
    "legal_actions",
    "load_exact_policy",
    "observe_exact",
    "save_exact_policy",
]
