"""Scalable neural processing unit for Step 1.

The model intentionally consumes the benchmark's compact 32x16 feature sequence
and produces structured logits. It is not a language model and has no vocabulary
embedding or text decoder.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import torch
from torch import nn

from .schema import FEATURE_WIDTH, SEQUENCE_LENGTH

NON_UNCERTAIN_LABELS: tuple[str, ...] = (
    "SIGNAL",
    "NO_SIGNAL",
    "CHANGE",
    "NO_CHANGE",
    "CONFLICT",
    "COMPATIBLE",
    "A_BEFORE_B",
    "B_BEFORE_A",
    "SAME_TIME",
    "RELEVANT",
    "NOT_RELEVANT",
)
LABEL_TO_INDEX = {label: index for index, label in enumerate(NON_UNCERTAIN_LABELS)}


@dataclass(frozen=True, slots=True)
class UnitConfig:
    """Structural configuration for one Step 1 unit."""

    d_model: int
    block_count: int
    attention_heads: int
    feed_forward_width: int
    dropout: float = 0.1
    sequence_length: int = SEQUENCE_LENGTH
    feature_width: int = FEATURE_WIDTH

    def validate(self) -> None:
        if self.d_model <= 0:
            raise ValueError("d_model must be positive")
        if self.block_count <= 0:
            raise ValueError("block_count must be positive")
        if self.attention_heads <= 0:
            raise ValueError("attention_heads must be positive")
        if self.feed_forward_width <= 0:
            raise ValueError("feed_forward_width must be positive")
        if self.d_model % self.attention_heads != 0:
            raise ValueError("d_model must be divisible by attention_heads")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if self.sequence_length != SEQUENCE_LENGTH:
            raise ValueError(
                f"Step 1 sequence length must remain {SEQUENCE_LENGTH} for fair comparison"
            )
        if self.feature_width != FEATURE_WIDTH:
            raise ValueError(
                f"Step 1 feature width must remain {FEATURE_WIDTH} for fair comparison"
            )


# Actual trainable parameter count with the current implementation: 10,148,108.
# This is the comfortably capable reference configuration, not a claim that 10M
# is an optimal unit size.
REFERENCE_10M_CONFIG = UnitConfig(
    d_model=256,
    block_count=11,
    attention_heads=8,
    feed_forward_width=1280,
    dropout=0.1,
)


class Step01Output(NamedTuple):
    label_logits: torch.Tensor
    uncertainty_logits: torch.Tensor


class Step01Unit(nn.Module):
    """Tiny-architecture-family baseline used for the minimum-unit size sweep."""

    def __init__(self, config: UnitConfig = REFERENCE_10M_CONFIG) -> None:
        super().__init__()
        config.validate()
        self.config = config

        self.input_projection = nn.Linear(config.feature_width, config.d_model)
        self.position_embedding = nn.Parameter(
            torch.zeros(1, config.sequence_length, config.d_model)
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.attention_heads,
            dim_feedforward=config.feed_forward_width,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=config.block_count,
        )
        self.final_norm = nn.LayerNorm(config.d_model)

        # The class head predicts only answerable labels. Uncertainty is a
        # separate binary decision so abstention quality can be measured directly.
        self.label_head = nn.Linear(config.d_model, len(NON_UNCERTAIN_LABELS))
        self.uncertainty_head = nn.Linear(config.d_model, 1)

        nn.init.trunc_normal_(self.position_embedding, std=0.02)

    def forward(self, features: torch.Tensor, mask: torch.Tensor) -> Step01Output:
        if features.ndim != 3:
            raise ValueError("features must have shape [batch, sequence, feature]")
        if mask.ndim != 2:
            raise ValueError("mask must have shape [batch, sequence]")
        if features.shape[:2] != mask.shape:
            raise ValueError("feature and mask batch/sequence dimensions must match")
        if features.shape[1] != self.config.sequence_length:
            raise ValueError(
                f"expected sequence length {self.config.sequence_length}, "
                f"got {features.shape[1]}"
            )
        if features.shape[2] != self.config.feature_width:
            raise ValueError(
                f"expected feature width {self.config.feature_width}, "
                f"got {features.shape[2]}"
            )

        mask = mask.to(dtype=torch.bool)
        if not torch.all(mask.any(dim=1)):
            raise ValueError("every sample must contain at least one valid sequence row")

        hidden = self.input_projection(features)
        hidden = hidden + self.position_embedding
        hidden = self.encoder(hidden, src_key_padding_mask=~mask)
        hidden = self.final_norm(hidden)

        weights = mask.unsqueeze(-1).to(dtype=hidden.dtype)
        pooled = (hidden * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)

        return Step01Output(
            label_logits=self.label_head(pooled),
            uncertainty_logits=self.uncertainty_head(pooled).squeeze(-1),
        )

    def trainable_parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)


def decode_predictions(
    output: Step01Output,
    *,
    uncertainty_threshold: float = 0.5,
) -> list[str]:
    """Convert structured model output into benchmark labels without task masking.

    Invalid task-specific labels are intentionally possible and are measured by
    the evaluation pipeline rather than hidden by a deterministic output mask.
    """

    if not 0.0 <= uncertainty_threshold <= 1.0:
        raise ValueError("uncertainty_threshold must be in [0, 1]")

    uncertain = torch.sigmoid(output.uncertainty_logits) >= uncertainty_threshold
    label_indices = output.label_logits.argmax(dim=-1)

    return [
        "UNCERTAIN" if is_uncertain else NON_UNCERTAIN_LABELS[index]
        for is_uncertain, index in zip(
            uncertain.tolist(),
            label_indices.tolist(),
            strict=True,
        )
    ]
