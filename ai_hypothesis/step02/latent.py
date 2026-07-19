"""Analysis-only latent extraction for frozen Step 1 units."""

from __future__ import annotations

import torch

from ai_hypothesis.step01.model import Step01Unit


def extract_pooled_latent(
    model: Step01Unit,
    features: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Return the shared pre-head pooled representation without changing forward.

    This mirrors ``Step01Unit.forward`` up to the masked mean pooling operation.
    The returned tensor is worker-local and must not be averaged as a raw latent
    vector across independently trained workers.
    """

    if features.ndim != 3:
        raise ValueError("features must have shape [batch, sequence, feature]")
    if mask.ndim != 2:
        raise ValueError("mask must have shape [batch, sequence]")
    if features.shape[:2] != mask.shape:
        raise ValueError("feature and mask batch/sequence dimensions must match")
    if features.shape[1] != model.config.sequence_length:
        raise ValueError(
            f"expected sequence length {model.config.sequence_length}, "
            f"got {features.shape[1]}"
        )
    if features.shape[2] != model.config.feature_width:
        raise ValueError(
            f"expected feature width {model.config.feature_width}, "
            f"got {features.shape[2]}"
        )

    mask = mask.to(dtype=torch.bool)
    if not torch.all(mask.any(dim=1)):
        raise ValueError("every sample must contain at least one valid sequence row")

    hidden = model.input_projection(features)
    hidden = hidden + model.position_embedding
    hidden = model.encoder(hidden, src_key_padding_mask=~mask)
    hidden = model.final_norm(hidden)

    weights = mask.unsqueeze(-1).to(dtype=hidden.dtype)
    return (hidden * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)


def summarize_worker_local_scalar_scores(
    scores: torch.Tensor,
    thresholds: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Summarize comparable per-worker scalar scores without raw latent mixing."""

    if scores.ndim != 2:
        raise ValueError("scores must be [workers, samples] scalar outputs")
    if thresholds.shape != (scores.shape[0],):
        raise ValueError("thresholds must have one scalar per worker")
    worker_thresholds = thresholds.to(device=scores.device, dtype=scores.dtype).unsqueeze(1)
    return {
        "mean_probability": scores.mean(dim=0),
        "max_probability": scores.max(dim=0).values,
        "fraction_above_worker_threshold": (scores >= worker_thresholds).to(torch.float32).mean(dim=0),
    }
