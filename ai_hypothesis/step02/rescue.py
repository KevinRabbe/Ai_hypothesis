"""Minority-rescue candidate extraction and validation-only gate utilities.

The Step 2 v0 reducer preserves strong minority evidence in its summary, but its
primary class is still chosen from population mean evidence. This module isolates
one narrower question without changing the production reducer:

    Can an inference-visible strong minority candidate be identified reliably
    enough to replace the current primary label when the population mean is wrong?

Ground truth is never accepted by the candidate builder or scorer. Truth is used
only by diagnostic calibration/measurement code outside inference.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import torch
import torch.nn.functional as F

from .evidence import EvidenceBatch, PopulationDecision, PopulationEvidenceSummary


MINORITY_RESCUE_FEATURE_NAMES: tuple[str, ...] = (
    "candidate_mean_evidence",
    "candidate_max_evidence",
    "candidate_topk_mean_evidence",
    "candidate_support_fraction",
    "candidate_peak_reliability",
    "candidate_peak_top_margin",
    "candidate_peak_worker_top_matches",
    "candidate_minus_primary_mean",
    "candidate_minus_primary_max",
    "candidate_minus_primary_topk_mean",
    "primary_margin",
    "disagreement_entropy",
    "mean_uncertainty",
    "mean_invalid_label_mass",
)


@dataclass(frozen=True, slots=True)
class MinorityCandidateBatch:
    """One strongest protected non-primary candidate per sample."""

    primary_label_indices: torch.Tensor
    candidate_label_indices: torch.Tensor
    candidate_exists: torch.Tensor
    features: torch.Tensor


@dataclass(frozen=True, slots=True)
class RescueGateConfig:
    """Frozen linear rescue gate calibrated only on development data."""

    version: str
    feature_names: tuple[str, ...]
    threshold: float
    weights: tuple[float, ...]
    bias: float
    feature_mean: tuple[float, ...]
    feature_std: tuple[float, ...]

    def validate(self) -> None:
        if not self.version:
            raise ValueError("version must be non-empty")
        if self.feature_names != MINORITY_RESCUE_FEATURE_NAMES:
            raise ValueError("feature_names do not match the minority rescue contract")
        if not 0.0 <= self.threshold <= 1.0:
            raise ValueError("threshold must be in [0, 1]")
        feature_count = len(self.feature_names)
        for name, values in (
            ("weights", self.weights),
            ("feature_mean", self.feature_mean),
            ("feature_std", self.feature_std),
        ):
            if len(values) != feature_count:
                raise ValueError(f"{name} must have one value per feature")
        if any(value <= 0.0 for value in self.feature_std):
            raise ValueError("feature_std values must be positive")

    def to_json(self) -> str:
        self.validate()
        return json.dumps(asdict(self), indent=2, sort_keys=True)

    @classmethod
    def from_json(cls, payload: str) -> "RescueGateConfig":
        data = json.loads(payload)
        for key in ("feature_names", "weights", "feature_mean", "feature_std"):
            data[key] = tuple(data[key])
        config = cls(**data)
        config.validate()
        return config

    def save(self, path: str | Path) -> None:
        Path(path).write_text(self.to_json(), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "RescueGateConfig":
        return cls.from_json(Path(path).read_text(encoding="utf-8"))


def _gather_label(values: torch.Tensor, label_indices: torch.Tensor) -> torch.Tensor:
    if values.ndim != 2:
        raise ValueError("values must have shape [batch, labels]")
    if label_indices.shape != (values.shape[0],):
        raise ValueError("label_indices must have one entry per batch sample")
    return values.gather(1, label_indices.unsqueeze(1)).squeeze(1)


def build_minority_candidates(
    evidence: EvidenceBatch,
    summary: PopulationEvidenceSummary,
    decision: PopulationDecision,
) -> MinorityCandidateBatch:
    """Build a high-recall minority proposal from inference-visible evidence only.

    Candidate generation and candidate acceptance are deliberately separate. The
    strongest protected non-primary label is proposed using maximum worker evidence
    so a single rare signal is not averaged away. A later calibrated gate decides
    whether that proposal is trustworthy enough to replace the primary label.
    """

    mean_evidence = summary.mean_evidence_per_label
    max_evidence = summary.max_evidence_per_label
    if mean_evidence.ndim != 2 or max_evidence.shape != mean_evidence.shape:
        raise ValueError("summary evidence tensors must have shape [batch, labels]")
    batch_size, label_count = mean_evidence.shape
    if decision.primary_label_indices.shape != (batch_size,):
        raise ValueError("decision primary indices must match summary batch size")
    if evidence.valid_label_mask.shape != (batch_size, label_count):
        raise ValueError("evidence valid-label mask does not match summary")
    if summary.protected_label_mask.shape != (batch_size, label_count):
        raise ValueError("protected-label mask does not match summary")
    if summary.top_k_evidence_per_label.shape[:2] != (batch_size, label_count):
        raise ValueError("top-k evidence tensor does not match summary")

    primary = decision.primary_label_indices
    label_positions = torch.arange(label_count, device=mean_evidence.device).unsqueeze(0)
    primary_mask = label_positions == primary.unsqueeze(1)
    candidate_mask = (
        evidence.valid_label_mask
        & summary.protected_label_mask
        & ~primary_mask
    )

    candidate_rank = max_evidence.masked_fill(~candidate_mask, float("-inf"))
    candidate_indices = candidate_rank.argmax(dim=-1)
    candidate_exists = candidate_mask.any(dim=-1)
    candidate_indices = torch.where(candidate_exists, candidate_indices, primary)

    topk_mean = summary.top_k_evidence_per_label.mean(dim=-1)
    support_fraction = summary.support_count_per_label.to(mean_evidence.dtype) / float(
        summary.population_width
    )

    candidate_mean = _gather_label(mean_evidence, candidate_indices)
    candidate_max = _gather_label(max_evidence, candidate_indices)
    candidate_topk = _gather_label(topk_mean, candidate_indices)
    candidate_support = _gather_label(support_fraction, candidate_indices)

    primary_mean = _gather_label(mean_evidence, primary)
    primary_max = _gather_label(max_evidence, primary)
    primary_topk = _gather_label(topk_mean, primary)

    batch_positions = torch.arange(batch_size, device=mean_evidence.device)
    peak_worker_ids = summary.top_k_worker_ids_per_label[
        batch_positions,
        candidate_indices,
        0,
    ]
    peak_reliability = evidence.reliability[peak_worker_ids, batch_positions]
    peak_top_margin = evidence.top_margin[peak_worker_ids, batch_positions]
    peak_top_matches = (
        evidence.top_valid_label_indices[peak_worker_ids, batch_positions]
        == candidate_indices
    ).to(mean_evidence.dtype)

    features = torch.stack(
        (
            candidate_mean,
            candidate_max,
            candidate_topk,
            candidate_support,
            peak_reliability,
            peak_top_margin,
            peak_top_matches,
            candidate_mean - primary_mean,
            candidate_max - primary_max,
            candidate_topk - primary_topk,
            decision.primary_margin,
            summary.disagreement_entropy,
            summary.mean_uncertainty,
            summary.mean_invalid_label_mass,
        ),
        dim=1,
    )
    features = torch.where(
        candidate_exists.unsqueeze(1),
        features,
        torch.zeros_like(features),
    )

    return MinorityCandidateBatch(
        primary_label_indices=primary,
        candidate_label_indices=candidate_indices,
        candidate_exists=candidate_exists,
        features=features,
    )


def fit_rescue_gate(
    features: torch.Tensor,
    improvement_targets: torch.Tensor,
    candidate_exists: torch.Tensor,
    *,
    steps: int = 600,
    learning_rate: float = 0.05,
    weight_decay: float = 1e-4,
) -> RescueGateConfig:
    """Fit a tiny logistic gate on development data only.

    ``improvement_targets`` is diagnostic supervision: true only when switching
    from the current primary label to the proposed candidate would fix the sample.
    It must never be constructed or consumed in inference code.
    """

    if features.ndim != 2 or features.shape[1] != len(MINORITY_RESCUE_FEATURE_NAMES):
        raise ValueError("features do not match the minority rescue feature contract")
    if improvement_targets.shape != (features.shape[0],):
        raise ValueError("improvement_targets must have one value per sample")
    if candidate_exists.shape != (features.shape[0],):
        raise ValueError("candidate_exists must have one value per sample")
    if steps <= 0:
        raise ValueError("steps must be positive")
    if learning_rate <= 0.0:
        raise ValueError("learning_rate must be positive")
    if weight_decay < 0.0:
        raise ValueError("weight_decay must be non-negative")

    mask = candidate_exists.to(dtype=torch.bool)
    x = features[mask].detach().to(device="cpu", dtype=torch.float32)
    y = improvement_targets[mask].detach().to(device="cpu", dtype=torch.float32)
    if x.shape[0] < 2:
        raise ValueError("at least two minority candidates are required for calibration")
    positive_count = int(y.sum().item())
    negative_count = int(y.numel() - positive_count)
    if positive_count == 0 or negative_count == 0:
        raise ValueError("calibration requires both improving and non-improving candidates")

    feature_mean = x.mean(dim=0)
    feature_std = x.std(dim=0, unbiased=False).clamp_min(1e-6)
    standardized = (x - feature_mean) / feature_std

    weights = torch.zeros(standardized.shape[1], dtype=torch.float32, requires_grad=True)
    bias = torch.zeros((), dtype=torch.float32, requires_grad=True)
    optimizer = torch.optim.Adam(
        (weights, bias),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    pos_weight = torch.tensor(negative_count / positive_count, dtype=torch.float32)

    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        logits = standardized @ weights + bias
        loss = F.binary_cross_entropy_with_logits(logits, y, pos_weight=pos_weight)
        loss.backward()
        optimizer.step()

    return RescueGateConfig(
        version="step02-minority-rescue-gate-v0",
        feature_names=MINORITY_RESCUE_FEATURE_NAMES,
        threshold=0.5,
        weights=tuple(float(value) for value in weights.detach()),
        bias=float(bias.detach()),
        feature_mean=tuple(float(value) for value in feature_mean),
        feature_std=tuple(float(value) for value in feature_std),
    )


def score_rescue_gate(
    features: torch.Tensor,
    candidate_exists: torch.Tensor,
    config: RescueGateConfig,
) -> torch.Tensor:
    """Return an inference-time probability that the minority switch is useful."""

    config.validate()
    if features.ndim != 2 or features.shape[1] != len(config.feature_names):
        raise ValueError("features do not match rescue gate config")
    if candidate_exists.shape != (features.shape[0],):
        raise ValueError("candidate_exists must have one value per sample")

    mean = torch.tensor(config.feature_mean, device=features.device, dtype=features.dtype)
    std = torch.tensor(config.feature_std, device=features.device, dtype=features.dtype)
    weights = torch.tensor(config.weights, device=features.device, dtype=features.dtype)
    standardized = (features - mean) / std.clamp_min(1e-6)
    scores = torch.sigmoid(standardized @ weights + config.bias)
    return torch.where(candidate_exists.to(dtype=torch.bool), scores, torch.zeros_like(scores))


def apply_rescue_gate(
    primary_label_indices: torch.Tensor,
    candidate_label_indices: torch.Tensor,
    candidate_exists: torch.Tensor,
    rescue_scores: torch.Tensor,
    *,
    threshold: float,
) -> torch.Tensor:
    """Switch primary A to minority candidate B only when the frozen gate accepts."""

    shape = primary_label_indices.shape
    if primary_label_indices.ndim != 1:
        raise ValueError("primary_label_indices must be one-dimensional")
    if candidate_label_indices.shape != shape:
        raise ValueError("candidate_label_indices must match primary_label_indices")
    if candidate_exists.shape != shape or rescue_scores.shape != shape:
        raise ValueError("candidate_exists/rescue_scores must match primary labels")
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be in [0, 1]")

    switch = candidate_exists.to(dtype=torch.bool) & (rescue_scores > threshold)
    return torch.where(switch, candidate_label_indices, primary_label_indices)


def rescue_threshold_metrics(
    rescue_scores: torch.Tensor,
    candidate_exists: torch.Tensor,
    primary_correct: torch.Tensor,
    candidate_correct: torch.Tensor,
    *,
    threshold: float,
) -> dict[str, float | int]:
    """Measure one threshold without exposing truth to inference-time functions."""

    shape = rescue_scores.shape
    if rescue_scores.ndim != 1:
        raise ValueError("rescue_scores must be one-dimensional")
    if any(tensor.shape != shape for tensor in (candidate_exists, primary_correct, candidate_correct)):
        raise ValueError("all rescue metric tensors must have matching shape")
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be in [0, 1]")

    exists = candidate_exists.to(dtype=torch.bool)
    primary_ok = primary_correct.to(dtype=torch.bool)
    candidate_ok = candidate_correct.to(dtype=torch.bool)
    switch = exists & (rescue_scores > threshold)
    gains = switch & ~primary_ok & candidate_ok
    harms = switch & primary_ok & ~candidate_ok
    opportunities = exists & ~primary_ok & candidate_ok
    final_correct = torch.where(switch, candidate_ok, primary_ok)

    total = int(rescue_scores.numel())
    switch_count = int(switch.sum().item())
    gain_count = int(gains.sum().item())
    harm_count = int(harms.sum().item())
    opportunity_count = int(opportunities.sum().item())
    accuracy_before = float(primary_ok.to(torch.float32).mean().item())
    accuracy_after = float(final_correct.to(torch.float32).mean().item())

    return {
        "threshold": float(threshold),
        "count": total,
        "candidate_count": int(exists.sum().item()),
        "switch_count": switch_count,
        "gain_count": gain_count,
        "harm_count": harm_count,
        "net_gain_count": gain_count - harm_count,
        "rescue_opportunity_count": opportunity_count,
        "rescue_recall": gain_count / opportunity_count if opportunity_count else 0.0,
        "switch_precision": gain_count / switch_count if switch_count else 0.0,
        "harm_rate_total": harm_count / total if total else 0.0,
        "accuracy_before": accuracy_before,
        "accuracy_after": accuracy_after,
        "accuracy_delta": accuracy_after - accuracy_before,
    }


def select_rescue_threshold(
    rescue_scores: torch.Tensor,
    candidate_exists: torch.Tensor,
    primary_correct: torch.Tensor,
    candidate_correct: torch.Tensor,
    *,
    max_harm_rate: float,
    thresholds: Sequence[float] | None = None,
) -> dict[str, object]:
    """Select a development threshold under an explicit total-harm budget."""

    if not 0.0 <= max_harm_rate <= 1.0:
        raise ValueError("max_harm_rate must be in [0, 1]")
    if thresholds is None:
        thresholds = tuple(index / 100.0 for index in range(101))
    if not thresholds:
        raise ValueError("thresholds must not be empty")

    sweep = [
        rescue_threshold_metrics(
            rescue_scores,
            candidate_exists,
            primary_correct,
            candidate_correct,
            threshold=float(threshold),
        )
        for threshold in thresholds
    ]
    eligible = [row for row in sweep if float(row["harm_rate_total"]) <= max_harm_rate]
    if not eligible:
        raise ValueError("no threshold satisfies the requested harm budget")

    selected = max(
        eligible,
        key=lambda row: (
            float(row["accuracy_after"]),
            -float(row["harm_rate_total"]),
            float(row["threshold"]),
        ),
    )
    return {
        "max_harm_rate": float(max_harm_rate),
        "selected": selected,
        "sweep": sweep,
    }
