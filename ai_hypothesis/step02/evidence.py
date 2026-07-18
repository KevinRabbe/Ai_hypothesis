"""Evidence construction and deterministic aggregation for Step 2.

This module keeps worker outputs continuous until the final decision. Majority vote
is deliberately not used as the primary aggregation rule.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import torch
import torch.nn.functional as F

from ai_hypothesis.step01.model import NON_UNCERTAIN_LABELS
from ai_hypothesis.step01.schema import TaskFamily, VALID_LABELS
from .population import PopulationOutput


@dataclass(frozen=True, slots=True)
class AggregationConfig:
    """Version-0 deterministic evidence settings.

    Threshold defaults are starting values for development and smoke tests. They
    must be calibrated on validation data and frozen before a formal test-set run.
    """

    eps: float = 1e-8
    evidence_clip: float = 8.0
    top_k: int = 3
    support_threshold: float = 0.5
    strong_evidence_threshold: float = 2.0
    min_primary_margin: float = 0.15
    max_mean_uncertainty: float = 0.5
    max_mean_invalid_mass: float = 0.5
    protected_conflict_mean_gap: float = 0.5

    def validate(self) -> None:
        if self.eps <= 0.0:
            raise ValueError("eps must be positive")
        if self.evidence_clip <= 0.0:
            raise ValueError("evidence_clip must be positive")
        if self.top_k <= 0:
            raise ValueError("top_k must be positive")
        if self.strong_evidence_threshold < self.support_threshold:
            raise ValueError(
                "strong_evidence_threshold must be >= support_threshold"
            )
        if self.min_primary_margin < 0.0:
            raise ValueError("min_primary_margin must be non-negative")
        if not 0.0 <= self.max_mean_uncertainty <= 1.0:
            raise ValueError("max_mean_uncertainty must be in [0, 1]")
        if not 0.0 <= self.max_mean_invalid_mass <= 1.0:
            raise ValueError("max_mean_invalid_mass must be in [0, 1]")
        if self.protected_conflict_mean_gap < 0.0:
            raise ValueError("protected_conflict_mean_gap must be non-negative")


@dataclass(frozen=True, slots=True)
class EvidenceBatch:
    """Per-worker evidence matrix for one population output batch."""

    label_probabilities_all: torch.Tensor
    valid_label_probabilities: torch.Tensor
    valid_label_mask: torch.Tensor
    invalid_label_mass: torch.Tensor
    uncertainty_probability: torch.Tensor
    reliability: torch.Tensor
    evidence_scores: torch.Tensor
    top_valid_label_indices: torch.Tensor
    top_margin: torch.Tensor


@dataclass(frozen=True, slots=True)
class PopulationEvidenceSummary:
    """Evidence-preserving population reduction without a final class decision."""

    population_width: int
    sum_evidence_per_label: torch.Tensor
    mean_evidence_per_label: torch.Tensor
    max_evidence_per_label: torch.Tensor
    top_k_evidence_per_label: torch.Tensor
    top_k_worker_ids_per_label: torch.Tensor
    support_count_per_label: torch.Tensor
    mean_uncertainty: torch.Tensor
    max_uncertainty: torch.Tensor
    uncertainty_quantiles: torch.Tensor
    mean_invalid_label_mass: torch.Tensor
    max_invalid_label_mass: torch.Tensor
    disagreement_entropy: torch.Tensor
    protected_label_mask: torch.Tensor


@dataclass(frozen=True, slots=True)
class PopulationDecision:
    """Final deterministic v0 decision plus diagnostics."""

    predictions: tuple[str, ...]
    primary_label_indices: torch.Tensor
    primary_margin: torch.Tensor
    unresolved_contradiction: torch.Tensor
    uncertainty_reasons: tuple[tuple[str, ...], ...]


def _valid_label_mask(
    tasks: Sequence[TaskFamily],
    *,
    device: torch.device,
) -> torch.Tensor:
    mask = torch.zeros(
        (len(tasks), len(NON_UNCERTAIN_LABELS)),
        dtype=torch.bool,
        device=device,
    )
    for sample_index, task in enumerate(tasks):
        valid = VALID_LABELS[task]
        for label_index, label in enumerate(NON_UNCERTAIN_LABELS):
            if label in valid:
                mask[sample_index, label_index] = True
    if not torch.all(mask.sum(dim=-1) >= 2):
        raise ValueError("every Step 2 task must have at least two non-UNCERTAIN labels")
    return mask


def build_evidence_matrix(
    output: PopulationOutput,
    tasks: Sequence[TaskFamily],
    config: AggregationConfig = AggregationConfig(),
) -> EvidenceBatch:
    """Convert raw population logits into the version-0 evidence contract."""

    config.validate()
    if output.label_logits.ndim != 3:
        raise ValueError("label_logits must have shape [workers, batch, labels]")
    if output.uncertainty_logits.ndim != 2:
        raise ValueError("uncertainty_logits must have shape [workers, batch]")

    workers, batch_size, label_count = output.label_logits.shape
    if label_count != len(NON_UNCERTAIN_LABELS):
        raise ValueError("population output label dimension does not match Step 1")
    if output.uncertainty_logits.shape != (workers, batch_size):
        raise ValueError("population label and uncertainty dimensions do not match")
    if len(tasks) != batch_size:
        raise ValueError("tasks length must match population output batch size")

    probabilities = torch.softmax(output.label_logits, dim=-1)
    valid_mask = _valid_label_mask(tasks, device=probabilities.device)
    valid_mask_workers = valid_mask.unsqueeze(0)

    valid_mass = (probabilities * valid_mask_workers).sum(dim=-1)
    invalid_mass = (1.0 - valid_mass).clamp(0.0, 1.0)
    valid_probabilities = (
        probabilities * valid_mask_workers
    ) / valid_mass.clamp_min(config.eps).unsqueeze(-1)

    uncertainty = torch.sigmoid(output.uncertainty_logits)
    reliability = (1.0 - uncertainty) * (1.0 - invalid_mass)

    log_valid = torch.log(valid_probabilities.clamp_min(config.eps))
    valid_count = valid_mask.sum(dim=-1).to(log_valid.dtype)
    sum_valid_logs = (log_valid * valid_mask_workers).sum(dim=-1)
    other_mean = (
        sum_valid_logs.unsqueeze(-1) - log_valid
    ) / (valid_count.unsqueeze(0).unsqueeze(-1) - 1.0)
    raw_support = log_valid - other_mean
    evidence_scores = (
        reliability.unsqueeze(-1)
        * raw_support.clamp(-config.evidence_clip, config.evidence_clip)
        * valid_mask_workers
    )

    masked_valid_probabilities = valid_probabilities.masked_fill(
        ~valid_mask_workers,
        float("-inf"),
    )
    top_values, top_indices = torch.topk(masked_valid_probabilities, k=2, dim=-1)
    top_margin = top_values[..., 0] - top_values[..., 1]

    return EvidenceBatch(
        label_probabilities_all=probabilities,
        valid_label_probabilities=valid_probabilities,
        valid_label_mask=valid_mask,
        invalid_label_mass=invalid_mass,
        uncertainty_probability=uncertainty,
        reliability=reliability,
        evidence_scores=evidence_scores,
        top_valid_label_indices=top_indices[..., 0],
        top_margin=top_margin,
    )


def aggregate_evidence(
    evidence: EvidenceBatch,
    config: AggregationConfig = AggregationConfig(),
) -> tuple[PopulationEvidenceSummary, PopulationDecision]:
    """Reduce a worker evidence matrix while preserving strong minority signals."""

    config.validate()
    scores = evidence.evidence_scores
    if scores.ndim != 3:
        raise ValueError("evidence_scores must have shape [workers, batch, labels]")
    population_width, batch_size, label_count = scores.shape
    if population_width <= 0:
        raise ValueError("population must contain at least one worker")

    sum_evidence = scores.sum(dim=0)
    mean_evidence = scores.mean(dim=0)
    max_evidence = scores.max(dim=0).values

    k = min(config.top_k, population_width)
    top_k_values, top_k_ids = torch.topk(scores, k=k, dim=0)
    top_k_values = top_k_values.permute(1, 2, 0).contiguous()
    top_k_ids = top_k_ids.permute(1, 2, 0).contiguous()

    support_count = (scores >= config.support_threshold).sum(dim=0)
    mean_uncertainty = evidence.uncertainty_probability.mean(dim=0)
    max_uncertainty = evidence.uncertainty_probability.max(dim=0).values
    quantiles = torch.quantile(
        evidence.uncertainty_probability,
        torch.tensor(
            [0.25, 0.5, 0.75],
            device=scores.device,
            dtype=scores.dtype,
        ),
        dim=0,
    ).transpose(0, 1)
    mean_invalid = evidence.invalid_label_mass.mean(dim=0)
    max_invalid = evidence.invalid_label_mass.max(dim=0).values

    worker_argmax = evidence.top_valid_label_indices
    argmax_one_hot = F.one_hot(worker_argmax, num_classes=label_count).to(scores.dtype)
    argmax_distribution = argmax_one_hot.mean(dim=0)
    entropy = -(
        argmax_distribution
        * torch.log(argmax_distribution.clamp_min(config.eps))
    ).sum(dim=-1)
    valid_count = evidence.valid_label_mask.sum(dim=-1).to(scores.dtype)
    disagreement_entropy = entropy / torch.log(valid_count)

    protected = (
        (max_evidence >= config.strong_evidence_threshold)
        & evidence.valid_label_mask
    )

    summary = PopulationEvidenceSummary(
        population_width=population_width,
        sum_evidence_per_label=sum_evidence,
        mean_evidence_per_label=mean_evidence,
        max_evidence_per_label=max_evidence,
        top_k_evidence_per_label=top_k_values,
        top_k_worker_ids_per_label=top_k_ids,
        support_count_per_label=support_count,
        mean_uncertainty=mean_uncertainty,
        max_uncertainty=max_uncertainty,
        uncertainty_quantiles=quantiles,
        mean_invalid_label_mass=mean_invalid,
        max_invalid_label_mass=max_invalid,
        disagreement_entropy=disagreement_entropy,
        protected_label_mask=protected,
    )

    masked_mean = mean_evidence.masked_fill(
        ~evidence.valid_label_mask,
        float("-inf"),
    )
    primary_values, primary_indices = torch.topk(masked_mean, k=2, dim=-1)
    primary_margin = primary_values[:, 0] - primary_values[:, 1]
    primary = primary_indices[:, 0]

    label_positions = torch.arange(label_count, device=scores.device).unsqueeze(0)
    primary_mask = label_positions == primary.unsqueeze(-1)
    protected_competitor = protected & ~primary_mask
    competitor_mean = masked_mean.masked_fill(
        ~protected_competitor,
        float("-inf"),
    ).max(dim=-1).values
    has_protected_competitor = protected_competitor.any(dim=-1)
    protected_gap = primary_values[:, 0] - competitor_mean
    unresolved_contradiction = (
        has_protected_competitor
        & (protected_gap <= config.protected_conflict_mean_gap)
    )

    predictions: list[str] = []
    reasons_per_sample: list[tuple[str, ...]] = []
    for sample_index in range(batch_size):
        reasons: list[str] = []
        if float(primary_margin[sample_index]) < config.min_primary_margin:
            reasons.append("low_primary_margin")
        if float(mean_uncertainty[sample_index]) > config.max_mean_uncertainty:
            reasons.append("high_mean_uncertainty")
        if float(mean_invalid[sample_index]) > config.max_mean_invalid_mass:
            reasons.append("high_invalid_label_mass")
        if bool(unresolved_contradiction[sample_index]):
            reasons.append("protected_minority_contradiction")

        if reasons:
            predictions.append("UNCERTAIN")
        else:
            predictions.append(NON_UNCERTAIN_LABELS[int(primary[sample_index])])
        reasons_per_sample.append(tuple(reasons))

    return summary, PopulationDecision(
        predictions=tuple(predictions),
        primary_label_indices=primary,
        primary_margin=primary_margin,
        unresolved_contradiction=unresolved_contradiction,
        uncertainty_reasons=tuple(reasons_per_sample),
    )
