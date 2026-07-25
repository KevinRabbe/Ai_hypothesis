"""Execute large-scope relevance worlds with frozen Step 1/2 workers.

The evaluator intentionally keeps world-level output continuous. It ranks inspected
windows by local RELEVANT evidence but does not introduce a detection threshold or
claim that the highest-scoring window should always be accepted as globally relevant.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Protocol

import torch

from ai_hypothesis.step01.model import LABEL_TO_INDEX, NON_UNCERTAIN_LABELS, Step01Output
from ai_hypothesis.step01.schema import TaskFamily
from ai_hypothesis.step02.evidence import AggregationConfig, build_evidence_matrix
from ai_hypothesis.step02.population import PopulationOutput

from .relevance import (
    LargeScopeRelevanceSample,
    diverse_worker_indices,
    inspection_prefix,
    same_worker_indices,
)


class ScopeWorkerMode(str, Enum):
    SAME_WORKER = "same_worker"
    DIVERSE_WORKERS = "diverse_workers"


class SelectedWorkerBank(Protocol):
    @property
    def population_width(self) -> int: ...

    def forward_selected(
        self,
        worker_indices: Sequence[int] | torch.Tensor,
        features: torch.Tensor,
        mask: torch.Tensor,
    ) -> Step01Output: ...


@dataclass(frozen=True, slots=True)
class WindowEvidence:
    window_index: int
    worker_index: int
    local_label: str
    relevant_evidence: float
    not_relevant_evidence: float
    uncertainty_probability: float
    invalid_label_mass: float
    top_margin: float


@dataclass(frozen=True, slots=True)
class ScopeEvaluation:
    seed: int
    width: int
    mode: ScopeWorkerMode
    inspected_window_indices: tuple[int, ...]
    worker_indices: tuple[int, ...]
    target_present: bool
    target_index: int | None
    target_inspected: bool
    candidate_window_index: int
    candidate_is_target: bool
    candidate_relevant_evidence: float
    target_relevant_evidence: float | None
    target_rank: int | None
    strongest_distractor_relevant_evidence: float | None
    window_evidence: tuple[WindowEvidence, ...]

    def validate(self) -> None:
        if self.width <= 0:
            raise ValueError("width must be positive")
        if len(self.inspected_window_indices) != self.width:
            raise ValueError("inspected-window count must match width")
        if len(self.worker_indices) != self.width:
            raise ValueError("worker-index count must match width")
        if len(self.window_evidence) != self.width:
            raise ValueError("window-evidence count must match width")
        if len(set(self.inspected_window_indices)) != self.width:
            raise ValueError("inspection plan must not duplicate windows")
        if self.candidate_window_index not in self.inspected_window_indices:
            raise ValueError("candidate must be an inspected window")
        if self.target_inspected != (
            self.target_index in self.inspected_window_indices
            if self.target_index is not None
            else False
        ):
            raise ValueError("target_inspected is inconsistent with inspection plan")
        if self.target_rank is not None and not 1 <= self.target_rank <= self.width:
            raise ValueError("target_rank must be inside inspected width")


def evaluate_scope_sample(
    bank: SelectedWorkerBank,
    sample: LargeScopeRelevanceSample,
    *,
    width: int,
    mode: ScopeWorkerMode | str,
    evidence_config: AggregationConfig = AggregationConfig(),
) -> ScopeEvaluation:
    """Evaluate one nested-width inspection prefix against one frozen worker bank."""

    sample.validate()
    mode = ScopeWorkerMode(mode)
    window_indices = inspection_prefix(sample, width)
    if mode is ScopeWorkerMode.SAME_WORKER:
        workers = same_worker_indices(
            seed=sample.seed,
            width=width,
            population_width=bank.population_width,
        )
    else:
        workers = diverse_worker_indices(
            seed=sample.seed,
            width=width,
            population_width=bank.population_width,
        )

    features = torch.tensor(
        [sample.windows[index].features for index in window_indices],
        dtype=torch.float32,
    )
    mask = torch.tensor(
        [sample.windows[index].mask for index in window_indices],
        dtype=torch.bool,
    )
    raw = bank.forward_selected(workers, features, mask)
    if raw.label_logits.shape[0] != width or raw.uncertainty_logits.shape[0] != width:
        raise ValueError("worker bank returned a batch size different from requested width")

    evidence = build_evidence_matrix(
        PopulationOutput(
            label_logits=raw.label_logits.unsqueeze(0),
            uncertainty_logits=raw.uncertainty_logits.unsqueeze(0),
        ),
        [TaskFamily.RELEVANCE] * width,
        evidence_config,
    )
    relevant_index = LABEL_TO_INDEX["RELEVANT"]
    not_relevant_index = LABEL_TO_INDEX["NOT_RELEVANT"]

    rows: list[WindowEvidence] = []
    relevant_scores: list[float] = []
    for batch_index, (window_index, worker_index) in enumerate(
        zip(window_indices, workers, strict=True)
    ):
        uncertainty = float(evidence.uncertainty_probability[0, batch_index].item())
        top_label_index = int(evidence.top_valid_label_indices[0, batch_index].item())
        local_label = (
            "UNCERTAIN"
            if uncertainty >= 0.5
            else NON_UNCERTAIN_LABELS[top_label_index]
        )
        relevant_score = float(
            evidence.evidence_scores[0, batch_index, relevant_index].item()
        )
        relevant_scores.append(relevant_score)
        rows.append(
            WindowEvidence(
                window_index=window_index,
                worker_index=worker_index,
                local_label=local_label,
                relevant_evidence=relevant_score,
                not_relevant_evidence=float(
                    evidence.evidence_scores[
                        0, batch_index, not_relevant_index
                    ].item()
                ),
                uncertainty_probability=uncertainty,
                invalid_label_mass=float(
                    evidence.invalid_label_mass[0, batch_index].item()
                ),
                top_margin=float(evidence.top_margin[0, batch_index].item()),
            )
        )

    candidate_batch_index = max(
        range(width),
        key=lambda index: (relevant_scores[index], -index),
    )
    candidate_window_index = window_indices[candidate_batch_index]
    target_inspected = (
        sample.target_index in window_indices if sample.target_index is not None else False
    )

    target_score: float | None = None
    target_rank: int | None = None
    if target_inspected:
        assert sample.target_index is not None
        target_batch_index = window_indices.index(sample.target_index)
        target_score = relevant_scores[target_batch_index]
        target_rank = 1 + sum(
            score > target_score for score in relevant_scores
        )

    distractor_scores = [
        score
        for window_index, score in zip(window_indices, relevant_scores, strict=True)
        if window_index != sample.target_index
    ]
    strongest_distractor = max(distractor_scores) if distractor_scores else None

    result = ScopeEvaluation(
        seed=sample.seed,
        width=width,
        mode=mode,
        inspected_window_indices=window_indices,
        worker_indices=workers,
        target_present=sample.target_present,
        target_index=sample.target_index,
        target_inspected=target_inspected,
        candidate_window_index=candidate_window_index,
        candidate_is_target=(
            sample.target_present and candidate_window_index == sample.target_index
        ),
        candidate_relevant_evidence=relevant_scores[candidate_batch_index],
        target_relevant_evidence=target_score,
        target_rank=target_rank,
        strongest_distractor_relevant_evidence=strongest_distractor,
        window_evidence=tuple(rows),
    )
    result.validate()
    return result


def evaluate_scope_widths(
    bank: SelectedWorkerBank,
    sample: LargeScopeRelevanceSample,
    *,
    widths: Sequence[int],
    modes: Sequence[ScopeWorkerMode | str] = (
        ScopeWorkerMode.SAME_WORKER,
        ScopeWorkerMode.DIVERSE_WORKERS,
    ),
    evidence_config: AggregationConfig = AggregationConfig(),
) -> tuple[ScopeEvaluation, ...]:
    """Evaluate nested widths/modes without changing the inspected prefix definition."""

    if not widths:
        raise ValueError("widths must not be empty")
    normalized_widths = tuple(int(width) for width in widths)
    if any(width <= 0 for width in normalized_widths):
        raise ValueError("widths must be positive")
    if len(set(normalized_widths)) != len(normalized_widths):
        raise ValueError("widths must be unique")
    if tuple(sorted(normalized_widths)) != normalized_widths:
        raise ValueError("widths must be supplied in increasing order")

    results: list[ScopeEvaluation] = []
    for mode in modes:
        for width in normalized_widths:
            results.append(
                evaluate_scope_sample(
                    bank,
                    sample,
                    width=width,
                    mode=mode,
                    evidence_config=evidence_config,
                )
            )
    return tuple(results)
