"""Threshold-free summaries for large-scope relevance evaluations."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from statistics import fmean

from .evaluate import ScopeEvaluation, ScopeWorkerMode


@dataclass(frozen=True, slots=True)
class ScopeConditionSummary:
    split: str
    mode: ScopeWorkerMode
    width: int
    world_count: int
    positive_world_count: int
    negative_world_count: int
    target_inspected_count: int
    target_retrieved_count: int
    target_coverage_rate: float
    target_retrieval_rate: float
    retrieval_given_inspected: float | None
    mean_target_rank_when_inspected: float | None
    mean_target_relevant_evidence_when_inspected: float | None
    mean_strongest_distractor_relevant_evidence: float | None
    mean_target_minus_distractor_evidence: float | None
    mean_candidate_relevant_evidence_positive: float | None
    mean_candidate_relevant_evidence_negative: float | None
    max_candidate_relevant_evidence_negative: float | None

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["mode"] = self.mode.value
        return payload


def summarize_scope_evaluations(
    evaluations: Sequence[ScopeEvaluation],
) -> tuple[ScopeConditionSummary, ...]:
    """Summarize observable retrieval/evidence behavior without choosing a threshold."""

    grouped: dict[tuple[str, ScopeWorkerMode, int], list[ScopeEvaluation]] = defaultdict(list)
    for evaluation in evaluations:
        evaluation.validate()
        grouped[(evaluation.split, evaluation.mode, evaluation.width)].append(evaluation)

    summaries: list[ScopeConditionSummary] = []
    for (split, mode, width), group in sorted(
        grouped.items(), key=lambda item: (item[0][0], item[0][1].value, item[0][2])
    ):
        positives = [result for result in group if result.target_present]
        negatives = [result for result in group if not result.target_present]
        inspected = [result for result in positives if result.target_inspected]
        retrieved = [result for result in positives if result.candidate_is_target]
        target_scores = [
            result.target_relevant_evidence
            for result in inspected
            if result.target_relevant_evidence is not None
        ]
        target_ranks = [
            result.target_rank for result in inspected if result.target_rank is not None
        ]
        distractor_scores = [
            result.strongest_distractor_relevant_evidence
            for result in group
            if result.strongest_distractor_relevant_evidence is not None
        ]
        target_minus_distractor = [
            result.target_relevant_evidence - result.strongest_distractor_relevant_evidence
            for result in inspected
            if result.target_relevant_evidence is not None
            and result.strongest_distractor_relevant_evidence is not None
        ]
        positive_candidate_scores = [
            result.candidate_relevant_evidence for result in positives
        ]
        negative_candidate_scores = [
            result.candidate_relevant_evidence for result in negatives
        ]

        summaries.append(
            ScopeConditionSummary(
                split=split,
                mode=mode,
                width=width,
                world_count=len(group),
                positive_world_count=len(positives),
                negative_world_count=len(negatives),
                target_inspected_count=len(inspected),
                target_retrieved_count=len(retrieved),
                target_coverage_rate=_rate(len(inspected), len(positives)),
                target_retrieval_rate=_rate(len(retrieved), len(positives)),
                retrieval_given_inspected=(
                    _rate(len(retrieved), len(inspected)) if inspected else None
                ),
                mean_target_rank_when_inspected=_mean(target_ranks),
                mean_target_relevant_evidence_when_inspected=_mean(target_scores),
                mean_strongest_distractor_relevant_evidence=_mean(distractor_scores),
                mean_target_minus_distractor_evidence=_mean(target_minus_distractor),
                mean_candidate_relevant_evidence_positive=_mean(positive_candidate_scores),
                mean_candidate_relevant_evidence_negative=_mean(negative_candidate_scores),
                max_candidate_relevant_evidence_negative=(
                    max(negative_candidate_scores) if negative_candidate_scores else None
                ),
            )
        )
    return tuple(summaries)


def _rate(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def _mean(values: Sequence[float | int]) -> float | None:
    return float(fmean(values)) if values else None
