"""Threshold-free bounded summaries for large-scope relevance evaluations."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass

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


@dataclass(slots=True)
class _RunningCondition:
    split: str
    mode: ScopeWorkerMode
    width: int
    world_count: int = 0
    positive_world_count: int = 0
    negative_world_count: int = 0
    target_inspected_count: int = 0
    target_retrieved_count: int = 0
    target_rank_sum: float = 0.0
    target_rank_count: int = 0
    target_score_sum: float = 0.0
    target_score_count: int = 0
    distractor_score_sum: float = 0.0
    distractor_score_count: int = 0
    target_gap_sum: float = 0.0
    target_gap_count: int = 0
    positive_candidate_sum: float = 0.0
    positive_candidate_count: int = 0
    negative_candidate_sum: float = 0.0
    negative_candidate_count: int = 0
    max_negative_candidate: float | None = None

    def add(self, result: ScopeEvaluation) -> None:
        self.world_count += 1
        if result.target_present:
            self.positive_world_count += 1
            self.positive_candidate_sum += result.candidate_relevant_evidence
            self.positive_candidate_count += 1
            if result.target_inspected:
                self.target_inspected_count += 1
                if result.target_rank is not None:
                    self.target_rank_sum += result.target_rank
                    self.target_rank_count += 1
                if result.target_relevant_evidence is not None:
                    self.target_score_sum += result.target_relevant_evidence
                    self.target_score_count += 1
                if (
                    result.target_relevant_evidence is not None
                    and result.strongest_distractor_relevant_evidence is not None
                ):
                    self.target_gap_sum += (
                        result.target_relevant_evidence
                        - result.strongest_distractor_relevant_evidence
                    )
                    self.target_gap_count += 1
            if result.candidate_is_target:
                self.target_retrieved_count += 1
        else:
            self.negative_world_count += 1
            score = result.candidate_relevant_evidence
            self.negative_candidate_sum += score
            self.negative_candidate_count += 1
            self.max_negative_candidate = (
                score
                if self.max_negative_candidate is None
                else max(self.max_negative_candidate, score)
            )

        if result.strongest_distractor_relevant_evidence is not None:
            self.distractor_score_sum += result.strongest_distractor_relevant_evidence
            self.distractor_score_count += 1

    def finish(self) -> ScopeConditionSummary:
        return ScopeConditionSummary(
            split=self.split,
            mode=self.mode,
            width=self.width,
            world_count=self.world_count,
            positive_world_count=self.positive_world_count,
            negative_world_count=self.negative_world_count,
            target_inspected_count=self.target_inspected_count,
            target_retrieved_count=self.target_retrieved_count,
            target_coverage_rate=_rate(
                self.target_inspected_count, self.positive_world_count
            ),
            target_retrieval_rate=_rate(
                self.target_retrieved_count, self.positive_world_count
            ),
            retrieval_given_inspected=(
                _rate(self.target_retrieved_count, self.target_inspected_count)
                if self.target_inspected_count
                else None
            ),
            mean_target_rank_when_inspected=_running_mean(
                self.target_rank_sum, self.target_rank_count
            ),
            mean_target_relevant_evidence_when_inspected=_running_mean(
                self.target_score_sum, self.target_score_count
            ),
            mean_strongest_distractor_relevant_evidence=_running_mean(
                self.distractor_score_sum, self.distractor_score_count
            ),
            mean_target_minus_distractor_evidence=_running_mean(
                self.target_gap_sum, self.target_gap_count
            ),
            mean_candidate_relevant_evidence_positive=_running_mean(
                self.positive_candidate_sum, self.positive_candidate_count
            ),
            mean_candidate_relevant_evidence_negative=_running_mean(
                self.negative_candidate_sum, self.negative_candidate_count
            ),
            max_candidate_relevant_evidence_negative=self.max_negative_candidate,
        )


class ScopeMetricsAccumulator:
    """Bounded streaming condition summaries for arbitrarily many worlds."""

    def __init__(self) -> None:
        self._conditions: dict[
            tuple[str, ScopeWorkerMode, int], _RunningCondition
        ] = {}

    def add(self, evaluation: ScopeEvaluation) -> None:
        evaluation.validate()
        key = (evaluation.split, evaluation.mode, evaluation.width)
        condition = self._conditions.get(key)
        if condition is None:
            condition = _RunningCondition(*key)
            self._conditions[key] = condition
        condition.add(evaluation)

    def summaries(self) -> tuple[ScopeConditionSummary, ...]:
        return tuple(
            self._conditions[key].finish()
            for key in sorted(
                self._conditions,
                key=lambda value: (value[0], value[1].value, value[2]),
            )
        )


def summarize_scope_evaluations(
    evaluations: Sequence[ScopeEvaluation],
) -> tuple[ScopeConditionSummary, ...]:
    accumulator = ScopeMetricsAccumulator()
    for evaluation in evaluations:
        accumulator.add(evaluation)
    return accumulator.summaries()


def _rate(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def _running_mean(total: float, count: int) -> float | None:
    return total / count if count else None
