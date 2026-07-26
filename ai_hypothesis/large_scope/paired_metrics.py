"""Bounded paired summaries for same-scope repeated-weight versus diverse-weight execution."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

from .evaluate import ScopeEvaluation, ScopeWorkerMode


@dataclass(frozen=True, slots=True)
class ScopePairedSummary:
    """Streaming diverse-minus-same comparison for one split/width."""

    split: str
    width: int
    pair_count: int
    positive_world_count: int
    negative_world_count: int
    target_inspected_count: int
    same_target_retrieved_count: int
    diverse_target_retrieved_count: int
    both_retrieved_count: int
    same_only_retrieved_count: int
    diverse_only_retrieved_count: int
    neither_retrieved_count: int
    retrieval_given_inspected_same: float | None
    retrieval_given_inspected_diverse: float | None
    retrieval_given_inspected_delta: float | None
    retrieval_discordant_count: int
    exact_retrieval_discordance_p_value: float | None
    mean_target_rank_delta_when_inspected: float | None
    se_target_rank_delta_when_inspected: float | None
    mean_target_relevant_evidence_delta_when_inspected: float | None
    se_target_relevant_evidence_delta_when_inspected: float | None
    mean_strongest_distractor_relevant_evidence_delta: float | None
    se_strongest_distractor_relevant_evidence_delta: float | None
    mean_target_minus_distractor_gap_delta_when_inspected: float | None
    se_target_minus_distractor_gap_delta_when_inspected: float | None
    mean_candidate_relevant_evidence_positive_delta: float | None
    se_candidate_relevant_evidence_positive_delta: float | None
    mean_candidate_relevant_evidence_negative_delta: float | None
    se_candidate_relevant_evidence_negative_delta: float | None

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["delta_definition"] = "diverse_workers_minus_same_worker"
        return payload


@dataclass(slots=True)
class _RunningMoment:
    count: int = 0
    mean: float = 0.0
    m2: float = 0.0

    def add(self, value: float) -> None:
        if not math.isfinite(value):
            raise ValueError("paired metric delta must be finite")
        self.count += 1
        delta = value - self.mean
        self.mean += delta / self.count
        self.m2 += delta * (value - self.mean)

    def mean_or_none(self) -> float | None:
        return self.mean if self.count else None

    def standard_error_or_none(self) -> float | None:
        if self.count <= 1:
            return None
        sample_variance = self.m2 / (self.count - 1)
        return math.sqrt(sample_variance / self.count)


@dataclass(slots=True)
class _RunningPair:
    split: str
    width: int
    pair_count: int = 0
    positive_world_count: int = 0
    negative_world_count: int = 0
    target_inspected_count: int = 0
    same_target_retrieved_count: int = 0
    diverse_target_retrieved_count: int = 0
    both_retrieved_count: int = 0
    same_only_retrieved_count: int = 0
    diverse_only_retrieved_count: int = 0
    neither_retrieved_count: int = 0
    target_rank_delta: _RunningMoment | None = None
    target_score_delta: _RunningMoment | None = None
    distractor_score_delta: _RunningMoment | None = None
    target_gap_delta: _RunningMoment | None = None
    positive_candidate_delta: _RunningMoment | None = None
    negative_candidate_delta: _RunningMoment | None = None

    def __post_init__(self) -> None:
        self.target_rank_delta = _RunningMoment()
        self.target_score_delta = _RunningMoment()
        self.distractor_score_delta = _RunningMoment()
        self.target_gap_delta = _RunningMoment()
        self.positive_candidate_delta = _RunningMoment()
        self.negative_candidate_delta = _RunningMoment()

    def add(self, same: ScopeEvaluation, diverse: ScopeEvaluation) -> None:
        _validate_pair(same, diverse)
        self.pair_count += 1
        candidate_delta = (
            diverse.candidate_relevant_evidence - same.candidate_relevant_evidence
        )

        assert self.positive_candidate_delta is not None
        assert self.negative_candidate_delta is not None
        assert self.target_rank_delta is not None
        assert self.target_score_delta is not None
        assert self.distractor_score_delta is not None
        assert self.target_gap_delta is not None

        same_distractor = same.strongest_distractor_relevant_evidence
        diverse_distractor = diverse.strongest_distractor_relevant_evidence
        if (same_distractor is None) != (diverse_distractor is None):
            raise ValueError("paired conditions disagree on distractor availability")
        if same_distractor is not None and diverse_distractor is not None:
            self.distractor_score_delta.add(diverse_distractor - same_distractor)

        if same.target_present:
            self.positive_world_count += 1
            self.positive_candidate_delta.add(candidate_delta)
            if same.target_inspected:
                self.target_inspected_count += 1
                same_retrieved = same.candidate_is_target
                diverse_retrieved = diverse.candidate_is_target
                self.same_target_retrieved_count += int(same_retrieved)
                self.diverse_target_retrieved_count += int(diverse_retrieved)
                if same_retrieved and diverse_retrieved:
                    self.both_retrieved_count += 1
                elif same_retrieved:
                    self.same_only_retrieved_count += 1
                elif diverse_retrieved:
                    self.diverse_only_retrieved_count += 1
                else:
                    self.neither_retrieved_count += 1

                if same.target_rank is None or diverse.target_rank is None:
                    raise ValueError("inspected paired target requires both target ranks")
                if (
                    same.target_relevant_evidence is None
                    or diverse.target_relevant_evidence is None
                ):
                    raise ValueError(
                        "inspected paired target requires both target evidence values"
                    )
                self.target_rank_delta.add(diverse.target_rank - same.target_rank)
                self.target_score_delta.add(
                    diverse.target_relevant_evidence - same.target_relevant_evidence
                )

                if same_distractor is not None and diverse_distractor is not None:
                    same_gap = same.target_relevant_evidence - same_distractor
                    diverse_gap = diverse.target_relevant_evidence - diverse_distractor
                    self.target_gap_delta.add(diverse_gap - same_gap)
        else:
            self.negative_world_count += 1
            self.negative_candidate_delta.add(candidate_delta)

    def finish(self) -> ScopePairedSummary:
        assert self.target_rank_delta is not None
        assert self.target_score_delta is not None
        assert self.distractor_score_delta is not None
        assert self.target_gap_delta is not None
        assert self.positive_candidate_delta is not None
        assert self.negative_candidate_delta is not None

        same_rate = _optional_rate(
            self.same_target_retrieved_count,
            self.target_inspected_count,
        )
        diverse_rate = _optional_rate(
            self.diverse_target_retrieved_count,
            self.target_inspected_count,
        )
        discordant = self.same_only_retrieved_count + self.diverse_only_retrieved_count
        return ScopePairedSummary(
            split=self.split,
            width=self.width,
            pair_count=self.pair_count,
            positive_world_count=self.positive_world_count,
            negative_world_count=self.negative_world_count,
            target_inspected_count=self.target_inspected_count,
            same_target_retrieved_count=self.same_target_retrieved_count,
            diverse_target_retrieved_count=self.diverse_target_retrieved_count,
            both_retrieved_count=self.both_retrieved_count,
            same_only_retrieved_count=self.same_only_retrieved_count,
            diverse_only_retrieved_count=self.diverse_only_retrieved_count,
            neither_retrieved_count=self.neither_retrieved_count,
            retrieval_given_inspected_same=same_rate,
            retrieval_given_inspected_diverse=diverse_rate,
            retrieval_given_inspected_delta=(
                diverse_rate - same_rate
                if same_rate is not None and diverse_rate is not None
                else None
            ),
            retrieval_discordant_count=discordant,
            exact_retrieval_discordance_p_value=_exact_discordance_p_value(
                self.same_only_retrieved_count,
                self.diverse_only_retrieved_count,
            ),
            mean_target_rank_delta_when_inspected=self.target_rank_delta.mean_or_none(),
            se_target_rank_delta_when_inspected=self.target_rank_delta.standard_error_or_none(),
            mean_target_relevant_evidence_delta_when_inspected=self.target_score_delta.mean_or_none(),
            se_target_relevant_evidence_delta_when_inspected=self.target_score_delta.standard_error_or_none(),
            mean_strongest_distractor_relevant_evidence_delta=self.distractor_score_delta.mean_or_none(),
            se_strongest_distractor_relevant_evidence_delta=self.distractor_score_delta.standard_error_or_none(),
            mean_target_minus_distractor_gap_delta_when_inspected=self.target_gap_delta.mean_or_none(),
            se_target_minus_distractor_gap_delta_when_inspected=self.target_gap_delta.standard_error_or_none(),
            mean_candidate_relevant_evidence_positive_delta=self.positive_candidate_delta.mean_or_none(),
            se_candidate_relevant_evidence_positive_delta=self.positive_candidate_delta.standard_error_or_none(),
            mean_candidate_relevant_evidence_negative_delta=self.negative_candidate_delta.mean_or_none(),
            se_candidate_relevant_evidence_negative_delta=self.negative_candidate_delta.standard_error_or_none(),
        )


class ScopePairedMetricsAccumulator:
    """Pair identical worlds/widths across worker modes with bounded pending state."""

    def __init__(self) -> None:
        self._pending: dict[tuple[str, int, int], ScopeEvaluation] = {}
        self._pairs: dict[tuple[str, int], _RunningPair] = {}

    def add(self, evaluation: ScopeEvaluation) -> None:
        evaluation.validate()
        key = (evaluation.split, evaluation.seed, evaluation.width)
        previous = self._pending.pop(key, None)
        if previous is None:
            self._pending[key] = evaluation
            return
        if previous.mode is evaluation.mode:
            self._pending[key] = previous
            raise ValueError("paired metrics received duplicate worker mode for one world")
        same = (
            previous
            if previous.mode is ScopeWorkerMode.SAME_WORKER
            else evaluation
        )
        diverse = (
            previous
            if previous.mode is ScopeWorkerMode.DIVERSE_WORKERS
            else evaluation
        )
        pair_key = (evaluation.split, evaluation.width)
        running = self._pairs.get(pair_key)
        if running is None:
            running = _RunningPair(*pair_key)
            self._pairs[pair_key] = running
        running.add(same, diverse)

    def summaries(self) -> tuple[ScopePairedSummary, ...]:
        if self._pending:
            raise ValueError(
                "paired metrics cannot finalize with unmatched same/diverse evaluations"
            )
        return tuple(
            self._pairs[key].finish()
            for key in sorted(self._pairs, key=lambda value: (value[0], value[1]))
        )


def _validate_pair(same: ScopeEvaluation, diverse: ScopeEvaluation) -> None:
    if same.mode is not ScopeWorkerMode.SAME_WORKER:
        raise ValueError("paired baseline must use same_worker mode")
    if diverse.mode is not ScopeWorkerMode.DIVERSE_WORKERS:
        raise ValueError("paired treatment must use diverse_workers mode")
    if (same.split, same.seed, same.width) != (
        diverse.split,
        diverse.seed,
        diverse.width,
    ):
        raise ValueError("paired evaluations must refer to the same world and width")
    if same.inspected_window_indices != diverse.inspected_window_indices:
        raise ValueError("paired evaluations must inspect identical windows")
    if (
        same.target_present != diverse.target_present
        or same.target_index != diverse.target_index
        or same.target_inspected != diverse.target_inspected
    ):
        raise ValueError("paired evaluations disagree on frozen target state")


def _optional_rate(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _exact_discordance_p_value(same_only: int, diverse_only: int) -> float | None:
    if same_only < 0 or diverse_only < 0:
        raise ValueError("discordant counts must be non-negative")
    total = same_only + diverse_only
    if total == 0:
        return None
    tail = min(same_only, diverse_only)
    probability = sum(math.comb(total, value) for value in range(tail + 1)) / (2**total)
    return min(1.0, 2.0 * probability)
