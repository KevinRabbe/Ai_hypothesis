"""Normalize direct and persistent large-scope outputs onto one evaluation surface."""

from __future__ import annotations

from dataclasses import dataclass

from .evaluate import ScopeEvaluation
from .persistent_experiment import PersistentScopeEvaluation


@dataclass(frozen=True, slots=True)
class ScopeEquivalenceObservation:
    split: str
    seed: int
    mode: str
    width: int
    max_abs_evidence_delta: float
    candidate_match: bool
    target_rank_match: bool
    target_evidence_presence_match: bool

    def within_tolerance(self, tolerance: float) -> bool:
        if tolerance < 0.0:
            raise ValueError("tolerance must be non-negative")
        return (
            self.max_abs_evidence_delta <= tolerance
            and self.candidate_match
            and self.target_rank_match
            and self.target_evidence_presence_match
        )


@dataclass(frozen=True, slots=True)
class ScopeEquivalenceSummary:
    world_count: int
    mismatch_count: int
    candidate_mismatch_count: int
    target_rank_mismatch_count: int
    target_evidence_presence_mismatch_count: int
    max_abs_evidence_delta: float
    tolerance: float

    @property
    def passed(self) -> bool:
        return self.mismatch_count == 0


class ScopeEquivalenceAccumulator:
    def __init__(self, *, tolerance: float = 1e-5) -> None:
        if tolerance < 0.0:
            raise ValueError("tolerance must be non-negative")
        self.tolerance = tolerance
        self.world_count = 0
        self.mismatch_count = 0
        self.candidate_mismatch_count = 0
        self.target_rank_mismatch_count = 0
        self.target_evidence_presence_mismatch_count = 0
        self.max_abs_evidence_delta = 0.0

    def add(self, observation: ScopeEquivalenceObservation) -> None:
        self.world_count += 1
        self.max_abs_evidence_delta = max(
            self.max_abs_evidence_delta,
            observation.max_abs_evidence_delta,
        )
        if not observation.candidate_match:
            self.candidate_mismatch_count += 1
        if not observation.target_rank_match:
            self.target_rank_mismatch_count += 1
        if not observation.target_evidence_presence_match:
            self.target_evidence_presence_mismatch_count += 1
        if not observation.within_tolerance(self.tolerance):
            self.mismatch_count += 1

    def summary(self) -> ScopeEquivalenceSummary:
        return ScopeEquivalenceSummary(
            world_count=self.world_count,
            mismatch_count=self.mismatch_count,
            candidate_mismatch_count=self.candidate_mismatch_count,
            target_rank_mismatch_count=self.target_rank_mismatch_count,
            target_evidence_presence_mismatch_count=(
                self.target_evidence_presence_mismatch_count
            ),
            max_abs_evidence_delta=self.max_abs_evidence_delta,
            tolerance=self.tolerance,
        )


def scope_evaluation_from_persistent(
    result: PersistentScopeEvaluation,
) -> ScopeEvaluation:
    """Project a non-redundant persistent baseline onto the direct metric contract."""

    width = len(result.window_evidence)
    if width <= 0:
        raise ValueError("persistent result contains no window evidence")
    if result.evidence_count != result.attempt_count:
        raise ValueError("persistent baseline requires one evidence record per attempt")
    if width != result.evidence_count:
        raise ValueError("persistent window-evidence count does not match evidence count")

    window_indices = tuple(row.window_index for row in result.window_evidence)
    if len(set(window_indices)) != len(window_indices):
        raise ValueError(
            "persistent-to-direct projection requires a non-redundant scope budget"
        )
    worker_indices = tuple(row.worker_index for row in result.window_evidence)
    if result.candidate_window_index is None:
        raise ValueError("persistent result has no candidate window")
    if result.candidate_relevant_evidence is None:
        raise ValueError("persistent result has no candidate evidence")

    projected = ScopeEvaluation(
        split=result.split,
        seed=result.seed,
        width=width,
        mode=result.mode,
        inspected_window_indices=window_indices,
        worker_indices=worker_indices,
        target_present=result.target_present,
        target_index=result.target_index,
        target_inspected=result.target_resolved,
        candidate_window_index=result.candidate_window_index,
        candidate_is_target=result.candidate_is_target,
        candidate_relevant_evidence=result.candidate_relevant_evidence,
        target_relevant_evidence=result.target_relevant_evidence,
        target_rank=result.target_rank,
        strongest_distractor_relevant_evidence=(
            result.strongest_distractor_relevant_evidence
        ),
        window_evidence=result.window_evidence,
    )
    projected.validate()
    return projected


def compare_scope_evaluations(
    direct: ScopeEvaluation,
    persistent: ScopeEvaluation,
) -> ScopeEquivalenceObservation:
    """Require identical experiment structure and measure numeric execution drift."""

    direct.validate()
    persistent.validate()
    for name, left, right in (
        ("split", direct.split, persistent.split),
        ("seed", direct.seed, persistent.seed),
        ("mode", direct.mode, persistent.mode),
        ("width", direct.width, persistent.width),
        (
            "inspected_window_indices",
            direct.inspected_window_indices,
            persistent.inspected_window_indices,
        ),
        ("worker_indices", direct.worker_indices, persistent.worker_indices),
        ("target_present", direct.target_present, persistent.target_present),
        ("target_index", direct.target_index, persistent.target_index),
        ("target_inspected", direct.target_inspected, persistent.target_inspected),
    ):
        if left != right:
            raise ValueError(f"direct/persistent structural mismatch for {name}")

    if len(direct.window_evidence) != len(persistent.window_evidence):
        raise ValueError("direct/persistent window evidence counts differ")

    max_delta = 0.0
    for direct_row, persistent_row in zip(
        direct.window_evidence,
        persistent.window_evidence,
        strict=True,
    ):
        if direct_row.window_index != persistent_row.window_index:
            raise ValueError("direct/persistent evidence window order differs")
        if direct_row.worker_index != persistent_row.worker_index:
            raise ValueError("direct/persistent evidence worker order differs")
        if direct_row.local_label != persistent_row.local_label:
            raise ValueError("direct/persistent local decoded label differs")
        for left, right in (
            (direct_row.relevant_evidence, persistent_row.relevant_evidence),
            (direct_row.not_relevant_evidence, persistent_row.not_relevant_evidence),
            (
                direct_row.uncertainty_probability,
                persistent_row.uncertainty_probability,
            ),
            (direct_row.invalid_label_mass, persistent_row.invalid_label_mass),
            (direct_row.top_margin, persistent_row.top_margin),
        ):
            max_delta = max(max_delta, abs(left - right))

    direct_target_present = direct.target_relevant_evidence is not None
    persistent_target_present = persistent.target_relevant_evidence is not None
    if direct_target_present and persistent_target_present:
        assert direct.target_relevant_evidence is not None
        assert persistent.target_relevant_evidence is not None
        max_delta = max(
            max_delta,
            abs(
                direct.target_relevant_evidence
                - persistent.target_relevant_evidence
            ),
        )
    max_delta = max(
        max_delta,
        abs(
            direct.candidate_relevant_evidence
            - persistent.candidate_relevant_evidence
        ),
    )
    if (
        direct.strongest_distractor_relevant_evidence is not None
        and persistent.strongest_distractor_relevant_evidence is not None
    ):
        max_delta = max(
            max_delta,
            abs(
                direct.strongest_distractor_relevant_evidence
                - persistent.strongest_distractor_relevant_evidence
            ),
        )

    return ScopeEquivalenceObservation(
        split=direct.split,
        seed=direct.seed,
        mode=direct.mode.value,
        width=direct.width,
        max_abs_evidence_delta=max_delta,
        candidate_match=(
            direct.candidate_window_index == persistent.candidate_window_index
            and direct.candidate_is_target == persistent.candidate_is_target
        ),
        target_rank_match=direct.target_rank == persistent.target_rank,
        target_evidence_presence_match=(
            direct_target_present == persistent_target_present
        ),
    )
