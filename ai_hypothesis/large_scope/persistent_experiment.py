"""Equal-budget persistent execution harness for the large-scope relevance workload.

The harness is an experiment-control layer, not a learned/adaptive scheduler. It uses
fixed per-step width plus the coverage-aware scope planner so persistent-runtime costs
and later routing policies can be compared against the direct fixed-prefix benchmark.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

from ai_hypothesis.runtime import (
    RuntimeControlLoop,
    SchedulerAction,
    SchedulerSignals,
    ScopeCoverageProjector,
    SQLiteResearchLedger,
    ThreadStateProjector,
    TracingScheduler,
)
from ai_hypothesis.step02.evidence import AggregationConfig

from .coverage_planner import CoverageAwareScopePlanner
from .evaluate import ScopeWorkerMode, SelectedWorkerBank, WindowEvidence
from .relevance import LARGE_SCOPE_BENCHMARK_VERSION, LargeScopeRelevanceSample, diverse_worker_indices, same_worker_indices
from .runtime_bridge import FixedScopeScheduler, LargeScopeRuntimeWorkerBank, large_scope_region_id


_RUNTIME_EVIDENCE_KIND = "LARGE_SCOPE_RELEVANCE_WINDOW"


class PersistentScopeWorkerSelector:
    """Restart-safe benchmark worker assignment for repeated persistent steps.

    `same_worker` always reuses one deterministic checkpoint.

    `diverse_workers` uses one deterministic cyclic ordering of all checkpoints. The
    next batch starts after `previous_worker_id`, which RuntimeControlLoop derives from
    durable ATTEMPT_STARTED history. No hidden round counter is required.
    """

    def __init__(
        self,
        runtime_bank: LargeScopeRuntimeWorkerBank,
        sample: LargeScopeRelevanceSample,
        mode: ScopeWorkerMode | str,
    ) -> None:
        sample.validate()
        self.runtime_bank = runtime_bank
        self.sample = sample
        self.mode = ScopeWorkerMode(mode)
        population_width = runtime_bank.bank.population_width
        if self.mode is ScopeWorkerMode.SAME_WORKER:
            index = same_worker_indices(
                seed=sample.seed,
                width=1,
                population_width=population_width,
                split=sample.split,
            )[0]
            self._ordered_worker_ids = (runtime_bank.worker_id_for_index(index),)
            self._allocation_width_limit = sample.config.window_count
        else:
            indices = diverse_worker_indices(
                seed=sample.seed,
                width=population_width,
                population_width=population_width,
                split=sample.split,
            )
            self._ordered_worker_ids = tuple(
                runtime_bank.worker_id_for_index(index) for index in indices
            )
            self._allocation_width_limit = min(
                sample.config.window_count,
                population_width,
            )
        self._position = {
            worker_id: index for index, worker_id in enumerate(self._ordered_worker_ids)
        }

    @property
    def population_width(self) -> int:
        """Maximum simultaneous benchmark attempt width, not distinct checkpoint count."""

        return self._allocation_width_limit

    def choose(
        self,
        action: SchedulerAction,
        *,
        previous_worker_id: str | None,
    ) -> str:
        return self.choose_many(
            action,
            previous_worker_id=previous_worker_id,
            count=1,
        )[0]

    def choose_many(
        self,
        _action: SchedulerAction,
        *,
        previous_worker_id: str | None,
        count: int,
    ) -> tuple[str, ...]:
        if count <= 0:
            raise ValueError("worker selection count must be positive")
        if count > self.population_width:
            raise ValueError("requested width exceeds persistent scope allocation limit")

        if self.mode is ScopeWorkerMode.SAME_WORKER:
            return (self._ordered_worker_ids[0],) * count

        if previous_worker_id is None:
            start = 0
        else:
            try:
                start = (self._position[previous_worker_id] + 1) % len(
                    self._ordered_worker_ids
                )
            except KeyError as error:
                raise ValueError(
                    "previous worker is outside the persistent diverse-worker plan"
                ) from error
        return tuple(
            self._ordered_worker_ids[(start + offset) % len(self._ordered_worker_ids)]
            for offset in range(count)
        )


@dataclass(frozen=True, slots=True)
class PersistentScopeEvaluation:
    thread_id: str
    split: str
    seed: int
    mode: ScopeWorkerMode
    step_width: int
    step_count: int
    attempt_count: int
    evidence_count: int
    scheduler_decision_count: int
    distinct_worker_count: int
    resolved_region_count: int
    expected_region_count: int
    coverage_fraction: float
    window_evidence: tuple[WindowEvidence, ...]
    target_present: bool
    target_index: int | None
    target_resolved: bool
    candidate_window_index: int | None
    candidate_is_target: bool
    candidate_relevant_evidence: float | None
    target_relevant_evidence: float | None
    target_rank: int | None
    strongest_distractor_relevant_evidence: float | None
    ledger_event_count: int

    @property
    def requested_window_evaluations(self) -> int:
        return self.step_width * self.step_count

    @property
    def duplicate_evidence_count(self) -> int:
        return self.evidence_count - len(
            {row.window_index for row in self.window_evidence}
        )


class PersistentScopeEvaluationProjector:
    """Project threshold-free persistent benchmark observations from the Research Ledger."""

    def __init__(
        self,
        sample: LargeScopeRelevanceSample,
        mode: ScopeWorkerMode | str,
        *,
        coverage_projector: ScopeCoverageProjector | None = None,
    ) -> None:
        sample.validate()
        self.sample = sample
        self.mode = ScopeWorkerMode(mode)
        self.coverage_projector = coverage_projector or ScopeCoverageProjector()
        self._expected_region_ids = tuple(
            large_scope_region_id(sample, index)
            for index in range(sample.config.window_count)
        )

    def project(
        self,
        events: Sequence,
        *,
        thread_id: str,
        step_width: int,
    ) -> PersistentScopeEvaluation:
        if not thread_id or not thread_id.strip():
            raise ValueError("thread_id must be non-empty")
        if step_width <= 0:
            raise ValueError("step_width must be positive")

        thread_events = tuple(event for event in events if event.thread_id == thread_id)
        attempts = [
            event
            for event in thread_events
            if event.event_type == "ATTEMPT_STARTED"
            and bool(event.payload.get("scope_region_ids"))
        ]
        decisions = [
            event
            for event in thread_events
            if event.event_type == "SCHEDULER_DECISION_RECORDED"
        ]
        worker_ids: list[str] = []
        for event in attempts:
            worker_id = event.payload.get("worker_id")
            if not isinstance(worker_id, str) or not worker_id:
                raise ValueError("scoped attempt is missing worker_id")
            if worker_id not in worker_ids:
                worker_ids.append(worker_id)

        rows: list[WindowEvidence] = []
        observation_sequences: list[int] = []
        for event in thread_events:
            if event.event_type != "EVIDENCE_ADDED":
                continue
            if event.payload.get("kind") != _RUNTIME_EVIDENCE_KIND:
                continue
            data = event.payload.get("data")
            if not isinstance(data, dict):
                raise ValueError("large-scope evidence data must be a mapping")
            if data.get("benchmark_version") != LARGE_SCOPE_BENCHMARK_VERSION:
                raise ValueError("large-scope evidence benchmark version mismatch")
            if data.get("split") != self.sample.split or data.get("world_seed") != self.sample.seed:
                raise ValueError("large-scope evidence belongs to a different benchmark world")
            if data.get("mode") != self.mode.value:
                raise ValueError("large-scope evidence worker mode mismatch")

            window_index = _required_non_negative_int(data, "window_index")
            worker_index = _required_non_negative_int(data, "worker_index")
            if window_index >= self.sample.config.window_count:
                raise ValueError("large-scope evidence window index is outside the world")
            local_label = _required_text(data, "local_label")
            relevant = _required_finite_float(data, "relevant_evidence")
            not_relevant = _required_finite_float(data, "not_relevant_evidence")
            invalid_mass = _required_finite_float(data, "invalid_label_mass")
            top_margin = _required_finite_float(data, "top_margin")
            uncertainty = event.payload.get("uncertainty")
            if isinstance(uncertainty, bool) or not isinstance(uncertainty, (int, float)):
                raise ValueError("large-scope evidence uncertainty must be numeric")
            uncertainty = float(uncertainty)
            if not math.isfinite(uncertainty) or not 0.0 <= uncertainty <= 1.0:
                raise ValueError("large-scope evidence uncertainty must be finite in [0, 1]")

            rows.append(
                WindowEvidence(
                    window_index=window_index,
                    worker_index=worker_index,
                    local_label=local_label,
                    relevant_evidence=relevant,
                    not_relevant_evidence=not_relevant,
                    uncertainty_probability=uncertainty,
                    invalid_label_mass=invalid_mass,
                    top_margin=top_margin,
                )
            )
            observation_sequences.append(event.sequence)

        coverage = self.coverage_projector.for_thread(events, thread_id)
        resolved = set(coverage.resolved_region_ids)
        resolved_count = sum(
            region_id in resolved for region_id in self._expected_region_ids
        )
        expected_count = len(self._expected_region_ids)
        coverage_fraction = resolved_count / expected_count if expected_count else 1.0

        region_best: dict[int, tuple[float, int]] = {}
        for row, sequence in zip(rows, observation_sequences, strict=True):
            current = region_best.get(row.window_index)
            candidate = (row.relevant_evidence, -sequence)
            if current is None or candidate > current:
                region_best[row.window_index] = candidate

        candidate_window_index: int | None = None
        candidate_score: float | None = None
        if region_best:
            candidate_window_index = max(
                region_best,
                key=lambda index: region_best[index],
            )
            candidate_score = region_best[candidate_window_index][0]

        target_region_id = (
            large_scope_region_id(self.sample, self.sample.target_index)
            if self.sample.target_index is not None
            else None
        )
        target_resolved = target_region_id in resolved if target_region_id is not None else False
        target_score: float | None = None
        target_rank: int | None = None
        if self.sample.target_index is not None and self.sample.target_index in region_best:
            target_score = region_best[self.sample.target_index][0]
            target_rank = 1 + sum(
                score_tuple[0] > target_score
                for index, score_tuple in region_best.items()
                if index != self.sample.target_index
            )

        distractor_scores = [
            score_tuple[0]
            for index, score_tuple in region_best.items()
            if index != self.sample.target_index
        ]
        strongest_distractor = max(distractor_scores) if distractor_scores else None

        if len(attempts) % step_width != 0:
            raise ValueError("persistent scope attempt count is not divisible by step_width")
        step_count = len(attempts) // step_width
        return PersistentScopeEvaluation(
            thread_id=thread_id,
            split=self.sample.split,
            seed=self.sample.seed,
            mode=self.mode,
            step_width=step_width,
            step_count=step_count,
            attempt_count=len(attempts),
            evidence_count=len(rows),
            scheduler_decision_count=len(decisions),
            distinct_worker_count=len(worker_ids),
            resolved_region_count=resolved_count,
            expected_region_count=expected_count,
            coverage_fraction=coverage_fraction,
            window_evidence=tuple(rows),
            target_present=self.sample.target_present,
            target_index=self.sample.target_index,
            target_resolved=target_resolved,
            candidate_window_index=candidate_window_index,
            candidate_is_target=(
                self.sample.target_present
                and candidate_window_index is not None
                and candidate_window_index == self.sample.target_index
            ),
            candidate_relevant_evidence=candidate_score,
            target_relevant_evidence=target_score,
            target_rank=target_rank,
            strongest_distractor_relevant_evidence=strongest_distractor,
            ledger_event_count=len(thread_events),
        )


class PersistentScopeExperiment:
    """Run a fixed number of persistent coverage-planned steps on a dedicated thread."""

    def __init__(
        self,
        *,
        ledger: SQLiteResearchLedger,
        sample: LargeScopeRelevanceSample,
        bank: SelectedWorkerBank,
        mode: ScopeWorkerMode | str,
        step_width: int,
        thread_id: str = "large-scope-runtime",
        evidence_config: AggregationConfig = AggregationConfig(),
    ) -> None:
        sample.validate()
        if step_width <= 0:
            raise ValueError("step_width must be positive")
        if step_width > sample.config.window_count:
            raise ValueError("step_width cannot exceed large-scope window_count")
        if not thread_id or not thread_id.strip():
            raise ValueError("thread_id must be non-empty")

        self.ledger = ledger
        self.sample = sample
        self.mode = ScopeWorkerMode(mode)
        self.step_width = step_width
        self.thread_id = thread_id
        self.runtime_bank = LargeScopeRuntimeWorkerBank(bank, evidence_config)
        self.worker_selector = PersistentScopeWorkerSelector(
            self.runtime_bank,
            sample,
            self.mode,
        )
        if step_width > self.worker_selector.population_width:
            raise ValueError("step_width exceeds persistent worker allocation limit")
        self.planner = CoverageAwareScopePlanner(
            ledger,
            sample,
            self.mode,
        )
        self.loop = RuntimeControlLoop(
            ledger=ledger,
            scheduler=TracingScheduler(ledger, FixedScopeScheduler(step_width)),
            worker_bank=self.runtime_bank,
            worker_ids=self.runtime_bank.worker_ids,
            worker_selector=self.worker_selector,
        )
        self._ensure_thread()

    def run_steps(self, step_count: int) -> PersistentScopeEvaluation:
        if step_count <= 0:
            raise ValueError("step_count must be positive")
        for _ in range(step_count):
            self.loop.run_once(
                signal_provider=self.planner.signal_provider(
                    lambda _state: SchedulerSignals(
                        importance=1.0,
                        recent_progress=1.0,
                    )
                ),
                context_provider=self.planner,
            )
        return self.evaluate()

    def evaluate(self) -> PersistentScopeEvaluation:
        return PersistentScopeEvaluationProjector(
            self.sample,
            self.mode,
        ).project(
            self.ledger.read_all_events(),
            thread_id=self.thread_id,
            step_width=self.step_width,
        )

    def _ensure_thread(self) -> None:
        states = ThreadStateProjector().project_all(self.ledger.read_all_events())
        by_id = {state.thread_id: state for state in states}
        other_active = [
            state.thread_id
            for state in states
            if state.thread_id != self.thread_id and state.status != "COMPLETE"
        ]
        if other_active:
            raise ValueError(
                "persistent scope experiment requires a dedicated ledger or no other active threads"
            )

        expected_metadata = {
            "benchmark_version": LARGE_SCOPE_BENCHMARK_VERSION,
            "large_scope_split": self.sample.split,
            "large_scope_world_seed": self.sample.seed,
            "large_scope_mode": self.mode.value,
            "large_scope_window_count": self.sample.config.window_count,
        }
        existing = by_id.get(self.thread_id)
        if existing is None:
            self.loop.create_thread(
                thread_id=self.thread_id,
                objective="Evaluate persistent large-scope relevance",
                metadata=expected_metadata,
            )
            return
        if existing.status == "COMPLETE" or existing.merged_into_thread_id is not None:
            raise ValueError("persistent scope benchmark thread is terminal")
        for key, expected in expected_metadata.items():
            if existing.metadata.get(key) != expected:
                raise ValueError(
                    f"existing persistent scope thread metadata mismatch for {key!r}"
                )


def _required_text(mapping: dict, key: str) -> str:
    value = mapping.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"large-scope evidence field {key!r} must be text")
    return value


def _required_non_negative_int(mapping: dict, key: str) -> int:
    value = mapping.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(
            f"large-scope evidence field {key!r} must be a non-negative integer"
        )
    return value


def _required_finite_float(mapping: dict, key: str) -> float:
    value = mapping.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"large-scope evidence field {key!r} must be numeric")
    resolved = float(value)
    if not math.isfinite(resolved):
        raise ValueError(f"large-scope evidence field {key!r} must be finite")
    return resolved
