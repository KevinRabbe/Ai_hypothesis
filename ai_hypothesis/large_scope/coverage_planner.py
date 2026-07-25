"""Coverage-aware scope planning for persistent large-scope experiments.

This module is benchmark-specific policy over generic runtime projections. It does not
change Scheduler v0 and does not teach the runtime what a benchmark window means.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Callable

from ai_hypothesis.runtime import (
    ProjectedState,
    SchedulerDecision,
    SchedulerSignals,
    ScopeCoverageProjector,
    SQLiteResearchLedger,
    WorkPreparation,
    WorkPreparationBatch,
)

from .evaluate import ScopeWorkerMode
from .relevance import LargeScopeRelevanceSample, inspection_order
from .runtime_bridge import large_scope_region_id


@dataclass(frozen=True, slots=True)
class CoverageScopePlan:
    thread_id: str
    width: int
    selected_window_indices: tuple[int, ...]
    selected_region_ids: tuple[str, ...]
    expected_region_count: int
    resolved_region_count: int
    missing_region_count: int

    @property
    def missing_coverage(self) -> float:
        if self.expected_region_count <= 0:
            return 0.0
        return self.missing_region_count / self.expected_region_count


class CoverageAwareScopePlanner:
    """Choose under-covered source regions from durable attempt history.

    Selection order is deterministic:
    1. never-attempted unresolved regions;
    2. attempted but still unresolved regions (for example crashed attempts);
    3. already resolved regions with the fewest resolved inspections.

    Ties retain the benchmark's deterministic inspection order. This maximizes new scope
    before retrying aborted work and only turns to redundancy after missing scope is
    exhausted or the requested width is larger than the remaining missing set.
    """

    def __init__(
        self,
        ledger: SQLiteResearchLedger,
        sample: LargeScopeRelevanceSample,
        mode: ScopeWorkerMode | str,
        *,
        projector: ScopeCoverageProjector | None = None,
    ) -> None:
        sample.validate()
        self.ledger = ledger
        self.sample = sample
        self.mode = ScopeWorkerMode(mode)
        self.projector = projector or ScopeCoverageProjector()
        self._inspection_order = inspection_order(
            sample.seed,
            sample.config.window_count,
            split=sample.split,
        )
        self._region_by_window = tuple(
            large_scope_region_id(sample, index)
            for index in range(sample.config.window_count)
        )
        self._window_by_region = {
            region_id: index for index, region_id in enumerate(self._region_by_window)
        }

    @property
    def expected_region_ids(self) -> tuple[str, ...]:
        return tuple(self._region_by_window[index] for index in self._inspection_order)

    def coverage_for(self, thread_id: str):
        return self.projector.for_thread(
            self.ledger.read_all_events(),
            thread_id,
        )

    def missing_region_ids(self, thread_id: str) -> tuple[str, ...]:
        coverage = self.coverage_for(thread_id)
        missing = set(
            coverage.missing_region_ids(
                self.expected_region_ids,
                require_resolved=True,
            )
        )
        return tuple(
            region_id for region_id in self.expected_region_ids if region_id in missing
        )

    def plan(self, thread_id: str, width: int) -> CoverageScopePlan:
        if not thread_id or not thread_id.strip():
            raise ValueError("thread_id must be non-empty")
        if width <= 0:
            raise ValueError("width must be positive")
        if width > self.sample.config.window_count:
            raise ValueError("width cannot exceed large-scope window_count")

        coverage = self.coverage_for(thread_id)
        by_region = {region.region_id: region for region in coverage.regions}

        def rank(window_index: int) -> tuple[int, int, int, int]:
            region_id = self._region_by_window[window_index]
            region = by_region.get(region_id)
            if region is None:
                # New scope is always preferred.
                return (0, 0, 0, self._inspection_order.index(window_index))
            if region.resolved_attempt_count == 0:
                # Aborted/unresolved work is retried only after never-seen scope.
                return (
                    1,
                    region.started_attempt_count,
                    region.aborted_attempt_count,
                    self._inspection_order.index(window_index),
                )
            # Once all missing work is exhausted, distribute redundancy toward the
            # least independently resolved regions first.
            return (
                2,
                region.resolved_attempt_count,
                region.started_attempt_count,
                self._inspection_order.index(window_index),
            )

        selected = tuple(sorted(self._inspection_order, key=rank)[:width])
        selected_regions = tuple(self._region_by_window[index] for index in selected)
        resolved = set(coverage.resolved_region_ids)
        resolved_count = sum(region_id in resolved for region_id in self.expected_region_ids)
        expected_count = len(self.expected_region_ids)
        return CoverageScopePlan(
            thread_id=thread_id,
            width=width,
            selected_window_indices=selected,
            selected_region_ids=selected_regions,
            expected_region_count=expected_count,
            resolved_region_count=resolved_count,
            missing_region_count=expected_count - resolved_count,
        )

    def augment_signals(
        self,
        state: ProjectedState,
        base: SchedulerSignals | None = None,
    ) -> SchedulerSignals:
        """Inject only observed missing-coverage pressure; preserve stronger caller input."""

        signals = base or SchedulerSignals()
        plan = self.plan(state.thread_id, 1)
        missing_pressure = plan.missing_coverage
        if missing_pressure <= signals.missing_coverage:
            return signals
        return replace(signals, missing_coverage=missing_pressure)

    def signal_provider(
        self,
        base_provider: Callable[[ProjectedState], SchedulerSignals] | None = None,
    ) -> Callable[[ProjectedState], SchedulerSignals]:
        def provide(state: ProjectedState) -> SchedulerSignals:
            base = base_provider(state) if base_provider is not None else SchedulerSignals()
            return self.augment_signals(state, base)

        return provide

    def __call__(
        self,
        state: ProjectedState,
        decision: SchedulerDecision,
    ) -> WorkPreparation | WorkPreparationBatch:
        plan = self.plan(state.thread_id, decision.width)
        items = tuple(
            self._preparation_for_window(window_index)
            for window_index in plan.selected_window_indices
        )
        if decision.width == 1:
            return items[0]
        return WorkPreparationBatch(items=items)

    def _preparation_for_window(self, window_index: int) -> WorkPreparation:
        window = self.sample.windows[window_index]
        region_id = self._region_by_window[window_index]
        return WorkPreparation(
            context={
                "large_scope_features": window.features,
                "large_scope_mask": window.mask,
                "large_scope_split": self.sample.split,
                "large_scope_world_seed": self.sample.seed,
                "large_scope_window_index": window_index,
                "large_scope_window_seed": self.sample.window_seeds[window_index],
                "large_scope_mode": self.mode.value,
            },
            reference_ids=(region_id,),
            scope_region_ids=(region_id,),
        )
