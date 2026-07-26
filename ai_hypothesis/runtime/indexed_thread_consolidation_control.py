"""Snapshot-pinned automatic thread-consolidation routing over indexed current state."""

from __future__ import annotations

from dataclasses import replace
from typing import Protocol

from .contracts import LedgerEvent, ProjectedState, SchedulerDecision
from .control import ContextProvider, SignalProvider, WorkPreparation, WorkPreparationBatch
from .indexed_control import IndexedRuntimeSnapshot, IndexedRuntimeSnapshotProvider
from .indexed_thread_consolidation import (
    IndexedThreadConsolidationPlanner,
    IndexedThreadConsolidationPressureProjector,
)
from .knowledge_index import SQLiteIndexedKnowledgeState
from .ledger import SQLiteResearchLedger
from .scheduler import SchedulerSignals
from .thread_consolidation import prepare_thread_consolidation_work


_THREAD_CONSOLIDATION_REASON = "THREAD_CONSOLIDATION"


class CapturedRevisionProvider(Protocol):
    @property
    def current_revision(self) -> int:
        ...


class PinnedIndexedRuntimeSnapshotProvider(IndexedRuntimeSnapshotProvider):
    """Indexed runtime snapshot provider that exposes its last successful frozen revision."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._captured_revision: int | None = None

    @property
    def current_revision(self) -> int:
        if self._captured_revision is None:
            raise RuntimeError("indexed runtime snapshot has not been captured yet")
        return self._captured_revision

    def capture(self) -> IndexedRuntimeSnapshot:
        snapshot = super().capture()
        self._captured_revision = snapshot.revision
        return snapshot


class IndexedThreadConsolidationControlAdapter:
    """Inject consolidation pressure and return a bounded context pinned at signal time."""

    def __init__(
        self,
        *,
        ledger: SQLiteResearchLedger,
        revision_provider: CapturedRevisionProvider,
        signal_fallback: SignalProvider,
        context_fallback: ContextProvider,
        planner: IndexedThreadConsolidationPlanner,
        pressure_projector: IndexedThreadConsolidationPressureProjector,
        knowledge: SQLiteIndexedKnowledgeState,
    ) -> None:
        if planner.ledger is not ledger or pressure_projector.ledger is not ledger:
            raise ValueError("indexed consolidation control must share one Research Ledger")
        if planner.knowledge is not knowledge or pressure_projector.knowledge is not knowledge:
            raise ValueError("indexed consolidation control must share one Knowledge State")
        if planner.lineage is not pressure_projector.lineage:
            raise ValueError("indexed consolidation planner and pressure must share one lineage view")
        if planner.config.minimum_source_deltas != pressure_projector.config.minimum_source_deltas:
            raise ValueError(
                "thread consolidation pressure minimum must match planner readiness minimum"
            )
        self.ledger = ledger
        self.revision_provider = revision_provider
        self.signal_fallback = signal_fallback
        self.context_fallback = context_fallback
        self.planner = planner
        self.pressure_projector = pressure_projector
        self.knowledge = knowledge
        self._cached_revision: int | None = None
        self._cached_overview = None
        self._preparation_by_thread_revision: dict[
            tuple[str, int], WorkPreparation
        ] = {}
        self._owned_route_by_thread_revision: set[tuple[str, int]] = set()

    def signals(self, state: ProjectedState) -> SchedulerSignals:
        state.validate()
        revision = self.revision_provider.current_revision
        self._refresh(revision)
        assert self._cached_overview is not None

        base = self.signal_fallback(state)
        base.validate()
        pressure = self._cached_overview.pressure_for(state.thread_id)
        key = (state.thread_id, state.revision)
        if pressure <= base.synthesis_need:
            self._owned_route_by_thread_revision.discard(key)
            self._preparation_by_thread_revision.pop(key, None)
            return base

        plan = self.planner.plan(sequence=revision, thread_id=state.thread_id)
        if not plan.ready:
            raise ValueError(
                "indexed consolidation pressure selected a thread without ready consolidation work"
            )
        knowledge = self.knowledge.project(
            self._boundary_events(revision),
            thread_id=state.thread_id,
        )
        preparation = prepare_thread_consolidation_work(plan, knowledge)
        preparation = replace(
            preparation,
            context={
                **dict(preparation.context),
                "synthesis_route": _THREAD_CONSOLIDATION_REASON,
                "consolidation_pressure_revision": revision,
            },
        )
        preparation.validate()
        self._preparation_by_thread_revision[key] = preparation
        self._owned_route_by_thread_revision.add(key)

        adjusted = replace(base, synthesis_need=pressure)
        adjusted.validate()
        return adjusted

    def context(
        self,
        state: ProjectedState,
        decision: SchedulerDecision,
    ) -> WorkPreparation | WorkPreparationBatch:
        state.validate()
        decision.validate()
        if decision.thread_id != state.thread_id:
            raise ValueError("scheduler decision and projected state refer to different threads")
        if _THREAD_CONSOLIDATION_REASON not in decision.reason_codes:
            return self.context_fallback(state, decision)

        if self._cached_revision != self.revision_provider.current_revision:
            raise RuntimeError(
                "thread consolidation route no longer matches the captured runtime snapshot"
            )
        key = (state.thread_id, decision.projection_revision)
        if key not in self._owned_route_by_thread_revision:
            raise ValueError(
                "thread consolidation route was not owned by the matching synthesis signal"
            )
        preparation = self._preparation_by_thread_revision.get(key)
        if preparation is None:
            raise RuntimeError(
                "thread consolidation route has no cached bounded preparation"
            )
        return preparation

    def owns_route(self, thread_id: str, projection_revision: int) -> bool:
        if self._cached_revision is None:
            return False
        try:
            current = self.revision_provider.current_revision
        except RuntimeError:
            return False
        return (
            self._cached_revision == current
            and (thread_id, projection_revision) in self._owned_route_by_thread_revision
        )

    def _refresh(self, revision: int) -> None:
        if revision == self._cached_revision and self._cached_overview is not None:
            return
        self._cached_overview = self.pressure_projector.project(sequence=revision)
        self._cached_revision = revision
        self._preparation_by_thread_revision.clear()
        self._owned_route_by_thread_revision.clear()

    def _boundary_events(self, revision: int) -> tuple[LedgerEvent, ...]:
        if revision == 0:
            return ()
        page = self.ledger.read_events(after_sequence=revision - 1, limit=1)
        if len(page) != 1 or page[0].sequence != revision:
            raise RuntimeError("cannot resolve exact consolidation control boundary event")
        return page
