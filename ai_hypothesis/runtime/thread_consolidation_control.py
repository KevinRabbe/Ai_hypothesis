"""Automatic control-plane routing for thread-level knowledge consolidation.

The generic scheduler only sees `synthesis_need`. This module projects consolidation demand
from one ledger snapshot, injects that generic signal, tags the final decision when the demand
actually came from thread consolidation, and supplies the bounded consolidation context.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Mapping, Protocol, Sequence

from .contracts import LedgerEvent, ProjectedState, SchedulerAction, SchedulerDecision
from .control import ContextProvider, SignalProvider, WorkPreparation, WorkPreparationBatch
from .integration_partition_lineage import PartitionedIntegrationLineageProjector
from .knowledge import KnowledgeRecord, KnowledgeStateProjector, KnowledgeStatus
from .ledger import SQLiteResearchLedger
from .scheduler import SchedulerSignals, SchedulableThread
from .thread_consolidation import (
    ThreadConsolidationPlanner,
    prepare_thread_consolidation_work,
)


_THREAD_CONSOLIDATION_KIND = "THREAD_CONSOLIDATION"
_THREAD_CONSOLIDATION_REASON = "THREAD_CONSOLIDATION"


class SchedulerLike(Protocol):
    def choose(
        self,
        candidates: Sequence[SchedulableThread],
        *,
        integration_backpressure: bool = False,
        max_width: int = 1,
    ) -> SchedulerDecision:
        ...


@dataclass(frozen=True, slots=True)
class ThreadConsolidationPressureConfig:
    full_pressure_count: int = 8
    minimum_source_deltas: int = 2

    def validate(self) -> None:
        if self.full_pressure_count <= 0:
            raise ValueError("full_pressure_count must be positive")
        if self.minimum_source_deltas <= 1:
            raise ValueError("minimum_source_deltas must be greater than one")
        if self.minimum_source_deltas > self.full_pressure_count:
            raise ValueError("minimum_source_deltas cannot exceed full_pressure_count")


@dataclass(frozen=True, slots=True)
class ThreadConsolidationPressureOverview:
    revision: int
    pending_source_count: Mapping[str, int]
    pending_partition_count: Mapping[str, int]
    thread_pressure: Mapping[str, float]
    incomplete_thread_ids: tuple[str, ...]

    def pressure_for(self, thread_id: str) -> float:
        return float(self.thread_pressure.get(thread_id, 0.0))

    def pending_sources_for(self, thread_id: str) -> int:
        return int(self.pending_source_count.get(thread_id, 0))

    def pending_partitions_for(self, thread_id: str) -> int:
        return int(self.pending_partition_count.get(thread_id, 0))

    def is_incomplete(self, thread_id: str) -> bool:
        return thread_id in self.incomplete_thread_ids


class ThreadConsolidationPressureProjector:
    """Project all thread consolidation pressure with one lineage + knowledge pass."""

    def __init__(
        self,
        config: ThreadConsolidationPressureConfig | None = None,
        *,
        lineage_projector: PartitionedIntegrationLineageProjector | None = None,
        knowledge_projector: KnowledgeStateProjector | None = None,
    ) -> None:
        self.config = config or ThreadConsolidationPressureConfig()
        self.config.validate()
        self.lineage_projector = lineage_projector or PartitionedIntegrationLineageProjector()
        self.knowledge_projector = knowledge_projector or KnowledgeStateProjector()

    def project(
        self,
        events: Sequence[LedgerEvent],
    ) -> ThreadConsolidationPressureOverview:
        lineage = self.lineage_projector.project(events)
        knowledge = self.knowledge_projector.project(events)
        record_by_id = {record.delta_id: record for record in knowledge.records}

        incomplete: set[str] = set()
        source_partition_by_thread: dict[str, dict[str, str]] = {}
        source_record_by_thread: dict[str, dict[str, KnowledgeRecord]] = {}

        for allocation in lineage.records:
            thread_id = allocation.allocation.thread_id
            if not allocation.provenance_complete:
                incomplete.add(thread_id)
                continue
            partition_index = source_partition_by_thread.setdefault(thread_id, {})
            record_index = source_record_by_thread.setdefault(thread_id, {})
            for partition_attempt in allocation.partition_attempts:
                partition_id = partition_attempt.partition.partition_id
                for delta_id in partition_attempt.knowledge_delta_ids:
                    record = record_by_id.get(delta_id)
                    if record is None:
                        raise ValueError(
                            "partition lineage references a knowledge delta missing from Knowledge State"
                        )
                    if record.thread_id != thread_id:
                        raise ValueError(
                            "partition-produced knowledge targets another Work Thread"
                        )
                    existing_partition = partition_index.get(delta_id)
                    if existing_partition is not None and existing_partition != partition_id:
                        raise ValueError(
                            "one knowledge delta is attributed to multiple integration partitions"
                        )
                    partition_index[delta_id] = partition_id
                    record_index[delta_id] = record

        active_by_thread: dict[str, dict[str, KnowledgeRecord]] = {}
        for thread_id, records in source_record_by_thread.items():
            if thread_id in incomplete:
                continue
            active_by_thread[thread_id] = {
                delta_id: record
                for delta_id, record in records.items()
                if record.status is not KnowledgeStatus.RETRACTED
            }

        consumed_by_thread: dict[str, set[str]] = {
            thread_id: set() for thread_id in active_by_thread
        }
        for consolidation in knowledge.records:
            thread_id = consolidation.thread_id
            if thread_id is None or thread_id not in active_by_thread:
                continue
            if consolidation.kind != _THREAD_CONSOLIDATION_KIND:
                continue
            if consolidation.status is KnowledgeStatus.RETRACTED:
                continue
            candidates = active_by_thread[thread_id]
            consumed = consumed_by_thread[thread_id]
            for reference_id in consolidation.source_reference_ids:
                candidate = candidates.get(reference_id)
                if candidate is None:
                    continue
                if candidate.created_sequence >= consolidation.created_sequence:
                    raise ValueError(
                        "thread consolidation references knowledge created after consolidation"
                    )
                consumed.add(reference_id)

        pending_source_count: dict[str, int] = {}
        pending_partition_count: dict[str, int] = {}
        pressure: dict[str, float] = {}
        for thread_id, active in active_by_thread.items():
            consumed = consumed_by_thread[thread_id]
            partition_index = source_partition_by_thread[thread_id]
            pending_ids = tuple(
                delta_id for delta_id in active if delta_id not in consumed
            )
            pending_source_count[thread_id] = len(pending_ids)
            pending_partition_count[thread_id] = len(
                {partition_index[delta_id] for delta_id in pending_ids}
            )
            pressure[thread_id] = (
                min(1.0, len(pending_ids) / float(self.config.full_pressure_count))
                if len(pending_ids) >= self.config.minimum_source_deltas
                else 0.0
            )

        return ThreadConsolidationPressureOverview(
            revision=knowledge.revision,
            pending_source_count=pending_source_count,
            pending_partition_count=pending_partition_count,
            thread_pressure=pressure,
            incomplete_thread_ids=tuple(sorted(incomplete)),
        )


class ThreadConsolidationControlAdapter:
    """Provide both synthesis signals and bounded consolidation context from one cached view."""

    def __init__(
        self,
        *,
        ledger: SQLiteResearchLedger,
        signal_fallback: SignalProvider,
        context_fallback: ContextProvider,
        pressure_projector: ThreadConsolidationPressureProjector | None = None,
        planner: ThreadConsolidationPlanner | None = None,
        knowledge_projector: KnowledgeStateProjector | None = None,
    ) -> None:
        self.ledger = ledger
        self.signal_fallback = signal_fallback
        self.context_fallback = context_fallback
        self.planner = planner or ThreadConsolidationPlanner()
        if pressure_projector is None:
            full_pressure_count = max(8, self.planner.config.minimum_source_deltas)
            pressure_projector = ThreadConsolidationPressureProjector(
                ThreadConsolidationPressureConfig(
                    full_pressure_count=full_pressure_count,
                    minimum_source_deltas=self.planner.config.minimum_source_deltas,
                ),
                lineage_projector=self.planner.lineage_projector,
                knowledge_projector=self.planner.knowledge_projector,
            )
        if (
            pressure_projector.config.minimum_source_deltas
            != self.planner.config.minimum_source_deltas
        ):
            raise ValueError(
                "consolidation pressure and planner minimum_source_deltas must match"
            )
        self.pressure_projector = pressure_projector
        self.knowledge_projector = (
            knowledge_projector or self.planner.knowledge_projector
        )
        self._cached_sequence: int | None = None
        self._cached_events: tuple[LedgerEvent, ...] = ()
        self._cached_overview: ThreadConsolidationPressureOverview | None = None
        self._snapshot_by_thread_revision: dict[
            tuple[str, int], tuple[LedgerEvent, ...]
        ] = {}
        self._pressure_revision_by_thread_revision: dict[
            tuple[str, int], int
        ] = {}
        self._owned_route_by_thread_revision: set[tuple[str, int]] = set()

    def signals(self, state: ProjectedState) -> SchedulerSignals:
        state.validate()
        self._refresh_if_needed()
        assert self._cached_overview is not None
        base = self.signal_fallback(state)
        base.validate()
        pressure = self._cached_overview.pressure_for(state.thread_id)
        key = (state.thread_id, state.revision)
        self._snapshot_by_thread_revision[key] = self._cached_events
        self._pressure_revision_by_thread_revision[key] = self._cached_overview.revision
        if pressure > base.synthesis_need:
            self._owned_route_by_thread_revision.add(key)
            adjusted = replace(base, synthesis_need=pressure)
            adjusted.validate()
            return adjusted
        self._owned_route_by_thread_revision.discard(key)
        return base

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

        key = (state.thread_id, decision.projection_revision)
        if key not in self._owned_route_by_thread_revision:
            raise ValueError(
                "thread consolidation route was not owned by the matching synthesis signal"
            )
        events = self._snapshot_by_thread_revision.get(key)
        if events is None:
            raise RuntimeError(
                "thread consolidation decision has no matching signal snapshot"
            )
        pressure_revision = self._pressure_revision_by_thread_revision.get(key)
        if pressure_revision is None:
            raise RuntimeError(
                "thread consolidation decision has no matching pressure revision"
            )
        plan = self.planner.plan(events, thread_id=state.thread_id)
        if not plan.ready:
            raise ValueError(
                "thread consolidation decision selected a thread without ready consolidation work"
            )
        knowledge = self.knowledge_projector.project(events)
        preparation = prepare_thread_consolidation_work(plan, knowledge)
        return replace(
            preparation,
            context={
                **dict(preparation.context),
                "synthesis_route": _THREAD_CONSOLIDATION_REASON,
                "consolidation_pressure_revision": pressure_revision,
            },
        )

    def owns_route(self, thread_id: str, projection_revision: int) -> bool:
        return (thread_id, projection_revision) in self._owned_route_by_thread_revision

    def _refresh_if_needed(self) -> None:
        sequence = self.ledger.latest_sequence()
        if sequence == self._cached_sequence and self._cached_overview is not None:
            return
        events = self.ledger.read_all_events()
        self._cached_events = events
        self._cached_overview = self.pressure_projector.project(events)
        self._cached_sequence = sequence
        # Bound route/snapshot memory to the current durable revision.
        self._snapshot_by_thread_revision.clear()
        self._pressure_revision_by_thread_revision.clear()
        self._owned_route_by_thread_revision.clear()


class ThreadConsolidationScheduler:
    """Tag generic synthesis decisions when consolidation supplied the winning demand."""

    def __init__(
        self,
        delegate: SchedulerLike,
        *,
        control: ThreadConsolidationControlAdapter,
    ) -> None:
        self.delegate = delegate
        self.control = control

    def choose(
        self,
        candidates: Sequence[SchedulableThread],
        *,
        integration_backpressure: bool = False,
        max_width: int = 1,
    ) -> SchedulerDecision:
        decision = self.delegate.choose(
            candidates,
            integration_backpressure=integration_backpressure,
            max_width=max_width,
        )
        decision.validate()
        if (
            decision.action is SchedulerAction.SYNTHESIZE
            and "SYNTHESIS_NEEDED" in decision.reason_codes
            and self.control.owns_route(
                decision.thread_id,
                decision.projection_revision,
            )
        ):
            reasons = tuple(
                dict.fromkeys((*decision.reason_codes, _THREAD_CONSOLIDATION_REASON))
            )
            routed = replace(decision, reason_codes=reasons)
            routed.validate()
            return routed
        return decision
