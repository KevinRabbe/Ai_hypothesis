"""Thread-consolidation planning over materialized lineage + Knowledge State.

This module replaces raw-ledger replay for the question "what partition-produced knowledge is
currently pending consolidation?" while preserving the existing ThreadConsolidationPlan and
ThreadConsolidationPressureOverview contracts.
"""

from __future__ import annotations

from collections.abc import Mapping

from .contracts import LedgerEvent
from .knowledge import KnowledgeRecord, KnowledgeStatus
from .knowledge_index import SQLiteIndexedKnowledgeState
from .ledger import SQLiteResearchLedger
from .partition_knowledge_index import SQLiteIndexedPartitionKnowledgeLineage
from .thread_consolidation import (
    ThreadConsolidationConfig,
    ThreadConsolidationPlan,
    ThreadConsolidationSource,
)
from .thread_consolidation_control import (
    ThreadConsolidationPressureConfig,
    ThreadConsolidationPressureOverview,
)


_THREAD_CONSOLIDATION_KIND = "THREAD_CONSOLIDATION"


class IndexedThreadConsolidationPlanner:
    """Build the existing bounded consolidation plan without replaying ledger history."""

    def __init__(
        self,
        *,
        ledger: SQLiteResearchLedger,
        lineage: SQLiteIndexedPartitionKnowledgeLineage,
        knowledge: SQLiteIndexedKnowledgeState,
        config: ThreadConsolidationConfig | None = None,
    ) -> None:
        if lineage.ledger is not ledger or knowledge.ledger is not ledger:
            raise ValueError("indexed consolidation views must share one Research Ledger")
        self.ledger = ledger
        self.lineage = lineage
        self.knowledge = knowledge
        self.config = config or ThreadConsolidationConfig()
        self.config.validate()

    def plan(self, *, sequence: int, thread_id: str) -> ThreadConsolidationPlan:
        if sequence < 0:
            raise ValueError("sequence must be non-negative")
        if not thread_id or not thread_id.strip():
            raise ValueError("thread_id must be non-empty")

        lineage = self.lineage.snapshot_through(sequence)
        missing = lineage.missing_provenance_for_thread(thread_id)
        if missing:
            raise ValueError(
                "selected Work Thread has partitioned integration history without durable provenance"
            )
        knowledge = self.knowledge.project(
            self._boundary_events(sequence),
            thread_id=thread_id,
        )
        record_by_id = {record.delta_id: record for record in knowledge.records}

        source_partition: dict[str, str] = {}
        source_records: dict[str, KnowledgeRecord] = {}
        for decision in lineage.decisions:
            if decision.thread_id != thread_id or not decision.provenance_complete:
                continue
            for source in decision.sources:
                record = record_by_id.get(source.delta_id)
                if record is None:
                    raise ValueError(
                        "partition lineage references a knowledge delta missing from Knowledge State"
                    )
                if record.thread_id != thread_id:
                    raise ValueError(
                        "partition-produced knowledge targets another Work Thread"
                    )
                existing_partition = source_partition.get(source.delta_id)
                if (
                    existing_partition is not None
                    and existing_partition != source.partition_id
                ):
                    raise ValueError(
                        "one knowledge delta is attributed to multiple integration partitions"
                    )
                source_partition[source.delta_id] = source.partition_id
                source_records[source.delta_id] = record

        active_sources = {
            delta_id: record
            for delta_id, record in source_records.items()
            if record.status is not KnowledgeStatus.RETRACTED
        }
        consumed = self._active_consolidation_sources(
            knowledge.records,
            thread_id=thread_id,
            candidate_ids=set(active_sources),
        )

        pending_by_partition: dict[str, list[KnowledgeRecord]] = {}
        for delta_id, record in active_sources.items():
            if delta_id in consumed:
                continue
            pending_by_partition.setdefault(source_partition[delta_id], []).append(record)
        for records in pending_by_partition.values():
            records.sort(key=lambda record: (record.created_sequence, record.delta_id))

        selected = self._round_robin_select(pending_by_partition)
        plan = ThreadConsolidationPlan(
            revision=knowledge.revision,
            thread_id=thread_id,
            selection_limit=self.config.selection_limit,
            minimum_source_deltas=self.config.minimum_source_deltas,
            pending_source_count=sum(len(records) for records in pending_by_partition.values()),
            pending_partition_count=len(pending_by_partition),
            selected_sources=tuple(
                ThreadConsolidationSource(
                    partition_id=partition_id,
                    delta_id=record.delta_id,
                    status=record.status,
                    created_sequence=record.created_sequence,
                )
                for partition_id, record in selected
            ),
        )
        plan.validate()
        return plan

    @staticmethod
    def _active_consolidation_sources(
        records: tuple[KnowledgeRecord, ...],
        *,
        thread_id: str,
        candidate_ids: set[str],
    ) -> set[str]:
        consumed: set[str] = set()
        record_by_id = {record.delta_id: record for record in records}
        for consolidation in records:
            if consolidation.thread_id != thread_id:
                continue
            if consolidation.kind != _THREAD_CONSOLIDATION_KIND:
                continue
            if consolidation.status is KnowledgeStatus.RETRACTED:
                continue
            for reference_id in consolidation.source_reference_ids:
                if reference_id not in candidate_ids:
                    continue
                source = record_by_id.get(reference_id)
                if source is None:
                    continue
                if source.created_sequence >= consolidation.created_sequence:
                    raise ValueError(
                        "thread consolidation references knowledge created after consolidation"
                    )
                consumed.add(reference_id)
        return consumed

    def _round_robin_select(
        self,
        pending_by_partition: Mapping[str, list[KnowledgeRecord]],
    ) -> tuple[tuple[str, KnowledgeRecord], ...]:
        queues = {
            partition_id: list(records)
            for partition_id, records in pending_by_partition.items()
            if records
        }
        partition_order = sorted(
            queues,
            key=lambda partition_id: (
                queues[partition_id][0].created_sequence,
                partition_id,
            ),
        )
        selected: list[tuple[str, KnowledgeRecord]] = []
        while partition_order and len(selected) < self.config.selection_limit:
            next_order: list[str] = []
            for partition_id in partition_order:
                queue = queues[partition_id]
                if queue and len(selected) < self.config.selection_limit:
                    selected.append((partition_id, queue.pop(0)))
                if queue:
                    next_order.append(partition_id)
            partition_order = next_order
        return tuple(selected)

    def _boundary_events(self, sequence: int) -> tuple[LedgerEvent, ...]:
        if sequence == 0:
            return ()
        page = self.ledger.read_events(after_sequence=sequence - 1, limit=1)
        if len(page) != 1 or page[0].sequence != sequence:
            raise RuntimeError("cannot resolve exact consolidation snapshot boundary event")
        return page


class IndexedThreadConsolidationPressureProjector:
    """Project all Work Thread consolidation pressure from materialized current state."""

    def __init__(
        self,
        *,
        ledger: SQLiteResearchLedger,
        lineage: SQLiteIndexedPartitionKnowledgeLineage,
        knowledge: SQLiteIndexedKnowledgeState,
        config: ThreadConsolidationPressureConfig | None = None,
    ) -> None:
        if lineage.ledger is not ledger or knowledge.ledger is not ledger:
            raise ValueError("indexed consolidation views must share one Research Ledger")
        self.ledger = ledger
        self.lineage = lineage
        self.knowledge = knowledge
        self.config = config or ThreadConsolidationPressureConfig()
        self.config.validate()

    def project(self, *, sequence: int) -> ThreadConsolidationPressureOverview:
        if sequence < 0:
            raise ValueError("sequence must be non-negative")
        lineage = self.lineage.snapshot_through(sequence)
        knowledge = self.knowledge.project(self._boundary_events(sequence))
        record_by_id = {record.delta_id: record for record in knowledge.records}

        incomplete: set[str] = {
            decision.thread_id
            for decision in lineage.decisions
            if not decision.provenance_complete
        }
        source_partition_by_thread: dict[str, dict[str, str]] = {}
        source_record_by_thread: dict[str, dict[str, KnowledgeRecord]] = {}
        for decision in lineage.decisions:
            thread_id = decision.thread_id
            if not decision.provenance_complete:
                continue
            partition_index = source_partition_by_thread.setdefault(thread_id, {})
            record_index = source_record_by_thread.setdefault(thread_id, {})
            for source in decision.sources:
                record = record_by_id.get(source.delta_id)
                if record is None:
                    raise ValueError(
                        "partition lineage references a knowledge delta missing from Knowledge State"
                    )
                if record.thread_id != thread_id:
                    raise ValueError(
                        "partition-produced knowledge targets another Work Thread"
                    )
                existing_partition = partition_index.get(source.delta_id)
                if (
                    existing_partition is not None
                    and existing_partition != source.partition_id
                ):
                    raise ValueError(
                        "one knowledge delta is attributed to multiple integration partitions"
                    )
                partition_index[source.delta_id] = source.partition_id
                record_index[source.delta_id] = record

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

    def _boundary_events(self, sequence: int) -> tuple[LedgerEvent, ...]:
        if sequence == 0:
            return ()
        page = self.ledger.read_events(after_sequence=sequence - 1, limit=1)
        if len(page) != 1 or page[0].sequence != sequence:
            raise RuntimeError("cannot resolve exact consolidation snapshot boundary event")
        return page
