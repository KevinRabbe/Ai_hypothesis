"""Bounded thread-level consolidation over partition-produced knowledge.

Raw evidence is integrated locally first. This planner then reconnects active knowledge deltas
across historical integration partitions using the same Work Item / Worker Runtime path.
Knowledge-to-knowledge references form the durable hierarchy; no separate hierarchy store is
introduced.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Sequence

from .contracts import LedgerEvent
from .control import WorkPreparation
from .integration_partition_lineage import PartitionedIntegrationLineageProjector
from .knowledge import (
    KnowledgeRecord,
    KnowledgeSnapshot,
    KnowledgeStateProjector,
    KnowledgeStatus,
)
from .knowledge_work import prepare_bounded_knowledge_work


_THREAD_CONSOLIDATION_KIND = "THREAD_CONSOLIDATION"


@dataclass(frozen=True, slots=True)
class ThreadConsolidationConfig:
    selection_limit: int = 32
    minimum_source_deltas: int = 2

    def validate(self) -> None:
        if self.selection_limit <= 0:
            raise ValueError("selection_limit must be positive")
        if self.minimum_source_deltas <= 1:
            raise ValueError("minimum_source_deltas must be greater than one")
        if self.minimum_source_deltas > self.selection_limit:
            raise ValueError("minimum_source_deltas cannot exceed selection_limit")


@dataclass(frozen=True, slots=True)
class ThreadConsolidationSource:
    partition_id: str
    delta_id: str
    created_sequence: int
    status: KnowledgeStatus


@dataclass(frozen=True, slots=True)
class ThreadConsolidationPlan:
    revision: int
    thread_id: str
    selection_limit: int
    minimum_source_deltas: int
    pending_source_count: int
    pending_partition_count: int
    selected_sources: tuple[ThreadConsolidationSource, ...]

    @property
    def selected_delta_ids(self) -> tuple[str, ...]:
        return tuple(source.delta_id for source in self.selected_sources)

    @property
    def selected_partition_ids(self) -> tuple[str, ...]:
        return tuple(
            dict.fromkeys(source.partition_id for source in self.selected_sources)
        )

    @property
    def ready(self) -> bool:
        return len(self.selected_sources) >= self.minimum_source_deltas

    def validate(self) -> None:
        if self.revision < 0:
            raise ValueError("revision must be non-negative")
        if not self.thread_id or not self.thread_id.strip():
            raise ValueError("thread_id must be non-empty")
        if self.selection_limit <= 0:
            raise ValueError("selection_limit must be positive")
        if self.minimum_source_deltas <= 1:
            raise ValueError("minimum_source_deltas must be greater than one")
        if self.minimum_source_deltas > self.selection_limit:
            raise ValueError("minimum_source_deltas cannot exceed selection_limit")
        if self.pending_source_count < 0 or self.pending_partition_count < 0:
            raise ValueError("pending consolidation counts must be non-negative")
        if len(self.selected_sources) > self.selection_limit:
            raise ValueError("selected consolidation sources exceed selection_limit")
        if len({source.delta_id for source in self.selected_sources}) != len(
            self.selected_sources
        ):
            raise ValueError("selected consolidation delta IDs must be unique")
        if len(self.selected_sources) > self.pending_source_count:
            raise ValueError("selected sources cannot exceed pending source count")
        selected_partition_count = len(self.selected_partition_ids)
        if selected_partition_count > self.pending_partition_count:
            raise ValueError("selected partitions cannot exceed pending partition count")
        for source in self.selected_sources:
            if not source.partition_id or not source.partition_id.strip():
                raise ValueError("source partition_id must be non-empty")
            if not source.delta_id or not source.delta_id.strip():
                raise ValueError("source delta_id must be non-empty")
            if source.created_sequence < 0:
                raise ValueError("source created_sequence must be non-negative")
            if source.status is KnowledgeStatus.RETRACTED:
                raise ValueError("retracted knowledge cannot be selected for consolidation")


class ThreadConsolidationPlanner:
    """Select active unconsolidated partition knowledge with cross-partition round-robin."""

    def __init__(
        self,
        config: ThreadConsolidationConfig | None = None,
        *,
        lineage_projector: PartitionedIntegrationLineageProjector | None = None,
        knowledge_projector: KnowledgeStateProjector | None = None,
    ) -> None:
        self.config = config or ThreadConsolidationConfig()
        self.config.validate()
        self.lineage_projector = (
            lineage_projector or PartitionedIntegrationLineageProjector()
        )
        self.knowledge_projector = knowledge_projector or KnowledgeStateProjector()

    def plan(
        self,
        events: Sequence[LedgerEvent],
        *,
        thread_id: str,
    ) -> ThreadConsolidationPlan:
        if not thread_id or not thread_id.strip():
            raise ValueError("thread_id must be non-empty")

        lineage = self.lineage_projector.project(events)
        relevant_lineage = tuple(
            allocation
            for allocation in lineage.records
            if allocation.allocation.thread_id == thread_id
        )
        missing_relevant = tuple(
            allocation.allocation.decision_id
            for allocation in relevant_lineage
            if not allocation.provenance_complete
        )
        if missing_relevant:
            raise ValueError(
                "selected Work Thread is missing durable allocation provenance"
            )

        knowledge = self.knowledge_projector.project(events)
        record_by_id = {record.delta_id: record for record in knowledge.records}

        source_partition: dict[str, str] = {}
        source_records: dict[str, KnowledgeRecord] = {}
        for allocation in relevant_lineage:
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
                    existing_partition = source_partition.get(delta_id)
                    if existing_partition is not None and existing_partition != partition_id:
                        raise ValueError(
                            "one knowledge delta is attributed to multiple integration partitions"
                        )
                    source_partition[delta_id] = partition_id
                    source_records[delta_id] = record

        active_sources = {
            delta_id: record
            for delta_id, record in source_records.items()
            if record.status is not KnowledgeStatus.RETRACTED
        }
        consumed = self._active_consolidation_sources(
            knowledge,
            thread_id=thread_id,
            candidate_records=active_sources,
        )

        pending_by_partition: dict[str, list[KnowledgeRecord]] = {}
        for delta_id, record in active_sources.items():
            if delta_id in consumed:
                continue
            partition_id = source_partition[delta_id]
            pending_by_partition.setdefault(partition_id, []).append(record)
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
                    partition_id=source_partition[record.delta_id],
                    delta_id=record.delta_id,
                    created_sequence=record.created_sequence,
                    status=record.status,
                )
                for record in selected
            ),
        )
        plan.validate()
        return plan

    @staticmethod
    def _active_consolidation_sources(
        knowledge: KnowledgeSnapshot,
        *,
        thread_id: str,
        candidate_records: dict[str, KnowledgeRecord],
    ) -> set[str]:
        consumed: set[str] = set()
        for record in knowledge.records:
            if record.thread_id != thread_id:
                continue
            if record.kind != _THREAD_CONSOLIDATION_KIND:
                continue
            if record.status is KnowledgeStatus.RETRACTED:
                continue
            for reference_id in record.source_reference_ids:
                candidate = candidate_records.get(reference_id)
                if candidate is None:
                    continue
                if candidate.created_sequence >= record.created_sequence:
                    raise ValueError(
                        "thread consolidation references knowledge created after consolidation"
                    )
                consumed.add(reference_id)
        return consumed

    def _round_robin_select(
        self,
        pending_by_partition: dict[str, list[KnowledgeRecord]],
    ) -> tuple[KnowledgeRecord, ...]:
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
        selected: list[KnowledgeRecord] = []
        while partition_order and len(selected) < self.config.selection_limit:
            next_order: list[str] = []
            for partition_id in partition_order:
                queue = queues[partition_id]
                if queue and len(selected) < self.config.selection_limit:
                    selected.append(queue.pop(0))
                if queue:
                    next_order.append(partition_id)
            partition_order = next_order
        return tuple(selected)


def prepare_thread_consolidation_work(
    plan: ThreadConsolidationPlan,
    knowledge: KnowledgeSnapshot,
) -> WorkPreparation:
    """Prepare one bounded same-thread consolidation attempt from current knowledge."""

    plan.validate()
    if not plan.ready:
        raise ValueError("thread consolidation requires at least two pending source deltas")
    if knowledge.revision != plan.revision:
        raise ValueError("knowledge snapshot revision does not match consolidation plan")

    preparation = prepare_bounded_knowledge_work(
        knowledge,
        plan.selected_delta_ids,
        limit=plan.selection_limit,
    )
    preparation = replace(
        preparation,
        context={
            **dict(preparation.context),
            "context_view": "SYNTHESIZE",
            "synthesis_mode": _THREAD_CONSOLIDATION_KIND,
            "source_partition_ids": list(plan.selected_partition_ids),
            "pending_source_count": plan.pending_source_count,
            "pending_partition_count": plan.pending_partition_count,
        },
        constraints={
            **dict(preparation.constraints),
            "knowledge_delta_kind": _THREAD_CONSOLIDATION_KIND,
            "reference_consumed_knowledge": True,
            "preserve_unresolved_contradictions": True,
            "output_remains_provisional": True,
        },
    )
    preparation.validate()
    return preparation
