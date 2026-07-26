"""Read-only information-volume telemetry across integration hierarchy levels.

The metrics describe how many durable information objects exist at each level and how those
objects reference one another. Count reduction is not treated as semantic correctness or
lossless compression; verification and rare-evidence retention remain separate questions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

from .contracts import LedgerEvent
from .integration_partition_lineage import PartitionedIntegrationLineageProjector
from .knowledge import KnowledgeRecord, KnowledgeStateProjector, KnowledgeStatus


_THREAD_CONSOLIDATION_KIND = "THREAD_CONSOLIDATION"


@dataclass(frozen=True, slots=True)
class IntegrationHierarchyTelemetrySnapshot:
    revision: int
    raw_evidence_count: int
    partition_lineage_complete: bool
    missing_partition_provenance_decision_ids: tuple[str, ...]
    partition_assignment_count: int
    unique_partition_id_count: int
    started_partition_attempt_count: int
    unstarted_partition_assignment_count: int
    partition_knowledge_delta_count: int
    active_partition_knowledge_delta_count: int
    partition_status_counts: Mapping[str, int]
    partition_source_reference_count: int
    unique_raw_evidence_referenced_by_partition_knowledge_count: int
    thread_consolidation_delta_count: int
    active_thread_consolidation_delta_count: int
    thread_consolidation_status_counts: Mapping[str, int]
    thread_partition_source_reference_count: int
    active_thread_partition_source_reference_count: int
    unique_active_partition_deltas_consumed_count: int
    pending_active_partition_delta_count: int
    active_retracted_partition_source_reference_count: int
    cross_thread_partition_source_reference_count: int
    non_partition_thread_source_reference_count: int
    mean_active_thread_consolidation_fan_in: float | None
    max_active_thread_consolidation_fan_in: int | None
    active_hierarchy_frontier_count: int

    def require_complete_partition_lineage(
        self,
    ) -> "IntegrationHierarchyTelemetrySnapshot":
        if not self.partition_lineage_complete:
            raise ValueError(
                "integration hierarchy telemetry is missing partition allocation provenance"
            )
        return self

    @property
    def partition_evidence_reference_fraction(self) -> float:
        if self.raw_evidence_count == 0:
            return 0.0
        return (
            self.unique_raw_evidence_referenced_by_partition_knowledge_count
            / self.raw_evidence_count
        )

    @property
    def partition_knowledge_per_raw_evidence(self) -> float | None:
        if self.raw_evidence_count == 0:
            return None
        return self.partition_knowledge_delta_count / self.raw_evidence_count

    @property
    def partition_to_frontier_count_reduction_factor(self) -> float | None:
        """Count-only lower/frontier ratio; this is not a semantic compression score."""

        if self.active_partition_knowledge_delta_count == 0:
            return None
        if self.active_hierarchy_frontier_count == 0:
            return None
        return (
            self.active_partition_knowledge_delta_count
            / self.active_hierarchy_frontier_count
        )

    @property
    def consumed_partition_deltas_per_active_thread_consolidation(self) -> float | None:
        if self.active_thread_consolidation_delta_count == 0:
            return None
        return (
            self.unique_active_partition_deltas_consumed_count
            / self.active_thread_consolidation_delta_count
        )


class IntegrationHierarchyTelemetryProjector:
    """Project global count/fan-in telemetry over evidence and two integration levels."""

    def __init__(
        self,
        *,
        lineage_projector: PartitionedIntegrationLineageProjector | None = None,
        knowledge_projector: KnowledgeStateProjector | None = None,
    ) -> None:
        self.lineage_projector = lineage_projector or PartitionedIntegrationLineageProjector()
        self.knowledge_projector = knowledge_projector or KnowledgeStateProjector()

    def project(
        self,
        events: Sequence[LedgerEvent],
    ) -> IntegrationHierarchyTelemetrySnapshot:
        revision, evidence_ids = self._evidence_index(events)
        lineage = self.lineage_projector.project(events)
        knowledge = self.knowledge_projector.project(events)
        if knowledge.revision != revision:
            raise ValueError("knowledge projection revision does not match ledger history")
        record_by_id = {record.delta_id: record for record in knowledge.records}

        partition_assignments = [
            partition
            for allocation in lineage.records
            if allocation.provenance_complete
            for partition in allocation.partitions
        ]
        unique_partition_ids = {
            partition.partition_id for partition in partition_assignments
        }
        started_partition_attempt_count = sum(
            len(allocation.partition_attempts)
            for allocation in lineage.records
            if allocation.provenance_complete
        )
        unstarted_partition_assignment_count = sum(
            len(allocation.unstarted_partition_ids)
            for allocation in lineage.records
            if allocation.provenance_complete
        )

        partition_delta_ids: list[str] = []
        for allocation in lineage.records:
            if not allocation.provenance_complete:
                continue
            for partition_attempt in allocation.partition_attempts:
                partition_delta_ids.extend(partition_attempt.knowledge_delta_ids)
        if len(set(partition_delta_ids)) != len(partition_delta_ids):
            raise ValueError(
                "one durable knowledge delta appears in multiple partition attempts"
            )

        partition_records: dict[str, KnowledgeRecord] = {}
        for delta_id in partition_delta_ids:
            record = record_by_id.get(delta_id)
            if record is None:
                raise ValueError(
                    "partition lineage references a knowledge delta missing from Knowledge State"
                )
            partition_records[delta_id] = record

        partition_status_counts = self._status_counts(partition_records.values())
        active_partition_records = {
            delta_id: record
            for delta_id, record in partition_records.items()
            if record.status is not KnowledgeStatus.RETRACTED
        }

        partition_source_reference_count = 0
        referenced_evidence: set[str] = set()
        for record in partition_records.values():
            for reference_id in record.source_reference_ids:
                if reference_id in evidence_ids:
                    partition_source_reference_count += 1
                    referenced_evidence.add(reference_id)

        thread_records = tuple(
            record
            for record in knowledge.records
            if record.kind == _THREAD_CONSOLIDATION_KIND
        )
        thread_status_counts = self._status_counts(thread_records)
        active_thread_records = tuple(
            record
            for record in thread_records
            if record.status is not KnowledgeStatus.RETRACTED
        )

        thread_partition_source_reference_count = 0
        active_thread_partition_source_reference_count = 0
        active_consumed_partition_ids: set[str] = set()
        active_retracted_source_refs = 0
        cross_thread_source_refs = 0
        non_partition_source_refs = 0
        active_fan_in: list[int] = []

        for record in thread_records:
            known_partition_refs: list[str] = []
            for reference_id in record.source_reference_ids:
                lower = partition_records.get(reference_id)
                if lower is None:
                    non_partition_source_refs += 1
                    continue
                if lower.created_sequence >= record.created_sequence:
                    raise ValueError(
                        "thread consolidation references partition knowledge created later"
                    )
                known_partition_refs.append(reference_id)
                thread_partition_source_reference_count += 1
                if lower.thread_id != record.thread_id:
                    cross_thread_source_refs += 1

            if record.status is KnowledgeStatus.RETRACTED:
                continue

            active_unique_refs = set(known_partition_refs)
            active_fan_in.append(len(active_unique_refs))
            active_thread_partition_source_reference_count += len(known_partition_refs)
            for reference_id in active_unique_refs:
                lower = partition_records[reference_id]
                if lower.status is KnowledgeStatus.RETRACTED:
                    active_retracted_source_refs += 1
                    continue
                active_consumed_partition_ids.add(reference_id)

        pending_active_partition_ids = (
            set(active_partition_records) - active_consumed_partition_ids
        )
        active_frontier_count = (
            len(pending_active_partition_ids) + len(active_thread_records)
        )

        return IntegrationHierarchyTelemetrySnapshot(
            revision=revision,
            raw_evidence_count=len(evidence_ids),
            partition_lineage_complete=lineage.provenance_complete,
            missing_partition_provenance_decision_ids=(
                lineage.missing_provenance_decision_ids
            ),
            partition_assignment_count=len(partition_assignments),
            unique_partition_id_count=len(unique_partition_ids),
            started_partition_attempt_count=started_partition_attempt_count,
            unstarted_partition_assignment_count=unstarted_partition_assignment_count,
            partition_knowledge_delta_count=len(partition_records),
            active_partition_knowledge_delta_count=len(active_partition_records),
            partition_status_counts=partition_status_counts,
            partition_source_reference_count=partition_source_reference_count,
            unique_raw_evidence_referenced_by_partition_knowledge_count=len(
                referenced_evidence
            ),
            thread_consolidation_delta_count=len(thread_records),
            active_thread_consolidation_delta_count=len(active_thread_records),
            thread_consolidation_status_counts=thread_status_counts,
            thread_partition_source_reference_count=(
                thread_partition_source_reference_count
            ),
            active_thread_partition_source_reference_count=(
                active_thread_partition_source_reference_count
            ),
            unique_active_partition_deltas_consumed_count=len(
                active_consumed_partition_ids
            ),
            pending_active_partition_delta_count=len(pending_active_partition_ids),
            active_retracted_partition_source_reference_count=(
                active_retracted_source_refs
            ),
            cross_thread_partition_source_reference_count=cross_thread_source_refs,
            non_partition_thread_source_reference_count=non_partition_source_refs,
            mean_active_thread_consolidation_fan_in=(
                sum(active_fan_in) / len(active_fan_in) if active_fan_in else None
            ),
            max_active_thread_consolidation_fan_in=(
                max(active_fan_in) if active_fan_in else None
            ),
            active_hierarchy_frontier_count=active_frontier_count,
        )

    @staticmethod
    def _evidence_index(
        events: Sequence[LedgerEvent],
    ) -> tuple[int, set[str]]:
        previous_sequence = -1
        revision = 0
        evidence_ids: set[str] = set()
        for event in events:
            event.validate()
            if event.sequence <= previous_sequence:
                raise ValueError("events must be in strictly increasing sequence order")
            previous_sequence = event.sequence
            revision = event.sequence
            if event.event_type != "EVIDENCE_ADDED":
                continue
            evidence_id = event.payload.get("evidence_id")
            if not isinstance(evidence_id, str) or not evidence_id:
                raise ValueError("EVIDENCE_ADDED is missing evidence_id")
            if evidence_id in evidence_ids:
                raise ValueError(f"duplicate durable evidence ID {evidence_id!r}")
            evidence_ids.add(evidence_id)
        return revision, evidence_ids

    @staticmethod
    def _status_counts(records: Sequence[KnowledgeRecord] | object) -> Mapping[str, int]:
        counts = {status.value: 0 for status in KnowledgeStatus}
        for record in records:  # type: ignore[union-attr]
            counts[record.status.value] += 1
        return counts
