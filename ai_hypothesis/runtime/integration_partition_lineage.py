"""Join partitioned integration allocations to attempts and knowledge outputs.

The partition allocation event freezes historical sharding/authority. This projector combines
that provenance with the existing integration-allocation outcome projection so later
consolidation can consume exact partition-produced knowledge without inferring old routing
from current configuration.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from .contracts import LedgerEvent
from .integration_allocation_outcomes import (
    IntegrationAllocationOutcome,
    IntegrationAllocationOutcomeProjector,
    IntegrationAttemptOutcome,
)


_PARTITION_ALLOCATION_EVENT = "INTEGRATION_PARTITION_ALLOCATION_RECORDED"
_PARTITION_ALLOCATION_SCHEMA = "integration-partition-allocation-v0"


@dataclass(frozen=True, slots=True)
class HistoricalIntegrationPartition:
    partition_id: str
    shard_index: int
    backlog_count: int
    oldest_pending_sequence: int
    evidence_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class PartitionAttemptLineage:
    partition: HistoricalIntegrationPartition
    attempt: IntegrationAttemptOutcome

    @property
    def knowledge_delta_ids(self) -> tuple[str, ...]:
        return self.attempt.knowledge_delta_ids


@dataclass(frozen=True, slots=True)
class PartitionedIntegrationLineage:
    allocation: IntegrationAllocationOutcome
    provenance_event_id: str | None
    provenance_sequence: int | None
    partition_plan_revision: int | None
    shard_count: int | None
    batch_limit: int | None
    partitions: tuple[HistoricalIntegrationPartition, ...]
    partition_attempts: tuple[PartitionAttemptLineage, ...]
    unstarted_partition_ids: tuple[str, ...]

    @property
    def provenance_complete(self) -> bool:
        return self.provenance_event_id is not None

    @property
    def knowledge_delta_ids(self) -> tuple[str, ...]:
        return tuple(
            delta_id
            for lineage in self.partition_attempts
            for delta_id in lineage.knowledge_delta_ids
        )


@dataclass(frozen=True, slots=True)
class PartitionedIntegrationLineageSnapshot:
    records: tuple[PartitionedIntegrationLineage, ...]
    missing_provenance_decision_ids: tuple[str, ...]

    @property
    def provenance_complete(self) -> bool:
        return not self.missing_provenance_decision_ids

    def require_complete(self) -> "PartitionedIntegrationLineageSnapshot":
        if self.missing_provenance_decision_ids:
            raise ValueError(
                "partitioned integration history is missing durable allocation provenance"
            )
        return self


@dataclass(frozen=True, slots=True)
class _ProvenanceRecord:
    event_id: str
    sequence: int
    decision_id: str
    thread_id: str
    partition_plan_revision: int
    shard_count: int
    batch_limit: int
    width: int
    partitions: tuple[HistoricalIntegrationPartition, ...]


class PartitionedIntegrationLineageProjector:
    """Project exact historical partition lineage for backpressure integration work."""

    def __init__(
        self,
        allocation_projector: IntegrationAllocationOutcomeProjector | None = None,
    ) -> None:
        self.allocation_projector = (
            allocation_projector or IntegrationAllocationOutcomeProjector()
        )

    def project(
        self,
        events: Sequence[LedgerEvent],
    ) -> PartitionedIntegrationLineageSnapshot:
        allocations = self.allocation_projector.project(events)
        partitioned = {
            allocation.decision_id: allocation
            for allocation in allocations
            if allocation.partitioned
        }
        decision_sequences = self._decision_sequences(events, set(partitioned))
        attempt_start_sequences = self._attempt_start_sequences(events, set(partitioned))
        provenance = self._provenance_records(
            events,
            partitioned=partitioned,
            decision_sequences=decision_sequences,
            attempt_start_sequences=attempt_start_sequences,
        )

        records: list[PartitionedIntegrationLineage] = []
        missing: list[str] = []
        for allocation in allocations:
            if not allocation.partitioned:
                continue
            record = provenance.get(allocation.decision_id)
            if record is None:
                missing.append(allocation.decision_id)
                records.append(
                    PartitionedIntegrationLineage(
                        allocation=allocation,
                        provenance_event_id=None,
                        provenance_sequence=None,
                        partition_plan_revision=None,
                        shard_count=None,
                        batch_limit=None,
                        partitions=(),
                        partition_attempts=(),
                        unstarted_partition_ids=(),
                    )
                )
                continue
            records.append(self._join(allocation, record))

        return PartitionedIntegrationLineageSnapshot(
            records=tuple(records),
            missing_provenance_decision_ids=tuple(missing),
        )

    @staticmethod
    def _decision_sequences(
        events: Sequence[LedgerEvent],
        decision_ids: set[str],
    ) -> dict[str, int]:
        sequences: dict[str, int] = {}
        for event in events:
            if event.event_type != "SCHEDULER_DECISION_RECORDED":
                continue
            decision_id = event.payload.get("decision_id")
            if not isinstance(decision_id, str) or decision_id not in decision_ids:
                continue
            if decision_id in sequences:
                raise ValueError("partitioned scheduler decision was recorded more than once")
            sequences[decision_id] = event.sequence
        missing = decision_ids - set(sequences)
        if missing:
            raise ValueError("partitioned integration outcome is missing scheduler trace")
        return sequences

    @staticmethod
    def _attempt_start_sequences(
        events: Sequence[LedgerEvent],
        decision_ids: set[str],
    ) -> dict[str, tuple[int, ...]]:
        by_decision: dict[str, list[int]] = {decision_id: [] for decision_id in decision_ids}
        for event in events:
            if event.event_type != "ATTEMPT_STARTED":
                continue
            decision_id = event.payload.get("scheduler_decision_id")
            if isinstance(decision_id, str) and decision_id in by_decision:
                by_decision[decision_id].append(event.sequence)
        return {
            decision_id: tuple(sequences)
            for decision_id, sequences in by_decision.items()
        }

    def _provenance_records(
        self,
        events: Sequence[LedgerEvent],
        *,
        partitioned: Mapping[str, IntegrationAllocationOutcome],
        decision_sequences: Mapping[str, int],
        attempt_start_sequences: Mapping[str, tuple[int, ...]],
    ) -> dict[str, _ProvenanceRecord]:
        records: dict[str, _ProvenanceRecord] = {}
        for event in events:
            if event.event_type != _PARTITION_ALLOCATION_EVENT:
                continue
            record = self._parse_provenance(event)
            allocation = partitioned.get(record.decision_id)
            if allocation is None:
                raise ValueError(
                    "partition allocation provenance references a non-partitioned decision"
                )
            if record.decision_id in records:
                raise ValueError(
                    "partition allocation provenance was recorded more than once"
                )
            if record.thread_id != allocation.thread_id:
                raise ValueError("partition allocation thread does not match scheduler decision")
            if record.width != allocation.width:
                raise ValueError("partition allocation width does not match scheduler decision")
            if event.sequence <= decision_sequences[record.decision_id]:
                raise ValueError("partition allocation provenance precedes scheduler decision")
            starts = attempt_start_sequences.get(record.decision_id, ())
            if starts and event.sequence >= min(starts):
                raise ValueError("partition allocation provenance was recorded after ATTEMPT_STARTED")
            records[record.decision_id] = record
        return records

    def _parse_provenance(self, event: LedgerEvent) -> _ProvenanceRecord:
        payload = event.payload
        if payload.get("schema") != _PARTITION_ALLOCATION_SCHEMA:
            raise ValueError("invalid partition allocation provenance schema")
        decision_id = self._text(payload, "decision_id")
        if event.thread_id is None:
            raise ValueError("partition allocation provenance is missing thread_id")
        partition_plan_revision = self._non_negative_int(
            payload,
            "partition_plan_revision",
        )
        shard_count = self._positive_int(payload, "shard_count")
        batch_limit = self._positive_int(payload, "batch_limit")
        width = self._positive_int(payload, "width")
        raw_partitions = payload.get("partitions")
        if not isinstance(raw_partitions, list) or len(raw_partitions) != width:
            raise ValueError("partition allocation provenance has invalid partition list")

        partitions: list[HistoricalIntegrationPartition] = []
        seen_partition_ids: set[str] = set()
        seen_evidence_ids: set[str] = set()
        for raw in raw_partitions:
            if not isinstance(raw, Mapping):
                raise ValueError("partition allocation entry must be an object")
            partition_id = self._text(raw, "partition_id")
            shard_index = self._non_negative_int(raw, "shard_index")
            backlog_count = self._positive_int(raw, "backlog_count")
            oldest_pending_sequence = self._non_negative_int(
                raw,
                "oldest_pending_sequence",
            )
            evidence_ids = self._string_tuple(raw, "evidence_ids")
            if not evidence_ids:
                raise ValueError("partition allocation entry must assign evidence")
            if len(evidence_ids) > batch_limit:
                raise ValueError("partition allocation entry exceeds recorded batch limit")
            if shard_index >= shard_count:
                raise ValueError("partition allocation shard index exceeds shard count")
            if partition_id in seen_partition_ids:
                raise ValueError("partition allocation contains duplicate partition IDs")
            overlap = seen_evidence_ids.intersection(evidence_ids)
            if overlap:
                raise ValueError("partition allocation contains overlapping evidence authority")
            seen_partition_ids.add(partition_id)
            seen_evidence_ids.update(evidence_ids)
            partitions.append(
                HistoricalIntegrationPartition(
                    partition_id=partition_id,
                    shard_index=shard_index,
                    backlog_count=backlog_count,
                    oldest_pending_sequence=oldest_pending_sequence,
                    evidence_ids=evidence_ids,
                )
            )

        if tuple(event.reference_ids) != tuple(
            partition.partition_id for partition in partitions
        ):
            raise ValueError(
                "partition allocation event references do not match payload partition order"
            )

        return _ProvenanceRecord(
            event_id=event.event_id,
            sequence=event.sequence,
            decision_id=decision_id,
            thread_id=event.thread_id,
            partition_plan_revision=partition_plan_revision,
            shard_count=shard_count,
            batch_limit=batch_limit,
            width=width,
            partitions=tuple(partitions),
        )

    @staticmethod
    def _join(
        allocation: IntegrationAllocationOutcome,
        provenance: _ProvenanceRecord,
    ) -> PartitionedIntegrationLineage:
        by_evidence: dict[tuple[str, ...], HistoricalIntegrationPartition] = {
            partition.evidence_ids: partition for partition in provenance.partitions
        }
        if len(by_evidence) != len(provenance.partitions):
            raise ValueError("partition allocation has ambiguous evidence assignments")

        used_partition_ids: set[str] = set()
        lineages: list[PartitionAttemptLineage] = []
        for attempt in allocation.attempts:
            partition = by_evidence.get(attempt.input_evidence_ids)
            if partition is None:
                raise ValueError(
                    "integration attempt input does not match durable partition allocation"
                )
            if partition.partition_id in used_partition_ids:
                raise ValueError("one durable integration partition was started more than once")
            used_partition_ids.add(partition.partition_id)
            lineages.append(PartitionAttemptLineage(partition=partition, attempt=attempt))

        unstarted = tuple(
            partition.partition_id
            for partition in provenance.partitions
            if partition.partition_id not in used_partition_ids
        )
        return PartitionedIntegrationLineage(
            allocation=allocation,
            provenance_event_id=provenance.event_id,
            provenance_sequence=provenance.sequence,
            partition_plan_revision=provenance.partition_plan_revision,
            shard_count=provenance.shard_count,
            batch_limit=provenance.batch_limit,
            partitions=provenance.partitions,
            partition_attempts=tuple(lineages),
            unstarted_partition_ids=unstarted,
        )

    @staticmethod
    def _text(payload: Mapping[str, Any], key: str) -> str:
        value = payload.get(key)
        if not isinstance(value, str) or not value:
            raise ValueError(f"partition allocation provenance has invalid {key}")
        return value

    @staticmethod
    def _positive_int(payload: Mapping[str, Any], key: str) -> int:
        value = payload.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"partition allocation provenance has invalid {key}")
        return value

    @staticmethod
    def _non_negative_int(payload: Mapping[str, Any], key: str) -> int:
        value = payload.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"partition allocation provenance has invalid {key}")
        return value

    @staticmethod
    def _string_tuple(payload: Mapping[str, Any], key: str) -> tuple[str, ...]:
        value = payload.get(key)
        if not isinstance(value, list) or any(
            not isinstance(item, str) or not item for item in value
        ):
            raise ValueError(f"partition allocation provenance has invalid {key}")
        return tuple(value)
