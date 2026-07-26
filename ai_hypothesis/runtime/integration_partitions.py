"""Mechanical partitioning of pending evidence for scalable integration.

This is deliberately not semantic clustering. Pending evidence remains owned by its source
Work Thread and is deterministically hash-sharded inside that thread. The projection keeps
full backlog counts but only a bounded next batch of compact evidence records per partition.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Mapping, Sequence

from .contracts import LedgerEvent
from .control import WorkPreparation
from .integration import PendingEvidence


@dataclass(frozen=True, slots=True)
class IntegrationPartitionConfig:
    shard_count: int = 8
    batch_limit: int = 32

    def validate(self) -> None:
        if self.shard_count <= 0:
            raise ValueError("shard_count must be positive")
        if self.batch_limit <= 0:
            raise ValueError("batch_limit must be positive")


@dataclass(frozen=True, slots=True)
class IntegrationPartition:
    partition_id: str
    thread_id: str | None
    shard_index: int
    shard_count: int
    backlog_count: int
    oldest_pending_sequence: int
    records: tuple[PendingEvidence, ...]

    @property
    def evidence_ids(self) -> tuple[str, ...]:
        return tuple(record.evidence_id for record in self.records)

    @property
    def causal_event_ids(self) -> tuple[str, ...]:
        return tuple(record.event_id for record in self.records)

    @property
    def buffered_count(self) -> int:
        return len(self.records)

    def validate(self, *, batch_limit: int) -> None:
        if not self.partition_id or not self.partition_id.strip():
            raise ValueError("partition_id must be non-empty")
        if self.thread_id is not None and not self.thread_id.strip():
            raise ValueError("thread_id must be non-empty when supplied")
        if self.shard_count <= 0:
            raise ValueError("shard_count must be positive")
        if not 0 <= self.shard_index < self.shard_count:
            raise ValueError("shard_index is outside shard_count")
        if self.backlog_count <= 0:
            raise ValueError("integration partitions must contain pending evidence")
        if self.oldest_pending_sequence < 0:
            raise ValueError("oldest_pending_sequence must be non-negative")
        if not self.records:
            raise ValueError("non-empty partition must buffer at least one pending record")
        if len(self.records) > batch_limit:
            raise ValueError("partition buffered records exceed batch_limit")
        if len({record.evidence_id for record in self.records}) != len(self.records):
            raise ValueError("partition buffered evidence IDs must be unique")
        if any(record.thread_id != self.thread_id for record in self.records):
            raise ValueError("partition contains evidence owned by another Work Thread")


@dataclass(frozen=True, slots=True)
class IntegrationPartitionPlan:
    revision: int
    shard_count: int
    batch_limit: int
    total_backlog_count: int
    partitions: tuple[IntegrationPartition, ...]

    def validate(self) -> None:
        if self.revision < 0:
            raise ValueError("revision must be non-negative")
        if self.shard_count <= 0:
            raise ValueError("shard_count must be positive")
        if self.batch_limit <= 0:
            raise ValueError("batch_limit must be positive")
        if self.total_backlog_count < 0:
            raise ValueError("total_backlog_count must be non-negative")
        if sum(partition.backlog_count for partition in self.partitions) != self.total_backlog_count:
            raise ValueError("partition backlog counts do not cover the global backlog exactly")
        partition_ids = [partition.partition_id for partition in self.partitions]
        if len(set(partition_ids)) != len(partition_ids):
            raise ValueError("partition IDs must be unique")
        for partition in self.partitions:
            if partition.shard_count != self.shard_count:
                raise ValueError("partition shard_count does not match plan")
            partition.validate(batch_limit=self.batch_limit)

    def for_thread(self, thread_id: str) -> tuple[IntegrationPartition, ...]:
        if not thread_id or not thread_id.strip():
            raise ValueError("thread_id must be non-empty")
        return tuple(
            partition for partition in self.partitions if partition.thread_id == thread_id
        )

    def get(self, partition_id: str) -> IntegrationPartition | None:
        if not partition_id or not partition_id.strip():
            raise ValueError("partition_id must be non-empty")
        for partition in self.partitions:
            if partition.partition_id == partition_id:
                return partition
        return None


@dataclass(slots=True)
class _MutablePartition:
    thread_id: str | None
    shard_index: int
    backlog_count: int = 0
    oldest_pending_sequence: int | None = None
    records: list[PendingEvidence] | None = None


class IntegrationPartitionProjector:
    """Derive deterministic thread-owned integration partitions from ledger history."""

    def __init__(self, config: IntegrationPartitionConfig | None = None) -> None:
        self.config = config or IntegrationPartitionConfig()
        self.config.validate()

    def project(self, events: Sequence[LedgerEvent]) -> IntegrationPartitionPlan:
        revision = 0
        previous_sequence = -1
        dispositioned: set[str] = set()

        # First pass resolves final disposition state. This prevents an early evidence record
        # from occupying a bounded partition buffer after a later disposition removed it.
        for event in events:
            event.validate()
            if event.sequence <= previous_sequence:
                raise ValueError("events must be in strictly increasing sequence order")
            previous_sequence = event.sequence
            revision = event.sequence
            if event.event_type == "INTEGRATION_DISPOSITION_RECORDED":
                dispositioned.update(event.reference_ids)

        partitions: dict[tuple[str | None, int], _MutablePartition] = {}
        seen_evidence: set[str] = set()

        # Second pass counts the full pending backlog while buffering only the next bounded
        # batch per partition.
        for event in events:
            if event.event_type != "EVIDENCE_ADDED":
                continue
            evidence_id = event.payload.get("evidence_id")
            if not isinstance(evidence_id, str) or not evidence_id:
                raise ValueError("EVIDENCE_ADDED is missing evidence_id")
            if evidence_id in seen_evidence:
                raise ValueError(f"duplicate durable evidence ID {evidence_id!r}")
            seen_evidence.add(evidence_id)
            if evidence_id in dispositioned:
                continue

            shard_index = self._shard_for(evidence_id)
            key = (event.thread_id, shard_index)
            mutable = partitions.get(key)
            if mutable is None:
                mutable = _MutablePartition(
                    thread_id=event.thread_id,
                    shard_index=shard_index,
                    records=[],
                )
                partitions[key] = mutable

            mutable.backlog_count += 1
            if mutable.oldest_pending_sequence is None:
                mutable.oldest_pending_sequence = event.sequence

            assert mutable.records is not None
            if len(mutable.records) >= self.config.batch_limit:
                continue
            data = event.payload.get("data")
            mutable.records.append(
                PendingEvidence(
                    evidence_id=evidence_id,
                    event_id=event.event_id,
                    sequence=event.sequence,
                    thread_id=event.thread_id,
                    kind=str(event.payload.get("kind", "UNKNOWN")),
                    summary=str(event.payload.get("summary", "")),
                    source_reference_ids=tuple(
                        reference_id
                        for reference_id in event.reference_ids
                        if reference_id != evidence_id
                    ),
                    strength=self._optional_float(event.payload.get("strength")),
                    uncertainty=self._optional_float(event.payload.get("uncertainty")),
                    data=dict(data) if isinstance(data, Mapping) else {},
                )
            )

        projected = tuple(
            self._freeze_partition(mutable)
            for _, mutable in sorted(
                partitions.items(),
                key=lambda item: (
                    "" if item[0][0] is None else item[0][0],
                    item[0][1],
                ),
            )
        )
        plan = IntegrationPartitionPlan(
            revision=revision,
            shard_count=self.config.shard_count,
            batch_limit=self.config.batch_limit,
            total_backlog_count=sum(partition.backlog_count for partition in projected),
            partitions=projected,
        )
        plan.validate()
        return plan

    def _freeze_partition(self, mutable: _MutablePartition) -> IntegrationPartition:
        if mutable.oldest_pending_sequence is None or mutable.records is None:
            raise ValueError("cannot freeze an empty integration partition")
        partition = IntegrationPartition(
            partition_id=self._partition_id(mutable.thread_id, mutable.shard_index),
            thread_id=mutable.thread_id,
            shard_index=mutable.shard_index,
            shard_count=self.config.shard_count,
            backlog_count=mutable.backlog_count,
            oldest_pending_sequence=mutable.oldest_pending_sequence,
            records=tuple(mutable.records),
        )
        partition.validate(batch_limit=self.config.batch_limit)
        return partition

    def _shard_for(self, evidence_id: str) -> int:
        digest = hashlib.sha256(evidence_id.encode("utf-8")).digest()
        return int.from_bytes(digest[:8], "big") % self.config.shard_count

    def _partition_id(self, thread_id: str | None, shard_index: int) -> str:
        owner = "<global>" if thread_id is None else thread_id
        owner_digest = hashlib.sha256(owner.encode("utf-8")).hexdigest()[:16]
        return (
            f"integration-partition-v0:{owner_digest}:"
            f"{shard_index:04d}-of-{self.config.shard_count:04d}"
        )

    @staticmethod
    def _optional_float(value: object) -> float | None:
        if value is None:
            return None
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError("durable evidence scalar must be numeric or null")
        return float(value)


def prepare_partition_integration_work(
    partition: IntegrationPartition,
    *,
    revision: int,
    batch_limit: int,
) -> WorkPreparation:
    """Turn one already-bounded integration partition into ordinary synthesis work."""

    if revision < 0:
        raise ValueError("revision must be non-negative")
    if batch_limit <= 0:
        raise ValueError("batch_limit must be positive")
    partition.validate(batch_limit=batch_limit)

    preparation = WorkPreparation(
        reference_ids=partition.evidence_ids,
        context={
            "context_view": "SYNTHESIZE",
            "synthesis_mode": "INTEGRATION_PARTITION",
            "integration_revision": revision,
            "integration_partition": {
                "partition_id": partition.partition_id,
                "thread_id": partition.thread_id,
                "shard_index": partition.shard_index,
                "shard_count": partition.shard_count,
                "backlog_count": partition.backlog_count,
                "oldest_pending_sequence": partition.oldest_pending_sequence,
            },
            "pending_evidence": tuple(
                record.to_context_record() for record in partition.records
            ),
            "causal_event_ids": list(partition.causal_event_ids),
        },
        constraints={
            "max_pending_evidence": batch_limit,
            "emit_structured_knowledge_deltas": True,
            "disposition_consumed_evidence": True,
            "preserve_source_thread_ownership": True,
        },
    )
    preparation.validate()
    return preparation
