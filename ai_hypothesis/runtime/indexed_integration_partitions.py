"""Partitioned raw-integration planning over the incremental integration materialization.

The existing IntegrationPartitionProjector remains the semantic authority for deterministic shard
assignment. Each newly observed evidence identity is classified by that projector exactly once;
normal scheduler cycles then use indexed pending counts and bounded canonical payload lookups.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Protocol, Sequence
from dataclasses import dataclass, replace

from .contracts import ProjectedState, SchedulerAction, SchedulerDecision
from .control import ContextProvider, WorkPreparation, WorkPreparationBatch
from .indexed_control import IndexedRuntimeIntegrationTracker
from .integration import PendingEvidence
from .integration_parallelism import IntegrationParallelismConfig
from .integration_partitions import IntegrationPartitionProjector
from .ledger import SQLiteResearchLedger
from .scheduler import SchedulableThread


_PARTITION_ALLOCATION_EVENT = "INTEGRATION_PARTITION_ALLOCATION_RECORDED"
_PARTITION_ALLOCATION_SCHEMA = "integration-partition-allocation-v0"


class CapturedRevisionProvider(Protocol):
    @property
    def current_revision(self) -> int:
        ...


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
class IndexedIntegrationPartition:
    partition_id: str
    thread_id: str
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

    def validate(self, *, batch_limit: int) -> None:
        if not self.partition_id or not self.partition_id.strip():
            raise ValueError("partition_id must be non-empty")
        if not self.thread_id or not self.thread_id.strip():
            raise ValueError("partition thread_id must be non-empty")
        if self.shard_count <= 0:
            raise ValueError("partition shard_count must be positive")
        if self.shard_index < 0 or self.shard_index >= self.shard_count:
            raise ValueError("partition shard_index must be inside shard_count")
        if self.backlog_count <= 0:
            raise ValueError("partition backlog_count must be positive")
        if self.oldest_pending_sequence < 0:
            raise ValueError("partition oldest_pending_sequence must be non-negative")
        if not self.records:
            raise ValueError("indexed integration partition must contain bounded records")
        if len(self.records) > batch_limit:
            raise ValueError("indexed integration partition exceeds batch limit")
        if len(self.records) > self.backlog_count:
            raise ValueError("partition record buffer cannot exceed backlog count")
        ids = self.evidence_ids
        if len(ids) != len(set(ids)):
            raise ValueError("partition evidence IDs must be unique")
        if any(record.thread_id != self.thread_id for record in self.records):
            raise ValueError("partition records must preserve source-thread ownership")


@dataclass(frozen=True, slots=True)
class IndexedIntegrationPartitionPlan:
    revision: int
    thread_id: str
    shard_count: int
    batch_limit: int
    partitions: tuple[IndexedIntegrationPartition, ...]

    def validate(self) -> None:
        if self.revision < 0:
            raise ValueError("partition plan revision must be non-negative")
        if not self.thread_id or not self.thread_id.strip():
            raise ValueError("partition plan thread_id must be non-empty")
        if self.shard_count <= 0:
            raise ValueError("partition plan shard_count must be positive")
        if self.batch_limit <= 0:
            raise ValueError("partition plan batch_limit must be positive")
        seen_ids: set[str] = set()
        seen_evidence: set[str] = set()
        for partition in self.partitions:
            partition.validate(batch_limit=self.batch_limit)
            if partition.thread_id != self.thread_id:
                raise ValueError("partition plan cannot mix Work Thread ownership")
            if partition.shard_count != self.shard_count:
                raise ValueError("partition shard_count must match plan")
            if partition.partition_id in seen_ids:
                raise ValueError("partition IDs must be unique inside one plan")
            overlap = seen_evidence.intersection(partition.evidence_ids)
            if overlap:
                raise ValueError("partition plan contains overlapping evidence authority")
            seen_ids.add(partition.partition_id)
            seen_evidence.update(partition.evidence_ids)

    def ordered_for_execution(self) -> tuple[IndexedIntegrationPartition, ...]:
        return tuple(
            sorted(
                self.partitions,
                key=lambda partition: (
                    -partition.backlog_count,
                    partition.oldest_pending_sequence,
                    partition.shard_index,
                    partition.partition_id,
                ),
            )
        )


class IndexedIntegrationPartitionPlanner:
    """Build exact deterministic partition plans from indexed pending evidence."""

    def __init__(
        self,
        *,
        ledger: SQLiteResearchLedger,
        integration: IndexedRuntimeIntegrationTracker,
        reference_projector: IntegrationPartitionProjector | None = None,
    ) -> None:
        if integration.ledger is not ledger:
            raise ValueError("indexed partition planner and integration index must share one ledger")
        self.ledger = ledger
        self.integration = integration
        self.reference_projector = reference_projector or IntegrationPartitionProjector()
        config = getattr(self.reference_projector, "config", None)
        if config is None:
            raise ValueError("reference partition projector must expose its config")
        self.shard_count = int(config.shard_count)
        self.batch_limit = int(config.batch_limit)
        if self.shard_count <= 0 or self.batch_limit <= 0:
            raise ValueError("reference partition configuration must be positive")
        self._create_schema()
        self._bind_config()

    def plan(self, *, sequence: int, thread_id: str) -> IndexedIntegrationPartitionPlan:
        if sequence < 0:
            raise ValueError("sequence must be non-negative")
        if not thread_id or not thread_id.strip():
            raise ValueError("thread_id must be non-empty")
        with self.integration._lock:
            self.integration.sync_through(sequence)
            self._ensure_pending_assignments()
            partitions = tuple(
                partition
                for shard_index in range(self.shard_count)
                if (
                    partition := self._partition_for_shard(
                        thread_id=thread_id,
                        shard_index=shard_index,
                        revision=sequence,
                    )
                )
                is not None
            )
            plan = IndexedIntegrationPartitionPlan(
                revision=sequence,
                thread_id=thread_id,
                shard_count=self.shard_count,
                batch_limit=self.batch_limit,
                partitions=partitions,
            )
            plan.validate()
            return plan

    def prepare_batch(
        self,
        plan: IndexedIntegrationPartitionPlan,
        *,
        width: int,
    ) -> tuple[WorkPreparationBatch, tuple[IndexedIntegrationPartition, ...]]:
        plan.validate()
        if width <= 0:
            raise ValueError("integration width must be positive")
        ordered = plan.ordered_for_execution()
        if width > len(ordered):
            raise ValueError("requested integration width exceeds available non-empty partitions")
        selected = ordered[:width]
        items = tuple(self._prepare_partition(plan, partition) for partition in selected)
        all_ids = [evidence_id for item in items for evidence_id in item.reference_ids]
        if len(all_ids) != len(set(all_ids)):
            raise ValueError("partitioned integration batch contains overlapping evidence authority")
        batch = WorkPreparationBatch(items=items)
        batch.validate(expected_width=width)
        return batch, selected

    def _ensure_pending_assignments(self) -> None:
        rows = self.integration._connection.execute(
            """
            SELECT e.evidence_id, e.event_id, e.thread_id
            FROM integration_evidence AS e
            LEFT JOIN integration_partition_assignment AS a
              ON a.evidence_id = e.evidence_id
            WHERE e.first_disposition_sequence IS NULL
              AND e.thread_id IS NOT NULL
              AND a.evidence_id IS NULL
            ORDER BY e.created_sequence, e.evidence_id
            """
        ).fetchall()
        if not rows:
            return
        with self.integration._connection:
            for row in rows:
                evidence_id = str(row["evidence_id"])
                event = self.ledger.get_event(str(row["event_id"]))
                if event is None or event.event_type != "EVIDENCE_ADDED":
                    raise RuntimeError("partition assignment references missing evidence event")
                if event.payload.get("evidence_id") != evidence_id:
                    raise RuntimeError("partition assignment evidence identity no longer matches ledger")
                if event.thread_id is None:
                    raise ValueError("partitioned integration requires thread-owned evidence")
                one = self.reference_projector.project((event,))
                matching = tuple(
                    partition
                    for partition in one.for_thread(event.thread_id)
                    if evidence_id in partition.evidence_ids
                )
                if len(matching) != 1:
                    raise RuntimeError("reference partition projector did not classify evidence exactly once")
                partition = matching[0]
                if partition.shard_count != self.shard_count:
                    raise RuntimeError("reference partition projector returned inconsistent shard_count")
                self.integration._connection.execute(
                    """
                    INSERT INTO integration_partition_assignment(
                        evidence_id, shard_count, shard_index, partition_id
                    ) VALUES (?, ?, ?, ?)
                    """,
                    (
                        evidence_id,
                        self.shard_count,
                        partition.shard_index,
                        partition.partition_id,
                    ),
                )

    def _partition_for_shard(
        self,
        *,
        thread_id: str,
        shard_index: int,
        revision: int,
    ) -> IndexedIntegrationPartition | None:
        summary = self.integration._connection.execute(
            """
            SELECT
                COUNT(*) AS backlog_count,
                MIN(e.created_sequence) AS oldest_pending_sequence,
                MIN(a.partition_id) AS min_partition_id,
                MAX(a.partition_id) AS max_partition_id
            FROM integration_evidence AS e
            JOIN integration_partition_assignment AS a
              ON a.evidence_id = e.evidence_id
            WHERE e.first_disposition_sequence IS NULL
              AND e.thread_id = ?
              AND a.shard_count = ?
              AND a.shard_index = ?
            """,
            (thread_id, self.shard_count, shard_index),
        ).fetchone()
        backlog_count = int(summary["backlog_count"] or 0)
        if backlog_count == 0:
            return None
        if summary["min_partition_id"] != summary["max_partition_id"]:
            raise RuntimeError("one thread/shard resolved to multiple stable partition IDs")
        partition_id = str(summary["min_partition_id"])
        rows = self.integration._connection.execute(
            """
            SELECT e.evidence_id, e.event_id, e.created_sequence
            FROM integration_evidence AS e
            JOIN integration_partition_assignment AS a
              ON a.evidence_id = e.evidence_id
            WHERE e.first_disposition_sequence IS NULL
              AND e.thread_id = ?
              AND a.shard_count = ?
              AND a.shard_index = ?
            ORDER BY e.created_sequence, e.evidence_id
            LIMIT ?
            """,
            (thread_id, self.shard_count, shard_index, self.batch_limit),
        ).fetchall()
        records = tuple(self._pending_record(row) for row in rows)
        return IndexedIntegrationPartition(
            partition_id=partition_id,
            thread_id=thread_id,
            shard_index=shard_index,
            shard_count=self.shard_count,
            backlog_count=backlog_count,
            oldest_pending_sequence=int(summary["oldest_pending_sequence"]),
            records=records,
        )

    def _pending_record(self, row) -> PendingEvidence:
        event = self.ledger.get_event(str(row["event_id"]))
        if event is None or event.event_type != "EVIDENCE_ADDED":
            raise RuntimeError("indexed partition references missing evidence event")
        evidence_id = str(row["evidence_id"])
        if event.payload.get("evidence_id") != evidence_id:
            raise RuntimeError("indexed partition evidence identity no longer matches ledger")
        data = event.payload.get("data")
        return PendingEvidence(
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
            strength=self.integration._optional_float(event.payload.get("strength")),
            uncertainty=self.integration._optional_float(event.payload.get("uncertainty")),
            data=dict(data) if isinstance(data, Mapping) else {},
        )

    @staticmethod
    def _prepare_partition(
        plan: IndexedIntegrationPartitionPlan,
        partition: IndexedIntegrationPartition,
    ) -> WorkPreparation:
        preparation = WorkPreparation(
            reference_ids=partition.evidence_ids,
            context={
                "context_view": "SYNTHESIZE",
                "synthesis_mode": "INTEGRATION_PARTITION",
                "integration_revision": plan.revision,
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
                "max_pending_evidence": plan.batch_limit,
                "emit_structured_knowledge_deltas": True,
                "disposition_consumed_evidence": True,
                "preserve_source_thread_ownership": True,
            },
        )
        preparation.validate()
        return preparation

    def _create_schema(self) -> None:
        with self.integration._lock, self.integration._connection:
            self.integration._connection.execute(
                """
                CREATE TABLE IF NOT EXISTS integration_partition_index_meta(
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                )
                """
            )
            self.integration._connection.execute(
                """
                CREATE TABLE IF NOT EXISTS integration_partition_assignment(
                    evidence_id TEXT PRIMARY KEY,
                    shard_count INTEGER NOT NULL,
                    shard_index INTEGER NOT NULL,
                    partition_id TEXT NOT NULL
                )
                """
            )
            self.integration._connection.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_integration_partition_pending
                ON integration_partition_assignment(shard_count, shard_index, partition_id)
                """
            )

    def _bind_config(self) -> None:
        with self.integration._lock, self.integration._connection:
            row = self.integration._connection.execute(
                "SELECT value FROM integration_partition_index_meta WHERE key = 'shard_count'"
            ).fetchone()
            current = int(row["value"]) if row is not None else None
            if current is not None and current != self.shard_count:
                self.integration._connection.execute("DELETE FROM integration_partition_assignment")
            self.integration._connection.execute(
                """
                INSERT INTO integration_partition_index_meta(key, value)
                VALUES ('shard_count', ?)
                ON CONFLICT(key) DO UPDATE SET value = excluded.value
                """,
                (str(self.shard_count),),
            )


@dataclass(frozen=True, slots=True)
class _CachedPartitionAllocation:
    revision: int
    decision_id: str
    thread_id: str
    plan: IndexedIntegrationPartitionPlan
    selected: tuple[IndexedIntegrationPartition, ...]


class IndexedPartitionedBackpressureScheduler:
    """Widen backpressure synthesis from one pinned indexed partition plan."""

    def __init__(
        self,
        delegate: SchedulerLike,
        *,
        planner: IndexedIntegrationPartitionPlanner,
        revision_provider: CapturedRevisionProvider,
        config: IntegrationParallelismConfig | None = None,
    ) -> None:
        self.delegate = delegate
        self.planner = planner
        self.revision_provider = revision_provider
        self.config = config or IntegrationParallelismConfig()
        self.config.validate()
        self._cached_revision: int | None = None
        self._allocations: dict[str, _CachedPartitionAllocation] = {}

    def choose(
        self,
        candidates: Sequence[SchedulableThread],
        *,
        integration_backpressure: bool = False,
        max_width: int = 1,
    ) -> SchedulerDecision:
        if max_width <= 0:
            raise ValueError("max_width must be positive")
        decision = self.delegate.choose(
            candidates,
            integration_backpressure=integration_backpressure,
            max_width=max_width,
        )
        decision.validate()
        if not self._is_partitionable_backpressure(decision, integration_backpressure):
            return decision

        revision = self.revision_provider.current_revision
        self._refresh_revision(revision)
        plan = self.planner.plan(sequence=revision, thread_id=decision.thread_id)
        available = len(plan.partitions)
        if available <= 0:
            return decision
        width = min(self.config.max_integration_width, max_width, available)
        selected = plan.ordered_for_execution()[:width]
        reasons = tuple(dict.fromkeys((*decision.reason_codes, "PARTITIONED_INTEGRATION")))
        widened = replace(decision, width=width, reason_codes=reasons)
        widened.validate()
        self._allocations[widened.decision_id] = _CachedPartitionAllocation(
            revision=revision,
            decision_id=widened.decision_id,
            thread_id=widened.thread_id,
            plan=plan,
            selected=selected,
        )
        return widened

    def allocation_for(self, decision: SchedulerDecision) -> _CachedPartitionAllocation:
        decision.validate()
        allocation = self._allocations.get(decision.decision_id)
        if allocation is None:
            raise ValueError("partitioned integration decision has no cached indexed allocation")
        if allocation.revision != self.revision_provider.current_revision:
            raise RuntimeError("partitioned integration allocation no longer matches runtime snapshot")
        if allocation.thread_id != decision.thread_id:
            raise ValueError("cached partition allocation targets another Work Thread")
        if len(allocation.selected) != decision.width:
            raise ValueError("cached partition allocation width does not match scheduler decision")
        return allocation

    def _refresh_revision(self, revision: int) -> None:
        if revision == self._cached_revision:
            return
        self._cached_revision = revision
        self._allocations.clear()

    @staticmethod
    def _is_partitionable_backpressure(
        decision: SchedulerDecision,
        integration_backpressure: bool,
    ) -> bool:
        return (
            integration_backpressure
            and decision.action is SchedulerAction.SYNTHESIZE
            and "BACKPRESSURE" in decision.reason_codes
        )


class IndexedPartitionedIntegrationContextRouter:
    """Return cached partition contexts and persist exact allocation provenance."""

    def __init__(
        self,
        *,
        ledger: SQLiteResearchLedger,
        scheduler: IndexedPartitionedBackpressureScheduler,
        planner: IndexedIntegrationPartitionPlanner,
        fallback: ContextProvider,
    ) -> None:
        if planner.ledger is not ledger:
            raise ValueError("indexed partition context and planner must share one ledger")
        self.ledger = ledger
        self.scheduler = scheduler
        self.planner = planner
        self.fallback = fallback

    def __call__(
        self,
        state: ProjectedState,
        decision: SchedulerDecision,
    ) -> WorkPreparation | WorkPreparationBatch:
        state.validate()
        decision.validate()
        if decision.thread_id != state.thread_id:
            raise ValueError("scheduler decision and projected state refer to different threads")
        if not self._is_partitioned_integration(decision):
            return self.fallback(state, decision)

        allocation = self.scheduler.allocation_for(decision)
        batch, selected = self.planner.prepare_batch(
            allocation.plan,
            width=decision.width,
        )
        if selected != allocation.selected:
            raise RuntimeError("cached indexed partition selection changed before context preparation")
        self._record_partition_allocation(state, decision, allocation.plan, selected)
        return batch

    def _record_partition_allocation(
        self,
        state: ProjectedState,
        decision: SchedulerDecision,
        plan: IndexedIntegrationPartitionPlan,
        partitions: Sequence[IndexedIntegrationPartition],
    ) -> None:
        if len(partitions) != decision.width:
            raise ValueError("partition provenance width does not match scheduler decision")
        event_id = self._allocation_event_id(decision.decision_id)
        reference_ids = tuple(partition.partition_id for partition in partitions)
        payload = {
            "schema": _PARTITION_ALLOCATION_SCHEMA,
            "decision_id": decision.decision_id,
            "decision_projection_revision": decision.projection_revision,
            "partition_plan_revision": plan.revision,
            "shard_count": plan.shard_count,
            "batch_limit": plan.batch_limit,
            "width": decision.width,
            "partitions": [
                {
                    "partition_id": partition.partition_id,
                    "shard_index": partition.shard_index,
                    "backlog_count": partition.backlog_count,
                    "oldest_pending_sequence": partition.oldest_pending_sequence,
                    "evidence_ids": list(partition.evidence_ids),
                }
                for partition in partitions
            ],
        }
        existing = self.ledger.get_event(event_id)
        if existing is not None:
            if (
                existing.event_type != _PARTITION_ALLOCATION_EVENT
                or existing.thread_id != state.thread_id
                or existing.reference_ids != reference_ids
                or self._assignment_identity(existing.payload)
                != self._assignment_identity(payload)
            ):
                raise ValueError("conflicting durable partition allocation provenance")
            return
        self.ledger.append_event(
            event_type=_PARTITION_ALLOCATION_EVENT,
            event_id=event_id,
            thread_id=state.thread_id,
            reference_ids=reference_ids,
            payload=payload,
        )

    @staticmethod
    def _assignment_identity(payload: Mapping[str, object]) -> tuple[object, ...]:
        raw_partitions = payload.get("partitions")
        if not isinstance(raw_partitions, list):
            return ("INVALID",)
        partitions: list[tuple[object, ...]] = []
        for raw in raw_partitions:
            if not isinstance(raw, Mapping):
                return ("INVALID",)
            evidence = raw.get("evidence_ids")
            if not isinstance(evidence, list):
                return ("INVALID",)
            partitions.append(
                (
                    raw.get("partition_id"),
                    raw.get("shard_index"),
                    tuple(evidence),
                )
            )
        return (
            payload.get("schema"),
            payload.get("decision_id"),
            payload.get("shard_count"),
            payload.get("batch_limit"),
            payload.get("width"),
            tuple(partitions),
        )

    @staticmethod
    def _allocation_event_id(decision_id: str) -> str:
        digest = hashlib.sha256(decision_id.encode("utf-8")).hexdigest()
        return f"integration-partition-allocation-v0:{digest}"

    @staticmethod
    def _is_partitioned_integration(decision: SchedulerDecision) -> bool:
        return (
            decision.action is SchedulerAction.SYNTHESIZE
            and "BACKPRESSURE" in decision.reason_codes
            and "PARTITIONED_INTEGRATION" in decision.reason_codes
        )
