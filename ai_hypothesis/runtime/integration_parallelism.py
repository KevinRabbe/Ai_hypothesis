"""Optional parallel integration over deterministic evidence partitions.

The baseline Scheduler v0 remains unchanged. This module wraps an existing scheduler and
context provider only for backpressure-driven synthesis so one hot Work Thread can spend
multiple homogeneous workers on non-overlapping integration partitions in one neural batch.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, replace
from typing import Protocol, Sequence

from .contracts import ProjectedState, SchedulerAction, SchedulerDecision
from .control import ContextProvider, WorkPreparation, WorkPreparationBatch
from .integration_partitions import (
    IntegrationPartition,
    IntegrationPartitionPlan,
    IntegrationPartitionProjector,
    prepare_partition_integration_work,
)
from .ledger import SQLiteResearchLedger
from .scheduler import SchedulableThread


_PARTITION_ALLOCATION_EVENT = "INTEGRATION_PARTITION_ALLOCATION_RECORDED"
_PARTITION_ALLOCATION_SCHEMA = "integration-partition-allocation-v0"


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
class IntegrationParallelismConfig:
    max_integration_width: int = 4

    def validate(self) -> None:
        if self.max_integration_width <= 0:
            raise ValueError("max_integration_width must be positive")


class IntegrationPartitionAllocator:
    """Select distinct thread-owned partitions from one immutable projection plan."""

    def __init__(self, projector: IntegrationPartitionProjector | None = None) -> None:
        self.projector = projector or IntegrationPartitionProjector()

    def plan(self, ledger: SQLiteResearchLedger) -> IntegrationPartitionPlan:
        return self.projector.project(ledger.read_all_events())

    @staticmethod
    def ordered_for_thread(
        plan: IntegrationPartitionPlan,
        thread_id: str,
    ) -> tuple[IntegrationPartition, ...]:
        partitions = plan.for_thread(thread_id)
        return tuple(
            sorted(
                partitions,
                key=lambda partition: (
                    -partition.backlog_count,
                    partition.oldest_pending_sequence,
                    partition.shard_index,
                    partition.partition_id,
                ),
            )
        )

    def available_width(self, plan: IntegrationPartitionPlan, thread_id: str) -> int:
        return len(self.ordered_for_thread(plan, thread_id))

    def prepare_batch(
        self,
        plan: IntegrationPartitionPlan,
        thread_id: str,
        *,
        width: int,
    ) -> WorkPreparationBatch:
        if width <= 0:
            raise ValueError("integration width must be positive")
        partitions = self.ordered_for_thread(plan, thread_id)
        if width > len(partitions):
            raise ValueError("requested integration width exceeds available non-empty partitions")

        selected = partitions[:width]
        items = tuple(
            prepare_partition_integration_work(plan, partition.partition_id)
            for partition in selected
        )
        reference_ids = [
            evidence_id
            for item in items
            for evidence_id in item.reference_ids
        ]
        if len(reference_ids) != len(set(reference_ids)):
            raise ValueError("partitioned integration batch contains overlapping evidence authority")

        batch = WorkPreparationBatch(items=items)
        batch.validate(expected_width=width)
        return batch


class PartitionedBackpressureScheduler:
    """Widen only backpressure synthesis using the currently available partition count."""

    def __init__(
        self,
        delegate: SchedulerLike,
        *,
        ledger: SQLiteResearchLedger,
        allocator: IntegrationPartitionAllocator,
        config: IntegrationParallelismConfig | None = None,
    ) -> None:
        self.delegate = delegate
        self.ledger = ledger
        self.allocator = allocator
        self.config = config or IntegrationParallelismConfig()
        self.config.validate()

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

        plan = self.allocator.plan(self.ledger)
        available = self.allocator.available_width(plan, decision.thread_id)
        if available <= 0:
            # Keep the delegate decision intact; the normal purpose router will reject an
            # impossible empty backpressure synthesis rather than hiding the inconsistency.
            return decision

        width = min(self.config.max_integration_width, max_width, available)
        reasons = tuple(dict.fromkeys((*decision.reason_codes, "PARTITIONED_INTEGRATION")))
        widened = replace(decision, width=width, reason_codes=reasons)
        widened.validate()
        return widened

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


class PartitionedIntegrationContextRouter:
    """Turn partitioned synthesis width into distinct bounded WorkPreparations."""

    def __init__(
        self,
        *,
        ledger: SQLiteResearchLedger,
        fallback: ContextProvider,
        allocator: IntegrationPartitionAllocator,
    ) -> None:
        fallback_ledger = getattr(fallback, "ledger", None)
        if fallback_ledger is not None and fallback_ledger is not ledger:
            raise ValueError("partitioned context router and fallback must use the same Research Ledger")
        self.ledger = ledger
        self.fallback = fallback
        self.allocator = allocator

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

        plan = self.allocator.plan(self.ledger)
        batch = self.allocator.prepare_batch(
            plan,
            state.thread_id,
            width=decision.width,
        )
        selected = self.allocator.ordered_for_thread(plan, state.thread_id)[: decision.width]
        self._record_partition_allocation(state, decision, plan, selected)
        return batch

    def _record_partition_allocation(
        self,
        state: ProjectedState,
        decision: SchedulerDecision,
        plan: IntegrationPartitionPlan,
        partitions: Sequence[IntegrationPartition],
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
                or dict(existing.payload) != payload
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
