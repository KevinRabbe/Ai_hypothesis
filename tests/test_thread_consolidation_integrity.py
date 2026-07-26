from __future__ import annotations

import unittest

from ai_hypothesis.runtime.contracts import LedgerEvent
from ai_hypothesis.runtime.integration_allocation_outcomes import IntegrationAllocationOutcome
from ai_hypothesis.runtime.integration_partition_lineage import (
    PartitionedIntegrationLineage,
    PartitionedIntegrationLineageSnapshot,
)
from ai_hypothesis.runtime.knowledge import KnowledgeSnapshot
from ai_hypothesis.runtime.thread_consolidation import ThreadConsolidationPlanner


_SCHEMA = "runtime-event-v0"


def _event(
    sequence: int,
    event_type: str,
    *,
    attempt_id: str | None = None,
    reference_ids: tuple[str, ...] = (),
    payload: dict | None = None,
) -> LedgerEvent:
    return LedgerEvent(
        event_id=f"event-{sequence}",
        event_type=event_type,
        sequence=sequence,
        payload_schema=_SCHEMA,
        thread_id="thread-a",
        attempt_id=attempt_id,
        reference_ids=reference_ids,
        payload=payload or {},
    )


class _FakeLineageProjector:
    def __init__(self, snapshot: PartitionedIntegrationLineageSnapshot) -> None:
        self.snapshot = snapshot

    def project(self, events):
        return self.snapshot


class _EmptyKnowledgeProjector:
    def project(self, events):
        return KnowledgeSnapshot(revision=0, records=())


class ThreadConsolidationIntegrityTests(unittest.TestCase):
    def test_missing_provenance_on_unrelated_thread_does_not_block_selected_thread(self) -> None:
        allocation_a = IntegrationAllocationOutcome(
            decision_id="decision-a",
            thread_id="thread-a",
            width=1,
            projection_revision=0,
            reason_codes=("BACKPRESSURE", "PARTITIONED_INTEGRATION"),
            attempts=(),
        )
        allocation_b = IntegrationAllocationOutcome(
            decision_id="decision-b",
            thread_id="thread-b",
            width=1,
            projection_revision=0,
            reason_codes=("BACKPRESSURE", "PARTITIONED_INTEGRATION"),
            attempts=(),
        )
        lineage = PartitionedIntegrationLineageSnapshot(
            records=(
                PartitionedIntegrationLineage(
                    allocation=allocation_a,
                    provenance_event_id="provenance-a",
                    provenance_sequence=1,
                    partition_plan_revision=0,
                    shard_count=8,
                    batch_limit=32,
                    partitions=(),
                    partition_attempts=(),
                    unstarted_partition_ids=(),
                ),
                PartitionedIntegrationLineage(
                    allocation=allocation_b,
                    provenance_event_id=None,
                    provenance_sequence=None,
                    partition_plan_revision=None,
                    shard_count=None,
                    batch_limit=None,
                    partitions=(),
                    partition_attempts=(),
                    unstarted_partition_ids=(),
                ),
            ),
            missing_provenance_decision_ids=("decision-b",),
        )
        planner = ThreadConsolidationPlanner(
            lineage_projector=_FakeLineageProjector(lineage),
            knowledge_projector=_EmptyKnowledgeProjector(),
        )

        plan = planner.plan((), thread_id="thread-a")

        self.assertEqual(plan.pending_source_count, 0)
        self.assertEqual(plan.pending_partition_count, 0)
        self.assertFalse(plan.ready)

    def test_consolidation_cannot_consume_knowledge_created_later(self) -> None:
        events = (
            _event(1, "EVIDENCE_ADDED", reference_ids=("e1",), payload={"evidence_id": "e1"}),
            _event(
                2,
                "SCHEDULER_DECISION_RECORDED",
                payload={
                    "decision_id": "decision-a",
                    "action": "SYNTHESIZE",
                    "purpose": "SYNTHESIZE",
                    "width": 1,
                    "reason_codes": ["BACKPRESSURE", "PARTITIONED_INTEGRATION"],
                    "projection_revision": 1,
                    "integration_backpressure": True,
                    "max_width": 1,
                },
            ),
            _event(
                3,
                "INTEGRATION_PARTITION_ALLOCATION_RECORDED",
                reference_ids=("partition-a",),
                payload={
                    "schema": "integration-partition-allocation-v0",
                    "decision_id": "decision-a",
                    "decision_projection_revision": 1,
                    "partition_plan_revision": 2,
                    "shard_count": 8,
                    "batch_limit": 2,
                    "width": 1,
                    "partitions": [
                        {
                            "partition_id": "partition-a",
                            "shard_index": 1,
                            "backlog_count": 1,
                            "oldest_pending_sequence": 1,
                            "evidence_ids": ["e1"],
                        }
                    ],
                },
            ),
            _event(
                4,
                "ATTEMPT_STARTED",
                attempt_id="attempt-a",
                reference_ids=("e1",),
                payload={
                    "work_item_id": "work-a",
                    "worker_id": "worker-a",
                    "purpose": "SYNTHESIZE",
                    "projection_revision": 1,
                    "scheduler_decision_id": "decision-a",
                    "scope_region_ids": [],
                },
            ),
            _event(
                5,
                "KNOWLEDGE_DELTA_RECORDED",
                reference_ids=("future-consolidation", "k1"),
                payload={
                    "delta_id": "future-consolidation",
                    "kind": "THREAD_CONSOLIDATION",
                    "summary": "invalid future reference",
                    "source_reference_ids": ["k1"],
                    "causal_event_ids": [],
                },
            ),
            _event(
                6,
                "KNOWLEDGE_DELTA_RECORDED",
                attempt_id="attempt-a",
                reference_ids=("k1", "e1"),
                payload={
                    "delta_id": "k1",
                    "kind": "INTEGRATION_SUMMARY",
                    "summary": "partition finding",
                    "source_reference_ids": ["e1"],
                    "causal_event_ids": [],
                },
            ),
            _event(7, "ATTEMPT_COMPLETED", attempt_id="attempt-a", payload={"progress_made": True}),
        )

        with self.assertRaisesRegex(ValueError, "knowledge created after consolidation"):
            ThreadConsolidationPlanner().plan(events, thread_id="thread-a")


if __name__ == "__main__":
    unittest.main()
