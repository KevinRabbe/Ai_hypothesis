from __future__ import annotations

import unittest

from ai_hypothesis.runtime import (
    AttemptResult,
    AttemptStatus,
    KnowledgeDelta,
    SQLiteResearchLedger,
    WorkItem,
    WorkPurpose,
    WorkerAssignment,
    WorkerRuntime,
)
from ai_hypothesis.runtime.contracts import LedgerEvent
from ai_hypothesis.runtime.knowledge import KnowledgeStateProjector, KnowledgeStatus
from ai_hypothesis.runtime.thread_consolidation import (
    ThreadConsolidationConfig,
    ThreadConsolidationPlanner,
    prepare_thread_consolidation_work,
)


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


def _partition_history() -> tuple[LedgerEvent, ...]:
    return (
        _event(1, "EVIDENCE_ADDED", reference_ids=("e1",), payload={"evidence_id": "e1"}),
        _event(2, "EVIDENCE_ADDED", reference_ids=("e2",), payload={"evidence_id": "e2"}),
        _event(3, "EVIDENCE_ADDED", reference_ids=("e3",), payload={"evidence_id": "e3"}),
        _event(4, "EVIDENCE_ADDED", reference_ids=("e4",), payload={"evidence_id": "e4"}),
        _event(
            5,
            "SCHEDULER_DECISION_RECORDED",
            payload={
                "decision_id": "decision-a",
                "action": "SYNTHESIZE",
                "purpose": "SYNTHESIZE",
                "width": 2,
                "reason_codes": ["BACKPRESSURE", "PARTITIONED_INTEGRATION"],
                "projection_revision": 4,
                "integration_backpressure": True,
                "max_width": 2,
            },
        ),
        _event(
            6,
            "INTEGRATION_PARTITION_ALLOCATION_RECORDED",
            reference_ids=("partition-a", "partition-b"),
            payload={
                "schema": "integration-partition-allocation-v0",
                "decision_id": "decision-a",
                "decision_projection_revision": 4,
                "partition_plan_revision": 5,
                "shard_count": 8,
                "batch_limit": 2,
                "width": 2,
                "partitions": [
                    {
                        "partition_id": "partition-a",
                        "shard_index": 1,
                        "backlog_count": 4,
                        "oldest_pending_sequence": 1,
                        "evidence_ids": ["e1", "e2"],
                    },
                    {
                        "partition_id": "partition-b",
                        "shard_index": 6,
                        "backlog_count": 3,
                        "oldest_pending_sequence": 3,
                        "evidence_ids": ["e3", "e4"],
                    },
                ],
            },
        ),
        _event(
            7,
            "ATTEMPT_STARTED",
            attempt_id="attempt-a",
            reference_ids=("e1", "e2"),
            payload={
                "work_item_id": "work-a",
                "worker_id": "worker-a",
                "purpose": "SYNTHESIZE",
                "projection_revision": 4,
                "scheduler_decision_id": "decision-a",
                "scope_region_ids": [],
            },
        ),
        _event(
            8,
            "ATTEMPT_STARTED",
            attempt_id="attempt-b",
            reference_ids=("e3", "e4"),
            payload={
                "work_item_id": "work-b",
                "worker_id": "worker-b",
                "purpose": "SYNTHESIZE",
                "projection_revision": 4,
                "scheduler_decision_id": "decision-a",
                "scope_region_ids": [],
            },
        ),
        _event(
            9,
            "KNOWLEDGE_DELTA_RECORDED",
            attempt_id="attempt-a",
            reference_ids=("k1", "e1"),
            payload={
                "delta_id": "k1",
                "kind": "INTEGRATION_SUMMARY",
                "summary": "partition A first finding",
                "source_reference_ids": ["e1"],
                "causal_event_ids": [],
            },
        ),
        _event(
            10,
            "KNOWLEDGE_DELTA_RECORDED",
            attempt_id="attempt-a",
            reference_ids=("k3", "e2"),
            payload={
                "delta_id": "k3",
                "kind": "INTEGRATION_SUMMARY",
                "summary": "partition A second finding",
                "source_reference_ids": ["e2"],
                "causal_event_ids": [],
            },
        ),
        _event(
            11,
            "KNOWLEDGE_DELTA_RECORDED",
            attempt_id="attempt-b",
            reference_ids=("k2", "e3", "e4"),
            payload={
                "delta_id": "k2",
                "kind": "INTEGRATION_SUMMARY",
                "summary": "partition B finding",
                "source_reference_ids": ["e3", "e4"],
                "causal_event_ids": [],
            },
        ),
        _event(12, "ATTEMPT_COMPLETED", attempt_id="attempt-a", payload={"progress_made": True}),
        _event(13, "ATTEMPT_COMPLETED", attempt_id="attempt-b", payload={"progress_made": True}),
    )


class ThreadConsolidationPlannerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.planner = ThreadConsolidationPlanner(
            ThreadConsolidationConfig(selection_limit=2, minimum_source_deltas=2)
        )

    def test_selects_across_partitions_before_taking_second_delta_from_same_partition(self) -> None:
        plan = self.planner.plan(_partition_history(), thread_id="thread-a")

        self.assertTrue(plan.ready)
        self.assertEqual(plan.pending_source_count, 3)
        self.assertEqual(plan.pending_partition_count, 2)
        self.assertEqual(plan.selected_delta_ids, ("k1", "k2"))
        self.assertEqual(plan.selected_partition_ids, ("partition-a", "partition-b"))
        self.assertEqual(
            tuple(source.status for source in plan.selected_sources),
            (KnowledgeStatus.PROVISIONAL, KnowledgeStatus.PROVISIONAL),
        )

    def test_preparation_contains_only_selected_compact_knowledge(self) -> None:
        events = _partition_history()
        plan = self.planner.plan(events, thread_id="thread-a")
        knowledge = KnowledgeStateProjector().project(events)

        preparation = prepare_thread_consolidation_work(plan, knowledge)

        self.assertEqual(preparation.reference_ids, ("k1", "k2"))
        self.assertEqual(preparation.context["context_view"], "SYNTHESIZE")
        self.assertEqual(preparation.context["synthesis_mode"], "THREAD_CONSOLIDATION")
        self.assertEqual(
            preparation.context["source_partition_ids"],
            ["partition-a", "partition-b"],
        )
        self.assertEqual(len(preparation.context["knowledge_records"]), 2)
        self.assertEqual(
            tuple(record["delta_id"] for record in preparation.context["knowledge_records"]),
            ("k1", "k2"),
        )
        self.assertEqual(
            preparation.constraints["knowledge_delta_kind"],
            "THREAD_CONSOLIDATION",
        )
        self.assertTrue(preparation.constraints["preserve_unresolved_contradictions"])
        self.assertTrue(preparation.constraints["output_remains_provisional"])

    def test_active_thread_consolidation_marks_only_referenced_sources_consumed(self) -> None:
        events = (
            *_partition_history(),
            _event(
                14,
                "KNOWLEDGE_DELTA_RECORDED",
                reference_ids=("thread-summary", "k1", "k2"),
                payload={
                    "delta_id": "thread-summary",
                    "kind": "THREAD_CONSOLIDATION",
                    "summary": "cross-partition synthesis",
                    "source_reference_ids": ["k1", "k2"],
                    "causal_event_ids": [],
                },
            ),
        )

        plan = self.planner.plan(events, thread_id="thread-a")

        self.assertEqual(plan.pending_source_count, 1)
        self.assertEqual(plan.selected_delta_ids, ("k3",))
        self.assertFalse(plan.ready)

    def test_retracting_consolidation_reopens_its_source_deltas(self) -> None:
        events = (
            *_partition_history(),
            _event(
                14,
                "KNOWLEDGE_DELTA_RECORDED",
                reference_ids=("thread-summary", "k1", "k2"),
                payload={
                    "delta_id": "thread-summary",
                    "kind": "THREAD_CONSOLIDATION",
                    "summary": "cross-partition synthesis",
                    "source_reference_ids": ["k1", "k2"],
                    "causal_event_ids": [],
                },
            ),
            _event(
                15,
                "KNOWLEDGE_ASSESSMENT_RECORDED",
                reference_ids=("thread-summary",),
                payload={"assessment": "RETRACTED", "reason": "bad synthesis"},
            ),
        )

        plan = self.planner.plan(events, thread_id="thread-a")

        self.assertEqual(plan.pending_source_count, 3)
        self.assertEqual(plan.selected_delta_ids, ("k1", "k2"))
        self.assertTrue(plan.ready)

    def test_retracted_partition_source_is_not_selected(self) -> None:
        events = (
            *_partition_history(),
            _event(
                14,
                "KNOWLEDGE_ASSESSMENT_RECORDED",
                reference_ids=("k1",),
                payload={"assessment": "RETRACTED", "reason": "invalid source"},
            ),
        )

        plan = self.planner.plan(events, thread_id="thread-a")

        self.assertEqual(plan.pending_source_count, 2)
        self.assertEqual(plan.selected_delta_ids, ("k3", "k2"))
        self.assertTrue(plan.ready)

    def test_missing_partition_provenance_blocks_consolidation_in_strict_mode(self) -> None:
        events = tuple(
            event
            for event in _partition_history()
            if event.event_type != "INTEGRATION_PARTITION_ALLOCATION_RECORDED"
        )

        with self.assertRaisesRegex(ValueError, "missing durable allocation provenance"):
            self.planner.plan(events, thread_id="thread-a")


class _ConsolidationBank:
    def execute_batch(self, requests):
        results = []
        for request in requests:
            refs = tuple(request.work_item.reference_ids)
            results.append(
                AttemptResult(
                    attempt_id=request.attempt_id,
                    work_item_id=request.work_item.work_item_id,
                    thread_id=request.work_item.thread_id,
                    worker_id=request.worker_id,
                    status=AttemptStatus.COMPLETED,
                    knowledge_deltas=(
                        KnowledgeDelta(
                            delta_id="thread-summary-runtime",
                            kind="THREAD_CONSOLIDATION",
                            summary="runtime cross-partition consolidation",
                            reference_ids=refs,
                            thread_id=request.work_item.thread_id,
                        ),
                    ),
                    progress_made=True,
                )
            )
        return tuple(results)


class ThreadConsolidationRuntimeTests(unittest.TestCase):
    def test_same_worker_runtime_can_create_higher_level_knowledge(self) -> None:
        ledger = SQLiteResearchLedger(":memory:")
        self.addCleanup(ledger.close)
        for source in _partition_history():
            ledger.append_event(
                event_type=source.event_type,
                thread_id=source.thread_id,
                attempt_id=source.attempt_id,
                reference_ids=source.reference_ids,
                parent_event_ids=source.parent_event_ids,
                payload=source.payload,
            )

        events = ledger.read_all_events()
        planner = ThreadConsolidationPlanner(
            ThreadConsolidationConfig(selection_limit=2, minimum_source_deltas=2)
        )
        plan = planner.plan(events, thread_id="thread-a")
        knowledge = KnowledgeStateProjector().project(events)
        preparation = prepare_thread_consolidation_work(plan, knowledge)
        item = WorkItem(
            work_item_id="thread-consolidation-work",
            thread_id="thread-a",
            objective="consolidate partition knowledge",
            purpose=WorkPurpose.SYNTHESIZE,
            projection_revision=knowledge.revision,
            reference_ids=preparation.reference_ids,
            context=preparation.context,
            constraints=preparation.constraints,
        )

        result = WorkerRuntime(ledger).run_attempt(
            WorkerAssignment(worker_id="worker-consolidator", work_item=item),
            _ConsolidationBank(),
        )

        self.assertEqual(result.status, AttemptStatus.COMPLETED)
        snapshot = KnowledgeStateProjector().project(ledger.read_all_events())
        summary = snapshot.get("thread-summary-runtime")
        self.assertIsNotNone(summary)
        self.assertEqual(summary.kind, "THREAD_CONSOLIDATION")
        self.assertEqual(summary.source_reference_ids, ("k1", "k2"))
        self.assertEqual(summary.status, KnowledgeStatus.PROVISIONAL)

        next_plan = planner.plan(ledger.read_all_events(), thread_id="thread-a")
        self.assertEqual(next_plan.pending_source_count, 1)
        self.assertEqual(next_plan.selected_delta_ids, ("k3",))
        self.assertFalse(next_plan.ready)


if __name__ == "__main__":
    unittest.main()
