from __future__ import annotations

import random
import tempfile
import unittest
from pathlib import Path

from ai_hypothesis.runtime.contracts import (
    AttemptResult,
    AttemptStatus,
    EvidenceDispositionKind,
    SchedulerAction,
    SchedulerDecision,
    WorkPurpose,
)
from ai_hypothesis.runtime.control import WorkPreparation
from ai_hypothesis.runtime.indexed_control import (
    IndexedRuntimeControlLoop,
    IndexedRuntimeIntegrationTracker,
    IndexedThreadRuntimeState,
)
from ai_hypothesis.runtime.indexed_integration_partitions import (
    IndexedIntegrationPartitionPlanner,
    IndexedPartitionedBackpressureScheduler,
    IndexedPartitionedIntegrationContextRouter,
)
from ai_hypothesis.runtime.indexed_thread_consolidation_control import (
    PinnedIndexedRuntimeSnapshotProvider,
)
from ai_hypothesis.runtime.integration_parallelism import (
    IntegrationParallelismConfig,
    IntegrationPartitionAllocator,
)
from ai_hypothesis.runtime.ledger import SQLiteResearchLedger
from ai_hypothesis.runtime.scheduler import SchedulerConfig, SchedulerSignals, SchedulerV0
from ai_hypothesis.runtime.scheduler_trace import TracingScheduler


class NoFullReplayLedger(SQLiteResearchLedger):
    def read_all_events(self, *args, **kwargs):
        raise AssertionError("indexed partitioned integration must not call read_all_events")


class RecordingBank:
    def __init__(self) -> None:
        self.requests = []

    def execute_batch(self, requests):
        self.requests.extend(requests)
        return tuple(
            AttemptResult(
                attempt_id=request.attempt_id,
                work_item_id=request.work_item.work_item_id,
                thread_id=request.work_item.thread_id,
                worker_id=request.worker_id,
                status=AttemptStatus.COMPLETED,
                progress_made=True,
            )
            for request in requests
        )


def append_thread(ledger: SQLiteResearchLedger, thread_id: str = "thread-a") -> None:
    ledger.append_event(
        event_type="THREAD_CREATED",
        thread_id=thread_id,
        payload={
            "objective": "integrate pending evidence",
            "purpose": "PROGRESS",
            "status": "ACTIVE",
        },
    )


def append_evidence(
    ledger: SQLiteResearchLedger,
    count: int,
    *,
    thread_id: str = "thread-a",
) -> tuple[str, ...]:
    ids = []
    for index in range(count):
        evidence_id = f"evidence-{index:03d}"
        ids.append(evidence_id)
        ledger.append_event(
            event_type="EVIDENCE_ADDED",
            thread_id=thread_id,
            reference_ids=(evidence_id, f"source-{index:03d}"),
            payload={
                "evidence_id": evidence_id,
                "kind": "OBSERVATION",
                "summary": f"observation {index}",
                "strength": 0.8,
                "uncertainty": 0.2,
                "data": {"index": index},
            },
        )
    return tuple(ids)


class IndexedIntegrationPartitionEquivalenceTests(unittest.TestCase):
    def test_indexed_plan_and_batch_match_replay_allocator(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            root = Path(tempdir)
            ledger = SQLiteResearchLedger(root / "ledger.sqlite3")
            integration = IndexedRuntimeIntegrationTracker(
                ledger,
                root / "integration.sqlite3",
            )
            try:
                append_thread(ledger)
                append_evidence(ledger, 24)
                sequence = ledger.latest_sequence()

                replay_allocator = IntegrationPartitionAllocator()
                replay_plan = replay_allocator.plan(ledger)
                replay_partitions = replay_allocator.ordered_for_thread(
                    replay_plan,
                    "thread-a",
                )

                indexed_planner = IndexedIntegrationPartitionPlanner(
                    ledger=ledger,
                    integration=integration,
                )
                indexed_plan = indexed_planner.plan(
                    sequence=sequence,
                    thread_id="thread-a",
                )
                indexed_partitions = indexed_plan.ordered_for_execution()

                self.assertEqual(
                    tuple(
                        (
                            partition.partition_id,
                            partition.shard_index,
                            partition.shard_count,
                            partition.backlog_count,
                            partition.oldest_pending_sequence,
                            partition.evidence_ids,
                            tuple(record.to_context_record() for record in partition.records),
                        )
                        for partition in indexed_partitions
                    ),
                    tuple(
                        (
                            partition.partition_id,
                            partition.shard_index,
                            partition.shard_count,
                            partition.backlog_count,
                            partition.oldest_pending_sequence,
                            partition.evidence_ids,
                            tuple(record.to_context_record() for record in partition.records),
                        )
                        for partition in replay_partitions
                    ),
                )

                width = min(3, len(indexed_partitions))
                indexed_batch, selected = indexed_planner.prepare_batch(
                    indexed_plan,
                    width=width,
                )
                replay_batch = replay_allocator.prepare_batch(
                    replay_plan,
                    "thread-a",
                    width=width,
                )
                self.assertEqual(indexed_batch, replay_batch)
                self.assertEqual(
                    tuple(partition.partition_id for partition in selected),
                    tuple(
                        partition.partition_id
                        for partition in replay_partitions[:width]
                    ),
                )
            finally:
                integration.close()
                ledger.close()


class IndexedPartitionedIntegrationRuntimeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        root = Path(self.tempdir.name)
        self.ledger = NoFullReplayLedger(root / "ledger.sqlite3")
        append_thread(self.ledger)
        append_evidence(self.ledger, 32)
        self.thread_state = IndexedThreadRuntimeState(
            self.ledger,
            root / "threads.sqlite3",
        )
        self.integration = IndexedRuntimeIntegrationTracker(
            self.ledger,
            root / "integration.sqlite3",
        )
        self.snapshot_provider = PinnedIndexedRuntimeSnapshotProvider(
            ledger=self.ledger,
            thread_state=self.thread_state,
            integration_tracker=self.integration,
            verification_tracker=None,
        )
        self.planner = IndexedIntegrationPartitionPlanner(
            ledger=self.ledger,
            integration=self.integration,
        )
        base = SchedulerV0(
            SchedulerConfig(exploration_probability=0.0),
            rng=random.Random(1),
        )
        self.partition_scheduler = IndexedPartitionedBackpressureScheduler(
            base,
            planner=self.planner,
            revision_provider=self.snapshot_provider,
            config=IntegrationParallelismConfig(max_integration_width=3),
        )
        self.scheduler = TracingScheduler(
            self.ledger,
            self.partition_scheduler,
        )
        self.router = IndexedPartitionedIntegrationContextRouter(
            ledger=self.ledger,
            scheduler=self.partition_scheduler,
            planner=self.planner,
            fallback=lambda _state, _decision: WorkPreparation(
                context={"context_view": "PROGRESS"}
            ),
        )
        self.bank = RecordingBank()
        self.loop = IndexedRuntimeControlLoop(
            ledger=self.ledger,
            scheduler=self.scheduler,
            worker_bank=self.bank,
            worker_ids=("worker-0", "worker-1", "worker-2", "worker-3"),
            snapshot_provider=self.snapshot_provider,
        )

    def tearDown(self) -> None:
        self.integration.close()
        self.thread_state.close()
        self.ledger.close()
        self.tempdir.cleanup()

    def test_backpressure_widens_one_thread_without_full_replay(self) -> None:
        step = self.loop.run_once(
            signal_provider=lambda _state: SchedulerSignals(recent_progress=1.0),
            context_provider=self.router,
            integration_backpressure=True,
        )

        self.assertEqual(step.decision.action, SchedulerAction.SYNTHESIZE)
        self.assertIn("BACKPRESSURE", step.decision.reason_codes)
        self.assertIn("PARTITIONED_INTEGRATION", step.decision.reason_codes)
        self.assertGreater(step.decision.width, 1)
        self.assertLessEqual(step.decision.width, 3)
        self.assertEqual(len(step.assignments), step.decision.width)
        self.assertEqual(len(self.bank.requests), step.decision.width)

        authority = [
            set(assignment.work_item.reference_ids)
            for assignment in step.assignments
        ]
        for index, left in enumerate(authority):
            self.assertTrue(left)
            for right in authority[index + 1 :]:
                self.assertTrue(left.isdisjoint(right))

        partition_ids = tuple(
            assignment.work_item.context["integration_partition"]["partition_id"]
            for assignment in step.assignments
        )
        self.assertEqual(len(partition_ids), len(set(partition_ids)))
        self.assertTrue(
            all(
                assignment.work_item.context["integration_revision"]
                == self.snapshot_provider.current_revision
                for assignment in step.assignments
            )
        )

        events = self.ledger.read_events(after_sequence=0, limit=10_000)
        traces = [
            event
            for event in events
            if event.event_type == "SCHEDULER_DECISION_RECORDED"
            and event.payload.get("decision_id") == step.decision.decision_id
        ]
        self.assertEqual(len(traces), 1)
        self.assertEqual(traces[0].payload["width"], step.decision.width)
        self.assertIn("PARTITIONED_INTEGRATION", traces[0].payload["reason_codes"])

        provenance = [
            event
            for event in events
            if event.event_type == "INTEGRATION_PARTITION_ALLOCATION_RECORDED"
            and event.payload.get("decision_id") == step.decision.decision_id
        ]
        self.assertEqual(len(provenance), 1)
        self.assertEqual(provenance[0].reference_ids, partition_ids)
        self.assertEqual(provenance[0].payload["width"], step.decision.width)
        self.assertEqual(
            tuple(
                tuple(partition["evidence_ids"])
                for partition in provenance[0].payload["partitions"]
            ),
            tuple(
                assignment.work_item.reference_ids
                for assignment in step.assignments
            ),
        )

    def test_context_reuses_scheduler_plan_and_provenance_is_idempotent(self) -> None:
        snapshot = self.snapshot_provider.capture()
        candidate_state = snapshot.states[0]
        decision = self.partition_scheduler.choose(
            (
                __import__(
                    "ai_hypothesis.runtime.scheduler",
                    fromlist=["SchedulableThread"],
                ).SchedulableThread(
                    state=candidate_state,
                    signals=SchedulerSignals(recent_progress=1.0),
                ),
            ),
            integration_backpressure=True,
            max_width=3,
        )
        first = self.router(candidate_state, decision)
        second = self.router(candidate_state, decision)
        self.assertEqual(first, second)

        event_id = self.router._allocation_event_id(decision.decision_id)
        event = self.ledger.get_event(event_id)
        self.assertIsNotNone(event)
        matching = [
            candidate
            for candidate in self.ledger.read_events(after_sequence=0, limit=10_000)
            if candidate.event_id == event_id
        ]
        self.assertEqual(len(matching), 1)

    def test_disposition_updates_next_indexed_partition_plan(self) -> None:
        snapshot = self.snapshot_provider.capture()
        before = self.planner.plan(
            sequence=snapshot.revision,
            thread_id="thread-a",
        )
        before_count = sum(partition.backlog_count for partition in before.partitions)
        self.assertEqual(before_count, 32)

        chosen = before.ordered_for_execution()[0]
        self.integration.record_disposition(
            chosen.evidence_ids,
            EvidenceDispositionKind.INTEGRATED,
            thread_id="thread-a",
        )
        current = self.ledger.latest_sequence()
        after = self.planner.plan(sequence=current, thread_id="thread-a")
        after_count = sum(partition.backlog_count for partition in after.partitions)
        self.assertEqual(after_count, before_count - len(chosen.evidence_ids))

    def test_forged_partition_reason_without_cached_allocation_is_rejected(self) -> None:
        snapshot = self.snapshot_provider.capture()
        state = snapshot.states[0]
        forged = SchedulerDecision(
            decision_id="forged",
            thread_id=state.thread_id,
            action=SchedulerAction.SYNTHESIZE,
            purpose=WorkPurpose.SYNTHESIZE,
            width=1,
            reason_codes=("BACKPRESSURE", "PARTITIONED_INTEGRATION"),
            projection_revision=state.revision,
        )
        forged.validate()
        with self.assertRaisesRegex(ValueError, "no cached indexed allocation"):
            self.router(state, forged)


if __name__ == "__main__":
    unittest.main()
