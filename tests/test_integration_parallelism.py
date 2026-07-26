from __future__ import annotations

import unittest

from ai_hypothesis.runtime import (
    AttemptResult,
    AttemptStatus,
    EvidenceDisposition,
    EvidenceDispositionKind,
    IntegrationTracker,
    KnowledgeDelta,
    ProjectedState,
    RuntimeControlLoop,
    SchedulerAction,
    SchedulerDecision,
    SchedulerSignals,
    SchedulableThread,
    SQLiteResearchLedger,
    WorkPreparation,
    WorkPurpose,
)
from ai_hypothesis.runtime.integration_parallelism import (
    IntegrationParallelismConfig,
    IntegrationPartitionAllocator,
    PartitionedBackpressureScheduler,
    PartitionedIntegrationContextRouter,
)
from ai_hypothesis.runtime.integration_partitions import (
    IntegrationPartitionConfig,
    IntegrationPartitionProjector,
)


class _FixedScheduler:
    def __init__(
        self,
        *,
        action: SchedulerAction = SchedulerAction.SYNTHESIZE,
        reasons: tuple[str, ...] = ("BACKPRESSURE",),
    ) -> None:
        self.action = action
        self.reasons = reasons

    def choose(
        self,
        candidates,
        *,
        integration_backpressure: bool = False,
        max_width: int = 1,
    ) -> SchedulerDecision:
        state = candidates[0].state
        purpose = {
            SchedulerAction.SYNTHESIZE: WorkPurpose.SYNTHESIZE,
            SchedulerAction.ADD_WIDTH: WorkPurpose.EXPLORE,
            SchedulerAction.VERIFY: WorkPurpose.VERIFY,
        }.get(self.action, state.purpose)
        return SchedulerDecision(
            decision_id=f"decision-{self.action.value}",
            thread_id=state.thread_id,
            action=self.action,
            purpose=purpose,
            width=1,
            reason_codes=self.reasons,
            projection_revision=state.revision,
        )


class _RecordingIntegrationBank:
    def __init__(self) -> None:
        self.calls = []

    def execute_batch(self, requests):
        self.calls.append(tuple(requests))
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
                            delta_id=f"delta-{request.attempt_id}",
                            kind="INTEGRATION_SUMMARY",
                            summary="integrated one bounded partition",
                            reference_ids=refs,
                            thread_id=request.work_item.thread_id,
                        ),
                    ),
                    evidence_dispositions=(
                        EvidenceDisposition(
                            evidence_ids=refs,
                            disposition=EvidenceDispositionKind.INTEGRATED,
                        ),
                    ),
                    progress_made=True,
                )
            )
        return tuple(results)


def _candidate(thread_id: str = "thread-a") -> SchedulableThread:
    return SchedulableThread(
        state=ProjectedState(
            revision=1,
            thread_id=thread_id,
            objective="integrate evidence",
            status="ACTIVE",
            purpose=WorkPurpose.PROGRESS,
        ),
        signals=SchedulerSignals(
            integration_backlog=1.0,
            recent_progress=1.0,
        ),
    )


def _append_evidence(ledger: SQLiteResearchLedger, thread_id: str, count: int) -> None:
    for index in range(count):
        evidence_id = f"evidence-{index}"
        ledger.append_event(
            event_type="EVIDENCE_ADDED",
            thread_id=thread_id,
            reference_ids=(evidence_id,),
            payload={
                "evidence_id": evidence_id,
                "kind": "OBSERVATION",
                "summary": f"observation {index}",
            },
        )


class IntegrationPartitionAllocatorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.ledger = SQLiteResearchLedger(":memory:")
        self.allocator = IntegrationPartitionAllocator(
            IntegrationPartitionProjector(
                IntegrationPartitionConfig(shard_count=8, batch_limit=2)
            )
        )

    def tearDown(self) -> None:
        self.ledger.close()

    def test_scheduler_width_is_capped_by_real_nonempty_partitions(self) -> None:
        _append_evidence(self.ledger, "thread-a", 32)
        plan = self.allocator.plan(self.ledger)
        available = self.allocator.available_width(plan, "thread-a")
        self.assertGreaterEqual(available, 3)

        scheduler = PartitionedBackpressureScheduler(
            _FixedScheduler(),
            ledger=self.ledger,
            allocator=self.allocator,
            config=IntegrationParallelismConfig(max_integration_width=4),
        )
        decision = scheduler.choose(
            (_candidate(),),
            integration_backpressure=True,
            max_width=3,
        )

        self.assertEqual(decision.width, 3)
        self.assertIn("PARTITIONED_INTEGRATION", decision.reason_codes)

    def test_non_synthesis_decision_is_not_widened(self) -> None:
        _append_evidence(self.ledger, "thread-a", 32)
        scheduler = PartitionedBackpressureScheduler(
            _FixedScheduler(
                action=SchedulerAction.ADD_WIDTH,
                reasons=("STRUCTURED_EXPLORATION", "BACKPRESSURE_EXPLORATION"),
            ),
            ledger=self.ledger,
            allocator=self.allocator,
        )
        decision = scheduler.choose(
            (_candidate(),),
            integration_backpressure=True,
            max_width=4,
        )
        self.assertEqual(decision.width, 1)
        self.assertNotIn("PARTITIONED_INTEGRATION", decision.reason_codes)

    def test_context_batch_has_disjoint_partition_authority(self) -> None:
        _append_evidence(self.ledger, "thread-a", 32)
        scheduler = PartitionedBackpressureScheduler(
            _FixedScheduler(),
            ledger=self.ledger,
            allocator=self.allocator,
            config=IntegrationParallelismConfig(max_integration_width=3),
        )
        decision = scheduler.choose(
            (_candidate(),),
            integration_backpressure=True,
            max_width=3,
        )
        router = PartitionedIntegrationContextRouter(
            ledger=self.ledger,
            allocator=self.allocator,
            fallback=lambda state, decision: WorkPreparation(context={"fallback": True}),
        )
        batch = router(_candidate().state, decision)

        self.assertEqual(len(batch.items), decision.width)
        seen: set[str] = set()
        partition_ids: set[str] = set()
        for item in batch.items:
            refs = set(item.reference_ids)
            self.assertTrue(refs)
            self.assertFalse(refs & seen)
            seen.update(refs)
            partition_ids.add(item.context["integration_partition"]["partition_id"])
        self.assertEqual(len(partition_ids), decision.width)


class PartitionedIntegrationControlLoopTests(unittest.TestCase):
    def test_one_backpressured_thread_executes_distinct_partitions_in_one_neural_batch(self) -> None:
        ledger = SQLiteResearchLedger(":memory:")
        self.addCleanup(ledger.close)
        allocator = IntegrationPartitionAllocator(
            IntegrationPartitionProjector(
                IntegrationPartitionConfig(shard_count=8, batch_limit=2)
            )
        )
        scheduler = PartitionedBackpressureScheduler(
            _FixedScheduler(),
            ledger=ledger,
            allocator=allocator,
            config=IntegrationParallelismConfig(max_integration_width=3),
        )
        bank = _RecordingIntegrationBank()
        tracker = IntegrationTracker(ledger)
        loop = RuntimeControlLoop(
            ledger=ledger,
            scheduler=scheduler,
            worker_bank=bank,
            worker_ids=("worker-0", "worker-1", "worker-2", "worker-3"),
            integration_tracker=tracker,
        )
        thread_id = loop.create_thread(
            objective="integrate a large evidence backlog",
            purpose=WorkPurpose.PROGRESS,
            thread_id="thread-a",
        )
        _append_evidence(ledger, thread_id, 32)

        partition_router = PartitionedIntegrationContextRouter(
            ledger=ledger,
            allocator=allocator,
            fallback=lambda state, decision: WorkPreparation(context={"fallback": True}),
        )
        before = tracker.snapshot(thread_id=thread_id)
        step = loop.run_once(
            signal_provider=lambda state: SchedulerSignals(
                integration_backlog=1.0,
                recent_progress=1.0,
            ),
            context_provider=partition_router,
            integration_backpressure=True,
        )
        after = tracker.snapshot(thread_id=thread_id)

        self.assertEqual(step.decision.action, SchedulerAction.SYNTHESIZE)
        self.assertEqual(step.decision.width, 3)
        self.assertEqual(len(step.assignments), 3)
        self.assertEqual(len(step.results), 3)
        self.assertEqual(len(bank.calls), 1)
        self.assertEqual(len(bank.calls[0]), 3)

        authority_sets = [set(request.work_item.reference_ids) for request in bank.calls[0]]
        for index, left in enumerate(authority_sets):
            self.assertTrue(left)
            for right in authority_sets[index + 1 :]:
                self.assertFalse(left & right)

        consumed = sum(len(authority) for authority in authority_sets)
        self.assertEqual(
            after.dispositioned_evidence_count - before.dispositioned_evidence_count,
            consumed,
        )
        self.assertEqual(
            after.knowledge_delta_count - before.knowledge_delta_count,
            step.decision.width,
        )


if __name__ == "__main__":
    unittest.main()
