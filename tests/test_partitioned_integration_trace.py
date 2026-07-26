from __future__ import annotations

import unittest

from ai_hypothesis.runtime import (
    ProjectedState,
    SchedulerAction,
    SchedulerDecision,
    SchedulerSignals,
    SchedulableThread,
    SQLiteResearchLedger,
    TracingScheduler,
    WorkPurpose,
)
from ai_hypothesis.runtime.integration_parallelism import (
    IntegrationParallelismConfig,
    IntegrationPartitionAllocator,
    PartitionedBackpressureScheduler,
)
from ai_hypothesis.runtime.integration_partitions import (
    IntegrationPartitionConfig,
    IntegrationPartitionProjector,
)


class _FixedBackpressureScheduler:
    def choose(
        self,
        candidates,
        *,
        integration_backpressure: bool = False,
        max_width: int = 1,
    ) -> SchedulerDecision:
        state = candidates[0].state
        return SchedulerDecision(
            decision_id="decision-partitioned",
            thread_id=state.thread_id,
            action=SchedulerAction.SYNTHESIZE,
            purpose=WorkPurpose.SYNTHESIZE,
            width=1,
            reason_codes=("BACKPRESSURE",),
            projection_revision=state.revision,
        )


class PartitionedIntegrationTraceTests(unittest.TestCase):
    def test_trace_records_final_widened_integration_decision(self) -> None:
        ledger = SQLiteResearchLedger(":memory:")
        self.addCleanup(ledger.close)
        for index in range(32):
            evidence_id = f"evidence-{index}"
            ledger.append_event(
                event_type="EVIDENCE_ADDED",
                thread_id="thread-a",
                reference_ids=(evidence_id,),
                payload={
                    "evidence_id": evidence_id,
                    "kind": "OBSERVATION",
                    "summary": f"observation {index}",
                },
            )

        allocator = IntegrationPartitionAllocator(
            IntegrationPartitionProjector(
                IntegrationPartitionConfig(shard_count=8, batch_limit=2)
            )
        )
        partitioned = PartitionedBackpressureScheduler(
            _FixedBackpressureScheduler(),
            ledger=ledger,
            allocator=allocator,
            config=IntegrationParallelismConfig(max_integration_width=3),
        )
        traced = TracingScheduler(ledger, partitioned)
        candidate = SchedulableThread(
            state=ProjectedState(
                revision=ledger.latest_sequence(),
                thread_id="thread-a",
                objective="integrate evidence",
                status="ACTIVE",
                purpose=WorkPurpose.PROGRESS,
            ),
            signals=SchedulerSignals(
                integration_backlog=1.0,
                recent_progress=1.0,
            ),
        )

        decision = traced.choose(
            (candidate,),
            integration_backpressure=True,
            max_width=4,
        )
        trace = ledger.read_all_events()[-1]

        self.assertEqual(decision.width, 3)
        self.assertEqual(trace.event_type, "SCHEDULER_DECISION_RECORDED")
        self.assertEqual(trace.payload["decision_id"], decision.decision_id)
        self.assertEqual(trace.payload["width"], 3)
        self.assertIn("PARTITIONED_INTEGRATION", trace.payload["reason_codes"])
        self.assertTrue(trace.payload["integration_backpressure"])
        self.assertEqual(trace.payload["max_width"], 4)


if __name__ == "__main__":
    unittest.main()
