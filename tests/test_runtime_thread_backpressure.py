"""Tests that integration pressure affects routing only during real backpressure."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from ai_hypothesis.runtime import (
    AttemptResult,
    AttemptStatus,
    IntegrationBackpressureConfig,
    IntegrationTracker,
    RuntimeControlLoop,
    SchedulerAction,
    SchedulerConfig,
    SchedulerSignals,
    SchedulerV0,
    SQLiteResearchLedger,
    WorkPreparation,
)


class _NoopWorkerBank:
    def execute_batch(self, requests):
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


def _signals(state):
    return SchedulerSignals(
        importance=1.0 if state.thread_id == "thread-a" else 0.1,
        recent_progress=1.0,
    )


def _prepare(_state, _decision):
    return WorkPreparation()


class ThreadLocalBackpressureTests(unittest.TestCase):
    def test_below_global_threshold_normal_priority_ignores_local_backlog(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with SQLiteResearchLedger(Path(directory) / "ledger.sqlite") as ledger:
                tracker = IntegrationTracker(
                    ledger,
                    IntegrationBackpressureConfig(
                        max_backlog_count=10,
                        max_backlog_age_sequences=100,
                    ),
                )
                loop = RuntimeControlLoop(
                    ledger=ledger,
                    scheduler=SchedulerV0(
                        SchedulerConfig(exploration_probability=0.0)
                    ),
                    worker_bank=_NoopWorkerBank(),
                    worker_ids=("worker-a", "worker-b"),
                    integration_tracker=tracker,
                )
                loop.create_thread(thread_id="thread-a", objective="High priority normal work")
                loop.create_thread(thread_id="thread-b", objective="Low priority with backlog")
                ledger.append_event(
                    event_type="EVIDENCE_ADDED",
                    thread_id="thread-b",
                    reference_ids=("evidence-b",),
                    payload={"evidence_id": "evidence-b"},
                )

                self.assertFalse(tracker.is_backpressured())
                step = loop.run_once(signal_provider=_signals, context_provider=_prepare)

                self.assertEqual(step.decision.thread_id, "thread-a")
                self.assertEqual(step.decision.action, SchedulerAction.CONTINUE)

    def test_global_backpressure_routes_synthesis_to_thread_with_backlog(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with SQLiteResearchLedger(Path(directory) / "ledger.sqlite") as ledger:
                tracker = IntegrationTracker(
                    ledger,
                    IntegrationBackpressureConfig(
                        max_backlog_count=0,
                        max_backlog_age_sequences=100,
                    ),
                )
                loop = RuntimeControlLoop(
                    ledger=ledger,
                    scheduler=SchedulerV0(
                        SchedulerConfig(exploration_probability=0.0)
                    ),
                    worker_bank=_NoopWorkerBank(),
                    worker_ids=("worker-a", "worker-b"),
                    integration_tracker=tracker,
                )
                loop.create_thread(thread_id="thread-a", objective="High priority normal work")
                loop.create_thread(thread_id="thread-b", objective="Low priority with backlog")
                ledger.append_event(
                    event_type="EVIDENCE_ADDED",
                    thread_id="thread-b",
                    reference_ids=("evidence-b",),
                    payload={"evidence_id": "evidence-b"},
                )

                self.assertTrue(tracker.is_backpressured())
                step = loop.run_once(signal_provider=_signals, context_provider=_prepare)

                self.assertEqual(step.decision.thread_id, "thread-b")
                self.assertEqual(step.decision.action, SchedulerAction.SYNTHESIZE)
                self.assertEqual(step.assignment.work_item.purpose.value, "SYNTHESIZE")


if __name__ == "__main__":
    unittest.main()
