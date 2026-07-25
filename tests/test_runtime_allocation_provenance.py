"""Tests durable scheduler-decision provenance on worker attempts."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from ai_hypothesis.runtime import (
    AttemptResult,
    AttemptStatus,
    RuntimeControlLoop,
    SchedulerConfig,
    SchedulerSignals,
    SQLiteResearchLedger,
    TracingSchedulerV0,
    WorkerAssignment,
    WorkerRuntime,
    WorkItem,
    WorkPreparation,
    WorkPurpose,
)


class _NoopBank:
    def execute_batch(self, requests):
        return tuple(
            AttemptResult(
                attempt_id=request.attempt_id,
                work_item_id=request.work_item.work_item_id,
                thread_id=request.work_item.thread_id,
                worker_id=request.worker_id,
                status=AttemptStatus.COMPLETED,
                progress_made=True,
                resource_usage={"units": 1},
            )
            for request in requests
        )


class AllocationProvenanceTests(unittest.TestCase):
    def test_control_loop_attempt_links_back_to_traced_scheduler_decision(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with SQLiteResearchLedger(Path(directory) / "ledger.sqlite") as ledger:
                scheduler = TracingSchedulerV0(
                    ledger,
                    SchedulerConfig(exploration_probability=0.0),
                )
                loop = RuntimeControlLoop(
                    ledger=ledger,
                    scheduler=scheduler,
                    worker_bank=_NoopBank(),
                    worker_ids=("worker-a", "worker-b"),
                )
                loop.create_thread(thread_id="thread-1", objective="Investigate")

                step = loop.run_once(
                    signal_provider=lambda _state: SchedulerSignals(
                        importance=1.0,
                        recent_progress=1.0,
                    ),
                    context_provider=lambda _state, _decision: WorkPreparation(),
                )

                started = [
                    event
                    for event in ledger.read_all_events()
                    if event.event_type == "ATTEMPT_STARTED"
                ]
                self.assertEqual(len(started), 1)
                self.assertEqual(
                    started[0].payload["scheduler_decision_id"],
                    step.decision.decision_id,
                )

    def test_explicit_work_item_decision_id_overrides_history_lookup(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with SQLiteResearchLedger(Path(directory) / "ledger.sqlite") as ledger:
                ledger.append_event(
                    event_type="SCHEDULER_DECISION_RECORDED",
                    thread_id="thread-1",
                    payload={
                        "decision_id": "historical-decision",
                        "action": "CONTINUE",
                        "purpose": "PROGRESS",
                        "width": 1,
                        "reason_codes": [],
                        "projection_revision": 7,
                        "integration_backpressure": False,
                        "max_width": 1,
                    },
                )
                runtime = WorkerRuntime(ledger)
                runtime.run_attempt(
                    WorkerAssignment(
                        worker_id="worker-a",
                        work_item=WorkItem(
                            work_item_id="work-1",
                            thread_id="thread-1",
                            objective="Replay explicit allocation",
                            purpose=WorkPurpose.PROGRESS,
                            projection_revision=7,
                            scheduler_decision_id="explicit-decision",
                        ),
                    ),
                    _NoopBank(),
                )

                started = [
                    event
                    for event in ledger.read_all_events()
                    if event.event_type == "ATTEMPT_STARTED"
                ]
                self.assertEqual(
                    started[0].payload["scheduler_decision_id"],
                    "explicit-decision",
                )


if __name__ == "__main__":
    unittest.main()
