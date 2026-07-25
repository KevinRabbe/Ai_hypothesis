"""Regression tests against silent truncation at the ledger page boundary."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from ai_hypothesis.runtime import (
    AttemptResult,
    AttemptStatus,
    IntegrationTracker,
    RuntimeControlLoop,
    SchedulerConfig,
    SchedulerSignals,
    SchedulerV0,
    SQLiteResearchLedger,
    WorkPreparation,
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
            )
            for request in requests
        )


class FullReplayTests(unittest.TestCase):
    def test_read_all_events_crosses_default_page_boundary(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with SQLiteResearchLedger(Path(directory) / "ledger.sqlite") as ledger:
                for index in range(1_205):
                    ledger.append_event(
                        event_type="TEST_EVENT",
                        payload={"index": index},
                    )

                first_page = ledger.read_events()
                complete = ledger.read_all_events()

                self.assertEqual(len(first_page), 1_000)
                self.assertEqual(len(complete), 1_205)
                self.assertEqual(complete[-1].payload["index"], 1_204)

    def test_control_loop_sees_thread_created_after_first_thousand_events(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with SQLiteResearchLedger(Path(directory) / "ledger.sqlite") as ledger:
                for index in range(1_001):
                    ledger.append_event(
                        event_type="BACKGROUND_EVENT",
                        payload={"index": index},
                    )
                loop = RuntimeControlLoop(
                    ledger=ledger,
                    scheduler=SchedulerV0(
                        SchedulerConfig(exploration_probability=0.0)
                    ),
                    worker_bank=_NoopBank(),
                    worker_ids=("worker-a",),
                )
                loop.create_thread(
                    thread_id="late-thread",
                    objective="Must remain visible after pagination",
                )

                step = loop.run_once(
                    signal_provider=lambda _state: SchedulerSignals(
                        importance=1.0,
                        recent_progress=1.0,
                    ),
                    context_provider=lambda _state, _decision: WorkPreparation(),
                )

                self.assertEqual(step.decision.thread_id, "late-thread")

    def test_integration_tracker_sees_evidence_after_first_thousand_events(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with SQLiteResearchLedger(Path(directory) / "ledger.sqlite") as ledger:
                for index in range(1_001):
                    ledger.append_event(
                        event_type="BACKGROUND_EVENT",
                        payload={"index": index},
                    )
                ledger.append_event(
                    event_type="EVIDENCE_ADDED",
                    thread_id="late-thread",
                    reference_ids=("late-evidence",),
                    payload={
                        "evidence_id": "late-evidence",
                        "kind": "LOCAL_FINDING",
                        "summary": "late but important",
                    },
                )
                tracker = IntegrationTracker(ledger)

                snapshot = tracker.snapshot()
                batch = tracker.pending_batch(limit=1)

                self.assertEqual(snapshot.backlog_evidence_ids, ("late-evidence",))
                self.assertEqual(batch.evidence_ids, ("late-evidence",))


if __name__ == "__main__":
    unittest.main()
