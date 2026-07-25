"""Batch-level invariants for Worker Runtime."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from typing import Sequence

from ai_hypothesis.runtime import (
    AttemptRequest,
    AttemptResult,
    AttemptStatus,
    SQLiteResearchLedger,
    WorkerAssignment,
    WorkerRuntime,
    WorkItem,
    WorkPurpose,
)


class MixedResultBank:
    def execute_batch(self, requests: Sequence[AttemptRequest]) -> Sequence[AttemptResult]:
        good, bad = requests
        return (
            AttemptResult(
                attempt_id=good.attempt_id,
                work_item_id=good.work_item.work_item_id,
                thread_id=good.work_item.thread_id,
                worker_id=good.worker_id,
                status=AttemptStatus.COMPLETED,
                evidence_refs=("evidence-good",),
                progress_made=True,
            ),
            AttemptResult(
                attempt_id="wrong-attempt-id",
                work_item_id=bad.work_item.work_item_id,
                thread_id=bad.work_item.thread_id,
                worker_id=bad.worker_id,
                status=AttemptStatus.COMPLETED,
            ),
        )


class WorkerRuntimeBatchTests(unittest.TestCase):
    def test_valid_peer_result_survives_invalid_batch_result(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with SQLiteResearchLedger(Path(directory) / "ledger.sqlite") as ledger:
                runtime = WorkerRuntime(ledger)
                assignments = (
                    WorkerAssignment(
                        worker_id="worker-good",
                        work_item=WorkItem(
                            work_item_id="work-good",
                            thread_id="thread-good",
                            objective="Useful work",
                            purpose=WorkPurpose.EXPLORE,
                            projection_revision=1,
                        ),
                    ),
                    WorkerAssignment(
                        worker_id="worker-bad",
                        work_item=WorkItem(
                            work_item_id="work-bad",
                            thread_id="thread-bad",
                            objective="Malformed peer",
                            purpose=WorkPurpose.EXPLORE,
                            projection_revision=1,
                        ),
                    ),
                )

                with self.assertRaisesRegex(ValueError, "mismatched attempt_id"):
                    runtime.run_batch(assignments, MixedResultBank())

                good_events = ledger.read_events(thread_id="thread-good")
                bad_events = ledger.read_events(thread_id="thread-bad")
                self.assertEqual(good_events[-1].event_type, "ATTEMPT_COMPLETED")
                self.assertTrue(
                    any(event.event_type == "EVIDENCE_ADDED" for event in good_events)
                )
                self.assertEqual(bad_events[-1].event_type, "ATTEMPT_INVALID_RESULT")


if __name__ == "__main__":
    unittest.main()
