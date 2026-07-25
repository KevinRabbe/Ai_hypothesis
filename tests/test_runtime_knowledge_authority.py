"""Tests canonical-state safety for knowledge assessment targets."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from ai_hypothesis.runtime import (
    AttemptResult,
    AttemptStatus,
    KnowledgeAssessment,
    KnowledgeAssessmentKind,
    SQLiteResearchLedger,
    WorkerAssignment,
    WorkerRuntime,
    WorkItem,
    WorkPurpose,
)


class _BogusAssessmentBank:
    def execute_batch(self, requests):
        return tuple(
            AttemptResult(
                attempt_id=request.attempt_id,
                work_item_id=request.work_item.work_item_id,
                thread_id=request.work_item.thread_id,
                worker_id=request.worker_id,
                status=AttemptStatus.COMPLETED,
                knowledge_assessments=(
                    KnowledgeAssessment(
                        delta_ids=("missing-delta",),
                        assessment=KnowledgeAssessmentKind.VERIFIED,
                        reason="The Work Item authorized an ID that does not exist",
                    ),
                ),
            )
            for request in requests
        )


class KnowledgeAuthorityTests(unittest.TestCase):
    def test_authorized_but_nonexistent_delta_is_rejected_before_persistence(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with SQLiteResearchLedger(Path(directory) / "ledger.sqlite") as ledger:
                runtime = WorkerRuntime(ledger)
                item = WorkItem(
                    work_item_id="work-verify",
                    thread_id="verification-thread",
                    objective="Verify one knowledge delta",
                    purpose=WorkPurpose.VERIFY,
                    projection_revision=1,
                    reference_ids=("missing-delta",),
                )

                with self.assertRaisesRegex(ValueError, "nonexistent knowledge delta"):
                    runtime.run_attempt(
                        WorkerAssignment("worker-verify", item),
                        _BogusAssessmentBank(),
                    )

                event_types = tuple(event.event_type for event in ledger.read_all_events())
                self.assertEqual(
                    event_types,
                    ("ATTEMPT_STARTED", "ATTEMPT_INVALID_RESULT"),
                )
                self.assertNotIn("KNOWLEDGE_ASSESSMENT_RECORDED", event_types)


if __name__ == "__main__":
    unittest.main()
