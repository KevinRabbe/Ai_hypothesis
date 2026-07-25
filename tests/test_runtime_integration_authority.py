"""Tests trust-boundary enforcement for learned integration outputs."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from ai_hypothesis.runtime import (
    AttemptResult,
    AttemptStatus,
    EvidenceDisposition,
    EvidenceDispositionKind,
    KnowledgeDelta,
    SQLiteResearchLedger,
    WorkerAssignment,
    WorkerRuntime,
    WorkItem,
    WorkPurpose,
)


def _item(work_id: str, thread_id: str, refs: tuple[str, ...]) -> WorkItem:
    return WorkItem(
        work_item_id=work_id,
        thread_id=thread_id,
        objective="Integrate bounded evidence",
        purpose=WorkPurpose.SYNTHESIZE,
        projection_revision=1,
        reference_ids=refs,
    )


class _UnauthorizedDispositionBank:
    def execute_batch(self, requests):
        request = requests[0]
        return (
            AttemptResult(
                attempt_id=request.attempt_id,
                work_item_id=request.work_item.work_item_id,
                thread_id=request.work_item.thread_id,
                worker_id=request.worker_id,
                status=AttemptStatus.COMPLETED,
                evidence_dispositions=(
                    EvidenceDisposition(
                        evidence_ids=("evidence-outside",),
                        disposition=EvidenceDispositionKind.INTEGRATED,
                    ),
                ),
            ),
        )


class IntegrationAuthorityTests(unittest.TestCase):
    def test_worker_cannot_disposition_evidence_outside_work_item(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with SQLiteResearchLedger(Path(directory) / "ledger.sqlite") as ledger:
                runtime = WorkerRuntime(ledger)
                with self.assertRaisesRegex(ValueError, "outside its Work Item authority"):
                    runtime.run_attempt(
                        WorkerAssignment(
                            "worker-a",
                            _item("work-1", "thread-1", ("evidence-1",)),
                        ),
                        _UnauthorizedDispositionBank(),
                    )

                event_types = tuple(event.event_type for event in ledger.read_all_events())
                self.assertEqual(event_types, ("ATTEMPT_STARTED", "ATTEMPT_INVALID_RESULT"))

    def test_worker_cannot_invent_causal_event_provenance(self) -> None:
        class Bank:
            def execute_batch(self, requests):
                request = requests[0]
                return (
                    AttemptResult(
                        attempt_id=request.attempt_id,
                        work_item_id=request.work_item.work_item_id,
                        thread_id=request.work_item.thread_id,
                        worker_id=request.worker_id,
                        status=AttemptStatus.COMPLETED,
                        knowledge_deltas=(
                            KnowledgeDelta(
                                delta_id="delta-1",
                                kind="SYNTHESIS",
                                summary="integrated evidence",
                                reference_ids=("evidence-1",),
                                causal_event_ids=("missing-event",),
                                thread_id="thread-1",
                            ),
                        ),
                    ),
                )

        with tempfile.TemporaryDirectory() as directory:
            with SQLiteResearchLedger(Path(directory) / "ledger.sqlite") as ledger:
                runtime = WorkerRuntime(ledger)
                with self.assertRaisesRegex(ValueError, "nonexistent causal event"):
                    runtime.run_attempt(
                        WorkerAssignment(
                            "worker-a",
                            _item("work-1", "thread-1", ("evidence-1",)),
                        ),
                        Bank(),
                    )

    def test_unauthorized_peer_does_not_discard_valid_batch_result(self) -> None:
        class MixedBank:
            def execute_batch(self, requests):
                good, bad = requests
                return (
                    AttemptResult(
                        attempt_id=good.attempt_id,
                        work_item_id=good.work_item.work_item_id,
                        thread_id=good.work_item.thread_id,
                        worker_id=good.worker_id,
                        status=AttemptStatus.COMPLETED,
                        progress_made=True,
                    ),
                    AttemptResult(
                        attempt_id=bad.attempt_id,
                        work_item_id=bad.work_item.work_item_id,
                        thread_id=bad.work_item.thread_id,
                        worker_id=bad.worker_id,
                        status=AttemptStatus.COMPLETED,
                        evidence_dispositions=(
                            EvidenceDisposition(
                                evidence_ids=("not-authorized",),
                                disposition=EvidenceDispositionKind.INVALID,
                            ),
                        ),
                    ),
                )

        with tempfile.TemporaryDirectory() as directory:
            with SQLiteResearchLedger(Path(directory) / "ledger.sqlite") as ledger:
                runtime = WorkerRuntime(ledger)
                with self.assertRaisesRegex(ValueError, "outside its Work Item authority"):
                    runtime.run_batch(
                        (
                            WorkerAssignment(
                                "worker-a",
                                _item("work-a", "thread-a", ("evidence-a",)),
                            ),
                            WorkerAssignment(
                                "worker-b",
                                _item("work-b", "thread-b", ("evidence-b",)),
                            ),
                        ),
                        MixedBank(),
                    )

                events = ledger.read_all_events()
                good_events = [event.event_type for event in events if event.thread_id == "thread-a"]
                bad_events = [event.event_type for event in events if event.thread_id == "thread-b"]
                self.assertIn("ATTEMPT_COMPLETED", good_events)
                self.assertIn("ATTEMPT_INVALID_RESULT", bad_events)
                self.assertNotIn("INTEGRATION_DISPOSITION_RECORDED", bad_events)


if __name__ == "__main__":
    unittest.main()
