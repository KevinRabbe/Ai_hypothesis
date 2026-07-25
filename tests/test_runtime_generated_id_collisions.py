"""Tests unique durable identities for generated evidence and knowledge."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from ai_hypothesis.runtime import (
    AttemptResult,
    AttemptStatus,
    EvidenceContribution,
    KnowledgeDelta,
    SQLiteResearchLedger,
    WorkerAssignment,
    WorkerRuntime,
    WorkItem,
    WorkPurpose,
)


def _item(index: int) -> WorkItem:
    return WorkItem(
        work_item_id=f"work-{index}",
        thread_id=f"thread-{index}",
        objective="Produce one bounded result",
        purpose=WorkPurpose.EXPLORE,
        projection_revision=1,
        reference_ids=(f"source-{index}",),
    )


class GeneratedIdCollisionTests(unittest.TestCase):
    def test_existing_evidence_id_cannot_be_reused_as_knowledge_delta_id(self) -> None:
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
                                delta_id="shared-id",
                                kind="TEST",
                                summary="should collide",
                                reference_ids=("source-1",),
                            ),
                        ),
                    ),
                )

        with tempfile.TemporaryDirectory() as directory:
            with SQLiteResearchLedger(Path(directory) / "ledger.sqlite") as ledger:
                ledger.append_event(
                    event_type="EVIDENCE_ADDED",
                    reference_ids=("shared-id",),
                    payload={
                        "evidence_id": "shared-id",
                        "kind": "TEST",
                        "summary": "existing evidence",
                    },
                )
                runtime = WorkerRuntime(ledger)

                with self.assertRaisesRegex(ValueError, "reused an existing durable"):
                    runtime.run_attempt(
                        WorkerAssignment("worker-1", _item(1)),
                        Bank(),
                    )

                delta_events = [
                    event
                    for event in ledger.read_all_events()
                    if event.event_type == "KNOWLEDGE_DELTA_RECORDED"
                ]
                self.assertEqual(delta_events, [])

    def test_colliding_batch_results_are_rejected_without_losing_valid_peer(self) -> None:
        class Bank:
            def execute_batch(self, requests):
                first, second, third = requests
                return (
                    AttemptResult(
                        attempt_id=first.attempt_id,
                        work_item_id=first.work_item.work_item_id,
                        thread_id=first.work_item.thread_id,
                        worker_id=first.worker_id,
                        status=AttemptStatus.COMPLETED,
                        evidence=(
                            EvidenceContribution(
                                evidence_id="collision-id",
                                kind="TEST",
                                summary="first collision",
                                reference_ids=("source-1",),
                            ),
                        ),
                    ),
                    AttemptResult(
                        attempt_id=second.attempt_id,
                        work_item_id=second.work_item.work_item_id,
                        thread_id=second.work_item.thread_id,
                        worker_id=second.worker_id,
                        status=AttemptStatus.COMPLETED,
                        knowledge_deltas=(
                            KnowledgeDelta(
                                delta_id="collision-id",
                                kind="TEST",
                                summary="second collision",
                                reference_ids=("source-2",),
                            ),
                        ),
                    ),
                    AttemptResult(
                        attempt_id=third.attempt_id,
                        work_item_id=third.work_item.work_item_id,
                        thread_id=third.work_item.thread_id,
                        worker_id=third.worker_id,
                        status=AttemptStatus.COMPLETED,
                        evidence=(
                            EvidenceContribution(
                                evidence_id="valid-id",
                                kind="TEST",
                                summary="valid independent result",
                                reference_ids=("source-3",),
                            ),
                        ),
                        progress_made=True,
                    ),
                )

        with tempfile.TemporaryDirectory() as directory:
            with SQLiteResearchLedger(Path(directory) / "ledger.sqlite") as ledger:
                runtime = WorkerRuntime(ledger)
                with self.assertRaisesRegex(ValueError, "colliding durable object IDs"):
                    runtime.run_batch(
                        (
                            WorkerAssignment("worker-1", _item(1)),
                            WorkerAssignment("worker-2", _item(2)),
                            WorkerAssignment("worker-3", _item(3)),
                        ),
                        Bank(),
                    )

                events = ledger.read_all_events()
                valid_events = [
                    event for event in events if event.thread_id == "thread-3"
                ]
                self.assertIn(
                    "EVIDENCE_ADDED",
                    tuple(event.event_type for event in valid_events),
                )
                self.assertIn(
                    "ATTEMPT_COMPLETED",
                    tuple(event.event_type for event in valid_events),
                )
                collision_events = [
                    event
                    for event in events
                    if event.event_type == "EVIDENCE_ADDED"
                    and event.payload.get("evidence_id") == "collision-id"
                ]
                self.assertEqual(collision_events, [])


if __name__ == "__main__":
    unittest.main()
