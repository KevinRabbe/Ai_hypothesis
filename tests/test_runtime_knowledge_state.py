"""Tests explicit knowledge state over append-only synthesis and assessment events."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from ai_hypothesis.runtime import (
    AttemptResult,
    AttemptStatus,
    KnowledgeAssessment,
    KnowledgeAssessmentKind,
    KnowledgeDelta,
    KnowledgeStateProjector,
    KnowledgeStatus,
    LedgerEvent,
    SQLiteResearchLedger,
    WorkerAssignment,
    WorkerRuntime,
    WorkItem,
    WorkPurpose,
)


class _SynthesisBank:
    def execute_batch(self, requests):
        return tuple(
            AttemptResult(
                attempt_id=request.attempt_id,
                work_item_id=request.work_item.work_item_id,
                thread_id=request.work_item.thread_id,
                worker_id=request.worker_id,
                status=AttemptStatus.COMPLETED,
                knowledge_deltas=(
                    KnowledgeDelta(
                        delta_id="delta-1",
                        kind="HYPOTHESIS_WEAKENED",
                        summary="H2 is weakened by evidence-1",
                        reference_ids=("evidence-1",),
                        thread_id="research-thread",
                    ),
                ),
                progress_made=True,
            )
            for request in requests
        )


class _AssessmentBank:
    def __init__(self, assessment: KnowledgeAssessmentKind, reason: str) -> None:
        self.assessment = assessment
        self.reason = reason

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
                        delta_ids=("delta-1",),
                        assessment=self.assessment,
                        reason=self.reason,
                    ),
                ),
                progress_made=True,
            )
            for request in requests
        )


def _item(
    *,
    work_id: str,
    thread_id: str,
    purpose: WorkPurpose,
    reference_ids: tuple[str, ...],
    parent_ids: tuple[str, ...] = (),
) -> WorkItem:
    return WorkItem(
        work_item_id=work_id,
        thread_id=thread_id,
        objective="Update bounded knowledge state",
        purpose=purpose,
        projection_revision=1,
        reference_ids=reference_ids,
        parent_ids=parent_ids,
    )


class KnowledgeStateTests(unittest.TestCase):
    def test_synthesis_starts_provisional_then_assessments_update_current_status(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with SQLiteResearchLedger(Path(directory) / "ledger.sqlite") as ledger:
                runtime = WorkerRuntime(ledger)
                projector = KnowledgeStateProjector()

                runtime.run_attempt(
                    WorkerAssignment(
                        "worker-synthesis",
                        _item(
                            work_id="work-synthesis",
                            thread_id="research-thread",
                            purpose=WorkPurpose.SYNTHESIZE,
                            reference_ids=("evidence-1",),
                        ),
                    ),
                    _SynthesisBank(),
                )
                provisional = projector.project(ledger.read_all_events())
                record = provisional.get("delta-1")
                self.assertIsNotNone(record)
                assert record is not None
                self.assertEqual(record.status, KnowledgeStatus.PROVISIONAL)
                self.assertEqual(record.source_reference_ids, ("evidence-1",))
                self.assertTrue(record.is_active)

                runtime.run_attempt(
                    WorkerAssignment(
                        "worker-verify",
                        _item(
                            work_id="work-verify",
                            thread_id="verification-thread",
                            purpose=WorkPurpose.VERIFY,
                            reference_ids=("delta-1",),
                        ),
                    ),
                    _AssessmentBank(
                        KnowledgeAssessmentKind.VERIFIED,
                        "Independent verification passed",
                    ),
                )
                verified = projector.project(ledger.read_all_events())
                verified_record = verified.get("delta-1")
                assert verified_record is not None
                self.assertEqual(verified_record.status, KnowledgeStatus.VERIFIED)
                self.assertEqual(
                    verified_record.assessment_reason,
                    "Independent verification passed",
                )

                runtime.run_attempt(
                    WorkerAssignment(
                        "worker-challenge",
                        _item(
                            work_id="work-challenge",
                            thread_id="challenge-thread",
                            purpose=WorkPurpose.CHALLENGE,
                            reference_ids=("delta-1",),
                        ),
                    ),
                    _AssessmentBank(
                        KnowledgeAssessmentKind.DISPUTED,
                        "A counterexample remains unresolved",
                    ),
                )
                disputed = projector.project(ledger.read_all_events())
                disputed_record = disputed.get("delta-1")
                assert disputed_record is not None
                self.assertEqual(disputed_record.status, KnowledgeStatus.DISPUTED)

                runtime.run_attempt(
                    WorkerAssignment(
                        "worker-retract",
                        _item(
                            work_id="work-retract",
                            thread_id="challenge-thread",
                            purpose=WorkPurpose.CHALLENGE,
                            reference_ids=("delta-1",),
                        ),
                    ),
                    _AssessmentBank(
                        KnowledgeAssessmentKind.RETRACTED,
                        "The source evidence was invalidated",
                    ),
                )
                retracted = projector.project(ledger.read_all_events())
                retracted_record = retracted.get("delta-1")
                assert retracted_record is not None
                self.assertEqual(retracted_record.status, KnowledgeStatus.RETRACTED)
                self.assertFalse(retracted_record.is_active)
                self.assertEqual(retracted.active_records, ())

                event_types = tuple(event.event_type for event in ledger.read_all_events())
                self.assertEqual(event_types.count("KNOWLEDGE_DELTA_RECORDED"), 1)
                self.assertEqual(event_types.count("KNOWLEDGE_ASSESSMENT_RECORDED"), 3)

    def test_thread_filter_applies_after_cross_thread_assessment(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with SQLiteResearchLedger(Path(directory) / "ledger.sqlite") as ledger:
                runtime = WorkerRuntime(ledger)
                runtime.run_attempt(
                    WorkerAssignment(
                        "worker-synthesis",
                        _item(
                            work_id="work-synthesis",
                            thread_id="research-thread",
                            purpose=WorkPurpose.SYNTHESIZE,
                            reference_ids=("evidence-1",),
                        ),
                    ),
                    _SynthesisBank(),
                )
                runtime.run_attempt(
                    WorkerAssignment(
                        "worker-verify",
                        _item(
                            work_id="work-verify",
                            thread_id="verification-thread",
                            purpose=WorkPurpose.VERIFY,
                            reference_ids=("delta-1",),
                        ),
                    ),
                    _AssessmentBank(
                        KnowledgeAssessmentKind.VERIFIED,
                        "Verified from a separate thread",
                    ),
                )

                snapshot = KnowledgeStateProjector().project(
                    ledger.read_all_events(),
                    thread_id="research-thread",
                )
                self.assertEqual(len(snapshot.records), 1)
                self.assertEqual(snapshot.records[0].status, KnowledgeStatus.VERIFIED)

    def test_worker_cannot_assess_delta_outside_work_item_authority(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with SQLiteResearchLedger(Path(directory) / "ledger.sqlite") as ledger:
                runtime = WorkerRuntime(ledger)
                with self.assertRaisesRegex(ValueError, "outside its Work Item authority"):
                    runtime.run_attempt(
                        WorkerAssignment(
                            "worker-verify",
                            _item(
                                work_id="work-verify",
                                thread_id="verification-thread",
                                purpose=WorkPurpose.VERIFY,
                                reference_ids=("different-delta",),
                            ),
                        ),
                        _AssessmentBank(
                            KnowledgeAssessmentKind.VERIFIED,
                            "Should be rejected",
                        ),
                    )

                event_types = tuple(event.event_type for event in ledger.read_all_events())
                self.assertEqual(
                    event_types,
                    ("ATTEMPT_STARTED", "ATTEMPT_INVALID_RESULT"),
                )

    def test_projector_rejects_assessment_before_delta_exists(self) -> None:
        events = (
            LedgerEvent(
                event_id="assessment-event",
                event_type="KNOWLEDGE_ASSESSMENT_RECORDED",
                sequence=1,
                payload_schema="runtime-event-v0",
                reference_ids=("unknown-delta",),
                payload={"assessment": "VERIFIED"},
            ),
        )
        with self.assertRaisesRegex(ValueError, "unknown delta"):
            KnowledgeStateProjector().project(events)


if __name__ == "__main__":
    unittest.main()
