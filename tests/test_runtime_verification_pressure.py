"""Tests generic verification pressure derived from current knowledge state."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from ai_hypothesis.runtime import (
    AttemptResult,
    AttemptStatus,
    KnowledgeAssessment,
    KnowledgeAssessmentKind,
    KnowledgeVerificationConfig,
    KnowledgeVerificationTracker,
    RuntimeControlLoop,
    SchedulerAction,
    SchedulerConfig,
    SchedulerSignals,
    SchedulerV0,
    SQLiteResearchLedger,
    WorkPreparation,
)


class _VerificationBank:
    def execute_batch(self, requests):
        results = []
        for request in requests:
            assessments = ()
            if request.work_item.purpose.value == "VERIFY":
                assessments = (
                    KnowledgeAssessment(
                        delta_ids=("delta-1",),
                        assessment=KnowledgeAssessmentKind.VERIFIED,
                        reason="independent verification passed",
                    ),
                )
            results.append(
                AttemptResult(
                    attempt_id=request.attempt_id,
                    work_item_id=request.work_item.work_item_id,
                    thread_id=request.work_item.thread_id,
                    worker_id=request.worker_id,
                    status=AttemptStatus.COMPLETED,
                    knowledge_assessments=assessments,
                    progress_made=True,
                )
            )
        return tuple(results)


class VerificationPressureTests(unittest.TestCase):
    def test_provisional_knowledge_triggers_verify_then_pressure_clears(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with SQLiteResearchLedger(Path(directory) / "ledger.sqlite") as ledger:
                ledger.append_event(
                    event_type="THREAD_CREATED",
                    thread_id="thread-1",
                    payload={
                        "objective": "Investigate H2",
                        "purpose": "PROGRESS",
                    },
                )
                ledger.append_event(
                    event_type="KNOWLEDGE_DELTA_RECORDED",
                    thread_id="thread-1",
                    reference_ids=("delta-1", "evidence-1"),
                    payload={
                        "delta_id": "delta-1",
                        "kind": "HYPOTHESIS_WEAKENED",
                        "summary": "H2 is weakened by evidence-1",
                        "source_reference_ids": ["evidence-1"],
                        "causal_event_ids": [],
                    },
                )
                tracker = KnowledgeVerificationTracker(
                    ledger,
                    KnowledgeVerificationConfig(full_pressure_count=1),
                )
                loop = RuntimeControlLoop(
                    ledger=ledger,
                    scheduler=SchedulerV0(
                        SchedulerConfig(exploration_probability=0.0)
                    ),
                    worker_bank=_VerificationBank(),
                    worker_ids=("worker-a", "worker-b"),
                    verification_tracker=tracker,
                )
                signals = lambda _state: SchedulerSignals(
                    importance=1.0,
                    recent_progress=1.0,
                )
                prepare = lambda _state, _decision: WorkPreparation(
                    reference_ids=("delta-1",)
                )

                before = tracker.overview()
                self.assertEqual(before.pressure_for("thread-1"), 1.0)

                first = loop.run_once(
                    signal_provider=signals,
                    context_provider=prepare,
                )
                self.assertEqual(first.decision.action, SchedulerAction.VERIFY)
                self.assertEqual(first.assignment.work_item.purpose.value, "VERIFY")

                after = tracker.overview()
                self.assertEqual(after.pressure_for("thread-1"), 0.0)
                self.assertEqual(after.unresolved_count, 0)

                second = loop.run_once(
                    signal_provider=signals,
                    context_provider=prepare,
                )
                self.assertEqual(second.decision.action, SchedulerAction.CONTINUE)

    def test_disputed_knowledge_remains_verification_work(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with SQLiteResearchLedger(Path(directory) / "ledger.sqlite") as ledger:
                ledger.append_event(
                    event_type="KNOWLEDGE_DELTA_RECORDED",
                    thread_id="thread-1",
                    reference_ids=("delta-1", "evidence-1"),
                    payload={
                        "delta_id": "delta-1",
                        "kind": "TEST",
                        "summary": "provisional claim",
                        "source_reference_ids": ["evidence-1"],
                        "causal_event_ids": [],
                    },
                )
                ledger.append_event(
                    event_type="KNOWLEDGE_ASSESSMENT_RECORDED",
                    reference_ids=("delta-1",),
                    payload={
                        "assessment": "DISPUTED",
                        "reason": "counterexample found",
                    },
                )
                tracker = KnowledgeVerificationTracker(
                    ledger,
                    KnowledgeVerificationConfig(full_pressure_count=1),
                )

                overview = tracker.overview()
                self.assertEqual(overview.pressure_for("thread-1"), 1.0)
                self.assertEqual(
                    overview.pending_delta_ids("thread-1")
                    if hasattr(overview, "pending_delta_ids")
                    else tuple(
                        record.delta_id
                        for record in overview.pending_for("thread-1", limit=10)
                    ),
                    ("delta-1",),
                )

    def test_verified_and_retracted_knowledge_do_not_add_pressure(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with SQLiteResearchLedger(Path(directory) / "ledger.sqlite") as ledger:
                for index, assessment in enumerate(("VERIFIED", "RETRACTED"), start=1):
                    delta_id = f"delta-{index}"
                    ledger.append_event(
                        event_type="KNOWLEDGE_DELTA_RECORDED",
                        thread_id="thread-1",
                        reference_ids=(delta_id, f"evidence-{index}"),
                        payload={
                            "delta_id": delta_id,
                            "kind": "TEST",
                            "summary": delta_id,
                            "source_reference_ids": [f"evidence-{index}"],
                            "causal_event_ids": [],
                        },
                    )
                    ledger.append_event(
                        event_type="KNOWLEDGE_ASSESSMENT_RECORDED",
                        reference_ids=(delta_id,),
                        payload={"assessment": assessment},
                    )
                tracker = KnowledgeVerificationTracker(
                    ledger,
                    KnowledgeVerificationConfig(full_pressure_count=1),
                )

                overview = tracker.overview()
                self.assertEqual(overview.unresolved_count, 0)
                self.assertEqual(overview.pressure_for("thread-1"), 0.0)


if __name__ == "__main__":
    unittest.main()
