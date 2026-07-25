"""Tests for the scale-invariant population runtime contracts."""

from __future__ import annotations

import unittest
from dataclasses import FrozenInstanceError

from ai_hypothesis.runtime import (
    AttemptResult,
    AttemptStatus,
    KnowledgeDelta,
    LedgerEvent,
    ProjectedState,
    SchedulerAction,
    SchedulerDecision,
    WorkItem,
    WorkPurpose,
)


class RuntimeContractTests(unittest.TestCase):
    def test_work_item_is_bounded_immutable_and_validatable(self) -> None:
        item = WorkItem(
            work_item_id="work-1",
            thread_id="thread-1",
            objective="Investigate alternative H2",
            purpose=WorkPurpose.EXPLORE,
            projection_revision=7,
            reference_ids=("evidence-1",),
            resource_budget={"max_steps": 1},
        )
        item.validate()

        with self.assertRaises(FrozenInstanceError):
            item.objective = "changed"  # type: ignore[misc]

    def test_attempt_result_preserves_failure_as_information(self) -> None:
        result = AttemptResult(
            attempt_id="attempt-1",
            work_item_id="work-1",
            thread_id="thread-1",
            worker_id="worker-7",
            status=AttemptStatus.FAILED,
            contradictions=("evidence-9 contradicts H2",),
            possibilities_eliminated=("H2",),
            progress_made=True,
        )
        result.validate()

        self.assertTrue(result.progress_made)
        self.assertEqual(result.possibilities_eliminated, ("H2",))

    def test_ledger_event_rejects_negative_sequence(self) -> None:
        event = LedgerEvent(
            event_id="event-1",
            event_type="EVIDENCE_ADDED",
            sequence=-1,
            payload_schema="runtime-event-v0",
        )

        with self.assertRaises(ValueError):
            event.validate()

    def test_projected_state_is_revisioned_and_rebuildable_contract(self) -> None:
        state = ProjectedState(
            revision=12,
            thread_id="thread-1",
            status="ACTIVE",
            purpose=WorkPurpose.VERIFY,
            reference_ids=("claim-2", "evidence-4"),
            contradiction_ids=("evidence-9",),
        )
        state.validate()
        self.assertEqual(state.revision, 12)

    def test_scheduler_decision_does_not_require_global_history(self) -> None:
        decision = SchedulerDecision(
            decision_id="decision-1",
            thread_id="thread-1",
            action=SchedulerAction.ADD_WIDTH,
            purpose=WorkPurpose.EXPLORE,
            reason_codes=("UNDER_COVERED",),
            projection_revision=12,
        )
        decision.validate()
        self.assertEqual(decision.action, SchedulerAction.ADD_WIDTH)

    def test_knowledge_delta_requires_recoverable_provenance(self) -> None:
        delta = KnowledgeDelta(
            delta_id="delta-1",
            kind="CONTRADICTION_ADDED",
            summary="H2 gained an independent contradiction",
            reference_ids=("evidence-9",),
            causal_event_ids=("event-8",),
            thread_id="thread-1",
        )
        delta.validate()

        missing_provenance = KnowledgeDelta(
            delta_id="delta-2",
            kind="SUMMARY",
            summary="Compressed state",
            reference_ids=(),
        )
        with self.assertRaises(ValueError):
            missing_provenance.validate()


if __name__ == "__main__":
    unittest.main()
