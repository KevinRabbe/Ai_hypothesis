"""Tests observable allocation outcomes without defining a reward function."""

from __future__ import annotations

import unittest

from ai_hypothesis.runtime import AllocationOutcomeProjector, LedgerEvent


def event(
    sequence: int,
    event_type: str,
    *,
    thread_id: str | None = None,
    attempt_id: str | None = None,
    reference_ids: tuple[str, ...] = (),
    payload: dict | None = None,
) -> LedgerEvent:
    return LedgerEvent(
        event_id=f"event-{sequence}",
        event_type=event_type,
        sequence=sequence,
        payload_schema="runtime-event-v0",
        thread_id=thread_id,
        attempt_id=attempt_id,
        reference_ids=reference_ids,
        payload=payload or {},
    )


def decision(sequence: int, *, width: int = 2) -> LedgerEvent:
    return event(
        sequence,
        "SCHEDULER_DECISION_RECORDED",
        thread_id="thread-1",
        payload={
            "decision_id": "decision-1",
            "action": "ADD_WIDTH" if width > 1 else "CONTINUE",
            "purpose": "EXPLORE" if width > 1 else "PROGRESS",
            "width": width,
            "reason_codes": ["STRUCTURED_EXPLORATION"] if width > 1 else ["PROGRESS"],
            "projection_revision": 7,
            "integration_backpressure": False,
            "max_width": 8,
        },
    )


def started(
    sequence: int,
    attempt_id: str,
    worker_id: str,
    work_item_id: str,
    *,
    decision_id: str | None = "decision-1",
    purpose: str = "EXPLORE",
) -> LedgerEvent:
    return event(
        sequence,
        "ATTEMPT_STARTED",
        thread_id="thread-1",
        attempt_id=attempt_id,
        payload={
            "work_item_id": work_item_id,
            "worker_id": worker_id,
            "purpose": purpose,
            "projection_revision": 7,
            "scheduler_decision_id": decision_id,
        },
    )


class AllocationOutcomeProjectorTests(unittest.TestCase):
    def test_width_two_decision_groups_attempt_outputs_and_crash(self) -> None:
        events = (
            decision(1, width=2),
            started(2, "attempt-a", "worker-a", "work-a"),
            event(
                3,
                "EVIDENCE_ADDED",
                thread_id="thread-1",
                attempt_id="attempt-a",
                reference_ids=("evidence-1",),
                payload={"evidence_id": "evidence-1"},
            ),
            event(
                4,
                "KNOWLEDGE_DELTA_RECORDED",
                thread_id="thread-1",
                attempt_id="attempt-a",
                reference_ids=("delta-1", "evidence-1"),
                payload={"delta_id": "delta-1"},
            ),
            event(
                5,
                "KNOWLEDGE_ASSESSMENT_RECORDED",
                thread_id="thread-1",
                attempt_id="attempt-a",
                reference_ids=("delta-old",),
                payload={"assessment": "VERIFIED"},
            ),
            event(
                6,
                "INTEGRATION_DISPOSITION_RECORDED",
                thread_id="thread-1",
                attempt_id="attempt-a",
                reference_ids=("evidence-old",),
                payload={"disposition": "INTEGRATED"},
            ),
            event(
                7,
                "CONTRADICTION_FOUND",
                thread_id="thread-1",
                attempt_id="attempt-a",
                reference_ids=("contradiction-1",),
            ),
            event(
                8,
                "POSSIBILITY_ELIMINATED",
                thread_id="thread-1",
                attempt_id="attempt-a",
                reference_ids=("possibility-1",),
            ),
            event(
                9,
                "OPEN_QUESTION_ADDED",
                thread_id="thread-1",
                attempt_id="attempt-a",
                payload={"question": "What remains unresolved?"},
            ),
            event(
                10,
                "FOLLOWUP_REQUESTED",
                thread_id="thread-1",
                attempt_id="attempt-a",
                payload={"request": "Inspect the boundary case"},
            ),
            event(
                11,
                "ATTEMPT_COMPLETED",
                thread_id="thread-1",
                attempt_id="attempt-a",
                payload={
                    "status": "COMPLETED",
                    "progress_made": True,
                    "resource_usage": {"compute_units": 12, "wall_ms": 4.5},
                },
            ),
            started(12, "attempt-b", "worker-b", "work-b"),
            event(
                13,
                "ATTEMPT_CRASHED",
                thread_id="thread-1",
                attempt_id="attempt-b",
                payload={"error_type": "RuntimeError", "error": "boom"},
            ),
        )

        outcomes = AllocationOutcomeProjector().project(events)
        self.assertEqual(len(outcomes), 1)
        outcome = outcomes[0]
        self.assertEqual(outcome.scheduler_decision_id, "decision-1")
        self.assertEqual(outcome.attempt_count, 2)
        self.assertEqual(outcome.allocated_width, 2)
        self.assertEqual(outcome.evidence_count, 1)
        self.assertEqual(outcome.knowledge_delta_count, 1)
        self.assertEqual(outcome.knowledge_assessment_count, 1)
        self.assertEqual(outcome.evidence_disposition_count, 1)
        self.assertEqual(outcome.contradiction_count, 1)
        self.assertEqual(outcome.possibility_elimination_count, 1)
        self.assertEqual(outcome.open_question_count, 1)
        self.assertEqual(outcome.followup_count, 1)

        completed, crashed = outcome.attempts
        self.assertEqual(completed.terminal_event_type, "ATTEMPT_COMPLETED")
        self.assertTrue(completed.progress_made)
        self.assertEqual(completed.resource_usage["compute_units"], 12)
        self.assertEqual(completed.evidence_ids, ("evidence-1",))
        self.assertEqual(completed.knowledge_delta_ids, ("delta-1",))
        self.assertEqual(crashed.terminal_event_type, "ATTEMPT_CRASHED")
        self.assertIsNone(crashed.progress_made)
        self.assertEqual(dict(crashed.resource_usage), {})

    def test_manual_untraced_attempt_is_outside_allocation_projection(self) -> None:
        events = (
            started(
                1,
                "manual-attempt",
                "worker-a",
                "manual-work",
                decision_id=None,
                purpose="PROGRESS",
            ),
            event(
                2,
                "ATTEMPT_COMPLETED",
                thread_id="thread-1",
                attempt_id="manual-attempt",
                payload={
                    "status": "COMPLETED",
                    "progress_made": True,
                    "resource_usage": {},
                },
            ),
        )
        self.assertEqual(AllocationOutcomeProjector().project(events), ())

    def test_explicit_provenance_without_local_trace_is_preserved(self) -> None:
        events = (
            started(
                1,
                "external-attempt",
                "worker-a",
                "external-work",
                decision_id="external-decision",
                purpose="PROGRESS",
            ),
            event(
                2,
                "ATTEMPT_FAILED",
                thread_id="thread-1",
                attempt_id="external-attempt",
                payload={
                    "status": "FAILED",
                    "progress_made": False,
                    "resource_usage": {"compute_units": 2},
                },
            ),
        )
        outcome = AllocationOutcomeProjector().project(events)[0]
        self.assertEqual(outcome.scheduler_decision_id, "external-decision")
        self.assertIsNone(outcome.action)
        self.assertEqual(outcome.attempt_count, 1)
        self.assertEqual(outcome.attempts[0].resource_usage["compute_units"], 2)

    def test_more_attempts_than_allocated_width_is_rejected(self) -> None:
        events = (
            decision(1, width=1),
            started(2, "attempt-a", "worker-a", "work-a", purpose="PROGRESS"),
            started(3, "attempt-b", "worker-b", "work-b", purpose="PROGRESS"),
        )
        with self.assertRaisesRegex(ValueError, "more attempts than allocated width"):
            AllocationOutcomeProjector().project(events)

    def test_output_after_terminal_is_rejected(self) -> None:
        events = (
            decision(1, width=1),
            started(2, "attempt-a", "worker-a", "work-a", purpose="PROGRESS"),
            event(
                3,
                "ATTEMPT_COMPLETED",
                thread_id="thread-1",
                attempt_id="attempt-a",
                payload={
                    "status": "COMPLETED",
                    "progress_made": True,
                    "resource_usage": {},
                },
            ),
            event(
                4,
                "EVIDENCE_ADDED",
                thread_id="thread-1",
                attempt_id="attempt-a",
                payload={"evidence_id": "late-evidence"},
            ),
        )
        with self.assertRaisesRegex(ValueError, "events after terminal"):
            AllocationOutcomeProjector().project(events)

    def test_out_of_order_events_are_rejected(self) -> None:
        events = (
            decision(2, width=1),
            started(1, "attempt-a", "worker-a", "work-a", purpose="PROGRESS"),
        )
        with self.assertRaisesRegex(ValueError, "strictly increasing"):
            AllocationOutcomeProjector().project(events)


if __name__ == "__main__":
    unittest.main()
