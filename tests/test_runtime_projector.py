"""Tests for deterministic Work Thread state projection."""

from __future__ import annotations

import unittest

from ai_hypothesis.runtime import LedgerEvent, ThreadStateProjector, WorkPurpose


def event(
    sequence: int,
    event_type: str,
    *,
    thread_id: str = "thread-1",
    reference_ids: tuple[str, ...] = (),
    payload: dict[str, object] | None = None,
) -> LedgerEvent:
    return LedgerEvent(
        event_id=f"event-{sequence}",
        event_type=event_type,
        sequence=sequence,
        payload_schema="runtime-event-v0",
        thread_id=thread_id,
        reference_ids=reference_ids,
        payload=payload or {},
    )


class ThreadStateProjectorTests(unittest.TestCase):
    def test_project_rebuilds_current_thread_state(self) -> None:
        projector = ThreadStateProjector()
        events = (
            event(
                1,
                "THREAD_CREATED",
                payload={"objective": "Investigate H2", "purpose": "EXPLORE"},
            ),
            event(2, "HYPOTHESIS_PROPOSED", reference_ids=("H2",)),
            event(3, "CONTRADICTION_FOUND", reference_ids=("E9",)),
            event(4, "OPEN_QUESTION_ADDED", payload={"question": "Does E9 generalize?"}),
            event(5, "DEPENDENCY_ADDED", reference_ids=("thread-8",)),
            event(6, "THREAD_PURPOSE_SET", payload={"purpose": "VERIFY"}),
            event(7, "THREAD_PAUSED"),
        )

        state = projector.project(events, thread_id="thread-1")

        self.assertEqual(state.revision, 7)
        self.assertEqual(state.objective, "Investigate H2")
        self.assertEqual(state.purpose, WorkPurpose.VERIFY)
        self.assertEqual(state.status, "PAUSED")
        self.assertEqual(state.hypothesis_ids, ("H2",))
        self.assertEqual(state.contradiction_ids, ("E9",))
        self.assertEqual(state.open_questions, ("Does E9 generalize?",))
        self.assertEqual(state.dependency_thread_ids, ("thread-8",))

    def test_rejected_and_resolved_items_are_removed(self) -> None:
        projector = ThreadStateProjector()
        events = (
            event(1, "THREAD_CREATED", payload={"objective": "Investigate H2"}),
            event(2, "HYPOTHESIS_PROPOSED", reference_ids=("H2",)),
            event(3, "HYPOTHESIS_REJECTED", reference_ids=("H2",)),
            event(4, "CONTRADICTION_FOUND", reference_ids=("E9",)),
            event(5, "CONTRADICTION_RESOLVED", reference_ids=("E9",)),
            event(6, "OPEN_QUESTION_ADDED", payload={"question": "Q1"}),
            event(7, "OPEN_QUESTION_RESOLVED", payload={"question": "Q1"}),
        )

        state = projector.project(events, thread_id="thread-1")
        self.assertEqual(state.hypothesis_ids, ())
        self.assertEqual(state.contradiction_ids, ())
        self.assertEqual(state.open_questions, ())

    def test_other_threads_do_not_pollute_projection(self) -> None:
        projector = ThreadStateProjector()
        events = (
            event(1, "THREAD_CREATED", payload={"objective": "Main"}),
            event(
                2,
                "THREAD_CREATED",
                thread_id="thread-2",
                payload={"objective": "Other"},
            ),
            event(3, "HYPOTHESIS_PROPOSED", thread_id="thread-2", reference_ids=("H99",)),
            event(4, "HYPOTHESIS_PROPOSED", reference_ids=("H2",)),
        )

        state = projector.project(events, thread_id="thread-1")
        self.assertEqual(state.objective, "Main")
        self.assertEqual(state.hypothesis_ids, ("H2",))
        self.assertEqual(state.revision, 4)

    def test_projection_requires_strict_event_order(self) -> None:
        projector = ThreadStateProjector()
        events = (
            event(2, "THREAD_CREATED", payload={"objective": "H2"}),
            event(1, "THREAD_PAUSED"),
        )

        with self.assertRaises(ValueError):
            projector.project(events, thread_id="thread-1")

    def test_projection_requires_creation_event(self) -> None:
        projector = ThreadStateProjector()
        with self.assertRaises(ValueError):
            projector.project((event(1, "THREAD_PAUSED"),), thread_id="thread-1")


if __name__ == "__main__":
    unittest.main()
