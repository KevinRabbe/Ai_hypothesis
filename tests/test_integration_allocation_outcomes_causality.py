from __future__ import annotations

import unittest

from ai_hypothesis.runtime.contracts import LedgerEvent
from ai_hypothesis.runtime.integration_allocation_outcomes import (
    IntegrationAllocationOutcomeProjector,
)


_SCHEMA = "runtime-event-v0"


def _event(
    sequence: int,
    event_type: str,
    *,
    attempt_id: str | None = None,
    reference_ids: tuple[str, ...] = (),
    payload: dict | None = None,
) -> LedgerEvent:
    return LedgerEvent(
        event_id=f"event-{sequence}",
        event_type=event_type,
        sequence=sequence,
        payload_schema=_SCHEMA,
        thread_id="thread-a",
        attempt_id=attempt_id,
        reference_ids=reference_ids,
        payload=payload or {},
    )


def _evidence(sequence: int, evidence_id: str) -> LedgerEvent:
    return _event(
        sequence,
        "EVIDENCE_ADDED",
        reference_ids=(evidence_id,),
        payload={"evidence_id": evidence_id},
    )


def _decision(sequence: int) -> LedgerEvent:
    return _event(
        sequence,
        "SCHEDULER_DECISION_RECORDED",
        payload={
            "decision_id": "decision-a",
            "action": "SYNTHESIZE",
            "purpose": "SYNTHESIZE",
            "width": 1,
            "reason_codes": ["BACKPRESSURE", "PARTITIONED_INTEGRATION"],
            "projection_revision": 1,
            "integration_backpressure": True,
            "max_width": 1,
        },
    )


def _started(sequence: int, refs: tuple[str, ...] = ("e1",)) -> LedgerEvent:
    return _event(
        sequence,
        "ATTEMPT_STARTED",
        attempt_id="attempt-a",
        reference_ids=refs,
        payload={
            "work_item_id": "work-a",
            "worker_id": "worker-a",
            "purpose": "SYNTHESIZE",
            "projection_revision": 1,
            "scheduler_decision_id": "decision-a",
            "scope_region_ids": [],
        },
    )


class IntegrationAllocationCausalityTests(unittest.TestCase):
    def setUp(self) -> None:
        self.projector = IntegrationAllocationOutcomeProjector()

    def test_output_before_attempt_start_is_rejected(self) -> None:
        events = (
            _evidence(1, "e1"),
            _decision(2),
            _event(
                3,
                "INTEGRATION_DISPOSITION_RECORDED",
                attempt_id="attempt-a",
                reference_ids=("e1",),
                payload={"disposition": "INTEGRATED"},
            ),
            _started(4),
        )

        with self.assertRaisesRegex(ValueError, "precedes ATTEMPT_STARTED"):
            self.projector.project(events)

    def test_attempt_start_before_scheduler_decision_is_rejected(self) -> None:
        events = (
            _evidence(1, "e1"),
            _started(2),
            _decision(3),
        )

        with self.assertRaisesRegex(ValueError, "precedes scheduler decision"):
            self.projector.project(events)

    def test_attempt_cannot_reference_evidence_created_later(self) -> None:
        events = (
            _decision(1),
            _started(2),
            _evidence(3, "e1"),
        )

        with self.assertRaisesRegex(ValueError, "evidence created after attempt start"):
            self.projector.project(events)

    def test_output_after_terminal_event_is_rejected(self) -> None:
        events = (
            _evidence(1, "e1"),
            _decision(2),
            _started(3),
            _event(
                4,
                "ATTEMPT_COMPLETED",
                attempt_id="attempt-a",
                payload={"progress_made": True},
            ),
            _event(
                5,
                "KNOWLEDGE_DELTA_RECORDED",
                attempt_id="attempt-a",
                reference_ids=("delta-a", "e1"),
                payload={
                    "delta_id": "delta-a",
                    "source_reference_ids": ["e1"],
                },
            ),
        )

        with self.assertRaisesRegex(ValueError, "output after terminal event"):
            self.projector.project(events)

    def test_second_terminal_event_is_rejected(self) -> None:
        events = (
            _evidence(1, "e1"),
            _decision(2),
            _started(3),
            _event(
                4,
                "ATTEMPT_FAILED",
                attempt_id="attempt-a",
                payload={"progress_made": False},
            ),
            _event(
                5,
                "ATTEMPT_COMPLETED",
                attempt_id="attempt-a",
                payload={"progress_made": True},
            ),
        )

        with self.assertRaisesRegex(ValueError, "more than one terminal event"):
            self.projector.project(events)


if __name__ == "__main__":
    unittest.main()
