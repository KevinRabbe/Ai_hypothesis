from __future__ import annotations

import unittest

from ai_hypothesis.runtime.contracts import LedgerEvent
from ai_hypothesis.runtime.integration_allocation_outcomes import (
    IntegrationAllocationOutcomeProjector,
    summarize_integration_allocations_by_width,
)


_SCHEMA = "runtime-event-v0"


def _event(
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
        payload_schema=_SCHEMA,
        thread_id=thread_id,
        attempt_id=attempt_id,
        reference_ids=reference_ids,
        payload=payload or {},
    )


def _evidence(sequence: int, evidence_id: str) -> LedgerEvent:
    return _event(
        sequence,
        "EVIDENCE_ADDED",
        thread_id="thread-a",
        reference_ids=(evidence_id,),
        payload={"evidence_id": evidence_id},
    )


def _decision(
    sequence: int,
    decision_id: str,
    *,
    width: int,
    partitioned: bool = True,
) -> LedgerEvent:
    reasons = ["BACKPRESSURE"]
    if partitioned:
        reasons.append("PARTITIONED_INTEGRATION")
    return _event(
        sequence,
        "SCHEDULER_DECISION_RECORDED",
        thread_id="thread-a",
        payload={
            "decision_id": decision_id,
            "action": "SYNTHESIZE",
            "purpose": "SYNTHESIZE",
            "width": width,
            "reason_codes": reasons,
            "projection_revision": 4,
            "integration_backpressure": True,
            "max_width": 8,
        },
    )


def _started(
    sequence: int,
    attempt_id: str,
    decision_id: str,
    worker_id: str,
    refs: tuple[str, ...],
) -> LedgerEvent:
    return _event(
        sequence,
        "ATTEMPT_STARTED",
        thread_id="thread-a",
        attempt_id=attempt_id,
        reference_ids=refs,
        payload={
            "work_item_id": f"work-{attempt_id}",
            "worker_id": worker_id,
            "purpose": "SYNTHESIZE",
            "projection_revision": 4,
            "scheduler_decision_id": decision_id,
            "scope_region_ids": [],
        },
    )


class IntegrationAllocationOutcomeProjectorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.projector = IntegrationAllocationOutcomeProjector()

    def test_projects_unique_absorption_and_knowledge_per_allocation(self) -> None:
        events = (
            _evidence(1, "e1"),
            _evidence(2, "e2"),
            _evidence(3, "e3"),
            _evidence(4, "e4"),
            _decision(5, "decision-a", width=2),
            _started(6, "attempt-a", "decision-a", "worker-a", ("e1", "e2")),
            _started(7, "attempt-b", "decision-a", "worker-b", ("e3", "e4")),
            _event(
                8,
                "INTEGRATION_DISPOSITION_RECORDED",
                thread_id="thread-a",
                attempt_id="attempt-a",
                reference_ids=("e1", "e2"),
                payload={"disposition": "INTEGRATED"},
            ),
            _event(
                9,
                "INTEGRATION_DISPOSITION_RECORDED",
                thread_id="thread-a",
                attempt_id="attempt-b",
                reference_ids=("e3",),
                payload={"disposition": "INTEGRATED"},
            ),
            _event(
                10,
                "KNOWLEDGE_DELTA_RECORDED",
                thread_id="thread-a",
                attempt_id="attempt-a",
                reference_ids=("delta-a", "e1", "e2"),
                payload={
                    "delta_id": "delta-a",
                    "source_reference_ids": ["e1", "e2"],
                },
            ),
            _event(
                11,
                "KNOWLEDGE_DELTA_RECORDED",
                thread_id="thread-a",
                attempt_id="attempt-b",
                reference_ids=("delta-b", "e3"),
                payload={
                    "delta_id": "delta-b",
                    "source_reference_ids": ["e3"],
                },
            ),
            _event(
                12,
                "ATTEMPT_COMPLETED",
                thread_id="thread-a",
                attempt_id="attempt-a",
                payload={"progress_made": True},
            ),
            _event(
                13,
                "ATTEMPT_COMPLETED",
                thread_id="thread-a",
                attempt_id="attempt-b",
                payload={"progress_made": True},
            ),
        )

        (outcome,) = self.projector.project(events)

        self.assertEqual(outcome.decision_id, "decision-a")
        self.assertEqual(outcome.width, 2)
        self.assertTrue(outcome.partitioned)
        self.assertEqual(outcome.started_attempt_count, 2)
        self.assertEqual(outcome.terminal_attempt_count, 2)
        self.assertEqual(outcome.width_utilization, 1.0)
        self.assertEqual(outcome.input_reference_count, 4)
        self.assertEqual(outcome.unique_input_evidence_count, 4)
        self.assertEqual(outcome.duplicate_input_authority_count, 0)
        self.assertEqual(outcome.disposition_reference_count, 3)
        self.assertEqual(outcome.unique_dispositioned_input_evidence_count, 3)
        self.assertEqual(outcome.duplicate_disposition_reference_count, 0)
        self.assertEqual(outcome.input_absorption_fraction, 0.75)
        self.assertEqual(outcome.knowledge_delta_count, 2)
        self.assertEqual(outcome.knowledge_referenced_input_evidence_count, 3)

    def test_overlapping_input_authority_is_visible_not_averaged_away(self) -> None:
        events = (
            _evidence(1, "e1"),
            _evidence(2, "e2"),
            _evidence(3, "e3"),
            _decision(4, "decision-a", width=2),
            _started(5, "attempt-a", "decision-a", "worker-a", ("e1", "e2")),
            _started(6, "attempt-b", "decision-a", "worker-b", ("e2", "e3")),
            _event(
                7,
                "INTEGRATION_DISPOSITION_RECORDED",
                thread_id="thread-a",
                attempt_id="attempt-a",
                reference_ids=("e2",),
                payload={"disposition": "INTEGRATED"},
            ),
            _event(
                8,
                "INTEGRATION_DISPOSITION_RECORDED",
                thread_id="thread-a",
                attempt_id="attempt-b",
                reference_ids=("e2",),
                payload={"disposition": "INTEGRATED"},
            ),
        )

        (outcome,) = self.projector.project(events)

        self.assertEqual(outcome.input_reference_count, 4)
        self.assertEqual(outcome.unique_input_evidence_count, 3)
        self.assertEqual(outcome.duplicate_input_authority_count, 1)
        self.assertEqual(outcome.disposition_reference_count, 2)
        self.assertEqual(outcome.unique_dispositioned_input_evidence_count, 1)
        self.assertEqual(outcome.duplicate_disposition_reference_count, 1)

    def test_non_evidence_authority_and_out_of_input_dispositions_remain_visible(self) -> None:
        events = (
            _evidence(1, "e1"),
            _decision(2, "decision-a", width=1, partitioned=False),
            _started(
                3,
                "attempt-a",
                "decision-a",
                "worker-a",
                ("e1", "document-a"),
            ),
            _event(
                4,
                "INTEGRATION_DISPOSITION_RECORDED",
                thread_id="thread-a",
                attempt_id="attempt-a",
                reference_ids=("e1", "generated-evidence"),
                payload={"disposition": "INTEGRATED"},
            ),
        )

        (outcome,) = self.projector.project(events)
        attempt = outcome.attempts[0]

        self.assertEqual(attempt.non_evidence_input_reference_count, 1)
        self.assertEqual(attempt.out_of_input_disposition_reference_count, 1)
        self.assertEqual(outcome.unique_dispositioned_input_evidence_count, 1)

    def test_started_attempts_cannot_exceed_traced_width(self) -> None:
        events = (
            _evidence(1, "e1"),
            _evidence(2, "e2"),
            _decision(3, "decision-a", width=1),
            _started(4, "attempt-a", "decision-a", "worker-a", ("e1",)),
            _started(5, "attempt-b", "decision-a", "worker-b", ("e2",)),
        )

        with self.assertRaisesRegex(ValueError, "more attempts than allocated width"):
            self.projector.project(events)

    def test_non_backpressure_synthesis_is_not_classified_as_integration_allocation(self) -> None:
        decision = _event(
            1,
            "SCHEDULER_DECISION_RECORDED",
            thread_id="thread-a",
            payload={
                "decision_id": "decision-final",
                "action": "SYNTHESIZE",
                "purpose": "SYNTHESIZE",
                "width": 1,
                "reason_codes": ["FINAL_SYNTHESIS"],
                "projection_revision": 0,
            },
        )
        self.assertEqual(self.projector.project((decision,)), ())

    def test_width_summary_keeps_unique_absorption_separate_from_raw_traffic(self) -> None:
        events = (
            _evidence(1, "e1"),
            _evidence(2, "e2"),
            _decision(3, "decision-1", width=1, partitioned=False),
            _started(4, "attempt-1", "decision-1", "worker-a", ("e1",)),
            _event(
                5,
                "INTEGRATION_DISPOSITION_RECORDED",
                thread_id="thread-a",
                attempt_id="attempt-1",
                reference_ids=("e1", "e1"),
                payload={"disposition": "INTEGRATED"},
            ),
            _decision(6, "decision-2", width=2, partitioned=True),
            _started(7, "attempt-2a", "decision-2", "worker-a", ("e1",)),
            _started(8, "attempt-2b", "decision-2", "worker-b", ("e2",)),
            _event(
                9,
                "INTEGRATION_DISPOSITION_RECORDED",
                thread_id="thread-a",
                attempt_id="attempt-2a",
                reference_ids=("e1",),
                payload={"disposition": "INTEGRATED"},
            ),
            _event(
                10,
                "INTEGRATION_DISPOSITION_RECORDED",
                thread_id="thread-a",
                attempt_id="attempt-2b",
                reference_ids=("e2",),
                payload={"disposition": "INTEGRATED"},
            ),
        )

        summaries = summarize_integration_allocations_by_width(
            self.projector.project(events)
        )

        self.assertEqual(tuple(summary.width for summary in summaries), (1, 2))
        width1, width2 = summaries
        self.assertEqual(width1.unique_dispositioned_input_evidence_total, 1)
        self.assertEqual(width1.disposition_reference_total, 2)
        self.assertEqual(width1.duplicate_disposition_reference_total, 1)
        self.assertEqual(width1.mean_input_absorption_fraction, 1.0)
        self.assertEqual(width2.partitioned_allocation_count, 1)
        self.assertEqual(width2.unique_input_evidence_total, 2)
        self.assertEqual(width2.unique_dispositioned_input_evidence_total, 2)
        self.assertEqual(width2.mean_input_absorption_fraction, 1.0)


if __name__ == "__main__":
    unittest.main()
