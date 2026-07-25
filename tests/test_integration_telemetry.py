from __future__ import annotations

import unittest

from ai_hypothesis.runtime.contracts import LedgerEvent
from ai_hypothesis.runtime.integration_telemetry import (
    IntegrationBandwidthWindow,
    IntegrationTelemetryProjector,
)


_SCHEMA = "runtime-event-v0"


def _event(
    sequence: int,
    event_type: str,
    *,
    thread_id: str | None = None,
    reference_ids: tuple[str, ...] = (),
    payload: dict | None = None,
) -> LedgerEvent:
    return LedgerEvent(
        event_id=f"event-{sequence}",
        event_type=event_type,
        sequence=sequence,
        payload_schema=_SCHEMA,
        thread_id=thread_id,
        reference_ids=reference_ids,
        payload=payload or {},
    )


class IntegrationTelemetryProjectorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.projector = IntegrationTelemetryProjector()
        self.events = (
            _event(
                1,
                "EVIDENCE_ADDED",
                thread_id="thread-a",
                reference_ids=("evidence-a",),
                payload={"evidence_id": "evidence-a"},
            ),
            _event(
                2,
                "EVIDENCE_ADDED",
                thread_id="thread-a",
                reference_ids=("evidence-b",),
                payload={"evidence_id": "evidence-b"},
            ),
            _event(
                3,
                "EVIDENCE_ADDED",
                thread_id="thread-b",
                reference_ids=("evidence-c",),
                payload={"evidence_id": "evidence-c"},
            ),
            _event(
                4,
                "INTEGRATION_DISPOSITION_RECORDED",
                thread_id="integrator",
                reference_ids=("evidence-a",),
                payload={"disposition": "INTEGRATED"},
            ),
            _event(
                5,
                "INTEGRATION_DISPOSITION_RECORDED",
                thread_id="integrator",
                reference_ids=("evidence-b",),
                payload={"disposition": "DUPLICATE"},
            ),
            _event(
                6,
                "KNOWLEDGE_DELTA_RECORDED",
                thread_id="thread-a",
                reference_ids=("delta-a", "evidence-a", "evidence-b", "document-7"),
                payload={
                    "delta_id": "delta-a",
                    "source_reference_ids": [
                        "evidence-a",
                        "evidence-b",
                        "document-7",
                    ],
                },
            ),
            _event(
                7,
                "INTEGRATION_DISPOSITION_RECORDED",
                thread_id="integrator",
                reference_ids=("evidence-a",),
                payload={"disposition": "INTEGRATED"},
            ),
            _event(
                8,
                "EVIDENCE_ADDED",
                thread_id="thread-a",
                reference_ids=("evidence-d",),
                payload={"evidence_id": "evidence-d"},
            ),
        )

    def test_projects_unique_resolution_backlog_and_knowledge_references(self) -> None:
        snapshot = self.projector.project(self.events)

        self.assertEqual(snapshot.revision, 8)
        self.assertEqual(snapshot.evidence_count, 4)
        self.assertEqual(snapshot.dispositioned_evidence_count, 2)
        self.assertEqual(snapshot.disposition_reference_count, 3)
        self.assertEqual(snapshot.backlog_count, 2)
        self.assertEqual(snapshot.knowledge_delta_count, 1)
        self.assertEqual(snapshot.knowledge_referenced_evidence_count, 2)
        self.assertEqual(snapshot.knowledge_source_reference_count, 2)
        self.assertEqual(snapshot.disposition_counts["INTEGRATED"], 1)
        self.assertEqual(snapshot.disposition_counts["DUPLICATE"], 1)
        self.assertEqual(snapshot.redisposition_count, 1)
        self.assertEqual(snapshot.unknown_disposition_reference_count, 0)
        self.assertEqual(snapshot.mean_disposition_latency_sequences, 3.0)
        self.assertEqual(snapshot.max_disposition_latency_sequences, 3)
        self.assertEqual(snapshot.mean_backlog_age_sequences, 2.5)
        self.assertEqual(snapshot.oldest_backlog_age_sequences, 5)
        self.assertEqual(snapshot.disposition_fraction, 0.5)
        self.assertEqual(snapshot.knowledge_reference_fraction, 0.5)
        self.assertEqual(snapshot.evidence_per_knowledge_delta, 2.0)

    def test_thread_projection_keeps_source_thread_semantics(self) -> None:
        thread_a = self.projector.project(self.events, thread_id="thread-a")
        thread_b = self.projector.project(self.events, thread_id="thread-b")

        self.assertEqual(thread_a.evidence_count, 3)
        self.assertEqual(thread_a.dispositioned_evidence_count, 2)
        self.assertEqual(thread_a.backlog_count, 1)
        self.assertEqual(thread_a.knowledge_delta_count, 1)
        self.assertEqual(thread_a.knowledge_referenced_evidence_count, 2)
        self.assertEqual(thread_a.oldest_backlog_age_sequences, 0)

        self.assertEqual(thread_b.evidence_count, 1)
        self.assertEqual(thread_b.dispositioned_evidence_count, 0)
        self.assertEqual(thread_b.backlog_count, 1)
        self.assertEqual(thread_b.knowledge_delta_count, 0)
        self.assertEqual(thread_b.knowledge_referenced_evidence_count, 0)
        self.assertEqual(thread_b.oldest_backlog_age_sequences, 5)

    def test_unknown_disposition_reference_is_visible_globally(self) -> None:
        snapshot = self.projector.project(
            (*self.events, _event(
                9,
                "INTEGRATION_DISPOSITION_RECORDED",
                reference_ids=("missing-evidence",),
                payload={"disposition": "INVALID"},
            ))
        )

        self.assertEqual(snapshot.unknown_disposition_reference_count, 1)
        self.assertEqual(snapshot.dispositioned_evidence_count, 2)
        self.assertEqual(snapshot.disposition_reference_count, 3)

    def test_duplicate_evidence_identity_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "duplicate durable evidence ID"):
            self.projector.project(
                (
                    self.events[0],
                    _event(
                        2,
                        "EVIDENCE_ADDED",
                        payload={"evidence_id": "evidence-a"},
                    ),
                )
            )


class IntegrationBandwidthWindowTests(unittest.TestCase):
    def test_rates_distinguish_unique_absorption_from_disposition_traffic(self) -> None:
        projector = IntegrationTelemetryProjector()
        previous = projector.project(
            (
                _event(1, "EVIDENCE_ADDED", payload={"evidence_id": "evidence-a"}),
                _event(2, "EVIDENCE_ADDED", payload={"evidence_id": "evidence-b"}),
                _event(3, "EVIDENCE_ADDED", payload={"evidence_id": "evidence-c"}),
            )
        )
        current = projector.project(
            (
                _event(1, "EVIDENCE_ADDED", payload={"evidence_id": "evidence-a"}),
                _event(2, "EVIDENCE_ADDED", payload={"evidence_id": "evidence-b"}),
                _event(3, "EVIDENCE_ADDED", payload={"evidence_id": "evidence-c"}),
                _event(
                    4,
                    "INTEGRATION_DISPOSITION_RECORDED",
                    reference_ids=("evidence-a",),
                    payload={"disposition": "INTEGRATED"},
                ),
                _event(
                    5,
                    "INTEGRATION_DISPOSITION_RECORDED",
                    reference_ids=("evidence-b",),
                    payload={"disposition": "DUPLICATE"},
                ),
                _event(
                    6,
                    "INTEGRATION_DISPOSITION_RECORDED",
                    reference_ids=("evidence-a",),
                    payload={"disposition": "INTEGRATED"},
                ),
                _event(
                    7,
                    "KNOWLEDGE_DELTA_RECORDED",
                    reference_ids=("delta-a", "evidence-a", "evidence-b"),
                    payload={
                        "delta_id": "delta-a",
                        "source_reference_ids": ["evidence-a", "evidence-b"],
                    },
                ),
                _event(8, "EVIDENCE_ADDED", payload={"evidence_id": "evidence-d"}),
            )
        )

        window = IntegrationBandwidthWindow.between(
            previous,
            current,
            elapsed_seconds=2.0,
        )

        self.assertEqual(window.evidence_generated, 1)
        self.assertEqual(window.evidence_dispositioned, 2)
        self.assertEqual(window.disposition_references_recorded, 3)
        self.assertEqual(window.knowledge_deltas_recorded, 1)
        self.assertEqual(window.backlog_delta, -1)
        self.assertEqual(window.evidence_per_second, 0.5)
        self.assertEqual(window.evidence_dispositioned_per_second, 1.0)
        self.assertEqual(window.disposition_references_per_second, 1.5)
        self.assertEqual(window.knowledge_deltas_per_second, 0.5)
        self.assertEqual(window.backlog_growth_per_second, -0.5)
        self.assertEqual(window.absorption_ratio, 2.0)

    def test_requires_positive_wall_clock_interval(self) -> None:
        snapshot = IntegrationTelemetryProjector().project(())
        with self.assertRaisesRegex(ValueError, "elapsed_seconds must be positive"):
            IntegrationBandwidthWindow.between(
                snapshot,
                snapshot,
                elapsed_seconds=0.0,
            )


if __name__ == "__main__":
    unittest.main()
