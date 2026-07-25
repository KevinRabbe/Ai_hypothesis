"""Tests for deterministic evidence integration backlog tracking."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from ai_hypothesis.runtime import (
    IntegrationBackpressureConfig,
    IntegrationDisposition,
    IntegrationTracker,
    KnowledgeDelta,
    SQLiteResearchLedger,
)


class IntegrationTrackerTests(unittest.TestCase):
    def test_backlog_tracks_only_generated_undispositioned_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with SQLiteResearchLedger(Path(directory) / "ledger.sqlite") as ledger:
                ledger.append_event(
                    event_type="EVIDENCE_ADDED",
                    thread_id="thread-1",
                    reference_ids=("evidence-1", "source-1"),
                    payload={"evidence_id": "evidence-1"},
                )
                ledger.append_event(
                    event_type="EVIDENCE_ADDED",
                    thread_id="thread-1",
                    reference_ids=("evidence-2", "source-2"),
                    payload={"evidence_id": "evidence-2"},
                )
                tracker = IntegrationTracker(ledger)

                initial = tracker.snapshot()
                self.assertEqual(initial.evidence_count, 2)
                self.assertEqual(initial.backlog_evidence_ids, ("evidence-1", "evidence-2"))

                tracker.record_disposition(
                    ("evidence-1",),
                    IntegrationDisposition.LOCAL_ONLY,
                    reason="Only relevant to this Work Thread",
                    thread_id="thread-1",
                )
                after = tracker.snapshot()
                self.assertEqual(after.dispositioned_evidence_count, 1)
                self.assertEqual(after.backlog_evidence_ids, ("evidence-2",))

    def test_knowledge_delta_retains_causal_references_without_implicitly_consuming_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with SQLiteResearchLedger(Path(directory) / "ledger.sqlite") as ledger:
                source_event = ledger.append_event(
                    event_type="EVIDENCE_ADDED",
                    thread_id="thread-1",
                    reference_ids=("evidence-1",),
                    payload={"evidence_id": "evidence-1"},
                )
                tracker = IntegrationTracker(ledger)
                delta_event = tracker.record_knowledge_delta(
                    KnowledgeDelta(
                        delta_id="delta-1",
                        kind="HYPOTHESIS_WEAKENED",
                        summary="H2 weakened by evidence-1",
                        reference_ids=("evidence-1",),
                        causal_event_ids=(source_event.event_id,),
                        thread_id="thread-1",
                    )
                )

                snapshot = tracker.snapshot()
                self.assertEqual(snapshot.knowledge_delta_count, 1)
                self.assertEqual(snapshot.backlog_evidence_ids, ("evidence-1",))
                self.assertIn("delta-1", delta_event.reference_ids)
                self.assertIn("evidence-1", delta_event.reference_ids)
                self.assertEqual(delta_event.parent_event_ids, (source_event.event_id,))

    def test_backpressure_uses_real_backlog_not_available_worker_count(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with SQLiteResearchLedger(Path(directory) / "ledger.sqlite") as ledger:
                tracker = IntegrationTracker(
                    ledger,
                    IntegrationBackpressureConfig(
                        max_backlog_count=0,
                        max_backlog_age_sequences=100,
                    ),
                )
                self.assertFalse(tracker.is_backpressured())

                ledger.append_event(
                    event_type="EVIDENCE_ADDED",
                    reference_ids=("evidence-1",),
                    payload={"evidence_id": "evidence-1"},
                )
                self.assertTrue(tracker.is_backpressured())

                tracker.record_disposition(
                    ("evidence-1",), IntegrationDisposition.INTEGRATED
                )
                self.assertFalse(tracker.is_backpressured())

    def test_oldest_backlog_age_grows_as_other_events_accumulate(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with SQLiteResearchLedger(Path(directory) / "ledger.sqlite") as ledger:
                ledger.append_event(
                    event_type="EVIDENCE_ADDED",
                    reference_ids=("evidence-1",),
                    payload={"evidence_id": "evidence-1"},
                )
                for index in range(4):
                    ledger.append_event(
                        event_type="THREAD_METADATA_UPDATED",
                        thread_id="thread-1",
                        payload={"index": index},
                    )
                tracker = IntegrationTracker(
                    ledger,
                    IntegrationBackpressureConfig(
                        max_backlog_count=100,
                        max_backlog_age_sequences=3,
                    ),
                )

                snapshot = tracker.snapshot()
                self.assertEqual(snapshot.oldest_backlog_age_sequences, 4)
                self.assertTrue(tracker.is_backpressured())


if __name__ == "__main__":
    unittest.main()
