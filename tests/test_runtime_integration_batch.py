"""Tests bounded oldest-first evidence intake for synthesis work."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from ai_hypothesis.runtime import (
    IntegrationDisposition,
    IntegrationTracker,
    SQLiteResearchLedger,
)


class IntegrationBatchTests(unittest.TestCase):
    def test_pending_batch_is_oldest_first_and_hard_limited(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with SQLiteResearchLedger(Path(directory) / "ledger.sqlite") as ledger:
                for index in range(5):
                    evidence_id = f"evidence-{index}"
                    ledger.append_event(
                        event_type="EVIDENCE_ADDED",
                        thread_id="thread-1",
                        reference_ids=(evidence_id, f"source-{index}"),
                        payload={
                            "evidence_id": evidence_id,
                            "kind": "LOCAL_FINDING",
                            "summary": f"finding {index}",
                            "strength": float(index),
                            "uncertainty": 0.1 * index,
                            "data": {"index": index},
                        },
                    )
                tracker = IntegrationTracker(ledger)

                batch = tracker.pending_batch(limit=2, thread_id="thread-1")

                self.assertEqual(batch.evidence_ids, ("evidence-0", "evidence-1"))
                self.assertEqual(len(batch.records), 2)
                self.assertEqual(batch.records[0].source_reference_ids, ("source-0",))
                self.assertEqual(batch.records[1].data, {"index": 1})

    def test_dispositioned_evidence_is_skipped_without_expanding_batch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with SQLiteResearchLedger(Path(directory) / "ledger.sqlite") as ledger:
                for index in range(4):
                    evidence_id = f"evidence-{index}"
                    ledger.append_event(
                        event_type="EVIDENCE_ADDED",
                        thread_id="thread-1",
                        reference_ids=(evidence_id,),
                        payload={
                            "evidence_id": evidence_id,
                            "kind": "LOCAL_FINDING",
                            "summary": evidence_id,
                        },
                    )
                tracker = IntegrationTracker(ledger)
                tracker.record_disposition(
                    ("evidence-0", "evidence-2"),
                    IntegrationDisposition.DUPLICATE,
                )

                batch = tracker.pending_batch(limit=2, thread_id="thread-1")

                self.assertEqual(batch.evidence_ids, ("evidence-1", "evidence-3"))
                self.assertEqual(len(batch.records), 2)

    def test_thread_filter_prevents_cross_thread_context_growth(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with SQLiteResearchLedger(Path(directory) / "ledger.sqlite") as ledger:
                for thread_id, evidence_id in (
                    ("thread-a", "evidence-a"),
                    ("thread-b", "evidence-b"),
                ):
                    ledger.append_event(
                        event_type="EVIDENCE_ADDED",
                        thread_id=thread_id,
                        reference_ids=(evidence_id,),
                        payload={
                            "evidence_id": evidence_id,
                            "kind": "LOCAL_FINDING",
                            "summary": evidence_id,
                        },
                    )
                tracker = IntegrationTracker(ledger)

                batch = tracker.pending_batch(limit=10, thread_id="thread-b")

                self.assertEqual(batch.evidence_ids, ("evidence-b",))

    def test_limit_must_be_positive(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with SQLiteResearchLedger(Path(directory) / "ledger.sqlite") as ledger:
                tracker = IntegrationTracker(ledger)
                with self.assertRaises(ValueError):
                    tracker.pending_batch(limit=0)


if __name__ == "__main__":
    unittest.main()
