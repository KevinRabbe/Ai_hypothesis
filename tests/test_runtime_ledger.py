"""Tests for the minimal durable Research Ledger."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from ai_hypothesis.runtime import SQLiteResearchLedger


class ResearchLedgerTests(unittest.TestCase):
    def test_append_assigns_monotonic_sequence_and_replays(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "ledger.sqlite"
            with SQLiteResearchLedger(path) as ledger:
                first = ledger.append_event(
                    event_type="THREAD_CREATED",
                    thread_id="thread-1",
                    payload={"objective": "Investigate H1"},
                )
                second = ledger.append_event(
                    event_type="EVIDENCE_ADDED",
                    thread_id="thread-1",
                    attempt_id="attempt-1",
                    reference_ids=("source-1",),
                    parent_event_ids=(first.event_id,),
                    payload={"claim": "observation-1"},
                )

                self.assertEqual(first.sequence, 1)
                self.assertEqual(second.sequence, 2)
                self.assertEqual(ledger.latest_sequence(), 2)
                self.assertEqual(ledger.read_events(), (first, second))

    def test_reopen_preserves_events(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "ledger.sqlite"
            with SQLiteResearchLedger(path) as ledger:
                event = ledger.append_event(
                    event_type="HYPOTHESIS_REJECTED",
                    thread_id="thread-2",
                    reference_ids=("hypothesis-2", "evidence-9"),
                    payload={"reason": "contradiction"},
                )

            with SQLiteResearchLedger(path) as reopened:
                loaded = reopened.get_event(event.event_id)
                self.assertEqual(loaded, event)
                self.assertEqual(reopened.latest_sequence(), event.sequence)

    def test_read_events_supports_checkpoint_and_thread_filter(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "ledger.sqlite"
            with SQLiteResearchLedger(path) as ledger:
                first = ledger.append_event(event_type="A", thread_id="thread-a")
                ledger.append_event(event_type="B", thread_id="thread-b")
                third = ledger.append_event(event_type="C", thread_id="thread-a")

                self.assertEqual(
                    ledger.read_events(after_sequence=first.sequence, thread_id="thread-a"),
                    (third,),
                )

    def test_append_rejects_unserializable_or_nonfinite_payload(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "ledger.sqlite"
            with SQLiteResearchLedger(path) as ledger:
                with self.assertRaises(ValueError):
                    ledger.append_event(event_type="BAD", payload={"value": object()})
                with self.assertRaises(ValueError):
                    ledger.append_event(event_type="BAD", payload={"value": float("nan")})
                self.assertEqual(ledger.latest_sequence(), 0)

    def test_read_validation_rejects_invalid_ranges(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "ledger.sqlite"
            with SQLiteResearchLedger(path) as ledger:
                with self.assertRaises(ValueError):
                    ledger.read_events(after_sequence=-1)
                with self.assertRaises(ValueError):
                    ledger.read_events(limit=0)


if __name__ == "__main__":
    unittest.main()
