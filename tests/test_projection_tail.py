from __future__ import annotations

import unittest

from ai_hypothesis.runtime.ledger import SQLiteResearchLedger
from ai_hypothesis.runtime.projection_tail import LedgerProjectionTail, ProjectionCheckpoint


def _append(ledger: SQLiteResearchLedger, value: int) -> None:
    ledger.append_event(
        event_type="TEST_EVENT",
        payload={"value": value},
    )


class ProjectionTailTests(unittest.TestCase):
    def test_streams_only_events_after_checkpoint_in_bounded_pages(self) -> None:
        ledger = SQLiteResearchLedger(":memory:")
        self.addCleanup(ledger.close)
        for value in range(7):
            _append(ledger, value)
        history = ledger.read_all_events()
        checkpoint = ProjectionCheckpoint(
            sequence=history[1].sequence,
            event_id=history[1].event_id,
        )
        tail = LedgerProjectionTail(ledger)

        pages = tuple(tail.iter_pages(checkpoint, page_size=2))

        self.assertEqual(tuple(len(page) for page in pages), (2, 2, 1))
        self.assertEqual(
            tuple(event.sequence for page in pages for event in page),
            tuple(event.sequence for event in history[2:]),
        )

    def test_exact_target_stops_before_later_canonical_events(self) -> None:
        ledger = SQLiteResearchLedger(":memory:")
        self.addCleanup(ledger.close)
        for value in range(5):
            _append(ledger, value)
        history = ledger.read_all_events()
        tail = LedgerProjectionTail(ledger)

        pages = tuple(
            tail.iter_pages(
                ProjectionCheckpoint(),
                target_sequence=history[2].sequence,
                page_size=10,
            )
        )

        flattened = tuple(event for page in pages for event in page)
        self.assertEqual(flattened, history[:3])

    def test_checkpoint_event_identity_is_validated(self) -> None:
        ledger = SQLiteResearchLedger(":memory:")
        self.addCleanup(ledger.close)
        _append(ledger, 1)
        event = ledger.read_all_events()[0]
        tail = LedgerProjectionTail(ledger)

        with self.assertRaisesRegex(RuntimeError, "checkpoint no longer matches"):
            tuple(
                tail.iter_pages(
                    ProjectionCheckpoint(
                        sequence=event.sequence,
                        event_id="wrong-event-id",
                    )
                )
            )

    def test_target_cannot_precede_checkpoint(self) -> None:
        ledger = SQLiteResearchLedger(":memory:")
        self.addCleanup(ledger.close)
        _append(ledger, 1)
        _append(ledger, 2)
        event = ledger.read_all_events()[-1]
        tail = LedgerProjectionTail(ledger)

        with self.assertRaisesRegex(ValueError, "ahead of requested ledger snapshot"):
            tuple(
                tail.iter_pages(
                    ProjectionCheckpoint(event.sequence, event.event_id),
                    target_sequence=event.sequence - 1,
                )
            )

    def test_requested_target_must_exist(self) -> None:
        ledger = SQLiteResearchLedger(":memory:")
        self.addCleanup(ledger.close)
        _append(ledger, 1)
        tail = LedgerProjectionTail(ledger)

        with self.assertRaisesRegex(RuntimeError, "not available"):
            tuple(
                tail.iter_pages(
                    ProjectionCheckpoint(),
                    target_sequence=99,
                )
            )

    def test_checkpoint_after_page_uses_last_event_identity(self) -> None:
        ledger = SQLiteResearchLedger(":memory:")
        self.addCleanup(ledger.close)
        _append(ledger, 1)
        _append(ledger, 2)
        history = ledger.read_all_events()

        checkpoint = LedgerProjectionTail.checkpoint_after(history)

        self.assertEqual(checkpoint.sequence, history[-1].sequence)
        self.assertEqual(checkpoint.event_id, history[-1].event_id)


if __name__ == "__main__":
    unittest.main()
