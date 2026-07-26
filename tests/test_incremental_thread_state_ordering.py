from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from ai_hypothesis.runtime.ledger import SQLiteResearchLedger
from ai_hypothesis.runtime.projector import ThreadStateProjector
from ai_hypothesis.runtime.thread_state_index import SQLiteIndexedThreadState


class IncrementalThreadStateOrderingTests(unittest.TestCase):
    def test_multi_target_event_preserves_reference_order(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            root = Path(tempdir)
            ledger = SQLiteResearchLedger(root / "ledger.sqlite3")
            try:
                for thread_id in ("p", "c1", "c2", "d1", "d2", "target", "s1", "s2"):
                    ledger.append_event(
                        event_type="THREAD_CREATED",
                        thread_id=thread_id,
                        payload={"objective": thread_id, "purpose": "EXPLORE"},
                    )

                ledger.append_event(
                    event_type="THREAD_FORKED",
                    thread_id="p",
                    reference_ids=("c2", "c1"),
                )
                ledger.append_event(
                    event_type="DEPENDENCY_ADDED",
                    thread_id="p",
                    reference_ids=("d2", "d1"),
                )
                ledger.append_event(
                    event_type="THREAD_MERGED",
                    thread_id="target",
                    reference_ids=("s2", "s1"),
                )

                history = ledger.read_all_events()
                expected = ThreadStateProjector().project_all(history)
                with SQLiteIndexedThreadState(ledger, root / "index.sqlite3") as indexed:
                    actual = indexed.project_all(history)
                    self.assertEqual(actual, expected)
                    p = indexed.snapshot(thread_id="p")
                    target = indexed.snapshot(thread_id="target")
                    self.assertEqual(p.child_thread_ids, ("c2", "c1"))
                    self.assertEqual(p.dependency_thread_ids, ("d2", "d1"))
                    self.assertEqual(target.merged_from_thread_ids, ("s2", "s1"))
            finally:
                ledger.close()


if __name__ == "__main__":
    unittest.main()
