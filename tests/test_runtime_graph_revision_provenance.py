"""Tests that cross-thread graph changes advance affected projected revisions."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from ai_hypothesis.runtime import SQLiteResearchLedger, ThreadStateProjector


class GraphRevisionProvenanceTests(unittest.TestCase):
    def test_fork_event_advances_child_revision(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with SQLiteResearchLedger(Path(directory) / "ledger.sqlite") as ledger:
                ledger.append_event(
                    event_type="THREAD_CREATED",
                    thread_id="parent",
                    payload={"objective": "parent", "purpose": "EXPLORE"},
                )
                child_created = ledger.append_event(
                    event_type="THREAD_CREATED",
                    thread_id="child",
                    payload={"objective": "child", "purpose": "EXPLORE"},
                )
                fork = ledger.append_event(
                    event_type="THREAD_FORKED",
                    thread_id="parent",
                    reference_ids=("child",),
                    payload={"child_thread_id": "child"},
                )

                states = {
                    state.thread_id: state
                    for state in ThreadStateProjector().project_all(
                        ledger.read_all_events()
                    )
                }
                self.assertLess(child_created.sequence, fork.sequence)
                self.assertEqual(states["child"].revision, fork.sequence)
                self.assertEqual(states["child"].parent_thread_ids, ("parent",))
                self.assertEqual(states["parent"].revision, fork.sequence)

    def test_merge_event_advances_source_revision_and_makes_it_terminal(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with SQLiteResearchLedger(Path(directory) / "ledger.sqlite") as ledger:
                ledger.append_event(
                    event_type="THREAD_CREATED",
                    thread_id="source",
                    payload={"objective": "source", "purpose": "PROGRESS"},
                )
                ledger.append_event(
                    event_type="THREAD_CREATED",
                    thread_id="target",
                    payload={"objective": "target", "purpose": "SYNTHESIZE"},
                )
                merge = ledger.append_event(
                    event_type="THREAD_MERGED",
                    thread_id="target",
                    reference_ids=("source",),
                    payload={"source_thread_ids": ["source"]},
                )

                states = {
                    state.thread_id: state
                    for state in ThreadStateProjector().project_all(
                        ledger.read_all_events()
                    )
                }
                self.assertEqual(states["source"].revision, merge.sequence)
                self.assertEqual(states["source"].status, "COMPLETE")
                self.assertEqual(states["source"].merged_into_thread_id, "target")
                self.assertEqual(states["target"].revision, merge.sequence)

    def test_relation_before_target_creation_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with SQLiteResearchLedger(Path(directory) / "ledger.sqlite") as ledger:
                ledger.append_event(
                    event_type="THREAD_CREATED",
                    thread_id="parent",
                    payload={"objective": "parent", "purpose": "EXPLORE"},
                )
                ledger.append_event(
                    event_type="THREAD_FORKED",
                    thread_id="parent",
                    reference_ids=("future-child",),
                    payload={"child_thread_id": "future-child"},
                )
                ledger.append_event(
                    event_type="THREAD_CREATED",
                    thread_id="future-child",
                    payload={"objective": "future", "purpose": "EXPLORE"},
                )

                with self.assertRaisesRegex(ValueError, "before creation"):
                    ThreadStateProjector().project_all(ledger.read_all_events())


if __name__ == "__main__":
    unittest.main()
