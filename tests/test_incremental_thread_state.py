from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from ai_hypothesis.runtime.ledger import SQLiteResearchLedger
from ai_hypothesis.runtime.projector import ThreadStateProjector
from ai_hypothesis.runtime.thread_state_index import SQLiteIndexedThreadState


class IncrementalThreadStateTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        root = Path(self.tempdir.name)
        self.ledger_path = root / "ledger.sqlite3"
        self.index_path = root / "thread-index.sqlite3"
        self.ledger = SQLiteResearchLedger(self.ledger_path)

    def tearDown(self) -> None:
        self.ledger.close()
        self.tempdir.cleanup()

    def _append(
        self,
        event_type: str,
        *,
        thread_id: str | None = None,
        reference_ids: tuple[str, ...] = (),
        payload: dict | None = None,
        attempt_id: str | None = None,
    ):
        return self.ledger.append_event(
            event_type=event_type,
            thread_id=thread_id,
            attempt_id=attempt_id,
            reference_ids=reference_ids,
            payload=payload or {},
        )

    def _create(self, thread_id: str, objective: str) -> None:
        self._append(
            "THREAD_CREATED",
            thread_id=thread_id,
            payload={"objective": objective, "purpose": "EXPLORE", "status": "ACTIVE"},
        )

    def test_matches_replay_for_thread_and_graph_state(self) -> None:
        self._create("root", "Investigate root")
        self._create("child", "Investigate child")
        self._append("THREAD_FORKED", thread_id="root", reference_ids=("child",))
        self._create("dep", "Dependency")
        self._append("DEPENDENCY_ADDED", thread_id="child", reference_ids=("dep",))
        self._append("HYPOTHESIS_PROPOSED", thread_id="child", reference_ids=("H1",))
        self._append("CONTRADICTION_FOUND", thread_id="child", reference_ids=("E9",))
        self._append("OPEN_QUESTION_ADDED", thread_id="child", payload={"question": "Q1"})
        self._append("THREAD_METADATA_UPDATED", thread_id="child", payload={"region": "north"})
        self._append(
            "ATTEMPT_STARTED",
            thread_id="child",
            attempt_id="attempt-1",
            payload={"worker_id": "worker-7"},
        )
        self._create("target", "Merge target")
        self._append("THREAD_MERGED", thread_id="target", reference_ids=("child",))

        history = self.ledger.read_all_events()
        expected = ThreadStateProjector().project_all(history)
        with SQLiteIndexedThreadState(self.ledger, self.index_path) as indexed:
            actual = indexed.project_all(history)
            self.assertEqual(actual, expected)
            self.assertEqual(indexed.last_worker_id("child"), "worker-7")
            child = indexed.snapshot(thread_id="child")
            self.assertEqual(child.merged_into_thread_id, "target")
            self.assertEqual(child.status, "COMPLETE")
            self.assertEqual(child.metadata["region"], "north")

    def test_duplicate_relations_keep_first_insertion_order(self) -> None:
        for thread_id in ("p", "c1", "c2", "d1", "d2", "target", "s1", "s2"):
            self._create(thread_id, thread_id)

        self._append("THREAD_FORKED", thread_id="p", reference_ids=("c1",))
        self._append("THREAD_FORKED", thread_id="p", reference_ids=("c2",))
        self._append("THREAD_FORKED", thread_id="p", reference_ids=("c1",))

        self._append("DEPENDENCY_ADDED", thread_id="p", reference_ids=("d1",))
        self._append("DEPENDENCY_ADDED", thread_id="p", reference_ids=("d2",))
        self._append("DEPENDENCY_ADDED", thread_id="p", reference_ids=("d1",))

        self._append("THREAD_MERGED", thread_id="target", reference_ids=("s1",))
        self._append("THREAD_MERGED", thread_id="target", reference_ids=("s2",))
        self._append("THREAD_MERGED", thread_id="target", reference_ids=("s1",))

        history = self.ledger.read_all_events()
        expected = ThreadStateProjector().project_all(history)
        with SQLiteIndexedThreadState(self.ledger, self.index_path) as indexed:
            self.assertEqual(indexed.project_all(history), expected)
            parent = indexed.snapshot(thread_id="p")
            target = indexed.snapshot(thread_id="target")
            self.assertEqual(parent.child_thread_ids, ("c1", "c2"))
            self.assertEqual(parent.dependency_thread_ids, ("d1", "d2"))
            self.assertEqual(target.merged_from_thread_ids, ("s1", "s2"))

    def test_dependency_remove_and_readd_moves_to_new_position(self) -> None:
        for thread_id in ("p", "d1", "d2"):
            self._create(thread_id, thread_id)
        self._append("DEPENDENCY_ADDED", thread_id="p", reference_ids=("d1",))
        self._append("DEPENDENCY_ADDED", thread_id="p", reference_ids=("d2",))
        self._append("DEPENDENCY_REMOVED", thread_id="p", reference_ids=("d1",))
        self._append("DEPENDENCY_ADDED", thread_id="p", reference_ids=("d1",))

        history = self.ledger.read_all_events()
        with SQLiteIndexedThreadState(self.ledger, self.index_path) as indexed:
            self.assertEqual(
                indexed.project_all(history),
                ThreadStateProjector().project_all(history),
            )
            self.assertEqual(
                indexed.snapshot(thread_id="p").dependency_thread_ids,
                ("d2", "d1"),
            )

    def test_exact_snapshot_rejects_cycle_but_later_resolution_can_sync(self) -> None:
        self._create("a", "A")
        self._create("b", "B")
        self._append("DEPENDENCY_ADDED", thread_id="a", reference_ids=("b",))
        cycle = self._append("DEPENDENCY_ADDED", thread_id="b", reference_ids=("a",))
        self._append("DEPENDENCY_REMOVED", thread_id="b", reference_ids=("a",))

        with SQLiteIndexedThreadState(self.ledger, self.index_path) as indexed:
            with self.assertRaisesRegex(ValueError, "dependency cycle"):
                indexed.sync_through(cycle.sequence)
            self.assertEqual(indexed.revision, 0)
            indexed.sync()
            self.assertEqual(
                indexed.snapshot_all(),
                ThreadStateProjector().project_all(self.ledger.read_all_events()),
            )

    def test_fork_before_child_creation_is_rejected(self) -> None:
        self._create("parent", "Parent")
        self._append("THREAD_FORKED", thread_id="parent", reference_ids=("child",))
        self._create("child", "Child")

        with SQLiteIndexedThreadState(self.ledger, self.index_path) as indexed:
            with self.assertRaisesRegex(ValueError, "before creation"):
                indexed.sync()
            self.assertEqual(indexed.revision, 0)

    def test_historical_snapshot_does_not_leak_future_events(self) -> None:
        self._create("t", "T")
        first = self._append("HYPOTHESIS_PROPOSED", thread_id="t", reference_ids=("H1",))
        old_history = self.ledger.read_all_events()
        self._append("HYPOTHESIS_PROPOSED", thread_id="t", reference_ids=("H2",))

        with SQLiteIndexedThreadState(self.ledger, self.index_path) as indexed:
            old = indexed.project_all(old_history)
            self.assertEqual(old[0].hypothesis_ids, ("H1",))
            self.assertEqual(indexed.revision, first.sequence)
            current = indexed.snapshot_all()
            self.assertEqual(current[0].hypothesis_ids, ("H1", "H2"))
            with self.assertRaisesRegex(ValueError, "ahead of requested ledger snapshot"):
                indexed.project_all(old_history)

    def test_restart_resumes_from_checkpoint_and_preserves_last_worker(self) -> None:
        self._create("t", "T")
        self._append(
            "ATTEMPT_STARTED",
            thread_id="t",
            attempt_id="a1",
            payload={"worker_id": "worker-a"},
        )
        with SQLiteIndexedThreadState(self.ledger, self.index_path) as first:
            first.sync()
            checkpoint = first.revision
            self.assertEqual(first.last_worker_id("t"), "worker-a")

        self._append("THREAD_PAUSED", thread_id="t")
        with SQLiteIndexedThreadState(self.ledger, self.index_path) as second:
            self.assertEqual(second.revision, checkpoint)
            second.sync()
            self.assertEqual(second.last_worker_id("t"), "worker-a")
            self.assertEqual(second.snapshot(thread_id="t").status, "PAUSED")

    def test_tail_paginates_beyond_one_thousand_events(self) -> None:
        self._create("t", "T")
        for index in range(1205):
            self._append(
                "THREAD_METADATA_UPDATED",
                thread_id="t",
                payload={"counter": index},
            )
        history = self.ledger.read_all_events(page_size=97)
        expected = ThreadStateProjector().project_all(history)
        with SQLiteIndexedThreadState(self.ledger, self.index_path) as indexed:
            indexed.sync(page_size=113)
            self.assertEqual(indexed.snapshot_all(), expected)
            self.assertEqual(indexed.snapshot(thread_id="t").metadata["counter"], 1204)

    def test_rebuild_is_lossless(self) -> None:
        self._create("t", "T")
        self._append("HYPOTHESIS_PROPOSED", thread_id="t", reference_ids=("H1",))
        with SQLiteIndexedThreadState(self.ledger, self.index_path) as indexed:
            before = indexed.snapshot_all()
            indexed.rebuild(page_size=1)
            self.assertEqual(indexed.snapshot_all(), before)

    def test_sidecar_must_not_reuse_canonical_ledger_file(self) -> None:
        with self.assertRaisesRegex(ValueError, "separate from the canonical ledger"):
            SQLiteIndexedThreadState(self.ledger, self.ledger_path)


if __name__ == "__main__":
    unittest.main()
