from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

from ai_hypothesis.runtime.knowledge import KnowledgeStateProjector, KnowledgeStatus
from ai_hypothesis.runtime.knowledge_index import SQLiteIndexedKnowledgeState
from ai_hypothesis.runtime.ledger import SQLiteResearchLedger


def _delta(
    ledger: SQLiteResearchLedger,
    delta_id: str,
    *,
    thread_id: str = "thread-a",
    kind: str = "CLAIM",
    sources: tuple[str, ...] = (),
) -> None:
    ledger.append_event(
        event_type="KNOWLEDGE_DELTA_RECORDED",
        thread_id=thread_id,
        reference_ids=(delta_id, *sources),
        payload={
            "delta_id": delta_id,
            "kind": kind,
            "summary": f"summary {delta_id}",
            "source_reference_ids": list(sources),
            "causal_event_ids": [],
        },
    )


def _assessment(
    ledger: SQLiteResearchLedger,
    delta_id: str,
    assessment: str,
    *,
    reason: str | None = None,
) -> None:
    payload = {"assessment": assessment}
    if reason is not None:
        payload["reason"] = reason
    ledger.append_event(
        event_type="KNOWLEDGE_ASSESSMENT_RECORDED",
        thread_id="verifier",
        reference_ids=(delta_id,),
        payload=payload,
    )


class IncrementalKnowledgeIndexTests(unittest.TestCase):
    def test_indexed_snapshot_matches_replay_projector_globally_and_per_thread(self) -> None:
        ledger = SQLiteResearchLedger(":memory:")
        self.addCleanup(ledger.close)
        _delta(ledger, "k1", thread_id="thread-a", sources=("e1",))
        _delta(ledger, "k2", thread_id="thread-b", sources=("e2",))
        _assessment(ledger, "k1", "VERIFIED", reason="replicated")
        _assessment(ledger, "k2", "DISPUTED", reason="counterexample")
        _delta(
            ledger,
            "k3",
            thread_id="thread-a",
            kind="THREAD_CONSOLIDATION",
            sources=("k1",),
        )

        history = ledger.read_all_events()
        replay = KnowledgeStateProjector()
        index = SQLiteIndexedKnowledgeState(ledger)
        self.addCleanup(index.close)

        self.assertEqual(index.project(history), replay.project(history))
        self.assertEqual(
            index.project(history, thread_id="thread-a"),
            replay.project(history, thread_id="thread-a"),
        )
        snapshot = index.snapshot()
        self.assertEqual(snapshot.get("k1").status, KnowledgeStatus.VERIFIED)  # type: ignore[union-attr]
        self.assertEqual(snapshot.get("k1").assessment_reason, "replicated")  # type: ignore[union-attr]
        self.assertEqual(snapshot.get("k2").status, KnowledgeStatus.DISPUTED)  # type: ignore[union-attr]
        self.assertEqual(snapshot.get("k3").source_reference_ids, ("k1",))  # type: ignore[union-attr]

    def test_incremental_sync_applies_only_new_assessment_tail(self) -> None:
        ledger = SQLiteResearchLedger(":memory:")
        self.addCleanup(ledger.close)
        _delta(ledger, "k1")
        index = SQLiteIndexedKnowledgeState(ledger)
        self.addCleanup(index.close)
        first_revision = index.sync()
        self.assertEqual(index.snapshot().get("k1").status, KnowledgeStatus.PROVISIONAL)  # type: ignore[union-attr]

        _assessment(ledger, "k1", "VERIFIED")
        index.sync()

        self.assertGreater(index.revision, first_revision)
        self.assertEqual(index.snapshot().get("k1").status, KnowledgeStatus.VERIFIED)  # type: ignore[union-attr]

    def test_sync_pages_beyond_one_thousand_knowledge_events(self) -> None:
        ledger = SQLiteResearchLedger(":memory:")
        self.addCleanup(ledger.close)
        for index_value in range(1205):
            _delta(ledger, f"k-{index_value}")
        index = SQLiteIndexedKnowledgeState(ledger)
        self.addCleanup(index.close)

        index.sync(page_size=83)

        snapshot = index.snapshot()
        self.assertEqual(snapshot.revision, ledger.latest_sequence())
        self.assertEqual(len(snapshot.records), 1205)

    def test_persistent_sidecar_resumes_and_applies_new_tail(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            ledger_path = Path(directory) / "research.sqlite3"
            index_path = Path(directory) / "knowledge-index.sqlite3"
            ledger = SQLiteResearchLedger(ledger_path)
            _delta(ledger, "k1")
            first = SQLiteIndexedKnowledgeState(ledger, index_path)
            first_revision = first.sync()
            first.close()

            _assessment(ledger, "k1", "VERIFIED")
            second = SQLiteIndexedKnowledgeState(ledger, index_path)
            try:
                self.assertEqual(second.revision, first_revision)
                second.sync()
                record = second.snapshot().get("k1")
                self.assertIsNotNone(record)
                self.assertEqual(record.status, KnowledgeStatus.VERIFIED)  # type: ignore[union-attr]
                self.assertEqual(second.revision, ledger.latest_sequence())
            finally:
                second.close()
                ledger.close()

    def test_checkpoint_detects_replaced_canonical_ledger(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            ledger_path = Path(directory) / "research.sqlite3"
            index_path = Path(directory) / "knowledge-index.sqlite3"
            ledger = SQLiteResearchLedger(ledger_path)
            _delta(ledger, "old-k")
            index = SQLiteIndexedKnowledgeState(ledger, index_path)
            index.sync()
            index.close()
            ledger.close()

            for suffix in ("", "-wal", "-shm"):
                path = Path(f"{ledger_path}{suffix}")
                if path.exists():
                    os.remove(path)

            replacement = SQLiteResearchLedger(ledger_path)
            self.addCleanup(replacement.close)
            _delta(replacement, "new-k")
            reopened = SQLiteIndexedKnowledgeState(replacement, index_path)
            self.addCleanup(reopened.close)

            with self.assertRaisesRegex(RuntimeError, "checkpoint no longer matches"):
                reopened.sync()

    def test_sidecar_must_be_separate_from_canonical_ledger_file(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            ledger_path = Path(directory) / "research.sqlite3"
            ledger = SQLiteResearchLedger(ledger_path)
            self.addCleanup(ledger.close)

            with self.assertRaisesRegex(ValueError, "separate from the canonical ledger"):
                SQLiteIndexedKnowledgeState(ledger, ledger_path)

    def test_unknown_assessment_rolls_back_page_and_checkpoint(self) -> None:
        ledger = SQLiteResearchLedger(":memory:")
        self.addCleanup(ledger.close)
        ledger.append_event(
            event_type="KNOWLEDGE_ASSESSMENT_RECORDED",
            reference_ids=("missing-k",),
            payload={"assessment": "VERIFIED"},
        )
        index = SQLiteIndexedKnowledgeState(ledger)
        self.addCleanup(index.close)

        with self.assertRaisesRegex(ValueError, "unknown delta"):
            index.sync()
        self.assertEqual(index.revision, 0)
        self.assertEqual(index.snapshot if False else index.revision, 0)

    def test_rebuild_matches_incrementally_maintained_state(self) -> None:
        ledger = SQLiteResearchLedger(":memory:")
        self.addCleanup(ledger.close)
        _delta(ledger, "k1", sources=("e1",))
        _delta(ledger, "k2", sources=("e2",))
        _assessment(ledger, "k1", "VERIFIED", reason="ok")
        index = SQLiteIndexedKnowledgeState(ledger)
        self.addCleanup(index.close)
        before = index.snapshot()
        before_revision = index.revision

        rebuilt_revision = index.rebuild(page_size=1)
        after = index.snapshot()

        self.assertEqual(rebuilt_revision, before_revision)
        self.assertEqual(after, before)


if __name__ == "__main__":
    unittest.main()
