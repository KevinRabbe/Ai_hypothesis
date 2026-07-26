from __future__ import annotations

import unittest

from ai_hypothesis.runtime.knowledge import KnowledgeStateProjector, KnowledgeStatus
from ai_hypothesis.runtime.knowledge_index import SQLiteIndexedKnowledgeState
from ai_hypothesis.runtime.knowledge_verification import KnowledgeVerificationTracker
from ai_hypothesis.runtime.ledger import SQLiteResearchLedger


def _delta(ledger: SQLiteResearchLedger, delta_id: str, *, thread_id: str = "thread-a") -> None:
    ledger.append_event(
        event_type="KNOWLEDGE_DELTA_RECORDED",
        thread_id=thread_id,
        reference_ids=(delta_id,),
        payload={
            "delta_id": delta_id,
            "kind": "CLAIM",
            "summary": delta_id,
            "source_reference_ids": [],
            "causal_event_ids": [],
        },
    )


def _assessment(ledger: SQLiteResearchLedger, delta_id: str, assessment: str) -> None:
    ledger.append_event(
        event_type="KNOWLEDGE_ASSESSMENT_RECORDED",
        thread_id="verifier",
        reference_ids=(delta_id,),
        payload={"assessment": assessment},
    )


class IncrementalKnowledgeIndexIntegrationTests(unittest.TestCase):
    def test_project_stops_at_supplied_snapshot_boundary(self) -> None:
        ledger = SQLiteResearchLedger(":memory:")
        self.addCleanup(ledger.close)
        _delta(ledger, "k1")
        old_history = ledger.read_all_events()
        _assessment(ledger, "k1", "VERIFIED")

        index = SQLiteIndexedKnowledgeState(ledger)
        self.addCleanup(index.close)
        old = index.project(old_history)
        self.assertEqual(old.revision, old_history[-1].sequence)
        self.assertEqual(old.get("k1").status, KnowledgeStatus.PROVISIONAL)  # type: ignore[union-attr]

        current_history = ledger.read_all_events()
        current = index.project(current_history)
        self.assertEqual(current.revision, current_history[-1].sequence)
        self.assertEqual(current.get("k1").status, KnowledgeStatus.VERIFIED)  # type: ignore[union-attr]

    def test_index_refuses_to_rewind_after_advancing_farther(self) -> None:
        ledger = SQLiteResearchLedger(":memory:")
        self.addCleanup(ledger.close)
        _delta(ledger, "k1")
        old_history = ledger.read_all_events()
        _assessment(ledger, "k1", "VERIFIED")

        index = SQLiteIndexedKnowledgeState(ledger)
        self.addCleanup(index.close)
        index.sync()

        with self.assertRaisesRegex(ValueError, "ahead of requested ledger snapshot"):
            index.project(old_history)

    def test_verification_tracker_accepts_indexed_knowledge_projector(self) -> None:
        ledger = SQLiteResearchLedger(":memory:")
        self.addCleanup(ledger.close)
        _delta(ledger, "k1", thread_id="thread-a")
        _delta(ledger, "k2", thread_id="thread-a")
        _assessment(ledger, "k1", "VERIFIED")

        index = SQLiteIndexedKnowledgeState(ledger)
        self.addCleanup(index.close)
        tracker = KnowledgeVerificationTracker(
            ledger,
            projector=index,  # type: ignore[arg-type]
        )
        overview = tracker.overview(ledger.read_all_events())

        self.assertEqual(overview.unresolved_count, 1)
        self.assertEqual(overview.pending_delta_ids("thread-a", limit=8), ("k2",))
        self.assertGreater(overview.pressure_for("thread-a"), 0.0)

    def test_index_and_replay_projections_remain_equal_across_assessment_changes(self) -> None:
        ledger = SQLiteResearchLedger(":memory:")
        self.addCleanup(ledger.close)
        _delta(ledger, "k1")
        index = SQLiteIndexedKnowledgeState(ledger)
        self.addCleanup(index.close)
        replay = KnowledgeStateProjector()

        for assessment in ("DISPUTED", "VERIFIED", "RETRACTED"):
            _assessment(ledger, "k1", assessment)
            history = ledger.read_all_events()
            self.assertEqual(index.project(history), replay.project(history))


if __name__ == "__main__":
    unittest.main()
