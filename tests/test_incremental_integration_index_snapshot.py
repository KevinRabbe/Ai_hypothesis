from __future__ import annotations

import unittest

from ai_hypothesis.runtime import (
    ProjectedState,
    SchedulerAction,
    SchedulerDecision,
    WorkPreparation,
    WorkPurpose,
)
from ai_hypothesis.runtime.context_views import PurposeContextRouter
from ai_hypothesis.runtime.integration_index import SQLiteIndexedIntegrationTracker
from ai_hypothesis.runtime.ledger import SQLiteResearchLedger


def _add_evidence(ledger: SQLiteResearchLedger, evidence_id: str) -> None:
    ledger.append_event(
        event_type="EVIDENCE_ADDED",
        thread_id="thread-a",
        reference_ids=(evidence_id,),
        payload={
            "evidence_id": evidence_id,
            "kind": "OBSERVATION",
            "summary": evidence_id,
        },
    )


class IncrementalIntegrationIndexSnapshotTests(unittest.TestCase):
    def test_overview_advances_only_through_supplied_snapshot_sequence(self) -> None:
        ledger = SQLiteResearchLedger(":memory:")
        self.addCleanup(ledger.close)
        _add_evidence(ledger, "e1")
        _add_evidence(ledger, "e2")
        first_two = ledger.read_events(limit=2)
        _add_evidence(ledger, "e3")

        index = SQLiteIndexedIntegrationTracker(ledger)
        self.addCleanup(index.close)
        old_overview = index.overview(first_two)

        self.assertEqual(old_overview.global_snapshot.revision, 2)
        self.assertEqual(old_overview.global_snapshot.evidence_count, 2)
        self.assertEqual(index.revision, 2)

        current = index.overview(ledger.read_all_events())
        self.assertEqual(current.global_snapshot.revision, 3)
        self.assertEqual(current.global_snapshot.evidence_count, 3)
        self.assertEqual(index.revision, 3)

    def test_index_refuses_to_rewind_to_older_scheduler_snapshot(self) -> None:
        ledger = SQLiteResearchLedger(":memory:")
        self.addCleanup(ledger.close)
        _add_evidence(ledger, "e1")
        old = ledger.read_all_events()
        _add_evidence(ledger, "e2")

        index = SQLiteIndexedIntegrationTracker(ledger)
        self.addCleanup(index.close)
        index.sync()

        with self.assertRaisesRegex(ValueError, "ahead of the requested ledger snapshot"):
            index.overview(old)

    def test_indexed_tracker_drives_existing_backpressure_context_router(self) -> None:
        ledger = SQLiteResearchLedger(":memory:")
        self.addCleanup(ledger.close)
        _add_evidence(ledger, "e1")
        _add_evidence(ledger, "e2")
        _add_evidence(ledger, "e3")

        index = SQLiteIndexedIntegrationTracker(ledger)
        self.addCleanup(index.close)
        router = PurposeContextRouter(
            ledger=ledger,
            fallback=lambda state, decision: WorkPreparation(context={"fallback": True}),
            integration_tracker=index,  # type: ignore[arg-type]
            integration_limit=2,
        )
        state = ProjectedState(
            revision=ledger.latest_sequence(),
            thread_id="thread-a",
            objective="integrate",
            status="ACTIVE",
            purpose=WorkPurpose.PROGRESS,
        )
        decision = SchedulerDecision(
            decision_id="decision-a",
            thread_id="thread-a",
            action=SchedulerAction.SYNTHESIZE,
            purpose=WorkPurpose.SYNTHESIZE,
            reason_codes=("BACKPRESSURE",),
            projection_revision=state.revision,
        )

        preparation = router(state, decision)

        self.assertEqual(preparation.reference_ids, ("e1", "e2"))
        self.assertEqual(len(preparation.context["pending_evidence"]), 2)
        self.assertEqual(preparation.context["synthesis_mode"], "INTEGRATION_BACKPRESSURE")


if __name__ == "__main__":
    unittest.main()
