from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from ai_hypothesis.runtime.indexed_control import (
    IndexedRuntimeIntegrationTracker,
    IndexedRuntimeSnapshotProvider,
    IndexedThreadRuntimeState,
)
from ai_hypothesis.runtime.knowledge_index import SQLiteIndexedKnowledgeState
from ai_hypothesis.runtime.knowledge_verification import KnowledgeVerificationTracker
from ai_hypothesis.runtime.ledger import SQLiteResearchLedger


class AppendAfterRevisionLedger(SQLiteResearchLedger):
    def __init__(self, path) -> None:
        super().__init__(path)
        self._append_thread_id: str | None = None

    def arm_future_metadata(self, thread_id: str) -> None:
        self._append_thread_id = thread_id

    def read_all_events(self, *args, **kwargs):
        raise AssertionError("indexed snapshot provider must not call read_all_events")

    def latest_sequence(self) -> int:
        revision = super().latest_sequence()
        if self._append_thread_id is not None:
            thread_id = self._append_thread_id
            self._append_thread_id = None
            super().append_event(
                event_type="THREAD_METADATA_UPDATED",
                thread_id=thread_id,
                payload={"future_event": True},
            )
        return revision


class IndexedRuntimeSnapshotIsolationTests(unittest.TestCase):
    def test_future_append_after_revision_capture_does_not_leak(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            root = Path(tempdir)
            ledger = AppendAfterRevisionLedger(root / "ledger.sqlite3")
            thread_state = IndexedThreadRuntimeState(
                ledger,
                root / "threads.sqlite3",
            )
            integration = IndexedRuntimeIntegrationTracker(
                ledger,
                root / "integration.sqlite3",
            )
            knowledge = SQLiteIndexedKnowledgeState(
                ledger,
                root / "knowledge.sqlite3",
            )
            verification = KnowledgeVerificationTracker(
                ledger,
                projector=knowledge,
            )
            try:
                ledger.append_event(
                    event_type="THREAD_CREATED",
                    thread_id="thread-a",
                    payload={
                        "objective": "snapshot isolation",
                        "purpose": "EXPLORE",
                    },
                )
                frozen_revision = super(AppendAfterRevisionLedger, ledger).latest_sequence()
                ledger.arm_future_metadata("thread-a")

                provider = IndexedRuntimeSnapshotProvider(
                    ledger=ledger,
                    thread_state=thread_state,
                    integration_tracker=integration,
                    verification_tracker=verification,
                )
                snapshot = provider.capture()

                self.assertEqual(snapshot.revision, frozen_revision)
                self.assertEqual(
                    snapshot.integration_overview.global_snapshot.revision,
                    frozen_revision,
                )
                self.assertEqual(snapshot.verification_overview.revision, frozen_revision)
                self.assertEqual(snapshot.states[0].revision, frozen_revision)
                self.assertNotIn("future_event", snapshot.states[0].metadata)
                self.assertEqual(ledger.latest_sequence(), frozen_revision + 1)

                # A later capture may now consume the event that was intentionally excluded.
                current = provider.capture()
                self.assertEqual(current.revision, frozen_revision + 1)
                self.assertTrue(current.states[0].metadata["future_event"])
            finally:
                knowledge.close()
                integration.close()
                thread_state.close()
                ledger.close()


if __name__ == "__main__":
    unittest.main()
