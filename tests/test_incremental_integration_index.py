from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

from ai_hypothesis.runtime import (
    EvidenceDispositionKind,
    IntegrationTracker,
    SQLiteResearchLedger,
)
from ai_hypothesis.runtime.integration_index import SQLiteIndexedIntegrationTracker


def _evidence(
    ledger: SQLiteResearchLedger,
    evidence_id: str,
    *,
    thread_id: str | None = "thread-a",
    summary: str | None = None,
) -> None:
    ledger.append_event(
        event_type="EVIDENCE_ADDED",
        thread_id=thread_id,
        reference_ids=(evidence_id, f"source-{evidence_id}"),
        payload={
            "evidence_id": evidence_id,
            "kind": "OBSERVATION",
            "summary": summary or evidence_id,
            "strength": 0.75,
            "uncertainty": 0.25,
            "data": {"value": evidence_id},
        },
    )


class _RecordingLedger:
    def __init__(self, delegate: SQLiteResearchLedger) -> None:
        self.delegate = delegate
        self.path = delegate.path
        self.schema_version = delegate.schema_version
        self.read_after_sequences: list[int] = []

    def read_events(self, *, after_sequence=0, limit=1000, thread_id=None):
        self.read_after_sequences.append(after_sequence)
        return self.delegate.read_events(
            after_sequence=after_sequence,
            limit=limit,
            thread_id=thread_id,
        )

    def get_event(self, event_id):
        return self.delegate.get_event(event_id)

    def append_event(self, **kwargs):
        return self.delegate.append_event(**kwargs)

    def latest_sequence(self):
        return self.delegate.latest_sequence()


class IncrementalIntegrationIndexTests(unittest.TestCase):
    def test_indexed_counts_match_replay_tracker_and_pending_payloads(self) -> None:
        ledger = SQLiteResearchLedger(":memory:")
        self.addCleanup(ledger.close)
        _evidence(ledger, "e1", thread_id="thread-a")
        _evidence(ledger, "e2", thread_id="thread-a")
        _evidence(ledger, "e3", thread_id="thread-b")
        ledger.append_event(
            event_type="INTEGRATION_DISPOSITION_RECORDED",
            thread_id="thread-a",
            reference_ids=("e1",),
            payload={"disposition": "INTEGRATED"},
        )
        ledger.append_event(
            event_type="KNOWLEDGE_DELTA_RECORDED",
            thread_id="thread-a",
            reference_ids=("k1", "e1"),
            payload={"delta_id": "k1", "kind": "CLAIM", "summary": "k1"},
        )

        replay = IntegrationTracker(ledger)
        indexed = SQLiteIndexedIntegrationTracker(ledger)
        self.addCleanup(indexed.close)

        replay_global = replay.snapshot()
        indexed_global = indexed.snapshot()
        self.assertEqual(indexed_global.revision, replay_global.revision)
        self.assertEqual(indexed_global.evidence_count, replay_global.evidence_count)
        self.assertEqual(
            indexed_global.dispositioned_evidence_count,
            replay_global.dispositioned_evidence_count,
        )
        self.assertEqual(indexed_global.backlog_count, replay_global.backlog_count)
        self.assertEqual(
            indexed_global.knowledge_delta_count,
            replay_global.knowledge_delta_count,
        )
        self.assertEqual(
            indexed_global.oldest_backlog_age_sequences,
            replay_global.oldest_backlog_age_sequences,
        )

        replay_a = replay.snapshot(thread_id="thread-a")
        indexed_a = indexed.snapshot(thread_id="thread-a")
        self.assertEqual(indexed_a.evidence_count, replay_a.evidence_count)
        self.assertEqual(indexed_a.backlog_count, replay_a.backlog_count)
        self.assertEqual(indexed_a.knowledge_delta_count, replay_a.knowledge_delta_count)

        replay_batch = replay.pending_batch(limit=8, thread_id="thread-a")
        indexed_batch = indexed.pending_batch(limit=8, thread_id="thread-a")
        self.assertEqual(indexed_batch.evidence_ids, replay_batch.evidence_ids)
        self.assertEqual(indexed_batch.causal_event_ids, replay_batch.causal_event_ids)
        self.assertEqual(
            indexed_batch.to_context_records(),
            replay_batch.to_context_records(),
        )

    def test_second_sync_starts_at_previous_checkpoint_not_sequence_zero(self) -> None:
        base = SQLiteResearchLedger(":memory:")
        self.addCleanup(base.close)
        recording = _RecordingLedger(base)
        _evidence(base, "e1")
        _evidence(base, "e2")
        index = SQLiteIndexedIntegrationTracker(recording)  # type: ignore[arg-type]
        self.addCleanup(index.close)

        first_revision = index.sync(page_size=1)
        self.assertEqual(first_revision, base.latest_sequence())
        recording.read_after_sequences.clear()

        _evidence(base, "e3")
        index.sync(page_size=1)

        self.assertTrue(recording.read_after_sequences)
        self.assertEqual(recording.read_after_sequences[0], first_revision)
        self.assertNotIn(0, recording.read_after_sequences)

    def test_sync_pages_beyond_one_thousand_events_without_truncation(self) -> None:
        ledger = SQLiteResearchLedger(":memory:")
        self.addCleanup(ledger.close)
        for index in range(1205):
            _evidence(ledger, f"e-{index}")
        tracker = SQLiteIndexedIntegrationTracker(ledger)
        self.addCleanup(tracker.close)

        tracker.sync(page_size=97)
        snapshot = tracker.snapshot()

        self.assertEqual(snapshot.revision, ledger.latest_sequence())
        self.assertEqual(snapshot.evidence_count, 1205)
        self.assertEqual(snapshot.backlog_count, 1205)

    def test_persistent_index_resumes_from_durable_checkpoint_after_restart(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            ledger_path = Path(directory) / "research.sqlite3"
            index_path = Path(directory) / "integration-index.sqlite3"
            ledger = SQLiteResearchLedger(ledger_path)
            _evidence(ledger, "e1")
            first = SQLiteIndexedIntegrationTracker(ledger, index_path)
            first_revision = first.sync()
            first.close()

            _evidence(ledger, "e2")
            second = SQLiteIndexedIntegrationTracker(ledger, index_path)
            try:
                self.assertEqual(second.revision, first_revision)
                second.sync()
                snapshot = second.snapshot()
                self.assertEqual(snapshot.evidence_count, 2)
                self.assertEqual(snapshot.revision, ledger.latest_sequence())
            finally:
                second.close()
                ledger.close()

    def test_checkpoint_detects_replaced_canonical_ledger(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            ledger_path = Path(directory) / "research.sqlite3"
            index_path = Path(directory) / "integration-index.sqlite3"
            ledger = SQLiteResearchLedger(ledger_path)
            _evidence(ledger, "old-evidence")
            index = SQLiteIndexedIntegrationTracker(ledger, index_path)
            index.sync()
            index.close()
            ledger.close()

            for suffix in ("", "-wal", "-shm"):
                path = Path(f"{ledger_path}{suffix}")
                if path.exists():
                    os.remove(path)

            replacement = SQLiteResearchLedger(ledger_path)
            self.addCleanup(replacement.close)
            _evidence(replacement, "replacement-evidence")
            reopened = SQLiteIndexedIntegrationTracker(replacement, index_path)
            self.addCleanup(reopened.close)

            with self.assertRaisesRegex(RuntimeError, "checkpoint no longer matches"):
                reopened.sync()

    def test_projection_storage_must_not_be_the_canonical_ledger_file(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            ledger_path = Path(directory) / "research.sqlite3"
            ledger = SQLiteResearchLedger(ledger_path)
            self.addCleanup(ledger.close)

            with self.assertRaisesRegex(ValueError, "separate from the canonical ledger"):
                SQLiteIndexedIntegrationTracker(ledger, ledger_path)

    def test_disposition_before_evidence_creation_is_rejected_incrementally(self) -> None:
        ledger = SQLiteResearchLedger(":memory:")
        self.addCleanup(ledger.close)
        ledger.append_event(
            event_type="INTEGRATION_DISPOSITION_RECORDED",
            reference_ids=("e1",),
            payload={"disposition": "INTEGRATED"},
        )
        _evidence(ledger, "e1")
        index = SQLiteIndexedIntegrationTracker(ledger)
        self.addCleanup(index.close)

        with self.assertRaisesRegex(ValueError, "precedes evidence creation"):
            index.sync()
        self.assertEqual(index.revision, 0)

    def test_unknown_disposition_and_redundant_disposition_traffic_remain_visible(self) -> None:
        ledger = SQLiteResearchLedger(":memory:")
        self.addCleanup(ledger.close)
        _evidence(ledger, "e1")
        ledger.append_event(
            event_type="INTEGRATION_DISPOSITION_RECORDED",
            reference_ids=("missing",),
            payload={"disposition": "INVALID"},
        )
        ledger.append_event(
            event_type="INTEGRATION_DISPOSITION_RECORDED",
            reference_ids=("e1", "e1"),
            payload={"disposition": "INTEGRATED"},
        )
        index = SQLiteIndexedIntegrationTracker(ledger)
        self.addCleanup(index.close)
        index.sync()

        self.assertEqual(index.unknown_disposition_reference_count(), 1)
        self.assertEqual(index.redisposition_reference_count(), 1)
        self.assertEqual(index.snapshot().dispositioned_evidence_count, 1)

    def test_record_helpers_are_read_your_writes_and_rebuild_is_lossless(self) -> None:
        ledger = SQLiteResearchLedger(":memory:")
        self.addCleanup(ledger.close)
        _evidence(ledger, "e1")
        index = SQLiteIndexedIntegrationTracker(ledger)
        self.addCleanup(index.close)
        self.assertEqual(index.snapshot().backlog_count, 1)

        index.record_disposition(("e1",), EvidenceDispositionKind.INTEGRATED)
        self.assertEqual(index.snapshot().backlog_count, 0)
        revision = index.revision

        rebuilt_revision = index.rebuild(page_size=1)
        self.assertEqual(rebuilt_revision, revision)
        self.assertEqual(index.snapshot().backlog_count, 0)
        self.assertEqual(index.snapshot().evidence_count, 1)


if __name__ == "__main__":
    unittest.main()
