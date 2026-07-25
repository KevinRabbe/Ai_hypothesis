"""Tests for rebuildable evidence-state projection."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from ai_hypothesis.runtime import (
    EvidenceStateProjector,
    EvidenceStatus,
    SQLiteResearchLedger,
)


def _add_evidence(
    ledger: SQLiteResearchLedger,
    *,
    evidence_id: str,
    thread_id: str,
    attempt_id: str,
    top_label: str,
    source_id: str = "source-1",
    strength: float = 1.0,
) -> None:
    ledger.append_event(
        event_type="EVIDENCE_ADDED",
        thread_id=thread_id,
        attempt_id=attempt_id,
        reference_ids=(evidence_id, source_id),
        payload={
            "evidence_id": evidence_id,
            "kind": "STEP02_LOCAL_CLASS_EVIDENCE",
            "summary": f"local evidence for {top_label}",
            "strength": strength,
            "uncertainty": 0.1,
            "data": {
                "top_label": top_label,
                "checkpoint_path": f"{attempt_id}.pt",
            },
        },
    )


class EvidenceStateProjectorTests(unittest.TestCase):
    def test_projection_preserves_all_independent_contributions(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with SQLiteResearchLedger(Path(directory) / "ledger.sqlite") as ledger:
                _add_evidence(
                    ledger,
                    evidence_id="evidence-majority-1",
                    thread_id="thread-1",
                    attempt_id="worker-1",
                    top_label="SIGNAL",
                )
                _add_evidence(
                    ledger,
                    evidence_id="evidence-majority-2",
                    thread_id="thread-1",
                    attempt_id="worker-2",
                    top_label="SIGNAL",
                )
                _add_evidence(
                    ledger,
                    evidence_id="evidence-minority",
                    thread_id="thread-1",
                    attempt_id="worker-3",
                    top_label="NO_SIGNAL",
                    strength=4.0,
                )

                projection = EvidenceStateProjector().project(ledger.read_events())

            self.assertEqual(len(projection.evidence), 3)
            self.assertEqual(
                tuple(state.evidence_id for state in projection.evidence),
                (
                    "evidence-majority-1",
                    "evidence-majority-2",
                    "evidence-minority",
                ),
            )
            minority = projection.get("evidence-minority")
            self.assertEqual(minority.data["top_label"], "NO_SIGNAL")
            self.assertEqual(minority.strength, 4.0)
            self.assertEqual(minority.status, EvidenceStatus.ACTIVE)
            self.assertEqual(minority.source_reference_ids, ("source-1",))

    def test_invalidation_and_supersession_do_not_delete_history(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with SQLiteResearchLedger(Path(directory) / "ledger.sqlite") as ledger:
                _add_evidence(
                    ledger,
                    evidence_id="evidence-invalid",
                    thread_id="thread-1",
                    attempt_id="attempt-1",
                    top_label="SIGNAL",
                )
                _add_evidence(
                    ledger,
                    evidence_id="evidence-old",
                    thread_id="thread-1",
                    attempt_id="attempt-2",
                    top_label="SIGNAL",
                )
                _add_evidence(
                    ledger,
                    evidence_id="evidence-new",
                    thread_id="thread-1",
                    attempt_id="attempt-3",
                    top_label="NO_SIGNAL",
                )
                ledger.append_event(
                    event_type="EVIDENCE_INVALIDATED",
                    thread_id="thread-1",
                    reference_ids=("evidence-invalid",),
                    payload={
                        "evidence_id": "evidence-invalid",
                        "reason": "source checksum failed",
                    },
                )
                ledger.append_event(
                    event_type="EVIDENCE_SUPERSEDED",
                    thread_id="thread-1",
                    reference_ids=("evidence-old", "evidence-new"),
                    payload={
                        "evidence_id": "evidence-old",
                        "superseded_by": "evidence-new",
                    },
                )

                projection = EvidenceStateProjector().project(ledger.read_events())

            invalid = projection.get("evidence-invalid")
            old = projection.get("evidence-old")
            new = projection.get("evidence-new")
            self.assertEqual(invalid.status, EvidenceStatus.INVALIDATED)
            self.assertEqual(invalid.invalidation_reason, "source checksum failed")
            self.assertEqual(old.status, EvidenceStatus.SUPERSEDED)
            self.assertEqual(old.superseded_by, "evidence-new")
            self.assertEqual(new.status, EvidenceStatus.ACTIVE)
            self.assertEqual(len(projection.evidence), 3)

    def test_bounded_queries_filter_without_aggregating(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with SQLiteResearchLedger(Path(directory) / "ledger.sqlite") as ledger:
                _add_evidence(
                    ledger,
                    evidence_id="evidence-a",
                    thread_id="thread-a",
                    attempt_id="attempt-a",
                    top_label="SIGNAL",
                    source_id="shared-source",
                )
                _add_evidence(
                    ledger,
                    evidence_id="evidence-b",
                    thread_id="thread-b",
                    attempt_id="attempt-b",
                    top_label="NO_SIGNAL",
                    source_id="shared-source",
                )
                ledger.append_event(
                    event_type="EVIDENCE_INVALIDATED",
                    thread_id="thread-b",
                    reference_ids=("evidence-b",),
                    payload={
                        "evidence_id": "evidence-b",
                        "reason": "failed verification",
                    },
                )

                projection = EvidenceStateProjector().project(ledger.read_events())

            self.assertEqual(
                tuple(
                    state.evidence_id
                    for state in projection.select(thread_id="thread-a")
                ),
                ("evidence-a",),
            )
            self.assertEqual(
                tuple(
                    state.evidence_id
                    for state in projection.select(status=EvidenceStatus.ACTIVE)
                ),
                ("evidence-a",),
            )
            self.assertEqual(
                len(
                    projection.select(
                        kind="STEP02_LOCAL_CLASS_EVIDENCE",
                        source_reference_id="shared-source",
                    )
                ),
                2,
            )

    def test_duplicate_and_invalid_lifecycle_events_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with SQLiteResearchLedger(Path(directory) / "ledger.sqlite") as ledger:
                _add_evidence(
                    ledger,
                    evidence_id="evidence-1",
                    thread_id="thread-1",
                    attempt_id="attempt-1",
                    top_label="SIGNAL",
                )
                _add_evidence(
                    ledger,
                    evidence_id="evidence-1",
                    thread_id="thread-1",
                    attempt_id="attempt-2",
                    top_label="NO_SIGNAL",
                )

                with self.assertRaisesRegex(ValueError, "added more than once"):
                    EvidenceStateProjector().project(ledger.read_events())

        with tempfile.TemporaryDirectory() as directory:
            with SQLiteResearchLedger(Path(directory) / "ledger.sqlite") as ledger:
                ledger.append_event(
                    event_type="EVIDENCE_INVALIDATED",
                    thread_id="thread-1",
                    reference_ids=("missing",),
                    payload={
                        "evidence_id": "missing",
                        "reason": "invalid",
                    },
                )

                with self.assertRaisesRegex(ValueError, "unknown evidence"):
                    EvidenceStateProjector().project(ledger.read_events())


if __name__ == "__main__":
    unittest.main()
