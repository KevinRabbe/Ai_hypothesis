"""Tests for append-only verification-state projection."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from ai_hypothesis.runtime import (
    EvidenceVerificationStatus,
    SQLiteResearchLedger,
    VerificationAttemptStatus,
    VerificationStateProjector,
)


def _add_evidence(
    ledger: SQLiteResearchLedger,
    evidence_id: str,
    *,
    thread_id: str = "thread-1",
) -> None:
    ledger.append_event(
        event_type="EVIDENCE_ADDED",
        thread_id=thread_id,
        attempt_id=f"attempt:{evidence_id}",
        reference_ids=(evidence_id, "source-1"),
        payload={
            "evidence_id": evidence_id,
            "kind": "TEST_EVIDENCE",
            "summary": f"evidence {evidence_id}",
            "strength": 1.0,
            "uncertainty": 0.1,
            "data": {},
        },
    )


def _request(
    ledger: SQLiteResearchLedger,
    verification_id: str,
    target_evidence_id: str,
    *,
    thread_id: str = "thread-1",
) -> None:
    ledger.append_event(
        event_type="VERIFICATION_REQUESTED",
        thread_id=thread_id,
        reference_ids=(target_evidence_id,),
        payload={
            "verification_id": verification_id,
            "target_evidence_id": target_evidence_id,
            "purpose": "INDEPENDENT_REPLICATION",
            "data": {"requested_independence": True},
        },
    )


def _finish(
    ledger: SQLiteResearchLedger,
    event_type: str,
    verification_id: str,
    target_evidence_id: str,
    *,
    result_evidence_ids: tuple[str, ...] = (),
    worker_id: str = "verifier-1",
) -> None:
    ledger.append_event(
        event_type=event_type,
        thread_id="thread-1",
        reference_ids=(target_evidence_id, *result_evidence_ids),
        payload={
            "verification_id": verification_id,
            "target_evidence_id": target_evidence_id,
            "result_evidence_ids": list(result_evidence_ids),
            "verifier_worker_id": worker_id,
            "notes": f"result from {worker_id}",
            "data": {"tool_check": "complete"},
        },
    )


class VerificationStateProjectorTests(unittest.TestCase):
    def test_pending_request_keeps_evidence_unpromoted(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with SQLiteResearchLedger(Path(directory) / "ledger.sqlite") as ledger:
                _add_evidence(ledger, "evidence-1")
                _request(ledger, "verification-1", "evidence-1")

                projection = VerificationStateProjector().project(ledger.read_events())

            attempt = projection.get_attempt("verification-1")
            summary = projection.summary_for("evidence-1")
            self.assertEqual(attempt.status, VerificationAttemptStatus.PENDING)
            self.assertEqual(summary.status, EvidenceVerificationStatus.PENDING)
            self.assertEqual(summary.pending_count, 1)
            self.assertEqual(attempt.data["requested_independence"], True)

    def test_passed_verification_preserves_result_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with SQLiteResearchLedger(Path(directory) / "ledger.sqlite") as ledger:
                _add_evidence(ledger, "evidence-1")
                _request(ledger, "verification-1", "evidence-1")
                _add_evidence(ledger, "verification-evidence")
                _finish(
                    ledger,
                    "VERIFICATION_PASSED",
                    "verification-1",
                    "evidence-1",
                    result_evidence_ids=("verification-evidence",),
                )

                projection = VerificationStateProjector().project(ledger.read_events())

            attempt = projection.get_attempt("verification-1")
            summary = projection.summary_for("evidence-1")
            self.assertEqual(attempt.status, VerificationAttemptStatus.CONFIRMED)
            self.assertEqual(attempt.result_evidence_ids, ("verification-evidence",))
            self.assertEqual(attempt.verifier_worker_id, "verifier-1")
            self.assertEqual(attempt.data["requested_independence"], True)
            self.assertEqual(attempt.data["tool_check"], "complete")
            self.assertEqual(summary.status, EvidenceVerificationStatus.CONFIRMED)
            self.assertEqual(summary.confirmed_count, 1)

    def test_failed_and_inconclusive_results_remain_distinct(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with SQLiteResearchLedger(Path(directory) / "ledger.sqlite") as ledger:
                _add_evidence(ledger, "evidence-rejected")
                _add_evidence(ledger, "evidence-inconclusive")
                _request(ledger, "verification-rejected", "evidence-rejected")
                _request(
                    ledger,
                    "verification-inconclusive",
                    "evidence-inconclusive",
                )
                _finish(
                    ledger,
                    "VERIFICATION_FAILED",
                    "verification-rejected",
                    "evidence-rejected",
                )
                _finish(
                    ledger,
                    "VERIFICATION_INCONCLUSIVE",
                    "verification-inconclusive",
                    "evidence-inconclusive",
                )

                projection = VerificationStateProjector().project(ledger.read_events())

            self.assertEqual(
                projection.summary_for("evidence-rejected").status,
                EvidenceVerificationStatus.REJECTED,
            )
            self.assertEqual(
                projection.summary_for("evidence-inconclusive").status,
                EvidenceVerificationStatus.INCONCLUSIVE,
            )

    def test_independent_pass_and_failure_produce_conflicted_summary(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with SQLiteResearchLedger(Path(directory) / "ledger.sqlite") as ledger:
                _add_evidence(ledger, "evidence-1")
                _request(ledger, "verification-pass", "evidence-1")
                _request(ledger, "verification-fail", "evidence-1")
                _finish(
                    ledger,
                    "VERIFICATION_PASSED",
                    "verification-pass",
                    "evidence-1",
                    worker_id="verifier-a",
                )
                _finish(
                    ledger,
                    "VERIFICATION_FAILED",
                    "verification-fail",
                    "evidence-1",
                    worker_id="verifier-b",
                )

                projection = VerificationStateProjector().project(ledger.read_events())

            summary = projection.summary_for("evidence-1")
            self.assertEqual(summary.status, EvidenceVerificationStatus.CONFLICTED)
            self.assertEqual(summary.confirmed_count, 1)
            self.assertEqual(summary.rejected_count, 1)
            self.assertEqual(len(summary.attempts), 2)

    def test_summary_for_unverified_evidence_is_explicit(self) -> None:
        projection = VerificationStateProjector().project(())

        summary = projection.summary_for("evidence-never-verified")

        self.assertEqual(summary.status, EvidenceVerificationStatus.UNVERIFIED)
        self.assertEqual(summary.attempts, ())

    def test_invalid_lifecycle_order_and_duplicate_terminal_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with SQLiteResearchLedger(Path(directory) / "ledger.sqlite") as ledger:
                _request_event = ledger.append_event(
                    event_type="VERIFICATION_REQUESTED",
                    thread_id="thread-1",
                    payload={
                        "verification_id": "verification-1",
                        "target_evidence_id": "missing-evidence",
                        "purpose": "INDEPENDENT_REPLICATION",
                    },
                )
                with self.assertRaisesRegex(ValueError, "unknown evidence"):
                    VerificationStateProjector().project(ledger.read_events())

        with tempfile.TemporaryDirectory() as directory:
            with SQLiteResearchLedger(Path(directory) / "ledger.sqlite") as ledger:
                _add_evidence(ledger, "evidence-1")
                _request(ledger, "verification-1", "evidence-1")
                _finish(
                    ledger,
                    "VERIFICATION_PASSED",
                    "verification-1",
                    "evidence-1",
                )
                _finish(
                    ledger,
                    "VERIFICATION_FAILED",
                    "verification-1",
                    "evidence-1",
                )
                with self.assertRaisesRegex(ValueError, "already terminal"):
                    VerificationStateProjector().project(ledger.read_events())

    def test_result_evidence_must_exist_before_terminal_event(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with SQLiteResearchLedger(Path(directory) / "ledger.sqlite") as ledger:
                _add_evidence(ledger, "evidence-1")
                _request(ledger, "verification-1", "evidence-1")
                _finish(
                    ledger,
                    "VERIFICATION_PASSED",
                    "verification-1",
                    "evidence-1",
                    result_evidence_ids=("missing-result",),
                )

                with self.assertRaisesRegex(ValueError, "unknown evidence"):
                    VerificationStateProjector().project(ledger.read_events())


if __name__ == "__main__":
    unittest.main()
