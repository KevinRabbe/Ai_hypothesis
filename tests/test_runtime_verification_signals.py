"""Tests for verification-aware evidence scheduler signals."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from ai_hypothesis.runtime import (
    EvidenceSignalConfig,
    EvidenceSignalProviderV0,
    ProjectedState,
    SQLiteResearchLedger,
    WorkPurpose,
)


def _state() -> ProjectedState:
    return ProjectedState(
        revision=1,
        thread_id="thread-1",
        objective="Investigate strong evidence",
        status="ACTIVE",
        purpose=WorkPurpose.EXPLORE,
    )


def _add_evidence(ledger: SQLiteResearchLedger) -> None:
    ledger.append_event(
        event_type="EVIDENCE_ADDED",
        thread_id="thread-1",
        attempt_id="attempt:evidence-1",
        reference_ids=("evidence-1", "source-1"),
        payload={
            "evidence_id": "evidence-1",
            "kind": "STEP02_LOCAL_CLASS_EVIDENCE",
            "summary": "strong local evidence",
            "strength": 3.0,
            "uncertainty": 0.1,
            "data": {"top_label": "SIGNAL"},
        },
    )


def _request(ledger: SQLiteResearchLedger, verification_id: str) -> None:
    ledger.append_event(
        event_type="VERIFICATION_REQUESTED",
        thread_id="thread-1",
        reference_ids=("evidence-1",),
        payload={
            "verification_id": verification_id,
            "target_evidence_id": "evidence-1",
            "purpose": "INDEPENDENT_REPLICATION",
        },
    )


def _finish(
    ledger: SQLiteResearchLedger,
    verification_id: str,
    event_type: str,
) -> None:
    ledger.append_event(
        event_type=event_type,
        thread_id="thread-1",
        reference_ids=("evidence-1",),
        payload={
            "verification_id": verification_id,
            "target_evidence_id": "evidence-1",
            "verifier_worker_id": f"worker:{verification_id}",
        },
    )


class VerificationAwareEvidenceSignalTests(unittest.TestCase):
    def test_pending_work_counts_toward_redundancy_target(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with SQLiteResearchLedger(Path(directory) / "ledger.sqlite") as ledger:
                _add_evidence(ledger)
                provider = EvidenceSignalProviderV0(
                    ledger,
                    config=EvidenceSignalConfig(
                        strong_evidence_threshold=2.0,
                        verification_redundancy_target=2,
                    ),
                )

                self.assertEqual(provider(_state()).verification_need, 1.0)

                _request(ledger, "verification-1")
                self.assertEqual(provider(_state()).verification_need, 1.0)

                _request(ledger, "verification-2")
                self.assertEqual(provider(_state()).verification_need, 0.0)

    def test_confirmed_plus_pending_does_not_schedule_third_duplicate(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with SQLiteResearchLedger(Path(directory) / "ledger.sqlite") as ledger:
                _add_evidence(ledger)
                _request(ledger, "verification-1")
                _finish(ledger, "verification-1", "VERIFICATION_PASSED")
                _request(ledger, "verification-2")
                provider = EvidenceSignalProviderV0(
                    ledger,
                    config=EvidenceSignalConfig(
                        strong_evidence_threshold=2.0,
                        verification_redundancy_target=2,
                    ),
                )

                signals = provider(_state())

            self.assertEqual(signals.verification_need, 0.0)
            self.assertEqual(signals.contradiction_severity, 0.0)

    def test_confirmed_evidence_stops_requesting_verification_at_target(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with SQLiteResearchLedger(Path(directory) / "ledger.sqlite") as ledger:
                _add_evidence(ledger)
                _request(ledger, "verification-1")
                _finish(ledger, "verification-1", "VERIFICATION_PASSED")
                provider = EvidenceSignalProviderV0(
                    ledger,
                    config=EvidenceSignalConfig(
                        strong_evidence_threshold=2.0,
                        verification_redundancy_target=1,
                    ),
                )

                signals = provider(_state())

            self.assertEqual(signals.verification_need, 0.0)
            self.assertEqual(signals.contradiction_severity, 0.0)

    def test_rejected_verification_creates_contradiction_and_more_verification_need(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with SQLiteResearchLedger(Path(directory) / "ledger.sqlite") as ledger:
                _add_evidence(ledger)
                _request(ledger, "verification-1")
                _finish(ledger, "verification-1", "VERIFICATION_FAILED")
                provider = EvidenceSignalProviderV0(ledger)

                signals = provider(_state())

            self.assertEqual(signals.contradiction_severity, 1.0)
            self.assertEqual(signals.verification_need, 1.0)

    def test_inconclusive_verification_remains_unresolved(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with SQLiteResearchLedger(Path(directory) / "ledger.sqlite") as ledger:
                _add_evidence(ledger)
                _request(ledger, "verification-1")
                _finish(ledger, "verification-1", "VERIFICATION_INCONCLUSIVE")
                provider = EvidenceSignalProviderV0(ledger)

                signals = provider(_state())

            self.assertEqual(signals.contradiction_severity, 0.0)
            self.assertEqual(signals.verification_need, 1.0)

    def test_conflicting_independent_verification_is_maximally_unresolved(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with SQLiteResearchLedger(Path(directory) / "ledger.sqlite") as ledger:
                _add_evidence(ledger)
                _request(ledger, "verification-pass")
                _request(ledger, "verification-fail")
                _finish(ledger, "verification-pass", "VERIFICATION_PASSED")
                _finish(ledger, "verification-fail", "VERIFICATION_FAILED")
                provider = EvidenceSignalProviderV0(ledger)

                signals = provider(_state())

            self.assertEqual(signals.contradiction_severity, 1.0)
            self.assertEqual(signals.verification_need, 1.0)


if __name__ == "__main__":
    unittest.main()
