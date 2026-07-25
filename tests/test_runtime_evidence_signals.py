"""Tests for provisional scheduler signals derived from evidence state."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from ai_hypothesis.runtime import (
    EvidenceSignalConfig,
    EvidenceSignalProviderV0,
    EvidenceStateProjector,
    ProjectedState,
    SQLiteResearchLedger,
    WorkPurpose,
)


def _state(
    thread_id: str = "thread-1",
    *,
    metadata: dict[str, object] | None = None,
    contradictions: tuple[str, ...] = (),
    dependencies: tuple[str, ...] = (),
) -> ProjectedState:
    return ProjectedState(
        revision=1,
        thread_id=thread_id,
        objective="Investigate local evidence",
        status="ACTIVE",
        purpose=WorkPurpose.EXPLORE,
        contradiction_ids=contradictions,
        dependency_thread_ids=dependencies,
        metadata=metadata or {},
    )


def _add_evidence(
    ledger: SQLiteResearchLedger,
    *,
    evidence_id: str,
    top_label: str,
    source_id: str,
    strength: float,
    uncertainty: float,
    thread_id: str = "thread-1",
) -> None:
    ledger.append_event(
        event_type="EVIDENCE_ADDED",
        thread_id=thread_id,
        attempt_id=f"attempt:{evidence_id}",
        reference_ids=(evidence_id, source_id),
        payload={
            "evidence_id": evidence_id,
            "kind": "STEP02_LOCAL_CLASS_EVIDENCE",
            "summary": f"local evidence for {top_label}",
            "strength": strength,
            "uncertainty": uncertainty,
            "data": {"top_label": top_label},
        },
    )


class _CountingProjector(EvidenceStateProjector):
    def __init__(self) -> None:
        self.calls = 0

    def project(self, events):
        self.calls += 1
        return super().project(events)


class EvidenceSignalProviderTests(unittest.TestCase):
    def test_missing_evidence_creates_uncertainty_and_coverage_need(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with SQLiteResearchLedger(Path(directory) / "ledger.sqlite") as ledger:
                ledger.append_event(
                    event_type="THREAD_CREATED",
                    thread_id="thread-1",
                    payload={
                        "objective": "Investigate local evidence",
                        "purpose": "EXPLORE",
                    },
                )
                provider = EvidenceSignalProviderV0(
                    ledger,
                    config=EvidenceSignalConfig(coverage_target=2),
                )

                signals = provider(
                    _state(metadata={"importance": 0.8, "estimated_cost": 0.2})
                )

            self.assertEqual(signals.uncertainty, 1.0)
            self.assertEqual(signals.missing_coverage, 1.0)
            self.assertEqual(signals.novelty, 1.0)
            self.assertEqual(signals.importance, 0.8)
            self.assertEqual(signals.estimated_cost, 0.2)

    def test_one_strong_contribution_requests_independent_verification(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with SQLiteResearchLedger(Path(directory) / "ledger.sqlite") as ledger:
                _add_evidence(
                    ledger,
                    evidence_id="evidence-1",
                    top_label="SIGNAL",
                    source_id="source-1",
                    strength=3.0,
                    uncertainty=0.1,
                )
                provider = EvidenceSignalProviderV0(
                    ledger,
                    config=EvidenceSignalConfig(
                        coverage_target=2,
                        strong_evidence_threshold=2.0,
                        verification_redundancy_target=2,
                    ),
                )

                signals = provider(_state())

            self.assertEqual(signals.verification_need, 1.0)
            self.assertEqual(signals.missing_coverage, 0.5)
            self.assertAlmostEqual(signals.uncertainty, 0.1)

    def test_balanced_minority_conflict_becomes_maximum_contradiction(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with SQLiteResearchLedger(Path(directory) / "ledger.sqlite") as ledger:
                _add_evidence(
                    ledger,
                    evidence_id="evidence-signal",
                    top_label="SIGNAL",
                    source_id="source-a",
                    strength=1.0,
                    uncertainty=0.1,
                )
                _add_evidence(
                    ledger,
                    evidence_id="evidence-no-signal",
                    top_label="NO_SIGNAL",
                    source_id="source-b",
                    strength=1.0,
                    uncertainty=0.1,
                )
                provider = EvidenceSignalProviderV0(ledger)

                signals = provider(_state())

            self.assertEqual(signals.contradiction_severity, 1.0)
            self.assertEqual(signals.verification_need, 1.0)

    def test_invalidated_evidence_stops_influencing_current_signals(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with SQLiteResearchLedger(Path(directory) / "ledger.sqlite") as ledger:
                _add_evidence(
                    ledger,
                    evidence_id="evidence-signal",
                    top_label="SIGNAL",
                    source_id="source-a",
                    strength=1.0,
                    uncertainty=0.2,
                )
                _add_evidence(
                    ledger,
                    evidence_id="evidence-conflict",
                    top_label="NO_SIGNAL",
                    source_id="source-b",
                    strength=1.0,
                    uncertainty=0.2,
                )
                provider = EvidenceSignalProviderV0(ledger)
                before = provider(_state())

                ledger.append_event(
                    event_type="EVIDENCE_INVALIDATED",
                    thread_id="thread-1",
                    reference_ids=("evidence-conflict",),
                    payload={
                        "evidence_id": "evidence-conflict",
                        "reason": "failed verification",
                    },
                )
                after = provider(_state())

            self.assertEqual(before.contradiction_severity, 1.0)
            self.assertEqual(after.contradiction_severity, 0.0)

    def test_progress_starvation_dependencies_and_metadata_are_bounded(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with SQLiteResearchLedger(Path(directory) / "ledger.sqlite") as ledger:
                created = ledger.append_event(
                    event_type="THREAD_CREATED",
                    thread_id="thread-1",
                    payload={
                        "objective": "Investigate local evidence",
                        "purpose": "EXPLORE",
                    },
                )
                ledger.append_event(
                    event_type="ATTEMPT_COMPLETED",
                    thread_id="thread-1",
                    parent_event_ids=(created.event_id,),
                    payload={"progress_made": True},
                )
                ledger.append_event(
                    event_type="ATTEMPT_FAILED",
                    thread_id="thread-1",
                    parent_event_ids=(created.event_id,),
                    payload={"progress_made": False},
                )
                ledger.append_event(
                    event_type="THREAD_CREATED",
                    thread_id="other-1",
                    payload={"objective": "Other", "purpose": "EXPLORE"},
                )
                ledger.append_event(
                    event_type="THREAD_CREATED",
                    thread_id="other-2",
                    payload={"objective": "Other", "purpose": "EXPLORE"},
                )
                provider = EvidenceSignalProviderV0(
                    ledger,
                    config=EvidenceSignalConfig(
                        dependency_target=2,
                        progress_window=2,
                        starvation_event_window=2,
                    ),
                )

                signals = provider(
                    _state(
                        metadata={"importance": 1.0, "estimated_cost": 0.25},
                        contradictions=("c1", "c2", "c3"),
                        dependencies=("d1", "d2"),
                    )
                )

            self.assertEqual(signals.recent_progress, 0.5)
            self.assertEqual(signals.starvation, 1.0)
            self.assertEqual(signals.dependency_impact, 1.0)
            self.assertEqual(signals.contradiction_severity, 1.0)
            self.assertEqual(signals.importance, 1.0)
            self.assertEqual(signals.estimated_cost, 0.25)

    def test_projection_cache_refreshes_only_when_ledger_changes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with SQLiteResearchLedger(Path(directory) / "ledger.sqlite") as ledger:
                ledger.append_event(
                    event_type="THREAD_CREATED",
                    thread_id="thread-1",
                    payload={
                        "objective": "Investigate local evidence",
                        "purpose": "EXPLORE",
                    },
                )
                projector = _CountingProjector()
                provider = EvidenceSignalProviderV0(
                    ledger,
                    evidence_projector=projector,
                )
                state = _state()

                provider(state)
                provider(state)
                self.assertEqual(projector.calls, 1)

                _add_evidence(
                    ledger,
                    evidence_id="evidence-1",
                    top_label="SIGNAL",
                    source_id="source-1",
                    strength=1.0,
                    uncertainty=0.2,
                )
                provider(state)
                self.assertEqual(projector.calls, 2)

    def test_invalid_metadata_is_rejected_at_signal_boundary(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with SQLiteResearchLedger(Path(directory) / "ledger.sqlite") as ledger:
                provider = EvidenceSignalProviderV0(ledger)

                with self.assertRaisesRegex(ValueError, "importance"):
                    provider(_state(metadata={"importance": 2.0}))


if __name__ == "__main__":
    unittest.main()
