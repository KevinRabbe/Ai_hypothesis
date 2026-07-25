"""Provisional Scheduler v0 signals derived from persistent evidence state.

The formulas are deliberately simple and replaceable. The stable boundary is the
callable ``ProjectedState -> SchedulerSignals`` contract used by the control loop.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any

from .contracts import LedgerEvent, ProjectedState
from .evidence_projector import (
    EvidenceProjection,
    EvidenceState,
    EvidenceStateProjector,
    EvidenceStatus,
)
from .ledger import SQLiteResearchLedger
from .scheduler import SchedulerSignals


@dataclass(frozen=True, slots=True)
class EvidenceSignalConfig:
    """Inspectable saturation points for EvidenceSignalProviderV0."""

    coverage_target: int = 4
    contradiction_target: int = 3
    dependency_target: int = 4
    progress_window: int = 4
    starvation_event_window: int = 50
    strong_evidence_threshold: float = 2.0
    verification_redundancy_target: int = 2

    def validate(self) -> None:
        for name, value in (
            ("coverage_target", self.coverage_target),
            ("contradiction_target", self.contradiction_target),
            ("dependency_target", self.dependency_target),
            ("progress_window", self.progress_window),
            ("starvation_event_window", self.starvation_event_window),
            ("verification_redundancy_target", self.verification_redundancy_target),
        ):
            if value <= 0:
                raise ValueError(f"{name} must be positive")
        if self.strong_evidence_threshold <= 0.0:
            raise ValueError("strong_evidence_threshold must be positive")


class EvidenceSignalProviderV0:
    """Derive bounded scheduler metadata from ledger/evidence projections.

    One provider may be reused for every thread in a control-loop cycle. It caches the
    replay result until the ledger sequence changes so global history is not rebuilt
    independently for each thread.
    """

    def __init__(
        self,
        ledger: SQLiteResearchLedger,
        *,
        config: EvidenceSignalConfig | None = None,
        evidence_projector: EvidenceStateProjector | None = None,
    ) -> None:
        self.ledger = ledger
        self.config = config or EvidenceSignalConfig()
        self.config.validate()
        self.evidence_projector = evidence_projector or EvidenceStateProjector()
        self._cached_sequence = -1
        self._cached_events: tuple[LedgerEvent, ...] = ()
        self._cached_evidence = EvidenceProjection(revision=0, evidence=())

    def __call__(self, state: ProjectedState) -> SchedulerSignals:
        state.validate()
        self._refresh()

        active = self._cached_evidence.select(
            thread_id=state.thread_id,
            status=EvidenceStatus.ACTIVE,
        )
        thread_events = tuple(
            event
            for event in self._cached_events
            if event.thread_id == state.thread_id
        )

        importance = _metadata_unit_interval(state.metadata, "importance", 0.5)
        estimated_cost = _metadata_unit_interval(
            state.metadata,
            "estimated_cost",
            0.0,
        )
        coverage_units = _coverage_units(active)
        missing_coverage = 1.0 - min(
            len(coverage_units) / self.config.coverage_target,
            1.0,
        )
        uncertainty = _mean_uncertainty(active)
        disagreement = _label_disagreement(active)
        explicit_contradiction = min(
            len(state.contradiction_ids) / self.config.contradiction_target,
            1.0,
        )
        contradiction_severity = max(disagreement, explicit_contradiction)
        novelty = min(
            len(coverage_units) / max(len(active), 1),
            1.0,
        ) if active else 1.0
        dependency_impact = min(
            len(state.dependency_thread_ids) / self.config.dependency_target,
            1.0,
        )
        recent_progress = _recent_progress(
            thread_events,
            window=self.config.progress_window,
        )
        verification_need = max(
            contradiction_severity,
            _unverified_strong_signal(
                active,
                threshold=self.config.strong_evidence_threshold,
                redundancy_target=self.config.verification_redundancy_target,
            ),
        )
        starvation = _starvation(
            thread_events,
            latest_sequence=self._cached_sequence,
            window=self.config.starvation_event_window,
        )

        signals = SchedulerSignals(
            importance=importance,
            uncertainty=uncertainty,
            contradiction_severity=contradiction_severity,
            missing_coverage=missing_coverage,
            novelty=novelty,
            dependency_impact=dependency_impact,
            recent_progress=recent_progress,
            verification_need=verification_need,
            starvation=starvation,
            estimated_cost=estimated_cost,
        )
        signals.validate()
        return signals

    def _refresh(self) -> None:
        latest_sequence = self.ledger.latest_sequence()
        if latest_sequence == self._cached_sequence:
            return
        events = self.ledger.read_events()
        self._cached_events = events
        self._cached_evidence = self.evidence_projector.project(events)
        self._cached_sequence = latest_sequence


def _metadata_unit_interval(
    metadata: Any,
    key: str,
    default: float,
) -> float:
    value = metadata.get(key, default)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"thread metadata {key!r} must be numeric")
    resolved = float(value)
    if not 0.0 <= resolved <= 1.0:
        raise ValueError(f"thread metadata {key!r} must be in [0, 1]")
    return resolved


def _coverage_units(evidence: tuple[EvidenceState, ...]) -> set[str]:
    units: set[str] = set()
    for state in evidence:
        if state.source_reference_ids:
            units.update(state.source_reference_ids)
        else:
            units.add(state.evidence_id)
    return units


def _mean_uncertainty(evidence: tuple[EvidenceState, ...]) -> float:
    if not evidence:
        return 1.0
    values = tuple(
        state.uncertainty
        for state in evidence
        if state.uncertainty is not None
    )
    if not values:
        return 0.5
    return sum(values) / len(values)


def _label_disagreement(evidence: tuple[EvidenceState, ...]) -> float:
    labels = tuple(
        label
        for state in evidence
        if isinstance((label := state.data.get("top_label")), str) and label
    )
    if len(labels) < 2:
        return 0.0
    largest_group = max(Counter(labels).values())
    return 1.0 - largest_group / len(labels)


def _unverified_strong_signal(
    evidence: tuple[EvidenceState, ...],
    *,
    threshold: float,
    redundancy_target: int,
) -> float:
    if len(evidence) >= redundancy_target:
        return 0.0
    strengths = tuple(
        max(state.strength, 0.0)
        for state in evidence
        if state.strength is not None
    )
    if not strengths:
        return 0.0
    return min(max(strengths) / threshold, 1.0)


def _recent_progress(
    events: tuple[LedgerEvent, ...],
    *,
    window: int,
) -> float:
    terminal = tuple(
        event
        for event in events
        if event.event_type
        in {"ATTEMPT_COMPLETED", "ATTEMPT_PARTIAL", "ATTEMPT_FAILED"}
    )[-window:]
    if not terminal:
        return 0.0
    progress_count = sum(
        1 for event in terminal if event.payload.get("progress_made") is True
    )
    return progress_count / len(terminal)


def _starvation(
    events: tuple[LedgerEvent, ...],
    *,
    latest_sequence: int,
    window: int,
) -> float:
    if not events or latest_sequence <= 0:
        return 0.0
    attempt_sequences = tuple(
        event.sequence
        for event in events
        if event.event_type
        in {"ATTEMPT_COMPLETED", "ATTEMPT_PARTIAL", "ATTEMPT_FAILED"}
    )
    baseline = max(attempt_sequences) if attempt_sequences else events[0].sequence
    return min(max(latest_sequence - baseline, 0) / window, 1.0)
