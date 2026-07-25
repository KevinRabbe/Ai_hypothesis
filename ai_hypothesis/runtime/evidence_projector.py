"""Rebuild bounded evidence state from append-only Research Ledger events.

This is a specialized view inside the existing State Projector role. The ledger remains
canonical; no evidence is deleted or overwritten when a later event invalidates or
supersedes it.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, replace
from enum import Enum
from typing import Any

from .contracts import LedgerEvent


class EvidenceStatus(str, Enum):
    ACTIVE = "ACTIVE"
    INVALIDATED = "INVALIDATED"
    SUPERSEDED = "SUPERSEDED"


@dataclass(frozen=True, slots=True)
class EvidenceState:
    """Current projected state of one immutable evidence contribution."""

    evidence_id: str
    kind: str
    summary: str
    strength: float | None
    uncertainty: float | None
    data: Mapping[str, Any]
    source_reference_ids: tuple[str, ...]
    thread_id: str | None
    attempt_id: str | None
    created_event_id: str
    created_sequence: int
    status: EvidenceStatus = EvidenceStatus.ACTIVE
    status_event_id: str | None = None
    status_sequence: int | None = None
    invalidation_reason: str | None = None
    superseded_by: str | None = None

    def validate(self) -> None:
        for name, value in (
            ("evidence_id", self.evidence_id),
            ("kind", self.kind),
            ("summary", self.summary),
            ("created_event_id", self.created_event_id),
        ):
            if not value or not value.strip():
                raise ValueError(f"{name} must be non-empty")
        if self.created_sequence < 0:
            raise ValueError("created_sequence must be non-negative")
        if any(not value or not value.strip() for value in self.source_reference_ids):
            raise ValueError("source_reference_ids must not contain empty IDs")
        if self.strength is not None and not math.isfinite(self.strength):
            raise ValueError("strength must be finite")
        if self.uncertainty is not None:
            if not math.isfinite(self.uncertainty):
                raise ValueError("uncertainty must be finite")
            if not 0.0 <= self.uncertainty <= 1.0:
                raise ValueError("uncertainty must be in [0, 1]")
        if self.status is EvidenceStatus.ACTIVE:
            if self.status_event_id is not None or self.status_sequence is not None:
                raise ValueError("active evidence must not have a terminal status event")
            if self.invalidation_reason is not None or self.superseded_by is not None:
                raise ValueError("active evidence must not have terminal status metadata")
        else:
            if not self.status_event_id:
                raise ValueError("inactive evidence requires status_event_id")
            if self.status_sequence is None or self.status_sequence < self.created_sequence:
                raise ValueError("inactive evidence requires a valid status_sequence")
        if self.status is EvidenceStatus.INVALIDATED and not self.invalidation_reason:
            raise ValueError("invalidated evidence requires a reason")
        if self.status is EvidenceStatus.SUPERSEDED:
            if not self.superseded_by:
                raise ValueError("superseded evidence requires superseded_by")
            if self.superseded_by == self.evidence_id:
                raise ValueError("evidence cannot supersede itself")


@dataclass(frozen=True, slots=True)
class EvidenceProjection:
    """Bounded rebuildable evidence view at one ledger revision."""

    revision: int
    evidence: tuple[EvidenceState, ...]

    def get(self, evidence_id: str) -> EvidenceState:
        for state in self.evidence:
            if state.evidence_id == evidence_id:
                return state
        raise KeyError(evidence_id)

    def select(
        self,
        *,
        thread_id: str | None = None,
        kind: str | None = None,
        status: EvidenceStatus | None = None,
        source_reference_id: str | None = None,
    ) -> tuple[EvidenceState, ...]:
        return tuple(
            state
            for state in self.evidence
            if (thread_id is None or state.thread_id == thread_id)
            and (kind is None or state.kind == kind)
            and (status is None or state.status is status)
            and (
                source_reference_id is None
                or source_reference_id in state.source_reference_ids
            )
        )


class EvidenceStateProjector:
    """Fold evidence lifecycle events into a queryable current view."""

    def project(self, events: Iterable[LedgerEvent]) -> EvidenceProjection:
        states: dict[str, EvidenceState] = {}
        revision = 0
        previous_sequence = -1

        for event in events:
            event.validate()
            if event.sequence <= previous_sequence:
                raise ValueError(
                    "events must be supplied in strictly increasing sequence order"
                )
            previous_sequence = event.sequence
            revision = event.sequence

            if event.event_type == "EVIDENCE_ADDED":
                state = self._added_state(event)
                if state.evidence_id in states:
                    raise ValueError(
                        f"evidence {state.evidence_id!r} was added more than once"
                    )
                states[state.evidence_id] = state
            elif event.event_type == "EVIDENCE_INVALIDATED":
                evidence_id = _evidence_id(event)
                current = _require_active(states, evidence_id, event.event_type)
                reason = _payload_text(event, "reason")
                states[evidence_id] = replace(
                    current,
                    status=EvidenceStatus.INVALIDATED,
                    status_event_id=event.event_id,
                    status_sequence=event.sequence,
                    invalidation_reason=reason,
                )
            elif event.event_type == "EVIDENCE_SUPERSEDED":
                evidence_id = _evidence_id(event)
                current = _require_active(states, evidence_id, event.event_type)
                replacement_id = _payload_text(event, "superseded_by")
                if replacement_id == evidence_id:
                    raise ValueError("evidence cannot supersede itself")
                if replacement_id not in states:
                    raise ValueError(
                        f"replacement evidence {replacement_id!r} does not exist"
                    )
                states[evidence_id] = replace(
                    current,
                    status=EvidenceStatus.SUPERSEDED,
                    status_event_id=event.event_id,
                    status_sequence=event.sequence,
                    superseded_by=replacement_id,
                )

        projected = tuple(states.values())
        for state in projected:
            state.validate()
        return EvidenceProjection(revision=revision, evidence=projected)

    @staticmethod
    def _added_state(event: LedgerEvent) -> EvidenceState:
        evidence_id = _payload_text(event, "evidence_id")
        kind = _payload_text(event, "kind")
        summary = _payload_text(event, "summary")
        strength = _optional_float(event, "strength")
        uncertainty = _optional_float(event, "uncertainty")
        data = event.payload.get("data", {})
        if not isinstance(data, Mapping):
            raise ValueError("EVIDENCE_ADDED data must be a mapping")

        references = tuple(
            reference
            for reference in event.reference_ids
            if reference != evidence_id
        )
        state = EvidenceState(
            evidence_id=evidence_id,
            kind=kind,
            summary=summary,
            strength=strength,
            uncertainty=uncertainty,
            data=dict(data),
            source_reference_ids=references,
            thread_id=event.thread_id,
            attempt_id=event.attempt_id,
            created_event_id=event.event_id,
            created_sequence=event.sequence,
        )
        state.validate()
        return state


def _payload_text(event: LedgerEvent, key: str) -> str:
    value = event.payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(
            f"{event.event_type} requires non-empty string payload field {key!r}"
        )
    return value


def _optional_float(event: LedgerEvent, key: str) -> float | None:
    value = event.payload.get(key)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{event.event_type} field {key!r} must be numeric or null")
    resolved = float(value)
    if not math.isfinite(resolved):
        raise ValueError(f"{event.event_type} field {key!r} must be finite")
    return resolved


def _evidence_id(event: LedgerEvent) -> str:
    payload_id = event.payload.get("evidence_id")
    if payload_id is not None:
        return _payload_text(event, "evidence_id")
    if not event.reference_ids:
        raise ValueError(f"{event.event_type} requires an evidence reference")
    evidence_id = event.reference_ids[0]
    if not evidence_id or not evidence_id.strip():
        raise ValueError(f"{event.event_type} evidence reference must be non-empty")
    return evidence_id


def _require_active(
    states: Mapping[str, EvidenceState],
    evidence_id: str,
    event_type: str,
) -> EvidenceState:
    try:
        state = states[evidence_id]
    except KeyError as error:
        raise ValueError(
            f"{event_type} references unknown evidence {evidence_id!r}"
        ) from error
    if state.status is not EvidenceStatus.ACTIVE:
        raise ValueError(
            f"{event_type} requires active evidence, got {state.status.value}"
        )
    return state
