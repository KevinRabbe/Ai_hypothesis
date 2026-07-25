"""Rebuild verification attempts and evidence-level verification summaries.

Verification is represented as append-only state separate from evidence existence. A
worker contribution remains evidence until explicit verification events establish a
result; this projector does not promote anything to accepted knowledge by policy.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from enum import Enum
from typing import Any

from .contracts import LedgerEvent


class VerificationAttemptStatus(str, Enum):
    PENDING = "PENDING"
    CONFIRMED = "CONFIRMED"
    REJECTED = "REJECTED"
    INCONCLUSIVE = "INCONCLUSIVE"


class EvidenceVerificationStatus(str, Enum):
    UNVERIFIED = "UNVERIFIED"
    PENDING = "PENDING"
    CONFIRMED = "CONFIRMED"
    REJECTED = "REJECTED"
    CONFLICTED = "CONFLICTED"
    INCONCLUSIVE = "INCONCLUSIVE"


@dataclass(frozen=True, slots=True)
class VerificationAttempt:
    """Projected lifecycle of one immutable verification request."""

    verification_id: str
    target_evidence_id: str
    purpose: str
    request_event_id: str
    request_sequence: int
    thread_id: str | None
    status: VerificationAttemptStatus = VerificationAttemptStatus.PENDING
    terminal_event_id: str | None = None
    terminal_sequence: int | None = None
    result_evidence_ids: tuple[str, ...] = ()
    verifier_worker_id: str | None = None
    notes: str | None = None
    data: Mapping[str, Any] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.data is None:
            object.__setattr__(self, "data", {})

    def validate(self) -> None:
        for name, value in (
            ("verification_id", self.verification_id),
            ("target_evidence_id", self.target_evidence_id),
            ("purpose", self.purpose),
            ("request_event_id", self.request_event_id),
        ):
            if not value or not value.strip():
                raise ValueError(f"{name} must be non-empty")
        if self.request_sequence < 0:
            raise ValueError("request_sequence must be non-negative")
        if any(not value or not value.strip() for value in self.result_evidence_ids):
            raise ValueError("result_evidence_ids must not contain empty IDs")
        if self.status is VerificationAttemptStatus.PENDING:
            if self.terminal_event_id is not None or self.terminal_sequence is not None:
                raise ValueError("pending verification must not have a terminal event")
            if self.result_evidence_ids or self.notes is not None:
                raise ValueError("pending verification must not have terminal result data")
        else:
            if not self.terminal_event_id:
                raise ValueError("terminal verification requires terminal_event_id")
            if self.terminal_sequence is None or self.terminal_sequence < self.request_sequence:
                raise ValueError("terminal verification requires a valid terminal_sequence")
        if self.verifier_worker_id is not None and not self.verifier_worker_id.strip():
            raise ValueError("verifier_worker_id must be non-empty when present")
        if self.notes is not None and not self.notes.strip():
            raise ValueError("notes must be non-empty when present")


@dataclass(frozen=True, slots=True)
class EvidenceVerificationSummary:
    """All independent verification attempts associated with one evidence record."""

    evidence_id: str
    status: EvidenceVerificationStatus
    attempts: tuple[VerificationAttempt, ...]

    @property
    def pending_count(self) -> int:
        return sum(
            attempt.status is VerificationAttemptStatus.PENDING
            for attempt in self.attempts
        )

    @property
    def confirmed_count(self) -> int:
        return sum(
            attempt.status is VerificationAttemptStatus.CONFIRMED
            for attempt in self.attempts
        )

    @property
    def rejected_count(self) -> int:
        return sum(
            attempt.status is VerificationAttemptStatus.REJECTED
            for attempt in self.attempts
        )

    @property
    def inconclusive_count(self) -> int:
        return sum(
            attempt.status is VerificationAttemptStatus.INCONCLUSIVE
            for attempt in self.attempts
        )


@dataclass(frozen=True, slots=True)
class VerificationProjection:
    """Rebuildable verification view at one ledger revision."""

    revision: int
    attempts: tuple[VerificationAttempt, ...]

    def get_attempt(self, verification_id: str) -> VerificationAttempt:
        for attempt in self.attempts:
            if attempt.verification_id == verification_id:
                return attempt
        raise KeyError(verification_id)

    def select(
        self,
        *,
        target_evidence_id: str | None = None,
        status: VerificationAttemptStatus | None = None,
        thread_id: str | None = None,
    ) -> tuple[VerificationAttempt, ...]:
        return tuple(
            attempt
            for attempt in self.attempts
            if (
                target_evidence_id is None
                or attempt.target_evidence_id == target_evidence_id
            )
            and (status is None or attempt.status is status)
            and (thread_id is None or attempt.thread_id == thread_id)
        )

    def summary_for(self, evidence_id: str) -> EvidenceVerificationSummary:
        attempts = self.select(target_evidence_id=evidence_id)
        statuses = {attempt.status for attempt in attempts}
        if (
            VerificationAttemptStatus.CONFIRMED in statuses
            and VerificationAttemptStatus.REJECTED in statuses
        ):
            status = EvidenceVerificationStatus.CONFLICTED
        elif VerificationAttemptStatus.CONFIRMED in statuses:
            status = EvidenceVerificationStatus.CONFIRMED
        elif VerificationAttemptStatus.REJECTED in statuses:
            status = EvidenceVerificationStatus.REJECTED
        elif VerificationAttemptStatus.PENDING in statuses:
            status = EvidenceVerificationStatus.PENDING
        elif VerificationAttemptStatus.INCONCLUSIVE in statuses:
            status = EvidenceVerificationStatus.INCONCLUSIVE
        else:
            status = EvidenceVerificationStatus.UNVERIFIED
        return EvidenceVerificationSummary(
            evidence_id=evidence_id,
            status=status,
            attempts=attempts,
        )


class VerificationStateProjector:
    """Fold verification request/result events into independent attempt state."""

    _TERMINAL_TYPES = {
        "VERIFICATION_PASSED": VerificationAttemptStatus.CONFIRMED,
        "VERIFICATION_FAILED": VerificationAttemptStatus.REJECTED,
        "VERIFICATION_INCONCLUSIVE": VerificationAttemptStatus.INCONCLUSIVE,
    }

    def project(self, events: Iterable[LedgerEvent]) -> VerificationProjection:
        known_evidence_ids: set[str] = set()
        attempts: dict[str, VerificationAttempt] = {}
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
                known_evidence_ids.add(_payload_text(event, "evidence_id"))
                continue
            if event.event_type == "VERIFICATION_REQUESTED":
                attempt = self._request(event, known_evidence_ids)
                if attempt.verification_id in attempts:
                    raise ValueError(
                        f"verification {attempt.verification_id!r} was requested more than once"
                    )
                attempts[attempt.verification_id] = attempt
                continue
            terminal_status = self._TERMINAL_TYPES.get(event.event_type)
            if terminal_status is None:
                continue
            verification_id = _payload_text(event, "verification_id")
            try:
                current = attempts[verification_id]
            except KeyError as error:
                raise ValueError(
                    f"{event.event_type} references unknown verification "
                    f"{verification_id!r}"
                ) from error
            if current.status is not VerificationAttemptStatus.PENDING:
                raise ValueError(
                    f"verification {verification_id!r} is already terminal"
                )
            target_id = event.payload.get("target_evidence_id")
            if target_id is not None:
                resolved_target = _payload_text(event, "target_evidence_id")
                if resolved_target != current.target_evidence_id:
                    raise ValueError(
                        f"verification {verification_id!r} target does not match request"
                    )
            result_ids = _payload_string_tuple(event, "result_evidence_ids")
            missing_results = tuple(
                evidence_id
                for evidence_id in result_ids
                if evidence_id not in known_evidence_ids
            )
            if missing_results:
                raise ValueError(
                    "verification result references unknown evidence: "
                    + ", ".join(missing_results)
                )
            verifier_worker_id = _optional_text(event, "verifier_worker_id")
            notes = _optional_text(event, "notes")
            data = event.payload.get("data", {})
            if not isinstance(data, Mapping):
                raise ValueError(f"{event.event_type} data must be a mapping")
            attempts[verification_id] = replace(
                current,
                status=terminal_status,
                terminal_event_id=event.event_id,
                terminal_sequence=event.sequence,
                result_evidence_ids=result_ids,
                verifier_worker_id=verifier_worker_id,
                notes=notes,
                data=dict(data),
            )

        projected = tuple(attempts.values())
        for attempt in projected:
            attempt.validate()
        return VerificationProjection(revision=revision, attempts=projected)

    @staticmethod
    def _request(
        event: LedgerEvent,
        known_evidence_ids: set[str],
    ) -> VerificationAttempt:
        verification_id = _payload_text(event, "verification_id")
        target_evidence_id = _payload_text(event, "target_evidence_id")
        if target_evidence_id not in known_evidence_ids:
            raise ValueError(
                f"VERIFICATION_REQUESTED references unknown evidence "
                f"{target_evidence_id!r}"
            )
        purpose = _payload_text(event, "purpose")
        data = event.payload.get("data", {})
        if not isinstance(data, Mapping):
            raise ValueError("VERIFICATION_REQUESTED data must be a mapping")
        attempt = VerificationAttempt(
            verification_id=verification_id,
            target_evidence_id=target_evidence_id,
            purpose=purpose,
            request_event_id=event.event_id,
            request_sequence=event.sequence,
            thread_id=event.thread_id,
            data=dict(data),
        )
        attempt.validate()
        return attempt


def _payload_text(event: LedgerEvent, key: str) -> str:
    value = event.payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(
            f"{event.event_type} requires non-empty string payload field {key!r}"
        )
    return value


def _optional_text(event: LedgerEvent, key: str) -> str | None:
    value = event.payload.get(key)
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ValueError(
            f"{event.event_type} field {key!r} must be a non-empty string or null"
        )
    return value


def _payload_string_tuple(event: LedgerEvent, key: str) -> tuple[str, ...]:
    value = event.payload.get(key, ())
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"{event.event_type} field {key!r} must be a list of IDs")
    resolved = tuple(value)
    if any(not isinstance(item, str) or not item.strip() for item in resolved):
        raise ValueError(f"{event.event_type} field {key!r} contains an invalid ID")
    return resolved
