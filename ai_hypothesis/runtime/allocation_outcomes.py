"""Project observable scheduler-allocation outcomes from append-only ledger history.

This module deliberately does not compute a reward, usefulness score, worker ranking,
or uncertainty reduction. It only joins scheduler decisions to the attempts and durable
outputs they caused so later analysis can define workload-specific value without losing
raw provenance.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any

from .contracts import LedgerEvent


_TERMINAL_ATTEMPT_EVENTS = frozenset(
    {
        "ATTEMPT_COMPLETED",
        "ATTEMPT_PARTIAL",
        "ATTEMPT_FAILED",
        "ATTEMPT_CRASHED",
        "ATTEMPT_INVALID_RESULT",
    }
)


@dataclass(frozen=True, slots=True)
class AttemptOutcome:
    scheduler_decision_id: str
    attempt_id: str
    work_item_id: str
    thread_id: str
    worker_id: str
    purpose: str
    projection_revision: int
    terminal_event_type: str | None
    progress_made: bool | None
    resource_usage: Mapping[str, Any]
    evidence_ids: tuple[str, ...]
    knowledge_delta_ids: tuple[str, ...]
    assessed_delta_ids: tuple[str, ...]
    dispositioned_evidence_ids: tuple[str, ...]
    contradiction_ids: tuple[str, ...]
    possibility_elimination_ids: tuple[str, ...]
    open_questions: tuple[str, ...]
    followups: tuple[str, ...]

    @property
    def evidence_count(self) -> int:
        return len(self.evidence_ids)

    @property
    def knowledge_delta_count(self) -> int:
        return len(self.knowledge_delta_ids)

    @property
    def knowledge_assessment_count(self) -> int:
        return len(self.assessed_delta_ids)

    @property
    def evidence_disposition_count(self) -> int:
        return len(self.dispositioned_evidence_ids)

    @property
    def contradiction_count(self) -> int:
        return len(self.contradiction_ids)

    @property
    def possibility_elimination_count(self) -> int:
        return len(self.possibility_elimination_ids)

    @property
    def open_question_count(self) -> int:
        return len(self.open_questions)

    @property
    def followup_count(self) -> int:
        return len(self.followups)


@dataclass(frozen=True, slots=True)
class AllocationOutcome:
    scheduler_decision_id: str
    thread_id: str | None
    action: str | None
    purpose: str | None
    allocated_width: int | None
    reason_codes: tuple[str, ...]
    projection_revision: int | None
    integration_backpressure: bool | None
    attempts: tuple[AttemptOutcome, ...]

    @property
    def attempt_count(self) -> int:
        return len(self.attempts)

    @property
    def evidence_count(self) -> int:
        return sum(attempt.evidence_count for attempt in self.attempts)

    @property
    def knowledge_delta_count(self) -> int:
        return sum(attempt.knowledge_delta_count for attempt in self.attempts)

    @property
    def knowledge_assessment_count(self) -> int:
        return sum(attempt.knowledge_assessment_count for attempt in self.attempts)

    @property
    def evidence_disposition_count(self) -> int:
        return sum(attempt.evidence_disposition_count for attempt in self.attempts)

    @property
    def contradiction_count(self) -> int:
        return sum(attempt.contradiction_count for attempt in self.attempts)

    @property
    def possibility_elimination_count(self) -> int:
        return sum(attempt.possibility_elimination_count for attempt in self.attempts)

    @property
    def open_question_count(self) -> int:
        return sum(attempt.open_question_count for attempt in self.attempts)

    @property
    def followup_count(self) -> int:
        return sum(attempt.followup_count for attempt in self.attempts)


@dataclass(slots=True)
class _MutableAttempt:
    scheduler_decision_id: str
    attempt_id: str
    work_item_id: str
    thread_id: str
    worker_id: str
    purpose: str
    projection_revision: int
    terminal_event_type: str | None = None
    progress_made: bool | None = None
    resource_usage: dict[str, Any] = field(default_factory=dict)
    evidence_ids: list[str] = field(default_factory=list)
    knowledge_delta_ids: list[str] = field(default_factory=list)
    assessed_delta_ids: list[str] = field(default_factory=list)
    dispositioned_evidence_ids: list[str] = field(default_factory=list)
    contradiction_ids: list[str] = field(default_factory=list)
    possibility_elimination_ids: list[str] = field(default_factory=list)
    open_questions: list[str] = field(default_factory=list)
    followups: list[str] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class _DecisionMetadata:
    thread_id: str | None
    action: str | None
    purpose: str | None
    allocated_width: int | None
    reason_codes: tuple[str, ...]
    projection_revision: int | None
    integration_backpressure: bool | None


class AllocationOutcomeProjector:
    """Join traced scheduler decisions to observable attempt outcomes in one pass."""

    def project(self, events: Iterable[LedgerEvent]) -> tuple[AllocationOutcome, ...]:
        decisions: dict[str, _DecisionMetadata] = {}
        decision_order: list[str] = []
        attempts: dict[str, _MutableAttempt] = {}
        attempts_by_decision: dict[str, list[str]] = {}
        previous_sequence = -1

        for event in events:
            event.validate()
            if event.sequence <= previous_sequence:
                raise ValueError("events must be supplied in strictly increasing sequence order")
            previous_sequence = event.sequence

            if event.event_type == "SCHEDULER_DECISION_RECORDED":
                decision_id = _required_text(event.payload, "decision_id", event.event_type)
                if decision_id in decisions:
                    raise ValueError(f"scheduler decision {decision_id!r} was recorded more than once")
                decisions[decision_id] = self._decision_metadata(event)
                decision_order.append(decision_id)
                continue

            if event.event_type == "ATTEMPT_STARTED":
                decision_id = event.payload.get("scheduler_decision_id")
                if decision_id is None:
                    # Non-scheduler/manual attempts remain valid ledger history but are
                    # outside this projection's allocation-analysis scope.
                    continue
                if not isinstance(decision_id, str) or not decision_id.strip():
                    raise ValueError("ATTEMPT_STARTED scheduler_decision_id must be text or null")
                if event.attempt_id is None:
                    raise ValueError("ATTEMPT_STARTED requires attempt_id")
                if event.thread_id is None:
                    raise ValueError("ATTEMPT_STARTED requires thread_id")
                if event.attempt_id in attempts:
                    raise ValueError(f"attempt {event.attempt_id!r} was started more than once")
                attempts[event.attempt_id] = _MutableAttempt(
                    scheduler_decision_id=decision_id,
                    attempt_id=event.attempt_id,
                    work_item_id=_required_text(event.payload, "work_item_id", event.event_type),
                    thread_id=event.thread_id,
                    worker_id=_required_text(event.payload, "worker_id", event.event_type),
                    purpose=_required_text(event.payload, "purpose", event.event_type),
                    projection_revision=_required_non_negative_int(
                        event.payload, "projection_revision", event.event_type
                    ),
                )
                attempts_by_decision.setdefault(decision_id, []).append(event.attempt_id)
                if decision_id not in decisions and decision_id not in decision_order:
                    # Explicit externally supplied provenance may not have a local
                    # scheduler trace. Preserve it as an outcome group with unknown
                    # decision metadata rather than dropping the attempt.
                    decision_order.append(decision_id)
                continue

            if event.attempt_id is None or event.attempt_id not in attempts:
                continue
            attempt = attempts[event.attempt_id]
            self._apply_attempt_event(attempt, event)

        outcomes: list[AllocationOutcome] = []
        for decision_id in decision_order:
            attempt_ids = attempts_by_decision.get(decision_id, ())
            if not attempt_ids:
                # A scheduler decision that never reached worker execution is still
                # useful trace data, but it has no observable attempt outcome yet.
                projected_attempts: tuple[AttemptOutcome, ...] = ()
            else:
                projected_attempts = tuple(
                    self._freeze_attempt(attempts[attempt_id]) for attempt_id in attempt_ids
                )
            metadata = decisions.get(
                decision_id,
                _DecisionMetadata(None, None, None, None, (), None, None),
            )
            outcomes.append(
                AllocationOutcome(
                    scheduler_decision_id=decision_id,
                    thread_id=metadata.thread_id,
                    action=metadata.action,
                    purpose=metadata.purpose,
                    allocated_width=metadata.allocated_width,
                    reason_codes=metadata.reason_codes,
                    projection_revision=metadata.projection_revision,
                    integration_backpressure=metadata.integration_backpressure,
                    attempts=projected_attempts,
                )
            )
        return tuple(outcomes)

    def for_decision(
        self,
        events: Iterable[LedgerEvent],
        decision_id: str,
    ) -> AllocationOutcome | None:
        if not decision_id or not decision_id.strip():
            raise ValueError("decision_id must be non-empty")
        for outcome in self.project(events):
            if outcome.scheduler_decision_id == decision_id:
                return outcome
        return None

    @staticmethod
    def _decision_metadata(event: LedgerEvent) -> _DecisionMetadata:
        payload = event.payload
        width = payload.get("width")
        if isinstance(width, bool) or not isinstance(width, int) or width <= 0:
            raise ValueError("SCHEDULER_DECISION_RECORDED width must be a positive integer")
        revision = _required_non_negative_int(
            payload, "projection_revision", event.event_type
        )
        reason_codes = payload.get("reason_codes", [])
        if not isinstance(reason_codes, list) or any(
            not isinstance(reason, str) or not reason for reason in reason_codes
        ):
            raise ValueError("scheduler reason_codes must be a string list")
        backpressure = payload.get("integration_backpressure")
        if not isinstance(backpressure, bool):
            raise ValueError("scheduler integration_backpressure must be boolean")
        purpose = payload.get("purpose")
        if purpose is not None and (not isinstance(purpose, str) or not purpose):
            raise ValueError("scheduler purpose must be text or null")
        return _DecisionMetadata(
            thread_id=event.thread_id,
            action=_required_text(payload, "action", event.event_type),
            purpose=purpose,
            allocated_width=width,
            reason_codes=tuple(reason_codes),
            projection_revision=revision,
            integration_backpressure=backpressure,
        )

    @staticmethod
    def _apply_attempt_event(attempt: _MutableAttempt, event: LedgerEvent) -> None:
        if event.event_type in _TERMINAL_ATTEMPT_EVENTS:
            if attempt.terminal_event_type is not None:
                raise ValueError(f"attempt {attempt.attempt_id!r} has multiple terminal events")
            attempt.terminal_event_type = event.event_type
            if event.event_type in {"ATTEMPT_COMPLETED", "ATTEMPT_PARTIAL", "ATTEMPT_FAILED"}:
                progress = event.payload.get("progress_made")
                if not isinstance(progress, bool):
                    raise ValueError(f"{event.event_type} progress_made must be boolean")
                usage = event.payload.get("resource_usage", {})
                if not isinstance(usage, Mapping):
                    raise ValueError(f"{event.event_type} resource_usage must be a mapping")
                attempt.progress_made = progress
                attempt.resource_usage = dict(usage)
            return

        if event.event_type == "EVIDENCE_ADDED":
            evidence_id = _required_text(event.payload, "evidence_id", event.event_type)
            _append_unique(attempt.evidence_ids, (evidence_id,))
        elif event.event_type == "KNOWLEDGE_DELTA_RECORDED":
            delta_id = _required_text(event.payload, "delta_id", event.event_type)
            _append_unique(attempt.knowledge_delta_ids, (delta_id,))
        elif event.event_type == "KNOWLEDGE_ASSESSMENT_RECORDED":
            _append_unique(attempt.assessed_delta_ids, event.reference_ids)
        elif event.event_type == "INTEGRATION_DISPOSITION_RECORDED":
            _append_unique(attempt.dispositioned_evidence_ids, event.reference_ids)
        elif event.event_type == "CONTRADICTION_FOUND":
            _append_unique(attempt.contradiction_ids, event.reference_ids)
        elif event.event_type == "POSSIBILITY_ELIMINATED":
            _append_unique(attempt.possibility_elimination_ids, event.reference_ids)
        elif event.event_type == "OPEN_QUESTION_ADDED":
            question = _required_text(event.payload, "question", event.event_type)
            attempt.open_questions.append(question)
        elif event.event_type == "FOLLOWUP_REQUESTED":
            followup = _required_text(event.payload, "request", event.event_type)
            attempt.followups.append(followup)

    @staticmethod
    def _freeze_attempt(attempt: _MutableAttempt) -> AttemptOutcome:
        return AttemptOutcome(
            scheduler_decision_id=attempt.scheduler_decision_id,
            attempt_id=attempt.attempt_id,
            work_item_id=attempt.work_item_id,
            thread_id=attempt.thread_id,
            worker_id=attempt.worker_id,
            purpose=attempt.purpose,
            projection_revision=attempt.projection_revision,
            terminal_event_type=attempt.terminal_event_type,
            progress_made=attempt.progress_made,
            resource_usage=dict(attempt.resource_usage),
            evidence_ids=tuple(attempt.evidence_ids),
            knowledge_delta_ids=tuple(attempt.knowledge_delta_ids),
            assessed_delta_ids=tuple(attempt.assessed_delta_ids),
            dispositioned_evidence_ids=tuple(attempt.dispositioned_evidence_ids),
            contradiction_ids=tuple(attempt.contradiction_ids),
            possibility_elimination_ids=tuple(attempt.possibility_elimination_ids),
            open_questions=tuple(attempt.open_questions),
            followups=tuple(attempt.followups),
        )


def _required_text(payload: Mapping[str, Any], key: str, event_type: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{event_type} requires non-empty text field {key!r}")
    return value


def _required_non_negative_int(
    payload: Mapping[str, Any], key: str, event_type: str
) -> int:
    value = payload.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{event_type} requires non-negative integer field {key!r}")
    return value


def _append_unique(target: list[str], values: Iterable[str]) -> None:
    present = set(target)
    for value in values:
        if not value or not value.strip():
            raise ValueError("outcome reference IDs must be non-empty")
        if value not in present:
            target.append(value)
            present.add(value)
