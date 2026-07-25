"""Rebuild bounded Work Thread state from append-only ledger events."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

from .contracts import LedgerEvent, ProjectedState, WorkPurpose


@dataclass(slots=True)
class _MutableThreadState:
    created: bool = False
    objective: str = ""
    status: str = "ACTIVE"
    purpose: WorkPurpose = WorkPurpose.EXPLORE
    revision: int = 0
    references: list[str] = field(default_factory=list)
    hypotheses: list[str] = field(default_factory=list)
    contradictions: list[str] = field(default_factory=list)
    open_questions: list[str] = field(default_factory=list)
    dependencies: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


class ThreadStateProjector:
    """Deterministically fold ledger history into rebuildable Work Thread views."""

    def project(self, events: Iterable[LedgerEvent], *, thread_id: str) -> ProjectedState:
        if not thread_id:
            raise ValueError("thread_id must be non-empty")
        for state in self.project_all(events):
            if state.thread_id == thread_id:
                return state
        raise ValueError(f"thread {thread_id!r} has no THREAD_CREATED event")

    def project_all(self, events: Iterable[LedgerEvent]) -> tuple[ProjectedState, ...]:
        """Project every created Work Thread with one ordered pass over shared history.

        Thread-tagged evidence or diagnostic events that never received a
        ``THREAD_CREATED`` event are not schedulable Work Threads and are ignored in
        the returned view.
        """

        mutable: dict[str, _MutableThreadState] = {}
        creation_order: list[str] = []
        previous_sequence = -1

        for event in events:
            event.validate()
            if event.sequence <= previous_sequence:
                raise ValueError(
                    "events must be supplied in strictly increasing sequence order"
                )
            previous_sequence = event.sequence
            if event.thread_id is None:
                continue

            state = mutable.setdefault(event.thread_id, _MutableThreadState())
            state.revision = event.sequence
            _extend_unique(state.references, event.reference_ids)
            self._apply_event(event.thread_id, state, event, creation_order)

        return tuple(
            self._freeze(thread_id, mutable[thread_id]) for thread_id in creation_order
        )

    @staticmethod
    def _apply_event(
        thread_id: str,
        state: _MutableThreadState,
        event: LedgerEvent,
        creation_order: list[str],
    ) -> None:
        if event.event_type == "THREAD_CREATED":
            if state.created:
                raise ValueError(f"thread {thread_id!r} was created more than once")
            state.objective = _require_payload_text(event, "objective")
            raw_purpose = event.payload.get("purpose", WorkPurpose.EXPLORE.value)
            try:
                state.purpose = WorkPurpose(str(raw_purpose))
            except ValueError as error:
                raise ValueError(f"invalid thread purpose {raw_purpose!r}") from error
            state.status = str(event.payload.get("status", "ACTIVE"))
            if not state.status:
                raise ValueError("THREAD_CREATED status must be non-empty")
            state.created = True
            creation_order.append(thread_id)
        elif event.event_type == "THREAD_PURPOSE_SET":
            state.purpose = WorkPurpose(_require_payload_text(event, "purpose"))
        elif event.event_type == "THREAD_STATUS_SET":
            state.status = _require_payload_text(event, "status")
        elif event.event_type == "THREAD_PAUSED":
            state.status = "PAUSED"
        elif event.event_type == "THREAD_COMPLETED":
            state.status = "COMPLETE"
        elif event.event_type == "HYPOTHESIS_PROPOSED":
            _extend_unique(state.hypotheses, event.reference_ids)
        elif event.event_type == "HYPOTHESIS_REJECTED":
            _remove_all(state.hypotheses, event.reference_ids)
        elif event.event_type == "CONTRADICTION_FOUND":
            _extend_unique(state.contradictions, event.reference_ids)
        elif event.event_type == "CONTRADICTION_RESOLVED":
            _remove_all(state.contradictions, event.reference_ids)
        elif event.event_type == "OPEN_QUESTION_ADDED":
            _extend_unique(
                state.open_questions,
                (_require_payload_text(event, "question"),),
            )
        elif event.event_type == "OPEN_QUESTION_RESOLVED":
            _remove_all(
                state.open_questions,
                (_require_payload_text(event, "question"),),
            )
        elif event.event_type == "DEPENDENCY_ADDED":
            _extend_unique(state.dependencies, event.reference_ids)
        elif event.event_type == "DEPENDENCY_REMOVED":
            _remove_all(state.dependencies, event.reference_ids)
        elif event.event_type == "THREAD_METADATA_UPDATED":
            state.metadata.update(event.payload)

    @staticmethod
    def _freeze(thread_id: str, state: _MutableThreadState) -> ProjectedState:
        projected = ProjectedState(
            revision=state.revision,
            thread_id=thread_id,
            objective=state.objective,
            status=state.status,
            purpose=state.purpose,
            reference_ids=tuple(state.references),
            hypothesis_ids=tuple(state.hypotheses),
            contradiction_ids=tuple(state.contradictions),
            open_questions=tuple(state.open_questions),
            dependency_thread_ids=tuple(state.dependencies),
            metadata=dict(state.metadata),
        )
        projected.validate()
        return projected


def _require_payload_text(event: LedgerEvent, key: str) -> str:
    value = event.payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(
            f"{event.event_type} requires non-empty string payload field {key!r}"
        )
    return value


def _extend_unique(target: list[str], values: Iterable[str]) -> None:
    present = set(target)
    for value in values:
        if value and value not in present:
            target.append(value)
            present.add(value)


def _remove_all(target: list[str], values: Iterable[str]) -> None:
    removals = set(values)
    if not removals:
        return
    target[:] = [value for value in target if value not in removals]
