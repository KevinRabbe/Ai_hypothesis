"""Rebuild bounded Work Thread state from append-only ledger events."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from .contracts import LedgerEvent, ProjectedState, WorkPurpose


class ThreadStateProjector:
    """Deterministically fold one thread's ledger history into current state."""

    def project(self, events: Iterable[LedgerEvent], *, thread_id: str) -> ProjectedState:
        if not thread_id:
            raise ValueError("thread_id must be non-empty")

        created = False
        objective = ""
        status = "ACTIVE"
        purpose = WorkPurpose.EXPLORE
        revision = 0
        previous_sequence = -1
        references: list[str] = []
        hypotheses: list[str] = []
        contradictions: list[str] = []
        open_questions: list[str] = []
        dependencies: list[str] = []
        metadata: dict[str, Any] = {}

        for event in events:
            event.validate()
            if event.sequence <= previous_sequence:
                raise ValueError("events must be supplied in strictly increasing sequence order")
            previous_sequence = event.sequence

            if event.thread_id != thread_id:
                continue

            revision = event.sequence
            _extend_unique(references, event.reference_ids)

            if event.event_type == "THREAD_CREATED":
                if created:
                    raise ValueError(f"thread {thread_id!r} was created more than once")
                objective = _require_payload_text(event, "objective")
                raw_purpose = event.payload.get("purpose", WorkPurpose.EXPLORE.value)
                try:
                    purpose = WorkPurpose(str(raw_purpose))
                except ValueError as error:
                    raise ValueError(f"invalid thread purpose {raw_purpose!r}") from error
                status = str(event.payload.get("status", "ACTIVE"))
                if not status:
                    raise ValueError("THREAD_CREATED status must be non-empty")
                created = True
            elif event.event_type == "THREAD_PURPOSE_SET":
                purpose = WorkPurpose(_require_payload_text(event, "purpose"))
            elif event.event_type == "THREAD_STATUS_SET":
                status = _require_payload_text(event, "status")
            elif event.event_type == "THREAD_PAUSED":
                status = "PAUSED"
            elif event.event_type == "THREAD_COMPLETED":
                status = "COMPLETE"
            elif event.event_type == "HYPOTHESIS_PROPOSED":
                _extend_unique(hypotheses, event.reference_ids)
            elif event.event_type == "HYPOTHESIS_REJECTED":
                _remove_all(hypotheses, event.reference_ids)
            elif event.event_type == "CONTRADICTION_FOUND":
                _extend_unique(contradictions, event.reference_ids)
            elif event.event_type == "CONTRADICTION_RESOLVED":
                _remove_all(contradictions, event.reference_ids)
            elif event.event_type == "OPEN_QUESTION_ADDED":
                _extend_unique(open_questions, (_require_payload_text(event, "question"),))
            elif event.event_type == "OPEN_QUESTION_RESOLVED":
                _remove_all(open_questions, (_require_payload_text(event, "question"),))
            elif event.event_type == "DEPENDENCY_ADDED":
                _extend_unique(dependencies, event.reference_ids)
            elif event.event_type == "DEPENDENCY_REMOVED":
                _remove_all(dependencies, event.reference_ids)
            elif event.event_type == "THREAD_METADATA_UPDATED":
                metadata.update(event.payload)

        if not created:
            raise ValueError(f"thread {thread_id!r} has no THREAD_CREATED event")

        state = ProjectedState(
            revision=revision,
            thread_id=thread_id,
            objective=objective,
            status=status,
            purpose=purpose,
            reference_ids=tuple(references),
            hypothesis_ids=tuple(hypotheses),
            contradiction_ids=tuple(contradictions),
            open_questions=tuple(open_questions),
            dependency_thread_ids=tuple(dependencies),
            metadata=metadata,
        )
        state.validate()
        return state


def _require_payload_text(event: LedgerEvent, key: str) -> str:
    value = event.payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{event.event_type} requires non-empty string payload field {key!r}")
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
