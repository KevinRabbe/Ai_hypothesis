"""Rebuild compact current knowledge from append-only delta and assessment events."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, replace
from enum import Enum

from .contracts import KnowledgeAssessmentKind, LedgerEvent


class KnowledgeStatus(str, Enum):
    PROVISIONAL = "PROVISIONAL"
    VERIFIED = "VERIFIED"
    DISPUTED = "DISPUTED"
    RETRACTED = "RETRACTED"


_ASSESSMENT_TO_STATUS = {
    KnowledgeAssessmentKind.VERIFIED: KnowledgeStatus.VERIFIED,
    KnowledgeAssessmentKind.DISPUTED: KnowledgeStatus.DISPUTED,
    KnowledgeAssessmentKind.RETRACTED: KnowledgeStatus.RETRACTED,
}


@dataclass(frozen=True, slots=True)
class KnowledgeRecord:
    delta_id: str
    kind: str
    summary: str
    source_reference_ids: tuple[str, ...]
    causal_event_ids: tuple[str, ...]
    thread_id: str | None
    created_event_id: str
    created_sequence: int
    status: KnowledgeStatus = KnowledgeStatus.PROVISIONAL
    assessment_reason: str | None = None
    assessment_event_id: str | None = None
    assessment_sequence: int | None = None

    @property
    def is_active(self) -> bool:
        return self.status is not KnowledgeStatus.RETRACTED


@dataclass(frozen=True, slots=True)
class KnowledgeSnapshot:
    revision: int
    records: tuple[KnowledgeRecord, ...]

    def get(self, delta_id: str) -> KnowledgeRecord | None:
        for record in self.records:
            if record.delta_id == delta_id:
                return record
        return None

    @property
    def active_records(self) -> tuple[KnowledgeRecord, ...]:
        return tuple(record for record in self.records if record.is_active)

    @property
    def verified_records(self) -> tuple[KnowledgeRecord, ...]:
        return tuple(
            record for record in self.records if record.status is KnowledgeStatus.VERIFIED
        )


class KnowledgeStateProjector:
    """Fold knowledge deltas and later assessments into current rebuildable state."""

    def project(
        self,
        events: Iterable[LedgerEvent],
        *,
        thread_id: str | None = None,
    ) -> KnowledgeSnapshot:
        records: dict[str, KnowledgeRecord] = {}
        order: list[str] = []
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

            if event.event_type == "KNOWLEDGE_DELTA_RECORDED":
                record = self._record_from_delta_event(event)
                if record.delta_id in records:
                    raise ValueError(
                        f"knowledge delta {record.delta_id!r} was recorded more than once"
                    )
                records[record.delta_id] = record
                order.append(record.delta_id)
            elif event.event_type == "KNOWLEDGE_ASSESSMENT_RECORDED":
                assessment = self._assessment_from_event(event)
                for delta_id in event.reference_ids:
                    try:
                        current = records[delta_id]
                    except KeyError as error:
                        raise ValueError(
                            f"knowledge assessment references unknown delta {delta_id!r}"
                        ) from error
                    records[delta_id] = replace(
                        current,
                        status=_ASSESSMENT_TO_STATUS[assessment],
                        assessment_reason=self._optional_reason(event),
                        assessment_event_id=event.event_id,
                        assessment_sequence=event.sequence,
                    )

        projected = tuple(records[delta_id] for delta_id in order)
        if thread_id is not None:
            if not thread_id:
                raise ValueError("thread_id must be non-empty when provided")
            projected = tuple(
                record for record in projected if record.thread_id == thread_id
            )
        return KnowledgeSnapshot(revision=revision, records=projected)

    @staticmethod
    def _record_from_delta_event(event: LedgerEvent) -> KnowledgeRecord:
        delta_id = _payload_text(event, "delta_id")
        kind = _payload_text(event, "kind")
        summary = _payload_text(event, "summary")
        source_reference_ids = _payload_string_tuple(
            event,
            "source_reference_ids",
            fallback=tuple(
                reference_id
                for reference_id in event.reference_ids
                if reference_id != delta_id
            ),
        )
        causal_event_ids = _payload_string_tuple(
            event,
            "causal_event_ids",
            fallback=(),
        )
        return KnowledgeRecord(
            delta_id=delta_id,
            kind=kind,
            summary=summary,
            source_reference_ids=source_reference_ids,
            causal_event_ids=causal_event_ids,
            thread_id=event.thread_id,
            created_event_id=event.event_id,
            created_sequence=event.sequence,
        )

    @staticmethod
    def _assessment_from_event(event: LedgerEvent) -> KnowledgeAssessmentKind:
        raw = _payload_text(event, "assessment")
        try:
            return KnowledgeAssessmentKind(raw)
        except ValueError as error:
            raise ValueError(f"invalid knowledge assessment {raw!r}") from error

    @staticmethod
    def _optional_reason(event: LedgerEvent) -> str | None:
        value = event.payload.get("reason")
        if value is None:
            return None
        if not isinstance(value, str) or not value.strip():
            raise ValueError("knowledge assessment reason must be a non-empty string")
        return value


def _payload_text(event: LedgerEvent, key: str) -> str:
    value = event.payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(
            f"{event.event_type} requires non-empty string payload field {key!r}"
        )
    return value


def _payload_string_tuple(
    event: LedgerEvent,
    key: str,
    *,
    fallback: tuple[str, ...],
) -> tuple[str, ...]:
    if key not in event.payload:
        return fallback
    value = event.payload[key]
    if not isinstance(value, list) or any(
        not isinstance(item, str) or not item for item in value
    ):
        raise ValueError(f"{event.event_type} payload {key!r} must be a string list")
    return tuple(value)
