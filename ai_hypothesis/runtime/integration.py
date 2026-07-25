"""Minimal information-integration tracking over the append-only Research Ledger.

This is intentionally not hierarchical integration. It provides the scale-invariant
bookkeeping needed to know which generated evidence is still awaiting a disposition,
which compact knowledge deltas were produced, and when backlog should apply
backpressure to Scheduler v0.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from collections.abc import Sequence

from .contracts import KnowledgeDelta, LedgerEvent
from .ledger import SQLiteResearchLedger


class IntegrationDisposition(str, Enum):
    INTEGRATED = "INTEGRATED"
    DUPLICATE = "DUPLICATE"
    IRRELEVANT = "IRRELEVANT"
    INVALID = "INVALID"
    LOCAL_ONLY = "LOCAL_ONLY"


@dataclass(frozen=True, slots=True)
class IntegrationBackpressureConfig:
    max_backlog_count: int = 1_000
    max_backlog_age_sequences: int = 10_000

    def validate(self) -> None:
        if self.max_backlog_count < 0:
            raise ValueError("max_backlog_count must be non-negative")
        if self.max_backlog_age_sequences < 0:
            raise ValueError("max_backlog_age_sequences must be non-negative")


@dataclass(frozen=True, slots=True)
class IntegrationSnapshot:
    revision: int
    evidence_count: int
    dispositioned_evidence_count: int
    backlog_evidence_ids: tuple[str, ...]
    knowledge_delta_count: int
    oldest_backlog_age_sequences: int

    @property
    def backlog_count(self) -> int:
        return len(self.backlog_evidence_ids)


class IntegrationTracker:
    """Record integration outcomes and derive deterministic backlog/backpressure state."""

    def __init__(
        self,
        ledger: SQLiteResearchLedger,
        config: IntegrationBackpressureConfig | None = None,
    ) -> None:
        self.ledger = ledger
        self.config = config or IntegrationBackpressureConfig()
        self.config.validate()

    def record_disposition(
        self,
        evidence_ids: Sequence[str],
        disposition: IntegrationDisposition,
        *,
        reason: str | None = None,
        thread_id: str | None = None,
    ) -> LedgerEvent:
        resolved_ids = tuple(evidence_ids)
        if not resolved_ids:
            raise ValueError("at least one evidence ID is required")
        if any(not evidence_id or not evidence_id.strip() for evidence_id in resolved_ids):
            raise ValueError("evidence IDs must be non-empty")
        payload = {"disposition": disposition.value}
        if reason is not None:
            if not reason.strip():
                raise ValueError("reason must be non-empty when provided")
            payload["reason"] = reason
        return self.ledger.append_event(
            event_type="INTEGRATION_DISPOSITION_RECORDED",
            thread_id=thread_id,
            reference_ids=resolved_ids,
            payload=payload,
        )

    def record_knowledge_delta(self, delta: KnowledgeDelta) -> LedgerEvent:
        delta.validate()
        references = (delta.delta_id, *delta.reference_ids)
        return self.ledger.append_event(
            event_type="KNOWLEDGE_DELTA_RECORDED",
            thread_id=delta.thread_id,
            reference_ids=references,
            parent_event_ids=delta.causal_event_ids,
            payload={
                "delta_id": delta.delta_id,
                "kind": delta.kind,
                "summary": delta.summary,
            },
        )

    def snapshot(self) -> IntegrationSnapshot:
        return self.project(self.ledger.read_events())

    def project(self, events: Sequence[LedgerEvent]) -> IntegrationSnapshot:
        evidence_sequences: dict[str, int] = {}
        dispositioned: set[str] = set()
        knowledge_delta_count = 0
        revision = 0
        previous_sequence = -1

        for event in events:
            event.validate()
            if event.sequence <= previous_sequence:
                raise ValueError("events must be in strictly increasing sequence order")
            previous_sequence = event.sequence
            revision = event.sequence

            if event.event_type == "EVIDENCE_ADDED":
                evidence_id = event.payload.get("evidence_id")
                if isinstance(evidence_id, str) and evidence_id:
                    evidence_sequences.setdefault(evidence_id, event.sequence)
            elif event.event_type == "INTEGRATION_DISPOSITION_RECORDED":
                dispositioned.update(event.reference_ids)
            elif event.event_type == "KNOWLEDGE_DELTA_RECORDED":
                knowledge_delta_count += 1

        backlog = tuple(
            evidence_id
            for evidence_id in evidence_sequences
            if evidence_id not in dispositioned
        )
        if backlog:
            oldest_sequence = min(evidence_sequences[evidence_id] for evidence_id in backlog)
            age = max(0, revision - oldest_sequence)
        else:
            age = 0

        return IntegrationSnapshot(
            revision=revision,
            evidence_count=len(evidence_sequences),
            dispositioned_evidence_count=sum(
                1 for evidence_id in evidence_sequences if evidence_id in dispositioned
            ),
            backlog_evidence_ids=backlog,
            knowledge_delta_count=knowledge_delta_count,
            oldest_backlog_age_sequences=age,
        )

    def is_backpressured(self) -> bool:
        snapshot = self.snapshot()
        return (
            snapshot.backlog_count > self.config.max_backlog_count
            or snapshot.oldest_backlog_age_sequences
            > self.config.max_backlog_age_sequences
        )
