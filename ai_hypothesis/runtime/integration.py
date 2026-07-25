"""Minimal information-integration tracking over the append-only Research Ledger.

This is intentionally not hierarchical integration. It provides the scale-invariant
bookkeeping needed to know which generated evidence is still awaiting a disposition,
which compact knowledge deltas were produced, and when backlog should apply
backpressure to Scheduler v0.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from .contracts import EvidenceDispositionKind, KnowledgeDelta, LedgerEvent
from .ledger import SQLiteResearchLedger

# Backward-compatible runtime-facing name. The semantic enum lives in contracts so
# integration workers can return it directly through AttemptResult.
IntegrationDisposition = EvidenceDispositionKind


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


@dataclass(frozen=True, slots=True)
class IntegrationOverview:
    """One-pass global + per-thread integration projection for scheduler use."""

    global_snapshot: IntegrationSnapshot
    thread_snapshots: Mapping[str, IntegrationSnapshot]
    thread_pressure: Mapping[str, float]
    global_backpressured: bool

    def pressure_for(self, thread_id: str) -> float:
        return float(self.thread_pressure.get(thread_id, 0.0))


@dataclass(frozen=True, slots=True)
class PendingEvidence:
    """Compact recoverable evidence record eligible for integration work."""

    evidence_id: str
    event_id: str
    sequence: int
    thread_id: str | None
    kind: str
    summary: str
    source_reference_ids: tuple[str, ...]
    strength: float | None
    uncertainty: float | None
    data: Mapping[str, Any]

    def to_context_record(self) -> dict[str, Any]:
        return {
            "evidence_id": self.evidence_id,
            "event_id": self.event_id,
            "sequence": self.sequence,
            "thread_id": self.thread_id,
            "kind": self.kind,
            "summary": self.summary,
            "source_reference_ids": list(self.source_reference_ids),
            "strength": self.strength,
            "uncertainty": self.uncertainty,
            "data": dict(self.data),
        }


@dataclass(frozen=True, slots=True)
class IntegrationBatch:
    """Fixed-size oldest-first slice of the unresolved evidence backlog."""

    revision: int
    records: tuple[PendingEvidence, ...]

    @property
    def evidence_ids(self) -> tuple[str, ...]:
        return tuple(record.evidence_id for record in self.records)

    @property
    def causal_event_ids(self) -> tuple[str, ...]:
        return tuple(record.event_id for record in self.records)

    def to_context_records(self) -> tuple[dict[str, Any], ...]:
        return tuple(record.to_context_record() for record in self.records)


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
        disposition: EvidenceDispositionKind,
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

    def snapshot(self, *, thread_id: str | None = None) -> IntegrationSnapshot:
        overview = self.overview(self.ledger.read_events())
        if thread_id is None:
            return overview.global_snapshot
        return overview.thread_snapshots.get(
            thread_id,
            self._empty_snapshot(overview.global_snapshot.revision),
        )

    def overview(self, events: Sequence[LedgerEvent]) -> IntegrationOverview:
        """Project global and all per-thread integration state in one history pass."""

        global_evidence: dict[str, int] = {}
        thread_evidence: dict[str, dict[str, int]] = {}
        dispositioned: set[str] = set()
        global_delta_count = 0
        thread_delta_counts: dict[str, int] = {}
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
                    global_evidence.setdefault(evidence_id, event.sequence)
                    if event.thread_id is not None:
                        thread_evidence.setdefault(event.thread_id, {}).setdefault(
                            evidence_id,
                            event.sequence,
                        )
            elif event.event_type == "INTEGRATION_DISPOSITION_RECORDED":
                dispositioned.update(event.reference_ids)
            elif event.event_type == "KNOWLEDGE_DELTA_RECORDED":
                global_delta_count += 1
                if event.thread_id is not None:
                    thread_delta_counts[event.thread_id] = (
                        thread_delta_counts.get(event.thread_id, 0) + 1
                    )

        global_snapshot = self._build_snapshot(
            evidence_sequences=global_evidence,
            dispositioned=dispositioned,
            knowledge_delta_count=global_delta_count,
            revision=revision,
        )
        thread_ids = set(thread_evidence) | set(thread_delta_counts)
        thread_snapshots = {
            thread_id: self._build_snapshot(
                evidence_sequences=thread_evidence.get(thread_id, {}),
                dispositioned=dispositioned,
                knowledge_delta_count=thread_delta_counts.get(thread_id, 0),
                revision=revision,
            )
            for thread_id in thread_ids
        }
        thread_pressure = {
            thread_id: self._pressure_for_snapshot(snapshot)
            for thread_id, snapshot in thread_snapshots.items()
        }
        return IntegrationOverview(
            global_snapshot=global_snapshot,
            thread_snapshots=thread_snapshots,
            thread_pressure=thread_pressure,
            global_backpressured=self._is_snapshot_backpressured(global_snapshot),
        )

    def pending_batch(
        self,
        *,
        limit: int,
        thread_id: str | None = None,
    ) -> IntegrationBatch:
        """Return at most ``limit`` oldest unresolved evidence records."""

        if limit <= 0:
            raise ValueError("limit must be positive")
        events = self.ledger.read_events()
        dispositioned = self._dispositioned_ids(events)
        records: list[PendingEvidence] = []
        revision = events[-1].sequence if events else 0

        for event in events:
            if event.event_type != "EVIDENCE_ADDED":
                continue
            if thread_id is not None and event.thread_id != thread_id:
                continue
            evidence_id = event.payload.get("evidence_id")
            if not isinstance(evidence_id, str) or not evidence_id:
                continue
            if evidence_id in dispositioned:
                continue

            references = tuple(
                reference_id
                for reference_id in event.reference_ids
                if reference_id != evidence_id
            )
            data = event.payload.get("data")
            records.append(
                PendingEvidence(
                    evidence_id=evidence_id,
                    event_id=event.event_id,
                    sequence=event.sequence,
                    thread_id=event.thread_id,
                    kind=str(event.payload.get("kind", "UNKNOWN")),
                    summary=str(event.payload.get("summary", "")),
                    source_reference_ids=references,
                    strength=self._optional_float(event.payload.get("strength")),
                    uncertainty=self._optional_float(event.payload.get("uncertainty")),
                    data=dict(data) if isinstance(data, Mapping) else {},
                )
            )
            if len(records) >= limit:
                break

        return IntegrationBatch(revision=revision, records=tuple(records))

    def project(
        self,
        events: Sequence[LedgerEvent],
        *,
        thread_id: str | None = None,
    ) -> IntegrationSnapshot:
        overview = self.overview(events)
        if thread_id is None:
            return overview.global_snapshot
        return overview.thread_snapshots.get(
            thread_id,
            self._empty_snapshot(overview.global_snapshot.revision),
        )

    def pressure(self, *, thread_id: str | None = None) -> float:
        return self._pressure_for_snapshot(self.snapshot(thread_id=thread_id))

    def is_backpressured(self) -> bool:
        return self._is_snapshot_backpressured(self.snapshot())

    def _pressure_for_snapshot(self, snapshot: IntegrationSnapshot) -> float:
        count_pressure = self._ratio(
            snapshot.backlog_count,
            self.config.max_backlog_count,
        )
        age_pressure = self._ratio(
            snapshot.oldest_backlog_age_sequences,
            self.config.max_backlog_age_sequences,
        )
        return max(count_pressure, age_pressure)

    def _is_snapshot_backpressured(self, snapshot: IntegrationSnapshot) -> bool:
        return (
            snapshot.backlog_count > self.config.max_backlog_count
            or snapshot.oldest_backlog_age_sequences
            > self.config.max_backlog_age_sequences
        )

    @staticmethod
    def _build_snapshot(
        *,
        evidence_sequences: Mapping[str, int],
        dispositioned: set[str],
        knowledge_delta_count: int,
        revision: int,
    ) -> IntegrationSnapshot:
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

    @staticmethod
    def _empty_snapshot(revision: int) -> IntegrationSnapshot:
        return IntegrationSnapshot(
            revision=revision,
            evidence_count=0,
            dispositioned_evidence_count=0,
            backlog_evidence_ids=(),
            knowledge_delta_count=0,
            oldest_backlog_age_sequences=0,
        )

    @staticmethod
    def _dispositioned_ids(events: Sequence[LedgerEvent]) -> set[str]:
        dispositioned: set[str] = set()
        for event in events:
            if event.event_type == "INTEGRATION_DISPOSITION_RECORDED":
                dispositioned.update(event.reference_ids)
        return dispositioned

    @staticmethod
    def _optional_float(value: object) -> float | None:
        if value is None:
            return None
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError("durable evidence scalar must be numeric or null")
        return float(value)

    @staticmethod
    def _ratio(value: int, limit: int) -> float:
        if value <= 0:
            return 0.0
        if limit <= 0:
            return 1.0
        return min(1.0, value / float(limit))
