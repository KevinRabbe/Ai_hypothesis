"""Derived verification pressure over current compact knowledge state.

This is a State Projector view, not a new durable subsystem. The policy is deliberately
simple and replaceable: provisional and disputed deltas count as unresolved knowledge.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from .contracts import LedgerEvent
from .knowledge import KnowledgeRecord, KnowledgeStateProjector, KnowledgeStatus
from .ledger import SQLiteResearchLedger


@dataclass(frozen=True, slots=True)
class KnowledgeVerificationConfig:
    """Provisional mapping from unresolved-count to scheduler pressure."""

    full_pressure_count: int = 8

    def validate(self) -> None:
        if self.full_pressure_count <= 0:
            raise ValueError("full_pressure_count must be positive")


@dataclass(frozen=True, slots=True)
class KnowledgeVerificationOverview:
    revision: int
    unresolved_count: int
    thread_unresolved: Mapping[str, tuple[KnowledgeRecord, ...]]
    thread_pressure: Mapping[str, float]

    def pressure_for(self, thread_id: str) -> float:
        return float(self.thread_pressure.get(thread_id, 0.0))

    def pending_for(
        self,
        thread_id: str,
        *,
        limit: int,
    ) -> tuple[KnowledgeRecord, ...]:
        if limit <= 0:
            raise ValueError("limit must be positive")
        return tuple(self.thread_unresolved.get(thread_id, ()))[:limit]

    def pending_delta_ids(self, thread_id: str, *, limit: int) -> tuple[str, ...]:
        return tuple(record.delta_id for record in self.pending_for(thread_id, limit=limit))


class KnowledgeVerificationTracker:
    """Project unresolved knowledge into bounded verification pressure."""

    _UNRESOLVED = frozenset(
        {KnowledgeStatus.PROVISIONAL, KnowledgeStatus.DISPUTED}
    )

    def __init__(
        self,
        ledger: SQLiteResearchLedger,
        config: KnowledgeVerificationConfig | None = None,
        *,
        projector: KnowledgeStateProjector | None = None,
    ) -> None:
        self.ledger = ledger
        self.config = config or KnowledgeVerificationConfig()
        self.config.validate()
        self.projector = projector or KnowledgeStateProjector()

    def overview(
        self,
        events: Sequence[LedgerEvent] | None = None,
    ) -> KnowledgeVerificationOverview:
        history = tuple(events) if events is not None else self.ledger.read_all_events()
        snapshot = self.projector.project(history)
        by_thread: dict[str, list[KnowledgeRecord]] = {}
        unresolved_count = 0

        for record in snapshot.records:
            if record.status not in self._UNRESOLVED:
                continue
            unresolved_count += 1
            if record.thread_id is not None:
                by_thread.setdefault(record.thread_id, []).append(record)

        thread_unresolved = {
            thread_id: tuple(records) for thread_id, records in by_thread.items()
        }
        thread_pressure = {
            thread_id: min(
                1.0,
                len(records) / float(self.config.full_pressure_count),
            )
            for thread_id, records in thread_unresolved.items()
        }
        return KnowledgeVerificationOverview(
            revision=snapshot.revision,
            unresolved_count=unresolved_count,
            thread_unresolved=thread_unresolved,
            thread_pressure=thread_pressure,
        )

    def pressure(self, *, thread_id: str) -> float:
        return self.overview().pressure_for(thread_id)

    def pending_delta_ids(
        self,
        *,
        thread_id: str,
        limit: int,
    ) -> tuple[str, ...]:
        return self.overview().pending_delta_ids(thread_id, limit=limit)
