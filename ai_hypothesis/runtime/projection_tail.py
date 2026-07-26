"""Reusable forward-only ledger-tail primitive for rebuildable materialized projections.

Projection stores own their derived state and durable checkpoint metadata. This helper owns
only canonical ledger checkpoint validation and bounded tail iteration, so multiple hot views
can reuse the same correctness rules without becoming a second source of truth.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from .contracts import LedgerEvent
from .ledger import SQLiteResearchLedger


@dataclass(frozen=True, slots=True)
class ProjectionCheckpoint:
    sequence: int = 0
    event_id: str = ""

    def validate(self) -> None:
        if self.sequence < 0:
            raise ValueError("projection checkpoint sequence must be non-negative")
        if self.sequence == 0:
            if self.event_id:
                raise ValueError("projection checkpoint at sequence zero cannot have event_id")
        elif not self.event_id or not self.event_id.strip():
            raise ValueError("nonzero projection checkpoint requires event_id")


class LedgerProjectionTail:
    """Validate one projection checkpoint and stream only the requested canonical tail."""

    def __init__(self, ledger: SQLiteResearchLedger) -> None:
        self.ledger = ledger

    @property
    def source_identity(self) -> str:
        if self.ledger.path == ":memory:":
            # An in-memory ledger has process-object identity rather than stable path identity.
            # Persistent projection reuse across process restart is therefore intentionally not
            # supported for this source form.
            return f":memory:{id(self.ledger)}"
        return str(Path(self.ledger.path).expanduser().resolve())

    def validate_checkpoint(self, checkpoint: ProjectionCheckpoint) -> None:
        checkpoint.validate()
        if checkpoint.sequence == 0:
            return
        event = self.ledger.get_event(checkpoint.event_id)
        if event is None or event.sequence != checkpoint.sequence:
            raise RuntimeError(
                "projection checkpoint no longer matches canonical Research Ledger"
            )

    def iter_pages(
        self,
        checkpoint: ProjectionCheckpoint,
        *,
        target_sequence: int | None = None,
        page_size: int = 1000,
    ) -> Iterator[tuple[LedgerEvent, ...]]:
        """Yield bounded canonical pages after checkpoint through an optional exact target."""

        checkpoint.validate()
        if page_size <= 0:
            raise ValueError("page_size must be positive")
        if target_sequence is not None:
            if target_sequence < 0:
                raise ValueError("target_sequence must be non-negative")
            if target_sequence < checkpoint.sequence:
                raise ValueError("projection checkpoint is ahead of requested ledger snapshot")

        self.validate_checkpoint(checkpoint)
        current = checkpoint.sequence
        if target_sequence == current:
            return

        while True:
            page = self.ledger.read_events(
                after_sequence=current,
                limit=page_size,
            )
            if not page:
                break
            eligible = (
                page
                if target_sequence is None
                else tuple(event for event in page if event.sequence <= target_sequence)
            )
            if not eligible:
                break

            previous = current
            for event in eligible:
                event.validate()
                if event.sequence <= previous:
                    raise ValueError(
                        "canonical projection tail must be in strictly increasing sequence order"
                    )
                previous = event.sequence
            yield tuple(eligible)
            current = eligible[-1].sequence

            if target_sequence is not None and current >= target_sequence:
                break
            if len(eligible) < len(page):
                break
            if len(page) < page_size:
                break

        if target_sequence is not None and current != target_sequence:
            raise RuntimeError(
                "requested projection snapshot sequence is not available in canonical ledger"
            )

    @staticmethod
    def checkpoint_after(page: tuple[LedgerEvent, ...]) -> ProjectionCheckpoint:
        if not page:
            raise ValueError("cannot create projection checkpoint from empty page")
        return ProjectionCheckpoint(
            sequence=page[-1].sequence,
            event_id=page[-1].event_id,
        )
