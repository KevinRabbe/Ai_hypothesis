"""Helpers for putting explicitly selected compact knowledge into bounded Work Items."""

from __future__ import annotations

from collections.abc import Sequence

from .control import WorkPreparation
from .knowledge import KnowledgeSnapshot


def prepare_bounded_knowledge_work(
    snapshot: KnowledgeSnapshot,
    delta_ids: Sequence[str],
    *,
    limit: int = 32,
) -> WorkPreparation:
    """Prepare only explicitly selected current knowledge for one bounded attempt."""

    records = snapshot.select(delta_ids, limit=limit)
    preparation = WorkPreparation(
        reference_ids=tuple(record.delta_id for record in records),
        context={
            "knowledge_revision": snapshot.revision,
            "knowledge_records": tuple(
                record.to_context_record() for record in records
            ),
        },
        constraints={"max_knowledge_records": limit},
    )
    preparation.validate()
    return preparation
