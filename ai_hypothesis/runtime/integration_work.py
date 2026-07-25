"""Helpers for constructing bounded synthesis Work Items from integration backlog."""

from __future__ import annotations

from .contracts import ProjectedState
from .control import WorkPreparation
from .integration import IntegrationTracker


def prepare_bounded_integration_work(
    tracker: IntegrationTracker,
    state: ProjectedState,
    *,
    limit: int = 32,
) -> WorkPreparation:
    """Select a fixed-size oldest-first evidence batch for one synthesis attempt.

    The full backlog remains in the Research Ledger. Only this bounded slice and its
    provenance enter the worker's active context.
    """

    state.validate()
    batch = tracker.pending_batch(limit=limit, thread_id=state.thread_id)
    preparation = WorkPreparation(
        reference_ids=batch.evidence_ids,
        context={
            "integration_revision": batch.revision,
            "pending_evidence": batch.to_context_records(),
            "causal_event_ids": list(batch.causal_event_ids),
        },
        constraints={"max_pending_evidence": limit},
    )
    preparation.validate()
    return preparation
