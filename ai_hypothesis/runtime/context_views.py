"""Purpose-driven bounded context views for homogeneous workers.

Roles come from the Work Item context, not from specialized worker architectures. The
router only intercepts runtime purposes whose canonical bounded projections already exist;
domain-specific exploration/progression/challenge/final-synthesis context remains injected
through the fallback provider.
"""

from __future__ import annotations

from dataclasses import replace

from .contracts import ProjectedState, SchedulerAction, SchedulerDecision
from .control import ContextProvider, WorkPreparation, WorkPreparationBatch
from .integration import IntegrationTracker
from .integration_work import prepare_bounded_integration_work
from .knowledge import KnowledgeStateProjector
from .knowledge_verification import KnowledgeVerificationTracker
from .knowledge_work import prepare_bounded_knowledge_work
from .ledger import SQLiteResearchLedger


class PurposeContextRouter:
    """Route scheduler purposes into bounded ledger-derived worker views.

    Backpressure-driven SYNTHESIZE decisions consume pending evidence. VERIFY decisions
    consume unresolved knowledge. Everything else is delegated unchanged to the caller's
    domain context provider.
    """

    def __init__(
        self,
        *,
        ledger: SQLiteResearchLedger,
        fallback: ContextProvider,
        integration_tracker: IntegrationTracker | None = None,
        verification_tracker: KnowledgeVerificationTracker | None = None,
        knowledge_projector: KnowledgeStateProjector | None = None,
        integration_limit: int = 32,
        verification_limit: int = 32,
    ) -> None:
        if integration_limit <= 0:
            raise ValueError("integration_limit must be positive")
        if verification_limit <= 0:
            raise ValueError("verification_limit must be positive")
        if integration_tracker is not None and integration_tracker.ledger is not ledger:
            raise ValueError("integration tracker must use the same Research Ledger")
        if verification_tracker is not None and verification_tracker.ledger is not ledger:
            raise ValueError("verification tracker must use the same Research Ledger")

        self.ledger = ledger
        self.fallback = fallback
        self.integration_tracker = integration_tracker
        self.verification_tracker = verification_tracker
        self.knowledge_projector = knowledge_projector or KnowledgeStateProjector()
        self.integration_limit = integration_limit
        self.verification_limit = verification_limit

    def __call__(
        self,
        state: ProjectedState,
        decision: SchedulerDecision,
    ) -> WorkPreparation | WorkPreparationBatch:
        state.validate()
        decision.validate()
        if decision.thread_id != state.thread_id:
            raise ValueError("scheduler decision and projected state refer to different threads")

        if self._is_backpressure_synthesis(decision):
            return self._prepare_integration_view(state)
        if decision.action is SchedulerAction.VERIFY:
            return self._prepare_verification_view(state)
        return self.fallback(state, decision)

    @staticmethod
    def _is_backpressure_synthesis(decision: SchedulerDecision) -> bool:
        return (
            decision.action is SchedulerAction.SYNTHESIZE
            and "BACKPRESSURE" in decision.reason_codes
        )

    def _prepare_integration_view(self, state: ProjectedState) -> WorkPreparation:
        if self.integration_tracker is None:
            raise RuntimeError(
                "backpressure synthesis requires an IntegrationTracker"
            )
        preparation = prepare_bounded_integration_work(
            self.integration_tracker,
            state,
            limit=self.integration_limit,
        )
        if not preparation.reference_ids:
            raise ValueError(
                "backpressure synthesis selected a thread with no pending evidence"
            )
        return replace(
            preparation,
            context={
                **dict(preparation.context),
                "context_view": "SYNTHESIZE",
                "synthesis_mode": "INTEGRATION_BACKPRESSURE",
            },
            constraints={
                **dict(preparation.constraints),
                "emit_structured_knowledge_deltas": True,
                "disposition_consumed_evidence": True,
            },
        )

    def _prepare_verification_view(self, state: ProjectedState) -> WorkPreparation:
        if self.verification_tracker is None:
            raise RuntimeError("VERIFY decisions require a KnowledgeVerificationTracker")

        events = self.ledger.read_all_events()
        overview = self.verification_tracker.overview(events)
        delta_ids = overview.pending_delta_ids(
            state.thread_id,
            limit=self.verification_limit,
        )
        if not delta_ids:
            raise ValueError("VERIFY decision selected a thread with no unresolved knowledge")

        snapshot = self.knowledge_projector.project(events)
        preparation = prepare_bounded_knowledge_work(
            snapshot,
            delta_ids,
            limit=self.verification_limit,
        )
        return replace(
            preparation,
            context={
                **dict(preparation.context),
                "context_view": "VERIFY",
                "verification_target_delta_ids": list(delta_ids),
            },
            constraints={
                **dict(preparation.constraints),
                "independent_verification": True,
                "hide_worker_identity": True,
                "hide_vote_counts": True,
            },
        )
