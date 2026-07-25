"""Append-only scheduler allocation traces without changing SchedulerDecision semantics."""

from __future__ import annotations

import random
from collections.abc import Sequence

from .contracts import SchedulerDecision
from .ledger import SQLiteResearchLedger
from .scheduler import SchedulerConfig, SchedulerV0, SchedulableThread


class TracingSchedulerV0(SchedulerV0):
    """Scheduler v0 with durable allocation traces for later analysis/learning."""

    def __init__(
        self,
        ledger: SQLiteResearchLedger,
        config: SchedulerConfig | None = None,
        *,
        rng: random.Random | None = None,
    ) -> None:
        super().__init__(config, rng=rng)
        self.ledger = ledger

    def choose(
        self,
        candidates: Sequence[SchedulableThread],
        *,
        integration_backpressure: bool = False,
        max_width: int = 1,
    ) -> SchedulerDecision:
        decision = super().choose(
            candidates,
            integration_backpressure=integration_backpressure,
            max_width=max_width,
        )
        self.ledger.append_event(
            event_type="SCHEDULER_DECISION_RECORDED",
            thread_id=decision.thread_id,
            reference_ids=decision.work_item_ids,
            payload={
                "decision_id": decision.decision_id,
                "action": decision.action.value,
                "purpose": decision.purpose.value if decision.purpose is not None else None,
                "width": decision.width,
                "reason_codes": list(decision.reason_codes),
                "projection_revision": decision.projection_revision,
                "integration_backpressure": bool(integration_backpressure),
                "max_width": int(max_width),
            },
        )
        return decision
