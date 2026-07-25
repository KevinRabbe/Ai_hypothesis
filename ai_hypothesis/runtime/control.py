"""Thin composition layer over the five durable population-runtime roles.

This module is not a sixth autonomous subsystem. It wires ledger, projection,
scheduling, worker selection, and bounded execution together while keeping domain
signals and Work Item preparation injected and replaceable.
"""

from __future__ import annotations

import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from .contracts import (
    AttemptResult,
    LedgerEvent,
    ProjectedState,
    SchedulerAction,
    SchedulerDecision,
    WorkItem,
    WorkPurpose,
)
from .integration import IntegrationTracker
from .ledger import SQLiteResearchLedger
from .projector import ThreadStateProjector
from .scheduler import SchedulerSignals, SchedulerV0, SchedulableThread
from .worker_runtime import WorkerAssignment, WorkerBank, WorkerRuntime


SignalProvider = Callable[[ProjectedState], SchedulerSignals]


@dataclass(frozen=True, slots=True)
class WorkPreparation:
    """Bounded data selected for one Work Item from potentially large global state."""

    context: Mapping[str, Any] = field(default_factory=dict)
    reference_ids: tuple[str, ...] = ()
    parent_ids: tuple[str, ...] = ()
    constraints: Mapping[str, Any] = field(default_factory=dict)
    resource_budget: Mapping[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        for name, values in (
            ("reference_ids", self.reference_ids),
            ("parent_ids", self.parent_ids),
        ):
            if any(not value for value in values):
                raise ValueError(f"{name} must not contain empty IDs")


ContextProvider = Callable[[ProjectedState, SchedulerDecision], WorkPreparation]


@dataclass(frozen=True, slots=True)
class ControlStep:
    """Observable result of one bounded scheduler/execution cycle."""

    state: ProjectedState
    decision: SchedulerDecision
    assignment: WorkerAssignment | None
    result: AttemptResult | None


class WorkerSelectorV0:
    """Minimal worker-selection policy owned by the scheduler/control layer."""

    def __init__(self, worker_ids: Sequence[str]) -> None:
        self.worker_ids = tuple(worker_ids)
        if not self.worker_ids:
            raise ValueError("worker selector requires at least one worker")
        if any(not worker_id or not worker_id.strip() for worker_id in self.worker_ids):
            raise ValueError("worker IDs must be non-empty")
        if len(set(self.worker_ids)) != len(self.worker_ids):
            raise ValueError("worker IDs must be unique")
        self._next_index = 0

    def choose(
        self,
        action: SchedulerAction,
        *,
        previous_worker_id: str | None,
    ) -> str:
        # Continuing productive depth may reuse the current worker. Width, challenge,
        # verification, synthesis, or explicit rotation prefer an independent weight.
        if action is SchedulerAction.CONTINUE and previous_worker_id in self.worker_ids:
            assert previous_worker_id is not None
            return previous_worker_id

        candidates = self.worker_ids
        if previous_worker_id in self.worker_ids and len(self.worker_ids) > 1:
            candidates = tuple(
                worker_id for worker_id in self.worker_ids if worker_id != previous_worker_id
            )

        selected = candidates[self._next_index % len(candidates)]
        self._next_index += 1
        return selected


class RuntimeControlLoop:
    """Run one bounded control step over persistent Work Threads."""

    def __init__(
        self,
        *,
        ledger: SQLiteResearchLedger,
        scheduler: SchedulerV0,
        worker_bank: WorkerBank,
        worker_ids: Sequence[str],
        projector: ThreadStateProjector | None = None,
        worker_runtime: WorkerRuntime | None = None,
        worker_selector: WorkerSelectorV0 | None = None,
        integration_tracker: IntegrationTracker | None = None,
    ) -> None:
        self.ledger = ledger
        self.scheduler = scheduler
        self.worker_bank = worker_bank
        self.projector = projector or ThreadStateProjector()
        self.worker_runtime = worker_runtime or WorkerRuntime(ledger)
        self.worker_selector = worker_selector or WorkerSelectorV0(worker_ids)
        self.integration_tracker = integration_tracker

    def create_thread(
        self,
        *,
        objective: str,
        purpose: WorkPurpose = WorkPurpose.EXPLORE,
        reference_ids: Sequence[str] = (),
        metadata: Mapping[str, Any] | None = None,
        thread_id: str | None = None,
    ) -> str:
        resolved_id = thread_id or uuid.uuid4().hex
        self.ledger.append_event(
            event_type="THREAD_CREATED",
            thread_id=resolved_id,
            reference_ids=tuple(reference_ids),
            payload={
                "objective": objective,
                "purpose": purpose.value,
                "status": "ACTIVE",
            },
        )
        if metadata:
            self.ledger.append_event(
                event_type="THREAD_METADATA_UPDATED",
                thread_id=resolved_id,
                payload=dict(metadata),
            )
        return resolved_id

    def run_once(
        self,
        *,
        signal_provider: SignalProvider,
        context_provider: ContextProvider,
        integration_backpressure: bool | None = None,
    ) -> ControlStep:
        events = self.ledger.read_events()
        thread_ids = self._thread_ids(events)
        if not thread_ids:
            raise ValueError("runtime has no Work Threads")

        states = tuple(
            self.projector.project(events, thread_id=thread_id) for thread_id in thread_ids
        )
        candidates = tuple(
            SchedulableThread(state=state, signals=signal_provider(state))
            for state in states
            if state.status != "COMPLETE"
        )
        if integration_backpressure is None:
            resolved_backpressure = (
                self.integration_tracker.is_backpressured()
                if self.integration_tracker is not None
                else False
            )
        else:
            resolved_backpressure = integration_backpressure

        decision = self.scheduler.choose(
            candidates,
            integration_backpressure=resolved_backpressure,
        )
        selected_state = next(
            state for state in states if state.thread_id == decision.thread_id
        )

        if decision.action is SchedulerAction.PAUSE:
            self.ledger.append_event(
                event_type="THREAD_PAUSED",
                thread_id=selected_state.thread_id,
                payload={"reason_codes": list(decision.reason_codes)},
            )
            return ControlStep(selected_state, decision, None, None)
        if decision.action is SchedulerAction.COMPLETE:
            self.ledger.append_event(
                event_type="THREAD_COMPLETED",
                thread_id=selected_state.thread_id,
                payload={"reason_codes": list(decision.reason_codes)},
            )
            return ControlStep(selected_state, decision, None, None)

        previous_worker_id = self._last_worker_id(events, selected_state.thread_id)
        worker_id = self.worker_selector.choose(
            decision.action,
            previous_worker_id=previous_worker_id,
        )
        preparation = context_provider(selected_state, decision)
        preparation.validate()
        item = WorkItem(
            work_item_id=uuid.uuid4().hex,
            thread_id=selected_state.thread_id,
            objective=selected_state.objective,
            purpose=decision.purpose or selected_state.purpose,
            projection_revision=selected_state.revision,
            reference_ids=preparation.reference_ids,
            parent_ids=preparation.parent_ids,
            context=dict(preparation.context),
            constraints=dict(preparation.constraints),
            resource_budget=dict(preparation.resource_budget),
        )
        item.validate()
        assignment = WorkerAssignment(worker_id=worker_id, work_item=item)
        result = self.worker_runtime.run_attempt(assignment, self.worker_bank)
        return ControlStep(selected_state, decision, assignment, result)

    @staticmethod
    def _thread_ids(events: Sequence[LedgerEvent]) -> tuple[str, ...]:
        seen: set[str] = set()
        ordered: list[str] = []
        for event in events:
            if event.event_type != "THREAD_CREATED" or event.thread_id is None:
                continue
            if event.thread_id not in seen:
                ordered.append(event.thread_id)
                seen.add(event.thread_id)
        return tuple(ordered)

    @staticmethod
    def _last_worker_id(events: Sequence[LedgerEvent], thread_id: str) -> str | None:
        for event in reversed(events):
            if event.thread_id != thread_id or event.event_type != "ATTEMPT_STARTED":
                continue
            worker_id = event.payload.get("worker_id")
            if isinstance(worker_id, str) and worker_id:
                return worker_id
        return None
