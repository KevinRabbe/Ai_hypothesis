"""Thin composition layer over the five durable population-runtime roles.

This module is not a sixth autonomous subsystem. It wires ledger, projection,
scheduling, worker selection, and bounded execution together while keeping domain
signals and Work Item preparation injected and replaceable.
"""

from __future__ import annotations

import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
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
from .integration import IntegrationOverview, IntegrationTracker
from .knowledge_verification import (
    KnowledgeVerificationOverview,
    KnowledgeVerificationTracker,
)
from .ledger import SQLiteResearchLedger
from .projector import ThreadStateProjector
from .scheduler import SchedulerSignals, SchedulerV0, SchedulableThread
from .worker_runtime import WorkerAssignment, WorkerBank, WorkerRuntime


SignalProvider = Callable[[ProjectedState], SchedulerSignals]


@dataclass(frozen=True, slots=True)
class WorkPreparation:
    context: Mapping[str, Any] = field(default_factory=dict)
    reference_ids: tuple[str, ...] = ()
    parent_ids: tuple[str, ...] = ()
    constraints: Mapping[str, Any] = field(default_factory=dict)
    resource_budget: Mapping[str, Any] = field(default_factory=dict)
    scope_region_ids: tuple[str, ...] = ()

    def validate(self) -> None:
        for name, values in (
            ("scope_region_ids", self.scope_region_ids),
            ("reference_ids", self.reference_ids),
            ("parent_ids", self.parent_ids),
        ):
            if any(not value or not value.strip() for value in values):
                raise ValueError(f"{name} must not contain empty IDs")
        if len(set(self.scope_region_ids)) != len(self.scope_region_ids):
            raise ValueError("scope_region_ids must be unique inside one WorkPreparation")


@dataclass(frozen=True, slots=True)
class WorkPreparationBatch:
    """Optional per-worker preparation for one scheduler allocation.

    A single WorkPreparation remains the compact replication contract: it is reused
    for every allocated worker. This batch form is used only when width should inspect
    different source regions or receive otherwise different bounded context.
    """

    items: tuple[WorkPreparation, ...]

    def validate(self, *, expected_width: int) -> None:
        if expected_width <= 0:
            raise ValueError("expected_width must be positive")
        if len(self.items) != expected_width:
            raise ValueError("WorkPreparationBatch size must match scheduler width")
        for item in self.items:
            item.validate()


ContextProvider = Callable[
    [ProjectedState, SchedulerDecision],
    WorkPreparation | WorkPreparationBatch,
]


@dataclass(frozen=True, slots=True)
class ControlStep:
    state: ProjectedState
    decision: SchedulerDecision
    assignments: tuple[WorkerAssignment, ...] = ()
    results: tuple[AttemptResult, ...] = ()

    @property
    def assignment(self) -> WorkerAssignment | None:
        return self.assignments[0] if self.assignments else None

    @property
    def result(self) -> AttemptResult | None:
        return self.results[0] if self.results else None


@dataclass(frozen=True, slots=True)
class ControlBatch:
    """Several independent control decisions executed through one WorkerBank batch."""

    steps: tuple[ControlStep, ...]

    @property
    def assignments(self) -> tuple[WorkerAssignment, ...]:
        return tuple(
            assignment
            for step in self.steps
            for assignment in step.assignments
        )

    @property
    def results(self) -> tuple[AttemptResult, ...]:
        return tuple(result for step in self.steps for result in step.results)

    @property
    def neural_attempt_count(self) -> int:
        return len(self.assignments)


class WorkerSelectorV0:
    def __init__(self, worker_ids: Sequence[str]) -> None:
        self.worker_ids = tuple(worker_ids)
        if not self.worker_ids:
            raise ValueError("worker selector requires at least one worker")
        if any(not worker_id or not worker_id.strip() for worker_id in self.worker_ids):
            raise ValueError("worker IDs must be non-empty")
        if len(set(self.worker_ids)) != len(self.worker_ids):
            raise ValueError("worker IDs must be unique")
        self._next_index = 0

    @property
    def population_width(self) -> int:
        return len(self.worker_ids)

    def choose(self, action: SchedulerAction, *, previous_worker_id: str | None) -> str:
        return self.choose_many(action, previous_worker_id=previous_worker_id, count=1)[0]

    def choose_many(self, action: SchedulerAction, *, previous_worker_id: str | None, count: int) -> tuple[str, ...]:
        if count <= 0:
            raise ValueError("worker selection count must be positive")
        if action is SchedulerAction.CONTINUE and count == 1 and previous_worker_id in self.worker_ids:
            assert previous_worker_id is not None
            return (previous_worker_id,)

        population_size = len(self.worker_ids)
        start = self._next_index % population_size
        rotated = [self.worker_ids[(start + offset) % population_size] for offset in range(population_size)]
        if previous_worker_id in rotated and population_size > 1:
            rotated.remove(previous_worker_id)
            rotated.append(previous_worker_id)
        selected = tuple(rotated[: min(count, population_size)])
        self._next_index = (self._next_index + 1) % population_size
        return selected


class RuntimeControlLoop:
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
        verification_tracker: KnowledgeVerificationTracker | None = None,
    ) -> None:
        self.ledger = ledger
        self.scheduler = scheduler
        self.worker_bank = worker_bank
        self.projector = projector or ThreadStateProjector()
        self.worker_runtime = worker_runtime or WorkerRuntime(ledger)
        self.worker_selector = worker_selector or WorkerSelectorV0(worker_ids)
        self.integration_tracker = integration_tracker
        self.verification_tracker = verification_tracker

    def create_thread(
        self,
        *,
        objective: str,
        purpose: WorkPurpose = WorkPurpose.EXPLORE,
        reference_ids: Sequence[str] = (),
        metadata: Mapping[str, Any] | None = None,
        thread_id: str | None = None,
    ) -> str:
        if not objective or not objective.strip():
            raise ValueError("objective must be non-empty")
        resolved_id = thread_id or uuid.uuid4().hex
        if not resolved_id or not resolved_id.strip():
            raise ValueError("thread_id must be non-empty")
        if resolved_id in self._state_index():
            raise ValueError(f"Work Thread {resolved_id!r} already exists")

        self.ledger.append_event(
            event_type="THREAD_CREATED",
            thread_id=resolved_id,
            reference_ids=tuple(reference_ids),
            payload={"objective": objective, "purpose": purpose.value, "status": "ACTIVE"},
        )
        if metadata:
            self.ledger.append_event(event_type="THREAD_METADATA_UPDATED", thread_id=resolved_id, payload=dict(metadata))
        return resolved_id

    def fork_thread(
        self,
        parent_thread_id: str,
        *,
        objective: str,
        purpose: WorkPurpose = WorkPurpose.EXPLORE,
        reference_ids: Sequence[str] = (),
        metadata: Mapping[str, Any] | None = None,
        child_thread_id: str | None = None,
    ) -> str:
        states = self._state_index()
        parent = self._require_state(states, parent_thread_id)
        if parent.status == "COMPLETE" or parent.merged_into_thread_id is not None:
            raise ValueError("cannot fork a completed or merged Work Thread")
        child_id = child_thread_id or uuid.uuid4().hex
        if child_id in states:
            raise ValueError(f"Work Thread {child_id!r} already exists")

        self.create_thread(
            objective=objective,
            purpose=purpose,
            reference_ids=reference_ids,
            metadata=metadata,
            thread_id=child_id,
        )
        self.ledger.append_event(
            event_type="THREAD_FORKED",
            thread_id=parent_thread_id,
            reference_ids=(child_id,),
            payload={"child_thread_id": child_id},
        )
        return child_id

    def add_dependency(self, thread_id: str, dependency_thread_id: str) -> None:
        states = self._state_index()
        state = self._require_state(states, thread_id)
        self._require_state(states, dependency_thread_id)
        if thread_id == dependency_thread_id:
            raise ValueError("thread cannot depend on itself")
        if dependency_thread_id in state.dependency_thread_ids:
            return
        adjacency = {
            candidate.thread_id: tuple(candidate.dependency_thread_ids)
            for candidate in states.values()
        }
        adjacency[thread_id] = (*adjacency.get(thread_id, ()), dependency_thread_id)
        self._assert_acyclic(adjacency, relation_name="dependency")
        self.ledger.append_event(
            event_type="DEPENDENCY_ADDED",
            thread_id=thread_id,
            reference_ids=(dependency_thread_id,),
        )

    def remove_dependency(self, thread_id: str, dependency_thread_id: str) -> None:
        states = self._state_index()
        state = self._require_state(states, thread_id)
        self._require_state(states, dependency_thread_id)
        if dependency_thread_id not in state.dependency_thread_ids:
            return
        self.ledger.append_event(
            event_type="DEPENDENCY_REMOVED",
            thread_id=thread_id,
            reference_ids=(dependency_thread_id,),
        )

    def merge_threads(self, target_thread_id: str, source_thread_ids: Sequence[str]) -> None:
        source_ids = tuple(dict.fromkeys(source_thread_ids))
        if not source_ids:
            raise ValueError("merge requires at least one source Work Thread")
        states = self._state_index()
        target = self._require_state(states, target_thread_id)
        if target.status == "COMPLETE" or target.merged_into_thread_id is not None:
            raise ValueError("merge target must be active and not already merged")
        for source_id in source_ids:
            if source_id == target_thread_id:
                raise ValueError("thread cannot merge into itself")
            source = self._require_state(states, source_id)
            if source.merged_into_thread_id is not None:
                raise ValueError(f"source Work Thread {source_id!r} is already merged")

        self.ledger.append_event(
            event_type="THREAD_MERGED",
            thread_id=target_thread_id,
            reference_ids=source_ids,
            payload={"source_thread_ids": list(source_ids)},
        )

    def run_once(
        self,
        *,
        signal_provider: SignalProvider,
        context_provider: ContextProvider,
        integration_backpressure: bool | None = None,
    ) -> ControlStep:
        batch = self.run_many(
            signal_provider=signal_provider,
            context_provider=context_provider,
            max_threads=1,
            max_attempts=self.worker_selector.population_width,
            integration_backpressure=integration_backpressure,
        )
        if not batch.steps:
            raise RuntimeError("single-step control produced no decision")
        return batch.steps[0]

    def run_many(
        self,
        *,
        signal_provider: SignalProvider,
        context_provider: ContextProvider,
        max_threads: int,
        max_attempts: int,
        integration_backpressure: bool | None = None,
    ) -> ControlBatch:
        """Schedule independent threads from one snapshot and execute one neural batch.

        Each Work Thread can receive at most one scheduler decision in a call. Graph,
        dependency, integration, and verification eligibility are computed once from the
        initial ledger snapshot. A dependency completed during this batch therefore does
        not unlock another thread until the next control call.

        Scheduler decisions remain independent and retain their own provenance. Only the
        resulting neural assignments are flattened into one WorkerRuntime/WorkerBank call.
        """

        if max_threads <= 0:
            raise ValueError("max_threads must be positive")
        if max_attempts <= 0:
            raise ValueError("max_attempts must be positive")

        events = self.ledger.read_all_events()
        states = self.projector.project_all(events)
        if not states:
            raise ValueError("runtime has no Work Threads")
        state_by_id = {state.thread_id: state for state in states}

        integration_overview = self.integration_tracker.overview(events) if self.integration_tracker is not None else None
        verification_overview = self.verification_tracker.overview(events) if self.verification_tracker is not None else None
        candidates = [
            SchedulableThread(
                state=state,
                signals=self._signals_for(
                    state,
                    signal_provider,
                    integration_overview,
                    verification_overview,
                ),
            )
            for state in states
            if state.status != "COMPLETE" and not self._is_dependency_blocked(state, state_by_id)
        ]
        if not candidates:
            if any(state.status != "COMPLETE" for state in states):
                raise ValueError("all active Work Threads are dependency-blocked")
            raise ValueError("runtime has no non-complete Work Threads")

        resolved_backpressure = (
            integration_overview.global_backpressured if integration_backpressure is None and integration_overview is not None
            else False if integration_backpressure is None
            else integration_backpressure
        )

        planned_steps: list[ControlStep] = []
        all_assignments: list[WorkerAssignment] = []
        assignment_counts: list[int] = []
        remaining_attempts = max_attempts

        while candidates and len(planned_steps) < max_threads:
            if remaining_attempts <= 0:
                break
            available_width = min(
                self.worker_selector.population_width,
                remaining_attempts,
            )
            decision = self.scheduler.choose(
                tuple(candidates),
                integration_backpressure=resolved_backpressure,
                max_width=available_width,
            )
            if decision.width > available_width:
                raise ValueError(
                    "scheduler decision width exceeds remaining neural-attempt budget"
                )

            selected_index = next(
                (
                    index
                    for index, candidate in enumerate(candidates)
                    if candidate.state.thread_id == decision.thread_id
                ),
                None,
            )
            if selected_index is None:
                raise ValueError("scheduler selected a thread outside the candidate snapshot")
            selected = candidates.pop(selected_index)
            selected_state = selected.state

            if decision.action is SchedulerAction.PAUSE:
                self.ledger.append_event(
                    event_type="THREAD_PAUSED",
                    thread_id=selected_state.thread_id,
                    payload={"reason_codes": list(decision.reason_codes)},
                )
                planned_steps.append(ControlStep(selected_state, decision))
                assignment_counts.append(0)
                continue
            if decision.action is SchedulerAction.COMPLETE:
                self.ledger.append_event(
                    event_type="THREAD_COMPLETED",
                    thread_id=selected_state.thread_id,
                    payload={"reason_codes": list(decision.reason_codes)},
                )
                planned_steps.append(ControlStep(selected_state, decision))
                assignment_counts.append(0)
                continue

            assignments, resolved_decision = self._prepare_assignments(
                selected_state,
                decision,
                events=events,
                context_provider=context_provider,
            )
            if len(assignments) != decision.width:
                raise RuntimeError("prepared assignment count does not match scheduler width")
            remaining_attempts -= len(assignments)
            all_assignments.extend(assignments)
            assignment_counts.append(len(assignments))
            planned_steps.append(
                ControlStep(
                    selected_state,
                    resolved_decision,
                    assignments=assignments,
                )
            )

        flat_results = self.worker_runtime.run_batch(
            tuple(all_assignments),
            self.worker_bank,
        )
        if len(flat_results) != len(all_assignments):
            raise RuntimeError("worker runtime returned a mismatched cross-thread result count")

        resolved_steps: list[ControlStep] = []
        result_offset = 0
        for step, assignment_count in zip(
            planned_steps,
            assignment_counts,
            strict=True,
        ):
            step_results = tuple(
                flat_results[result_offset : result_offset + assignment_count]
            )
            result_offset += assignment_count
            resolved_steps.append(replace(step, results=step_results))
        if result_offset != len(flat_results):
            raise RuntimeError("cross-thread result partition did not consume all results")
        return ControlBatch(tuple(resolved_steps))

    def _prepare_assignments(
        self,
        state: ProjectedState,
        decision: SchedulerDecision,
        *,
        events: Sequence[LedgerEvent],
        context_provider: ContextProvider,
    ) -> tuple[tuple[WorkerAssignment, ...], SchedulerDecision]:
        raw_preparation = context_provider(state, decision)
        preparations = self._preparations_for_width(raw_preparation, decision.width)
        selected_worker_ids = self.worker_selector.choose_many(
            decision.action,
            previous_worker_id=self._last_worker_id(events, state.thread_id),
            count=decision.width,
        )
        if len(selected_worker_ids) != decision.width:
            raise RuntimeError("scheduler allocated more distinct workers than available")

        assignments: list[WorkerAssignment] = []
        for worker_id, preparation in zip(
            selected_worker_ids,
            preparations,
            strict=True,
        ):
            item = WorkItem(
                work_item_id=uuid.uuid4().hex,
                thread_id=state.thread_id,
                objective=state.objective,
                purpose=decision.purpose or state.purpose,
                projection_revision=state.revision,
                reference_ids=preparation.reference_ids,
                parent_ids=preparation.parent_ids,
                context=dict(preparation.context),
                constraints=dict(preparation.constraints),
                resource_budget=dict(preparation.resource_budget),
                scope_region_ids=preparation.scope_region_ids,
            )
            item.validate()
            assignments.append(WorkerAssignment(worker_id=worker_id, work_item=item))

        resolved_decision = replace(
            decision,
            work_item_ids=tuple(
                assignment.work_item.work_item_id for assignment in assignments
            ),
        )
        resolved_decision.validate()
        return tuple(assignments), resolved_decision

    @staticmethod
    def _preparations_for_width(
        preparation: WorkPreparation | WorkPreparationBatch,
        width: int,
    ) -> tuple[WorkPreparation, ...]:
        if width <= 0:
            raise ValueError("scheduler width must be positive")
        if isinstance(preparation, WorkPreparationBatch):
            preparation.validate(expected_width=width)
            return preparation.items
        if not isinstance(preparation, WorkPreparation):
            raise TypeError("context_provider must return WorkPreparation or WorkPreparationBatch")
        preparation.validate()
        return (preparation,) * width

    def _state_index(self) -> dict[str, ProjectedState]:
        return {
            state.thread_id: state
            for state in self.projector.project_all(self.ledger.read_all_events())
        }

    @staticmethod
    def _require_state(
        states: Mapping[str, ProjectedState], thread_id: str
    ) -> ProjectedState:
        try:
            return states[thread_id]
        except KeyError as error:
            raise ValueError(f"unknown Work Thread {thread_id!r}") from error

    @staticmethod
    def _is_dependency_blocked(
        state: ProjectedState, state_by_id: Mapping[str, ProjectedState]
    ) -> bool:
        return any(
            state_by_id[dependency_id].status != "COMPLETE"
            for dependency_id in state.dependency_thread_ids
        )

    @staticmethod
    def _signals_for(
        state: ProjectedState,
        signal_provider: SignalProvider,
        integration_overview: IntegrationOverview | None,
        verification_overview: KnowledgeVerificationOverview | None,
    ) -> SchedulerSignals:
        signals = signal_provider(state)
        if integration_overview is not None:
            pressure = integration_overview.pressure_for(state.thread_id)
            if pressure > signals.integration_backlog:
                signals = replace(signals, integration_backlog=pressure)
        if verification_overview is not None:
            verification = verification_overview.pressure_for(state.thread_id)
            if verification > signals.verification_need:
                signals = replace(signals, verification_need=verification)
        return signals

    @staticmethod
    def _assert_acyclic(
        adjacency: Mapping[str, Sequence[str]], *, relation_name: str
    ) -> None:
        visiting: set[str] = set()
        visited: set[str] = set()

        def visit(thread_id: str) -> None:
            if thread_id in visited:
                return
            if thread_id in visiting:
                raise ValueError(f"{relation_name} cycle detected at {thread_id!r}")
            visiting.add(thread_id)
            for target_id in adjacency.get(thread_id, ()):
                visit(target_id)
            visiting.remove(thread_id)
            visited.add(thread_id)

        for thread_id in adjacency:
            visit(thread_id)

    @staticmethod
    def _last_worker_id(events: Sequence[LedgerEvent], thread_id: str) -> str | None:
        for event in reversed(events):
            if event.thread_id == thread_id and event.event_type == "ATTEMPT_STARTED":
                worker_id = event.payload.get("worker_id")
                if isinstance(worker_id, str) and worker_id:
                    return worker_id
        return None
