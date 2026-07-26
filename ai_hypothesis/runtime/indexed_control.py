"""Runtime control path driven by incremental materialized projections.

The baseline RuntimeControlLoop remains unchanged. This optional variant captures one exact
canonical revision and obtains current thread state, integration pressure, verification pressure,
and worker continuity from rebuildable indexes rather than a full Research Ledger replay.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, replace
from types import MappingProxyType
from typing import Mapping, Sequence

from .contracts import AttemptResult, LedgerEvent, ProjectedState, SchedulerAction, SchedulerDecision, WorkItem
from .control import (
    ContextProvider,
    ControlBatch,
    ControlStep,
    RuntimeControlLoop,
    SignalProvider,
    WorkerSelectorV0,
)
from .integration_index import IndexedIntegrationOverview, SQLiteIndexedIntegrationTracker
from .knowledge_index import SQLiteIndexedKnowledgeState
from .knowledge_verification import KnowledgeVerificationOverview, KnowledgeVerificationTracker
from .ledger import SQLiteResearchLedger
from .scheduler import SchedulableThread
from .thread_state_index import SQLiteIndexedThreadState
from .worker_runtime import WorkerAssignment, WorkerBank, WorkerRuntime


@dataclass(frozen=True, slots=True)
class IndexedThreadRuntimeSnapshot:
    revision: int
    states: tuple[ProjectedState, ...]
    last_worker_ids: Mapping[str, str | None]


class IndexedThreadRuntimeState(SQLiteIndexedThreadState):
    """Thread materialization with one atomic scheduler-facing snapshot method."""

    def capture_through(self, sequence: int) -> IndexedThreadRuntimeSnapshot:
        if sequence < 0:
            raise ValueError("sequence must be non-negative")
        with self._lock:
            self._sync_to(target_sequence=sequence, page_size=1000)
            states = self._snapshot_all_unlocked()
            rows = self._connection.execute(
                """
                SELECT thread_id, last_worker_id
                FROM thread_state_record
                WHERE created = 1
                """
            ).fetchall()
            last_workers = {
                str(row["thread_id"]): (
                    str(row["last_worker_id"])
                    if row["last_worker_id"] is not None
                    else None
                )
                for row in rows
            }
            return IndexedThreadRuntimeSnapshot(
                revision=sequence,
                states=states,
                last_worker_ids=MappingProxyType(last_workers),
            )


@dataclass(frozen=True, slots=True)
class IndexedRuntimeSnapshot:
    revision: int
    states: tuple[ProjectedState, ...]
    last_worker_ids: Mapping[str, str | None]
    integration_overview: IndexedIntegrationOverview | None
    verification_overview: KnowledgeVerificationOverview | None


class IndexedRuntimeSnapshotProvider:
    """Freeze all scheduler-facing derived views at one canonical ledger revision."""

    def __init__(
        self,
        *,
        ledger: SQLiteResearchLedger,
        thread_state: IndexedThreadRuntimeState,
        integration_tracker: SQLiteIndexedIntegrationTracker | None = None,
        verification_tracker: KnowledgeVerificationTracker | None = None,
    ) -> None:
        if thread_state.ledger is not ledger:
            raise ValueError("thread runtime state must use the same Research Ledger")
        if integration_tracker is not None and integration_tracker.ledger is not ledger:
            raise ValueError("integration tracker must use the same Research Ledger")
        if verification_tracker is not None:
            if verification_tracker.ledger is not ledger:
                raise ValueError("verification tracker must use the same Research Ledger")
            if not isinstance(verification_tracker.projector, SQLiteIndexedKnowledgeState):
                raise ValueError(
                    "indexed runtime verification requires SQLiteIndexedKnowledgeState"
                )
        self.ledger = ledger
        self.thread_state = thread_state
        self.integration_tracker = integration_tracker
        self.verification_tracker = verification_tracker

    def capture(self) -> IndexedRuntimeSnapshot:
        revision = self.ledger.latest_sequence()
        boundary = self._boundary_events(revision)
        thread_snapshot = self.thread_state.capture_through(revision)
        integration = (
            self.integration_tracker.overview(boundary)
            if self.integration_tracker is not None
            else None
        )
        verification = (
            self.verification_tracker.overview(boundary)
            if self.verification_tracker is not None
            else None
        )

        if thread_snapshot.revision != revision:
            raise RuntimeError("thread runtime snapshot revision mismatch")
        if integration is not None and integration.global_snapshot.revision != revision:
            raise RuntimeError("integration runtime snapshot revision mismatch")
        if verification is not None and verification.revision != revision:
            raise RuntimeError("verification runtime snapshot revision mismatch")

        return IndexedRuntimeSnapshot(
            revision=revision,
            states=thread_snapshot.states,
            last_worker_ids=thread_snapshot.last_worker_ids,
            integration_overview=integration,
            verification_overview=verification,
        )

    def _boundary_events(self, revision: int) -> tuple[LedgerEvent, ...]:
        if revision == 0:
            return ()
        page = self.ledger.read_events(after_sequence=revision - 1, limit=1)
        if len(page) != 1 or page[0].sequence != revision:
            raise RuntimeError("cannot resolve exact runtime snapshot boundary event")
        return page


class IndexedRuntimeControlLoop(RuntimeControlLoop):
    """RuntimeControlLoop variant that never needs a full-history scheduler snapshot."""

    def __init__(
        self,
        *,
        ledger: SQLiteResearchLedger,
        scheduler,
        worker_bank: WorkerBank,
        worker_ids: Sequence[str],
        snapshot_provider: IndexedRuntimeSnapshotProvider,
        worker_runtime: WorkerRuntime | None = None,
        worker_selector: WorkerSelectorV0 | None = None,
    ) -> None:
        if snapshot_provider.ledger is not ledger:
            raise ValueError("indexed control loop and snapshot provider must share one ledger")
        super().__init__(
            ledger=ledger,
            scheduler=scheduler,
            worker_bank=worker_bank,
            worker_ids=worker_ids,
            projector=snapshot_provider.thread_state,
            worker_runtime=worker_runtime,
            worker_selector=worker_selector,
            integration_tracker=None,
            verification_tracker=None,
        )
        self.snapshot_provider = snapshot_provider

    def run_many(
        self,
        *,
        signal_provider: SignalProvider,
        context_provider: ContextProvider,
        max_threads: int,
        max_attempts: int,
        integration_backpressure: bool | None = None,
    ) -> ControlBatch:
        if max_threads <= 0:
            raise ValueError("max_threads must be positive")
        if max_attempts <= 0:
            raise ValueError("max_attempts must be positive")

        snapshot = self.snapshot_provider.capture()
        states = snapshot.states
        if not states:
            raise ValueError("runtime has no Work Threads")
        state_by_id = {state.thread_id: state for state in states}

        candidates = [
            SchedulableThread(
                state=state,
                signals=self._signals_for(
                    state,
                    signal_provider,
                    snapshot.integration_overview,
                    snapshot.verification_overview,
                ),
            )
            for state in states
            if state.status != "COMPLETE"
            and not self._is_dependency_blocked(state, state_by_id)
        ]
        if not candidates:
            if any(state.status != "COMPLETE" for state in states):
                raise ValueError("all active Work Threads are dependency-blocked")
            raise ValueError("runtime has no non-complete Work Threads")

        resolved_backpressure = (
            snapshot.integration_overview.global_backpressured
            if integration_backpressure is None
            and snapshot.integration_overview is not None
            else False
            if integration_backpressure is None
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

            assignments, resolved_decision = self._prepare_indexed_assignments(
                selected_state,
                decision,
                previous_worker_id=snapshot.last_worker_ids.get(selected_state.thread_id),
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

        flat_results = self.worker_runtime.run_batch(tuple(all_assignments), self.worker_bank)
        if len(flat_results) != len(all_assignments):
            raise RuntimeError("worker runtime returned a mismatched cross-thread result count")

        resolved_steps: list[ControlStep] = []
        result_offset = 0
        for step, assignment_count in zip(planned_steps, assignment_counts, strict=True):
            step_results = tuple(
                flat_results[result_offset : result_offset + assignment_count]
            )
            result_offset += assignment_count
            resolved_steps.append(replace(step, results=step_results))
        if result_offset != len(flat_results):
            raise RuntimeError("cross-thread result partition did not consume all results")
        return ControlBatch(tuple(resolved_steps))

    def _prepare_indexed_assignments(
        self,
        state: ProjectedState,
        decision: SchedulerDecision,
        *,
        previous_worker_id: str | None,
        context_provider: ContextProvider,
    ) -> tuple[tuple[WorkerAssignment, ...], SchedulerDecision]:
        raw_preparation = context_provider(state, decision)
        preparations = self._preparations_for_width(raw_preparation, decision.width)
        selected_worker_ids = self.worker_selector.choose_many(
            decision.action,
            previous_worker_id=previous_worker_id,
            count=decision.width,
        )
        if len(selected_worker_ids) != decision.width:
            raise RuntimeError("scheduler allocated more distinct workers than available")

        assignments: list[WorkerAssignment] = []
        for worker_id, preparation in zip(selected_worker_ids, preparations, strict=True):
            item = WorkItem(
                work_item_id=uuid.uuid4().hex,
                thread_id=state.thread_id,
                objective=state.objective,
                purpose=decision.purpose or state.purpose,
                projection_revision=state.revision,
                scheduler_decision_id=decision.decision_id,
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

    def _state_index(self) -> dict[str, ProjectedState]:
        return {
            state.thread_id: state
            for state in self.snapshot_provider.thread_state.snapshot_all()
        }
