"""Bounded worker-attempt orchestration over the stable runtime contracts."""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import Protocol, Sequence

from .contracts import AttemptResult, AttemptStatus, WorkItem
from .ledger import SQLiteResearchLedger


class AttemptExecutor(Protocol):
    """Injected learned/tool executor behind the stable Worker Runtime boundary."""

    def __call__(
        self,
        attempt_id: str,
        worker_id: str,
        work_item: WorkItem,
    ) -> AttemptResult: ...


@dataclass(frozen=True, slots=True)
class WorkerAssignment:
    worker_id: str
    work_item: WorkItem

    def validate(self) -> None:
        if not self.worker_id or not self.worker_id.strip():
            raise ValueError("worker_id must be non-empty")
        self.work_item.validate()


class WorkerRuntime:
    """Execute bounded attempts and convert their useful output into ledger events."""

    def __init__(self, ledger: SQLiteResearchLedger) -> None:
        self._ledger = ledger

    def run_attempt(
        self,
        assignment: WorkerAssignment,
        executor: AttemptExecutor,
    ) -> AttemptResult:
        assignment.validate()
        attempt_id = uuid.uuid4().hex
        item = assignment.work_item

        started = self._ledger.append_event(
            event_type="ATTEMPT_STARTED",
            thread_id=item.thread_id,
            attempt_id=attempt_id,
            reference_ids=item.reference_ids,
            payload={
                "work_item_id": item.work_item_id,
                "worker_id": assignment.worker_id,
                "purpose": item.purpose.value,
                "projection_revision": item.projection_revision,
            },
        )

        try:
            result = executor(attempt_id, assignment.worker_id, item)
        except Exception as error:
            self._ledger.append_event(
                event_type="ATTEMPT_CRASHED",
                thread_id=item.thread_id,
                attempt_id=attempt_id,
                parent_event_ids=(started.event_id,),
                payload={
                    "work_item_id": item.work_item_id,
                    "worker_id": assignment.worker_id,
                    "error_type": type(error).__name__,
                    "error": str(error),
                },
            )
            raise

        result.validate()
        self._validate_result(result, assignment, attempt_id)
        self._commit_result(result, parent_event_id=started.event_id)
        return result

    def run_batch(
        self,
        assignments: Sequence[WorkerAssignment],
        executor: AttemptExecutor,
    ) -> tuple[AttemptResult, ...]:
        """Day-1 sequential batch boundary; execution may be vectorized behind it later."""

        return tuple(self.run_attempt(assignment, executor) for assignment in assignments)

    @staticmethod
    def _validate_result(
        result: AttemptResult,
        assignment: WorkerAssignment,
        attempt_id: str,
    ) -> None:
        item = assignment.work_item
        if result.attempt_id != attempt_id:
            raise ValueError("executor returned a mismatched attempt_id")
        if result.work_item_id != item.work_item_id:
            raise ValueError("executor returned a mismatched work_item_id")
        if result.thread_id != item.thread_id:
            raise ValueError("executor returned a mismatched thread_id")
        if result.worker_id != assignment.worker_id:
            raise ValueError("executor returned a mismatched worker_id")

    def _commit_result(self, result: AttemptResult, *, parent_event_id: str) -> None:
        common = {
            "thread_id": result.thread_id,
            "attempt_id": result.attempt_id,
            "parent_event_ids": (parent_event_id,),
        }

        for observation in result.observations:
            self._ledger.append_event(
                event_type="OBSERVATION_RECORDED",
                payload={"observation": observation},
                **common,
            )

        if result.evidence_refs:
            self._ledger.append_event(
                event_type="EVIDENCE_ADDED",
                reference_ids=result.evidence_refs,
                **common,
            )

        for event_type, references in (
            ("HYPOTHESIS_PROPOSED", result.hypotheses_proposed),
            ("HYPOTHESIS_STRENGTHENED", result.hypotheses_strengthened),
            ("HYPOTHESIS_WEAKENED", result.hypotheses_weakened),
            ("HYPOTHESIS_REJECTED", result.hypotheses_rejected),
            ("CONTRADICTION_FOUND", result.contradictions),
            ("POSSIBILITY_ELIMINATED", result.possibilities_eliminated),
        ):
            if references:
                self._ledger.append_event(
                    event_type=event_type,
                    reference_ids=references,
                    **common,
                )

        for question in result.open_questions:
            self._ledger.append_event(
                event_type="OPEN_QUESTION_ADDED",
                payload={"question": question},
                **common,
            )

        for request in result.requested_followups:
            self._ledger.append_event(
                event_type="FOLLOWUP_REQUESTED",
                payload={"request": request},
                **common,
            )

        for side_effect in result.side_effects:
            self._ledger.append_event(
                event_type="SIDE_EFFECT_RECORDED",
                payload={"side_effect": side_effect},
                **common,
            )

        terminal_type = (
            "ATTEMPT_FAILED" if result.status is AttemptStatus.FAILED else "ATTEMPT_COMPLETED"
        )
        self._ledger.append_event(
            event_type=terminal_type,
            payload={
                "status": result.status.value,
                "progress_made": result.progress_made,
                "resource_usage": dict(result.resource_usage),
            },
            **common,
        )
