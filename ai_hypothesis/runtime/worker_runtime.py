"""Bounded worker-attempt orchestration over the stable runtime contracts."""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import Protocol, Sequence

from .contracts import AttemptResult, AttemptStatus, WorkItem
from .ledger import SQLiteResearchLedger


@dataclass(frozen=True, slots=True)
class AttemptRequest:
    """One runtime-owned attempt identity sent to the Worker Bank."""

    attempt_id: str
    worker_id: str
    work_item: WorkItem


class WorkerBank(Protocol):
    """Batch-capable learned execution boundary.

    Day-1 implementations may execute requests sequentially. GPU implementations
    may vectorize the same requests without changing Worker Runtime callers.
    """

    def execute_batch(
        self,
        requests: Sequence[AttemptRequest],
    ) -> Sequence[AttemptResult]: ...


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
        worker_bank: WorkerBank,
    ) -> AttemptResult:
        return self.run_batch((assignment,), worker_bank)[0]

    def run_batch(
        self,
        assignments: Sequence[WorkerAssignment],
        worker_bank: WorkerBank,
    ) -> tuple[AttemptResult, ...]:
        if not assignments:
            return ()

        requests: list[AttemptRequest] = []
        started_event_ids: dict[str, str] = {}
        for assignment in assignments:
            assignment.validate()
            request = AttemptRequest(
                attempt_id=uuid.uuid4().hex,
                worker_id=assignment.worker_id,
                work_item=assignment.work_item,
            )
            requests.append(request)
            started_event_ids[request.attempt_id] = self._record_started(request)

        try:
            results = tuple(worker_bank.execute_batch(tuple(requests)))
        except Exception as error:
            for request in requests:
                self._record_execution_error(
                    request,
                    started_event_ids[request.attempt_id],
                    "ATTEMPT_CRASHED",
                    error,
                )
            raise

        if len(results) != len(requests):
            error = ValueError(
                f"worker bank returned {len(results)} results for {len(requests)} requests"
            )
            for request in requests:
                self._record_execution_error(
                    request,
                    started_event_ids[request.attempt_id],
                    "ATTEMPT_INVALID_RESULT",
                    error,
                )
            raise error

        valid_pairs: list[tuple[AttemptRequest, AttemptResult]] = []
        validation_errors: list[Exception] = []
        for request, result in zip(requests, results, strict=True):
            try:
                result.validate()
                self._validate_result(result, request)
            except Exception as error:
                self._record_execution_error(
                    request,
                    started_event_ids[request.attempt_id],
                    "ATTEMPT_INVALID_RESULT",
                    error,
                )
                validation_errors.append(error)
            else:
                valid_pairs.append((request, result))

        # Never discard useful completed work merely because a peer result in the
        # same batch is malformed. Commit every valid result before surfacing error.
        for request, result in valid_pairs:
            self._commit_result(
                result,
                parent_event_id=started_event_ids[request.attempt_id],
            )

        if validation_errors:
            raise validation_errors[0]

        return results

    def _record_started(self, request: AttemptRequest) -> str:
        item = request.work_item
        event = self._ledger.append_event(
            event_type="ATTEMPT_STARTED",
            thread_id=item.thread_id,
            attempt_id=request.attempt_id,
            reference_ids=item.reference_ids,
            payload={
                "work_item_id": item.work_item_id,
                "worker_id": request.worker_id,
                "purpose": item.purpose.value,
                "projection_revision": item.projection_revision,
            },
        )
        return event.event_id

    def _record_execution_error(
        self,
        request: AttemptRequest,
        parent_event_id: str,
        event_type: str,
        error: Exception,
    ) -> None:
        item = request.work_item
        self._ledger.append_event(
            event_type=event_type,
            thread_id=item.thread_id,
            attempt_id=request.attempt_id,
            parent_event_ids=(parent_event_id,),
            payload={
                "work_item_id": item.work_item_id,
                "worker_id": request.worker_id,
                "error_type": type(error).__name__,
                "error": str(error),
            },
        )

    @staticmethod
    def _validate_result(result: AttemptResult, request: AttemptRequest) -> None:
        item = request.work_item
        if result.attempt_id != request.attempt_id:
            raise ValueError("worker bank returned a mismatched attempt_id")
        if result.work_item_id != item.work_item_id:
            raise ValueError("worker bank returned a mismatched work_item_id")
        if result.thread_id != item.thread_id:
            raise ValueError("worker bank returned a mismatched thread_id")
        if result.worker_id != request.worker_id:
            raise ValueError("worker bank returned a mismatched worker_id")

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

        for contribution in result.evidence:
            references = (contribution.evidence_id, *contribution.reference_ids)
            self._ledger.append_event(
                event_type="EVIDENCE_ADDED",
                reference_ids=references,
                payload={
                    "evidence_id": contribution.evidence_id,
                    "kind": contribution.kind,
                    "summary": contribution.summary,
                    "strength": contribution.strength,
                    "uncertainty": contribution.uncertainty,
                    "data": dict(contribution.data),
                },
                **common,
            )

        if result.evidence_refs:
            self._ledger.append_event(
                event_type="EVIDENCE_REFERENCED",
                reference_ids=result.evidence_refs,
                **common,
            )

        for delta in result.knowledge_deltas:
            self._ledger.append_event(
                event_type="KNOWLEDGE_DELTA_RECORDED",
                thread_id=delta.thread_id or result.thread_id,
                attempt_id=result.attempt_id,
                reference_ids=(delta.delta_id, *delta.reference_ids),
                parent_event_ids=(parent_event_id, *delta.causal_event_ids),
                payload={
                    "delta_id": delta.delta_id,
                    "kind": delta.kind,
                    "summary": delta.summary,
                },
            )

        for disposition in result.evidence_dispositions:
            payload = {"disposition": disposition.disposition.value}
            if disposition.reason is not None:
                payload["reason"] = disposition.reason
            self._ledger.append_event(
                event_type="INTEGRATION_DISPOSITION_RECORDED",
                reference_ids=disposition.evidence_ids,
                payload=payload,
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

        terminal_type = {
            AttemptStatus.COMPLETED: "ATTEMPT_COMPLETED",
            AttemptStatus.PARTIAL: "ATTEMPT_PARTIAL",
            AttemptStatus.FAILED: "ATTEMPT_FAILED",
        }[result.status]
        self._ledger.append_event(
            event_type=terminal_type,
            payload={
                "status": result.status.value,
                "progress_made": result.progress_made,
                "resource_usage": dict(result.resource_usage),
            },
            **common,
        )
