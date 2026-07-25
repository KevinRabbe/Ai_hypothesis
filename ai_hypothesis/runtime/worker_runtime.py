"""Bounded worker-attempt orchestration over the stable runtime contracts."""

from __future__ import annotations

import uuid
from collections import Counter
from dataclasses import dataclass
from typing import Protocol, Sequence

from .contracts import AttemptResult, AttemptStatus, WorkItem
from .ledger import SQLiteResearchLedger


@dataclass(frozen=True, slots=True)
class AttemptRequest:
    attempt_id: str
    worker_id: str
    work_item: WorkItem


class WorkerBank(Protocol):
    def execute_batch(self, requests: Sequence[AttemptRequest]) -> Sequence[AttemptResult]: ...


@dataclass(frozen=True, slots=True)
class WorkerAssignment:
    worker_id: str
    work_item: WorkItem

    def validate(self) -> None:
        if not self.worker_id or not self.worker_id.strip():
            raise ValueError("worker_id must be non-empty")
        self.work_item.validate()


class WorkerRuntime:
    def __init__(self, ledger: SQLiteResearchLedger) -> None:
        self._ledger = ledger

    def run_attempt(self, assignment: WorkerAssignment, worker_bank: WorkerBank) -> AttemptResult:
        return self.run_batch((assignment,), worker_bank)[0]

    def run_batch(self, assignments: Sequence[WorkerAssignment], worker_bank: WorkerBank) -> tuple[AttemptResult, ...]:
        if not assignments:
            return ()

        scheduler_decisions = (
            self._latest_scheduler_decisions()
            if any(
                assignment.work_item.scheduler_decision_id is None
                for assignment in assignments
            )
            else {}
        )
        requests: list[AttemptRequest] = []
        started_event_ids: dict[str, str] = {}
        for assignment in assignments:
            assignment.validate()
            request = AttemptRequest(uuid.uuid4().hex, assignment.worker_id, assignment.work_item)
            requests.append(request)
            item = assignment.work_item
            decision_id = item.scheduler_decision_id or scheduler_decisions.get(
                (item.thread_id, item.projection_revision)
            )
            started_event_ids[request.attempt_id] = self._record_started(
                request,
                scheduler_decision_id=decision_id,
            )

        try:
            results = tuple(worker_bank.execute_batch(tuple(requests)))
        except Exception as error:
            for request in requests:
                self._record_execution_error(request, started_event_ids[request.attempt_id], "ATTEMPT_CRASHED", error)
            raise

        if len(results) != len(requests):
            error = ValueError(f"worker bank returned {len(results)} results for {len(requests)} requests")
            for request in requests:
                self._record_execution_error(request, started_event_ids[request.attempt_id], "ATTEMPT_INVALID_RESULT", error)
            raise error

        needs_identity_history = any(
            result.evidence or result.knowledge_deltas or result.knowledge_assessments
            for result in results
        )
        recorded_evidence_ids, recorded_delta_ids = (
            self._recorded_generated_ids() if needs_identity_history else (set(), set())
        )
        generated_counts = Counter(
            [
                contribution.evidence_id
                for result in results
                for contribution in result.evidence
            ]
            + [
                delta.delta_id
                for result in results
                for delta in result.knowledge_deltas
            ]
        )
        batch_collisions = {
            object_id for object_id, count in generated_counts.items() if count > 1
        }
        recorded_generated_ids = recorded_evidence_ids | recorded_delta_ids

        valid_pairs: list[tuple[AttemptRequest, AttemptResult]] = []
        validation_errors: list[Exception] = []
        for request, result in zip(requests, results, strict=True):
            try:
                result.validate()
                self._validate_result(
                    result,
                    request,
                    recorded_generated_ids=recorded_generated_ids,
                    recorded_delta_ids=recorded_delta_ids,
                    batch_collisions=batch_collisions,
                )
            except Exception as error:
                self._record_execution_error(request, started_event_ids[request.attempt_id], "ATTEMPT_INVALID_RESULT", error)
                validation_errors.append(error)
            else:
                valid_pairs.append((request, result))

        for request, result in valid_pairs:
            self._commit_result(result, parent_event_id=started_event_ids[request.attempt_id])
        if validation_errors:
            raise validation_errors[0]
        return results

    def _record_started(
        self,
        request: AttemptRequest,
        *,
        scheduler_decision_id: str | None,
    ) -> str:
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
                "scheduler_decision_id": scheduler_decision_id,
            },
        )
        return event.event_id

    def _record_execution_error(self, request: AttemptRequest, parent_event_id: str, event_type: str, error: Exception) -> None:
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

    def _validate_result(
        self,
        result: AttemptResult,
        request: AttemptRequest,
        *,
        recorded_generated_ids: set[str],
        recorded_delta_ids: set[str],
        batch_collisions: set[str],
    ) -> None:
        item = request.work_item
        if result.attempt_id != request.attempt_id:
            raise ValueError("worker bank returned a mismatched attempt_id")
        if result.work_item_id != item.work_item_id:
            raise ValueError("worker bank returned a mismatched work_item_id")
        if result.thread_id != item.thread_id:
            raise ValueError("worker bank returned a mismatched thread_id")
        if result.worker_id != request.worker_id:
            raise ValueError("worker bank returned a mismatched worker_id")

        generated_ids = {
            contribution.evidence_id for contribution in result.evidence
        } | {delta.delta_id for delta in result.knowledge_deltas}
        if generated_ids & batch_collisions:
            raise ValueError("worker batch produced colliding durable object IDs")
        if generated_ids & recorded_generated_ids:
            raise ValueError("worker reused an existing durable evidence or knowledge ID")

        new_evidence_ids = {contribution.evidence_id for contribution in result.evidence}
        authorized_reference_ids = set(item.reference_ids) | new_evidence_ids
        for disposition in result.evidence_dispositions:
            unauthorized = set(disposition.evidence_ids) - authorized_reference_ids
            if unauthorized:
                raise ValueError("worker attempted to disposition evidence outside its Work Item authority")
        for delta in result.knowledge_deltas:
            unauthorized = set(delta.reference_ids) - authorized_reference_ids
            if unauthorized:
                raise ValueError("knowledge delta references information outside its Work Item authority")
            if delta.thread_id is not None and delta.thread_id != item.thread_id and delta.thread_id not in item.parent_ids:
                raise ValueError("knowledge delta targets a thread outside its Work Item authority")
            for causal_event_id in delta.causal_event_ids:
                if self._ledger.get_event(causal_event_id) is None:
                    raise ValueError("knowledge delta references a nonexistent causal event")
        for assessment in result.knowledge_assessments:
            unauthorized = set(assessment.delta_ids) - set(item.reference_ids)
            if unauthorized:
                raise ValueError("worker attempted to assess knowledge outside its Work Item authority")
            if not set(assessment.delta_ids) <= recorded_delta_ids:
                raise ValueError("worker attempted to assess a nonexistent knowledge delta")

    def _latest_scheduler_decisions(self) -> dict[tuple[str, int], str]:
        decisions: dict[tuple[str, int], str] = {}
        for event in self._ledger.read_all_events():
            if event.event_type != "SCHEDULER_DECISION_RECORDED" or event.thread_id is None:
                continue
            decision_id = event.payload.get("decision_id")
            revision = event.payload.get("projection_revision")
            if (
                isinstance(decision_id, str)
                and decision_id
                and isinstance(revision, int)
                and revision >= 0
            ):
                decisions[(event.thread_id, revision)] = decision_id
        return decisions

    def _recorded_generated_ids(self) -> tuple[set[str], set[str]]:
        evidence_ids: set[str] = set()
        delta_ids: set[str] = set()
        for event in self._ledger.read_all_events():
            if event.event_type == "EVIDENCE_ADDED":
                evidence_id = event.payload.get("evidence_id")
                if isinstance(evidence_id, str) and evidence_id:
                    evidence_ids.add(evidence_id)
            elif event.event_type == "KNOWLEDGE_DELTA_RECORDED":
                delta_id = event.payload.get("delta_id")
                if isinstance(delta_id, str) and delta_id:
                    delta_ids.add(delta_id)
        return evidence_ids, delta_ids

    def _commit_result(self, result: AttemptResult, *, parent_event_id: str) -> None:
        common = {"thread_id": result.thread_id, "attempt_id": result.attempt_id, "parent_event_ids": (parent_event_id,)}
        for observation in result.observations:
            self._ledger.append_event(event_type="OBSERVATION_RECORDED", payload={"observation": observation}, **common)
        for contribution in result.evidence:
            self._ledger.append_event(
                event_type="EVIDENCE_ADDED",
                reference_ids=(contribution.evidence_id, *contribution.reference_ids),
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
            self._ledger.append_event(event_type="EVIDENCE_REFERENCED", reference_ids=result.evidence_refs, **common)
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
                    "source_reference_ids": list(delta.reference_ids),
                    "causal_event_ids": list(delta.causal_event_ids),
                },
            )
        for assessment in result.knowledge_assessments:
            payload = {"assessment": assessment.assessment.value}
            if assessment.reason is not None:
                payload["reason"] = assessment.reason
            self._ledger.append_event(
                event_type="KNOWLEDGE_ASSESSMENT_RECORDED",
                reference_ids=assessment.delta_ids,
                payload=payload,
                **common,
            )
        for disposition in result.evidence_dispositions:
            payload = {"disposition": disposition.disposition.value}
            if disposition.reason is not None:
                payload["reason"] = disposition.reason
            self._ledger.append_event(event_type="INTEGRATION_DISPOSITION_RECORDED", reference_ids=disposition.evidence_ids, payload=payload, **common)
        for event_type, references in (
            ("HYPOTHESIS_PROPOSED", result.hypotheses_proposed),
            ("HYPOTHESIS_STRENGTHENED", result.hypotheses_strengthened),
            ("HYPOTHESIS_WEAKENED", result.hypotheses_weakened),
            ("HYPOTHESIS_REJECTED", result.hypotheses_rejected),
            ("CONTRADICTION_FOUND", result.contradictions),
            ("POSSIBILITY_ELIMINATED", result.possibilities_eliminated),
        ):
            if references:
                self._ledger.append_event(event_type=event_type, reference_ids=references, **common)
        for question in result.open_questions:
            self._ledger.append_event(event_type="OPEN_QUESTION_ADDED", payload={"question": question}, **common)
        for followup in result.requested_followups:
            self._ledger.append_event(event_type="FOLLOWUP_REQUESTED", payload={"request": followup}, **common)
        for side_effect in result.side_effects:
            self._ledger.append_event(event_type="SIDE_EFFECT_RECORDED", payload={"side_effect": side_effect}, **common)
        terminal_type = {AttemptStatus.COMPLETED: "ATTEMPT_COMPLETED", AttemptStatus.PARTIAL: "ATTEMPT_PARTIAL", AttemptStatus.FAILED: "ATTEMPT_FAILED"}[result.status]
        self._ledger.append_event(
            event_type=terminal_type,
            payload={"status": result.status.value, "progress_made": result.progress_made, "resource_usage": dict(result.resource_usage)},
            **common,
        )
