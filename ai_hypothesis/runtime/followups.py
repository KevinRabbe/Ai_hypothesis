"""Turn durable worker follow-up requests into persistent child Work Threads.

Workers only request follow-up work. Deterministic runtime code owns Work Thread
identity and graph mutation so retries cannot create duplicate branches.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass

from .contracts import LedgerEvent, WorkPurpose
from .ledger import SQLiteResearchLedger
from .projector import ThreadStateProjector


@dataclass(frozen=True, slots=True)
class FollowupRequest:
    request_event_id: str
    sequence: int
    parent_thread_id: str
    objective: str


@dataclass(frozen=True, slots=True)
class FollowupSnapshot:
    revision: int
    pending: tuple[FollowupRequest, ...]
    materialized_request_event_ids: tuple[str, ...]


class FollowupMaterializer:
    """Boundedly materialize oldest pending follow-ups as child Work Threads."""

    def __init__(self, ledger: SQLiteResearchLedger) -> None:
        self.ledger = ledger

    def snapshot(self) -> FollowupSnapshot:
        return self.project(self.ledger.read_all_events())

    def project(self, events: tuple[LedgerEvent, ...]) -> FollowupSnapshot:
        requests: list[FollowupRequest] = []
        materialized: set[str] = set()
        created_threads: set[str] = set()
        revision = 0
        previous_sequence = -1

        for event in events:
            event.validate()
            if event.sequence <= previous_sequence:
                raise ValueError("events must be in strictly increasing sequence order")
            previous_sequence = event.sequence
            revision = event.sequence

            if event.event_type == "THREAD_CREATED" and event.thread_id is not None:
                created_threads.add(event.thread_id)
            elif event.event_type == "FOLLOWUP_REQUESTED":
                if event.thread_id is None:
                    raise ValueError("FOLLOWUP_REQUESTED requires a parent thread")
                request = event.payload.get("request")
                if not isinstance(request, str) or not request.strip():
                    raise ValueError("FOLLOWUP_REQUESTED requires a non-empty request")
                requests.append(
                    FollowupRequest(
                        request_event_id=event.event_id,
                        sequence=event.sequence,
                        parent_thread_id=event.thread_id,
                        objective=request,
                    )
                )
            elif event.event_type == "FOLLOWUP_MATERIALIZED":
                if len(event.parent_event_ids) != 1:
                    raise ValueError(
                        "FOLLOWUP_MATERIALIZED requires exactly one request parent event"
                    )
                materialized.add(event.parent_event_ids[0])

        for request in requests:
            if request.parent_thread_id not in created_threads:
                raise ValueError(
                    f"follow-up request {request.request_event_id!r} references missing "
                    f"Work Thread {request.parent_thread_id!r}"
                )

        pending = tuple(
            request
            for request in requests
            if request.request_event_id not in materialized
        )
        return FollowupSnapshot(
            revision=revision,
            pending=pending,
            materialized_request_event_ids=tuple(sorted(materialized)),
        )

    def materialize(
        self,
        *,
        limit: int,
        purpose: WorkPurpose = WorkPurpose.EXPLORE,
    ) -> tuple[str, ...]:
        if limit <= 0:
            raise ValueError("limit must be positive")

        events = self.ledger.read_all_events()
        snapshot = self.project(events)
        selected = snapshot.pending[:limit]
        if not selected:
            return ()

        states = {
            state.thread_id: state
            for state in ThreadStateProjector().project_all(events)
        }
        existing_forks = {
            (state.thread_id, child_id)
            for state in states.values()
            for child_id in state.child_thread_ids
        }
        children: list[str] = []

        for request in selected:
            child_id = self.child_thread_id(request.request_event_id)
            if child_id not in states:
                self.ledger.append_event(
                    event_type="THREAD_CREATED",
                    thread_id=child_id,
                    payload={
                        "objective": request.objective,
                        "purpose": purpose.value,
                        "status": "ACTIVE",
                        "source_followup_event_id": request.request_event_id,
                    },
                )

            edge = (request.parent_thread_id, child_id)
            if edge not in existing_forks:
                self.ledger.append_event(
                    event_type="THREAD_FORKED",
                    thread_id=request.parent_thread_id,
                    reference_ids=(child_id,),
                    parent_event_ids=(request.request_event_id,),
                    payload={
                        "child_thread_id": child_id,
                        "source_followup_event_id": request.request_event_id,
                    },
                )
                existing_forks.add(edge)

            self.ledger.append_event(
                event_type="FOLLOWUP_MATERIALIZED",
                thread_id=request.parent_thread_id,
                reference_ids=(child_id,),
                parent_event_ids=(request.request_event_id,),
                payload={
                    "child_thread_id": child_id,
                    "purpose": purpose.value,
                },
            )
            states[child_id] = states.get(child_id)  # identity is now reserved durably
            children.append(child_id)

        return tuple(children)

    @staticmethod
    def child_thread_id(request_event_id: str) -> str:
        if not request_event_id or not request_event_id.strip():
            raise ValueError("request_event_id must be non-empty")
        return "followup-" + uuid.uuid5(
            uuid.NAMESPACE_URL,
            f"ai-hypothesis-followup:{request_event_id}",
        ).hex
