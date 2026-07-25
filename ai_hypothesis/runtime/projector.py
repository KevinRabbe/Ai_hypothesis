"""Rebuild bounded Work Thread and Work Graph state from append-only ledger events."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any

from .contracts import LedgerEvent, ProjectedState, WorkPurpose


@dataclass(slots=True)
class _MutableThreadState:
    created: bool = False
    objective: str = ""
    status: str = "ACTIVE"
    purpose: WorkPurpose = WorkPurpose.EXPLORE
    revision: int = 0
    references: list[str] = field(default_factory=list)
    hypotheses: list[str] = field(default_factory=list)
    contradictions: list[str] = field(default_factory=list)
    open_questions: list[str] = field(default_factory=list)
    dependencies: list[str] = field(default_factory=list)
    parents: list[str] = field(default_factory=list)
    children: list[str] = field(default_factory=list)
    merged_from: list[str] = field(default_factory=list)
    merged_into: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class ThreadStateProjector:
    """Deterministically fold ledger history into rebuildable Work Thread views."""

    def project(self, events: Iterable[LedgerEvent], *, thread_id: str) -> ProjectedState:
        if not thread_id:
            raise ValueError("thread_id must be non-empty")
        for state in self.project_all(events):
            if state.thread_id == thread_id:
                return state
        raise ValueError(f"thread {thread_id!r} has no THREAD_CREATED event")

    def project_all(self, events: Iterable[LedgerEvent]) -> tuple[ProjectedState, ...]:
        """Project every created Work Thread with one ordered pass plus graph checks."""

        mutable: dict[str, _MutableThreadState] = {}
        creation_order: list[str] = []
        fork_edges: list[tuple[str, str]] = []
        merge_edges: list[tuple[str, str]] = []
        previous_sequence = -1

        for event in events:
            event.validate()
            if event.sequence <= previous_sequence:
                raise ValueError(
                    "events must be supplied in strictly increasing sequence order"
                )
            previous_sequence = event.sequence

            if event.event_type == "THREAD_FORKED":
                parent_id = self._require_thread_id(event)
                child_ids = self._relation_targets(event, "THREAD_FORKED")
                fork_edges.extend((parent_id, child_id) for child_id in child_ids)
            elif event.event_type == "THREAD_MERGED":
                target_id = self._require_thread_id(event)
                source_ids = self._relation_targets(event, "THREAD_MERGED")
                merge_edges.extend((source_id, target_id) for source_id in source_ids)

            if event.thread_id is None:
                continue

            state = mutable.setdefault(event.thread_id, _MutableThreadState())
            state.revision = event.sequence
            _extend_unique(state.references, event.reference_ids)
            self._apply_thread_event(event.thread_id, state, event, creation_order)

        created_ids = {thread_id for thread_id, state in mutable.items() if state.created}
        self._apply_forks(mutable, created_ids, fork_edges)
        self._apply_merges(mutable, created_ids, merge_edges)
        self._validate_dependencies(mutable, created_ids)
        self._validate_acyclic(
            {thread_id: tuple(mutable[thread_id].children) for thread_id in created_ids},
            relation_name="fork ancestry",
        )
        self._validate_acyclic(
            {thread_id: tuple(mutable[thread_id].dependencies) for thread_id in created_ids},
            relation_name="dependency",
        )

        return tuple(
            self._freeze(thread_id, mutable[thread_id]) for thread_id in creation_order
        )

    @staticmethod
    def _apply_thread_event(
        thread_id: str,
        state: _MutableThreadState,
        event: LedgerEvent,
        creation_order: list[str],
    ) -> None:
        if event.event_type == "THREAD_CREATED":
            if state.created:
                raise ValueError(f"thread {thread_id!r} was created more than once")
            state.objective = _require_payload_text(event, "objective")
            raw_purpose = event.payload.get("purpose", WorkPurpose.EXPLORE.value)
            try:
                state.purpose = WorkPurpose(str(raw_purpose))
            except ValueError as error:
                raise ValueError(f"invalid thread purpose {raw_purpose!r}") from error
            state.status = str(event.payload.get("status", "ACTIVE"))
            if not state.status:
                raise ValueError("THREAD_CREATED status must be non-empty")
            state.created = True
            creation_order.append(thread_id)
        elif event.event_type == "THREAD_PURPOSE_SET":
            state.purpose = WorkPurpose(_require_payload_text(event, "purpose"))
        elif event.event_type == "THREAD_STATUS_SET":
            state.status = _require_payload_text(event, "status")
        elif event.event_type == "THREAD_PAUSED":
            state.status = "PAUSED"
        elif event.event_type == "THREAD_COMPLETED":
            state.status = "COMPLETE"
        elif event.event_type == "HYPOTHESIS_PROPOSED":
            _extend_unique(state.hypotheses, event.reference_ids)
        elif event.event_type == "HYPOTHESIS_REJECTED":
            _remove_all(state.hypotheses, event.reference_ids)
        elif event.event_type == "CONTRADICTION_FOUND":
            _extend_unique(state.contradictions, event.reference_ids)
        elif event.event_type == "CONTRADICTION_RESOLVED":
            _remove_all(state.contradictions, event.reference_ids)
        elif event.event_type == "OPEN_QUESTION_ADDED":
            _extend_unique(
                state.open_questions,
                (_require_payload_text(event, "question"),),
            )
        elif event.event_type == "OPEN_QUESTION_RESOLVED":
            _remove_all(
                state.open_questions,
                (_require_payload_text(event, "question"),),
            )
        elif event.event_type == "DEPENDENCY_ADDED":
            _extend_unique(state.dependencies, event.reference_ids)
        elif event.event_type == "DEPENDENCY_REMOVED":
            _remove_all(state.dependencies, event.reference_ids)
        elif event.event_type == "THREAD_METADATA_UPDATED":
            state.metadata.update(event.payload)

    @staticmethod
    def _apply_forks(
        mutable: Mapping[str, _MutableThreadState],
        created_ids: set[str],
        fork_edges: Iterable[tuple[str, str]],
    ) -> None:
        for parent_id, child_id in fork_edges:
            _require_created_relation(parent_id, child_id, created_ids, "THREAD_FORKED")
            if parent_id == child_id:
                raise ValueError("thread cannot fork itself")
            _extend_unique(mutable[parent_id].children, (child_id,))
            _extend_unique(mutable[child_id].parents, (parent_id,))

    @staticmethod
    def _apply_merges(
        mutable: Mapping[str, _MutableThreadState],
        created_ids: set[str],
        merge_edges: Iterable[tuple[str, str]],
    ) -> None:
        for source_id, target_id in merge_edges:
            _require_created_relation(source_id, target_id, created_ids, "THREAD_MERGED")
            if source_id == target_id:
                raise ValueError("thread cannot merge into itself")
            source = mutable[source_id]
            if source.merged_into is not None and source.merged_into != target_id:
                raise ValueError(
                    f"thread {source_id!r} was merged into multiple targets"
                )
            source.merged_into = target_id
            _extend_unique(mutable[target_id].merged_from, (source_id,))

    @staticmethod
    def _validate_dependencies(
        mutable: Mapping[str, _MutableThreadState], created_ids: set[str]
    ) -> None:
        for thread_id in created_ids:
            for dependency_id in mutable[thread_id].dependencies:
                if dependency_id not in created_ids:
                    raise ValueError(
                        f"dependency from {thread_id!r} references missing thread "
                        f"{dependency_id!r}"
                    )
                if dependency_id == thread_id:
                    raise ValueError("thread cannot depend on itself")

    @staticmethod
    def _validate_acyclic(
        adjacency: Mapping[str, tuple[str, ...]], *, relation_name: str
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
    def _require_thread_id(event: LedgerEvent) -> str:
        if event.thread_id is None:
            raise ValueError(f"{event.event_type} requires thread_id")
        return event.thread_id

    @staticmethod
    def _relation_targets(event: LedgerEvent, event_type: str) -> tuple[str, ...]:
        if not event.reference_ids:
            raise ValueError(f"{event_type} requires at least one target thread ID")
        if any(not reference_id for reference_id in event.reference_ids):
            raise ValueError(f"{event_type} target IDs must be non-empty")
        return event.reference_ids

    @staticmethod
    def _freeze(thread_id: str, state: _MutableThreadState) -> ProjectedState:
        projected = ProjectedState(
            revision=state.revision,
            thread_id=thread_id,
            objective=state.objective,
            status=state.status,
            purpose=state.purpose,
            reference_ids=tuple(state.references),
            hypothesis_ids=tuple(state.hypotheses),
            contradiction_ids=tuple(state.contradictions),
            open_questions=tuple(state.open_questions),
            dependency_thread_ids=tuple(state.dependencies),
            parent_thread_ids=tuple(state.parents),
            child_thread_ids=tuple(state.children),
            merged_from_thread_ids=tuple(state.merged_from),
            merged_into_thread_id=state.merged_into,
            metadata=dict(state.metadata),
        )
        projected.validate()
        return projected


def _require_created_relation(
    source_id: str,
    target_id: str,
    created_ids: set[str],
    event_type: str,
) -> None:
    if source_id not in created_ids:
        raise ValueError(f"{event_type} references missing thread {source_id!r}")
    if target_id not in created_ids:
        raise ValueError(f"{event_type} references missing thread {target_id!r}")


def _require_payload_text(event: LedgerEvent, key: str) -> str:
    value = event.payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(
            f"{event.event_type} requires non-empty string payload field {key!r}"
        )
    return value


def _extend_unique(target: list[str], values: Iterable[str]) -> None:
    present = set(target)
    for value in values:
        if value and value not in present:
            target.append(value)
            present.add(value)


def _remove_all(target: list[str], values: Iterable[str]) -> None:
    removals = set(values)
    if not removals:
        return
    target[:] = [value for value in target if value not in removals]
