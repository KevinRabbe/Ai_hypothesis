"""Bounded reference-only context resolution across Work Graph relationships.

Graph relationships never copy histories. This resolver only selects related thread
identities and compact knowledge delta IDs; callers still choose which relation types
and knowledge statuses are relevant to a specific Work Item.
"""

from __future__ import annotations

from collections.abc import Sequence

from .contracts import ProjectedState
from .knowledge import KnowledgeSnapshot, KnowledgeStatus


class WorkGraphContextResolver:
    """Resolve bounded related-thread and knowledge references without copying state."""

    def __init__(
        self,
        states: Sequence[ProjectedState],
        knowledge: KnowledgeSnapshot,
    ) -> None:
        resolved_states = tuple(states)
        self.states = {state.thread_id: state for state in resolved_states}
        if len(self.states) != len(resolved_states):
            raise ValueError("ProjectedState thread IDs must be unique")
        self.knowledge = knowledge

    def related_thread_ids(
        self,
        thread_id: str,
        *,
        include_self: bool = False,
        include_parents: bool = False,
        include_children: bool = False,
        include_dependencies: bool = False,
        include_merged_sources: bool = False,
        include_merge_target: bool = False,
        limit: int,
    ) -> tuple[str, ...]:
        if limit <= 0:
            raise ValueError("limit must be positive")
        try:
            state = self.states[thread_id]
        except KeyError as error:
            raise ValueError(f"unknown Work Thread {thread_id!r}") from error

        candidates: list[str] = []
        if include_self:
            candidates.append(thread_id)
        if include_parents:
            candidates.extend(state.parent_thread_ids)
        if include_children:
            candidates.extend(state.child_thread_ids)
        if include_dependencies:
            candidates.extend(state.dependency_thread_ids)
        if include_merged_sources:
            candidates.extend(state.merged_from_thread_ids)
        if include_merge_target and state.merged_into_thread_id is not None:
            candidates.append(state.merged_into_thread_id)

        ordered: list[str] = []
        seen: set[str] = set()
        for candidate in candidates:
            if candidate in seen:
                continue
            if candidate not in self.states:
                raise ValueError(
                    f"Work Graph relationship references unknown thread {candidate!r}"
                )
            ordered.append(candidate)
            seen.add(candidate)
            if len(ordered) >= limit:
                break
        return tuple(ordered)

    def related_knowledge_delta_ids(
        self,
        thread_id: str,
        *,
        statuses: Sequence[KnowledgeStatus],
        include_self: bool = False,
        include_parents: bool = False,
        include_children: bool = False,
        include_dependencies: bool = False,
        include_merged_sources: bool = False,
        include_merge_target: bool = False,
        thread_limit: int,
        knowledge_limit: int,
    ) -> tuple[str, ...]:
        if knowledge_limit <= 0:
            raise ValueError("knowledge_limit must be positive")
        allowed_statuses = frozenset(statuses)
        if not allowed_statuses:
            return ()
        related = set(
            self.related_thread_ids(
                thread_id,
                include_self=include_self,
                include_parents=include_parents,
                include_children=include_children,
                include_dependencies=include_dependencies,
                include_merged_sources=include_merged_sources,
                include_merge_target=include_merge_target,
                limit=thread_limit,
            )
        )
        selected: list[str] = []
        for record in self.knowledge.records:
            if record.thread_id not in related:
                continue
            if record.status not in allowed_statuses:
                continue
            selected.append(record.delta_id)
            if len(selected) >= knowledge_limit:
                break
        return tuple(selected)
