"""Tests reference-only knowledge reuse across Work Graph relationships."""

from __future__ import annotations

import unittest

from ai_hypothesis.runtime.contracts import ProjectedState, WorkPurpose
from ai_hypothesis.runtime.graph_context import WorkGraphContextResolver
from ai_hypothesis.runtime.knowledge import (
    KnowledgeRecord,
    KnowledgeSnapshot,
    KnowledgeStatus,
)


def _state(thread_id: str, **relationships) -> ProjectedState:
    return ProjectedState(
        revision=10,
        thread_id=thread_id,
        objective=thread_id,
        status="ACTIVE",
        purpose=WorkPurpose.PROGRESS,
        **relationships,
    )


def _knowledge(
    delta_id: str,
    thread_id: str,
    status: KnowledgeStatus,
    sequence: int,
) -> KnowledgeRecord:
    return KnowledgeRecord(
        delta_id=delta_id,
        kind="TEST",
        summary=delta_id,
        source_reference_ids=(f"evidence-{delta_id}",),
        causal_event_ids=(),
        thread_id=thread_id,
        created_event_id=f"event-{delta_id}",
        created_sequence=sequence,
        status=status,
    )


class WorkGraphContextResolverTests(unittest.TestCase):
    def test_forked_child_can_reference_parent_knowledge_without_copying_state(self) -> None:
        states = (
            _state("parent", child_thread_ids=("child",)),
            _state("child", parent_thread_ids=("parent",)),
        )
        snapshot = KnowledgeSnapshot(
            revision=12,
            records=(
                _knowledge("delta-parent", "parent", KnowledgeStatus.VERIFIED, 1),
            ),
        )
        resolver = WorkGraphContextResolver(states, snapshot)

        threads = resolver.related_thread_ids(
            "child",
            include_parents=True,
            limit=4,
        )
        deltas = resolver.related_knowledge_delta_ids(
            "child",
            statuses=(KnowledgeStatus.VERIFIED,),
            include_parents=True,
            thread_limit=4,
            knowledge_limit=4,
        )

        self.assertEqual(threads, ("parent",))
        self.assertEqual(deltas, ("delta-parent",))
        self.assertEqual(states[1].reference_ids, ())

    def test_merge_target_can_reference_source_knowledge(self) -> None:
        states = (
            _state("source-a", merged_into_thread_id="target"),
            _state("source-b", merged_into_thread_id="target"),
            _state(
                "target",
                merged_from_thread_ids=("source-a", "source-b"),
            ),
        )
        snapshot = KnowledgeSnapshot(
            revision=20,
            records=(
                _knowledge("delta-a", "source-a", KnowledgeStatus.VERIFIED, 1),
                _knowledge("delta-b", "source-b", KnowledgeStatus.PROVISIONAL, 2),
            ),
        )
        resolver = WorkGraphContextResolver(states, snapshot)

        verified = resolver.related_knowledge_delta_ids(
            "target",
            statuses=(KnowledgeStatus.VERIFIED,),
            include_merged_sources=True,
            thread_limit=8,
            knowledge_limit=8,
        )
        all_active = resolver.related_knowledge_delta_ids(
            "target",
            statuses=(KnowledgeStatus.VERIFIED, KnowledgeStatus.PROVISIONAL),
            include_merged_sources=True,
            thread_limit=8,
            knowledge_limit=8,
        )

        self.assertEqual(verified, ("delta-a",))
        self.assertEqual(all_active, ("delta-a", "delta-b"))

    def test_dependency_context_is_opt_in_not_automatic(self) -> None:
        states = (
            _state("dependency"),
            _state("dependent", dependency_thread_ids=("dependency",)),
        )
        snapshot = KnowledgeSnapshot(
            revision=5,
            records=(
                _knowledge("delta-dependency", "dependency", KnowledgeStatus.VERIFIED, 1),
            ),
        )
        resolver = WorkGraphContextResolver(states, snapshot)

        none = resolver.related_knowledge_delta_ids(
            "dependent",
            statuses=(KnowledgeStatus.VERIFIED,),
            thread_limit=4,
            knowledge_limit=4,
        )
        explicit = resolver.related_knowledge_delta_ids(
            "dependent",
            statuses=(KnowledgeStatus.VERIFIED,),
            include_dependencies=True,
            thread_limit=4,
            knowledge_limit=4,
        )

        self.assertEqual(none, ())
        self.assertEqual(explicit, ("delta-dependency",))

    def test_thread_and_knowledge_limits_are_hard_caps(self) -> None:
        states = (
            _state("p1"),
            _state("p2"),
            _state("p3"),
            _state("child", parent_thread_ids=("p1", "p2", "p3")),
        )
        snapshot = KnowledgeSnapshot(
            revision=50,
            records=tuple(
                _knowledge(
                    f"delta-{index}",
                    "p1" if index < 5 else "p2",
                    KnowledgeStatus.VERIFIED,
                    index + 1,
                )
                for index in range(10)
            ),
        )
        resolver = WorkGraphContextResolver(states, snapshot)

        threads = resolver.related_thread_ids(
            "child",
            include_parents=True,
            limit=2,
        )
        deltas = resolver.related_knowledge_delta_ids(
            "child",
            statuses=(KnowledgeStatus.VERIFIED,),
            include_parents=True,
            thread_limit=2,
            knowledge_limit=3,
        )

        self.assertEqual(threads, ("p1", "p2"))
        self.assertEqual(deltas, ("delta-0", "delta-1", "delta-2"))

    def test_retracted_knowledge_is_only_returned_when_explicitly_requested(self) -> None:
        states = (
            _state("parent"),
            _state("child", parent_thread_ids=("parent",)),
        )
        snapshot = KnowledgeSnapshot(
            revision=3,
            records=(
                _knowledge("delta-old", "parent", KnowledgeStatus.RETRACTED, 1),
            ),
        )
        resolver = WorkGraphContextResolver(states, snapshot)

        normal = resolver.related_knowledge_delta_ids(
            "child",
            statuses=(KnowledgeStatus.VERIFIED,),
            include_parents=True,
            thread_limit=2,
            knowledge_limit=2,
        )
        explicit = resolver.related_knowledge_delta_ids(
            "child",
            statuses=(KnowledgeStatus.RETRACTED,),
            include_parents=True,
            thread_limit=2,
            knowledge_limit=2,
        )

        self.assertEqual(normal, ())
        self.assertEqual(explicit, ("delta-old",))

    def test_unknown_thread_is_rejected(self) -> None:
        resolver = WorkGraphContextResolver(
            (_state("known"),),
            KnowledgeSnapshot(revision=0, records=()),
        )
        with self.assertRaisesRegex(ValueError, "unknown Work Thread"):
            resolver.related_thread_ids(
                "missing",
                include_self=True,
                limit=1,
            )


if __name__ == "__main__":
    unittest.main()
