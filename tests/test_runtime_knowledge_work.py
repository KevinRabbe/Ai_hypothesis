"""Tests explicit bounded selection of current knowledge for worker context."""

from __future__ import annotations

import unittest

from ai_hypothesis.runtime import (
    KnowledgeRecord,
    KnowledgeSnapshot,
    KnowledgeStatus,
    prepare_bounded_knowledge_work,
)


def _record(index: int, status: KnowledgeStatus) -> KnowledgeRecord:
    return KnowledgeRecord(
        delta_id=f"delta-{index}",
        kind="TEST_KNOWLEDGE",
        summary=f"knowledge {index}",
        source_reference_ids=(f"evidence-{index}",),
        causal_event_ids=(f"event-{index}",),
        thread_id="thread-1",
        created_event_id=f"delta-event-{index}",
        created_sequence=index + 1,
        status=status,
        assessment_reason="checked" if status is not KnowledgeStatus.PROVISIONAL else None,
    )


class BoundedKnowledgeWorkTests(unittest.TestCase):
    def test_explicit_selection_is_hard_capped_and_preserves_request_order(self) -> None:
        snapshot = KnowledgeSnapshot(
            revision=100,
            records=tuple(
                _record(index, KnowledgeStatus.PROVISIONAL)
                for index in range(20)
            ),
        )

        preparation = prepare_bounded_knowledge_work(
            snapshot,
            ("delta-9", "delta-2", "delta-15", "delta-4"),
            limit=3,
        )

        self.assertEqual(
            preparation.reference_ids,
            ("delta-9", "delta-2", "delta-15"),
        )
        records = preparation.context["knowledge_records"]
        self.assertEqual(len(records), 3)
        self.assertEqual(
            tuple(record["delta_id"] for record in records),
            ("delta-9", "delta-2", "delta-15"),
        )
        self.assertEqual(preparation.context["knowledge_revision"], 100)
        self.assertEqual(preparation.constraints["max_knowledge_records"], 3)

    def test_current_status_and_provenance_are_visible_in_bounded_context(self) -> None:
        snapshot = KnowledgeSnapshot(
            revision=12,
            records=(
                _record(1, KnowledgeStatus.VERIFIED),
                _record(2, KnowledgeStatus.DISPUTED),
                _record(3, KnowledgeStatus.RETRACTED),
            ),
        )

        preparation = prepare_bounded_knowledge_work(
            snapshot,
            ("delta-2", "delta-3"),
            limit=2,
        )
        records = preparation.context["knowledge_records"]

        self.assertEqual(records[0]["status"], "DISPUTED")
        self.assertEqual(records[0]["source_reference_ids"], ["evidence-2"])
        self.assertEqual(records[1]["status"], "RETRACTED")
        self.assertEqual(records[1]["causal_event_ids"], ["event-3"])

    def test_unknown_requested_delta_is_not_silently_dropped(self) -> None:
        snapshot = KnowledgeSnapshot(
            revision=1,
            records=(_record(1, KnowledgeStatus.PROVISIONAL),),
        )
        with self.assertRaisesRegex(ValueError, "unknown knowledge delta"):
            prepare_bounded_knowledge_work(
                snapshot,
                ("missing-delta",),
                limit=1,
            )

    def test_limit_must_be_positive(self) -> None:
        snapshot = KnowledgeSnapshot(revision=0, records=())
        with self.assertRaises(ValueError):
            prepare_bounded_knowledge_work(snapshot, (), limit=0)


if __name__ == "__main__":
    unittest.main()
