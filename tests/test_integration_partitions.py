from __future__ import annotations

import unittest

from ai_hypothesis.runtime.contracts import LedgerEvent
from ai_hypothesis.runtime.integration_partitions import (
    IntegrationPartitionConfig,
    IntegrationPartitionProjector,
    prepare_partition_integration_work,
)


_SCHEMA = "runtime-event-v0"


def _event(
    sequence: int,
    event_type: str,
    *,
    thread_id: str | None = None,
    reference_ids: tuple[str, ...] = (),
    payload: dict | None = None,
) -> LedgerEvent:
    return LedgerEvent(
        event_id=f"event-{sequence}",
        event_type=event_type,
        sequence=sequence,
        payload_schema=_SCHEMA,
        thread_id=thread_id,
        reference_ids=reference_ids,
        payload=payload or {},
    )


def _evidence(sequence: int, evidence_id: str, thread_id: str) -> LedgerEvent:
    return _event(
        sequence,
        "EVIDENCE_ADDED",
        thread_id=thread_id,
        reference_ids=(evidence_id, f"source-{evidence_id}"),
        payload={
            "evidence_id": evidence_id,
            "kind": "OBSERVATION",
            "summary": f"summary {evidence_id}",
            "strength": 0.75,
            "uncertainty": 0.25,
            "data": {"token": evidence_id},
        },
    )


class IntegrationPartitionProjectorTests(unittest.TestCase):
    def test_partitions_cover_full_backlog_but_buffer_only_bounded_records(self) -> None:
        events = tuple(_evidence(index + 1, f"evidence-{index}", "thread-a") for index in range(9))
        projector = IntegrationPartitionProjector(
            IntegrationPartitionConfig(shard_count=4, batch_limit=2)
        )

        plan = projector.project(events)

        self.assertEqual(plan.total_backlog_count, 9)
        self.assertEqual(sum(partition.backlog_count for partition in plan.partitions), 9)
        self.assertTrue(all(partition.buffered_count <= 2 for partition in plan.partitions))
        # Pigeonhole principle: 9 records across 4 shards means at least one shard owns >2.
        self.assertTrue(
            any(partition.backlog_count > partition.buffered_count for partition in plan.partitions)
        )
        buffered_ids = [
            evidence_id
            for partition in plan.partitions
            for evidence_id in partition.evidence_ids
        ]
        self.assertEqual(len(buffered_ids), len(set(buffered_ids)))

    def test_projection_is_deterministic_for_same_history_and_config(self) -> None:
        events = tuple(_evidence(index + 1, f"evidence-{index}", "thread-a") for index in range(7))
        projector = IntegrationPartitionProjector(
            IntegrationPartitionConfig(shard_count=8, batch_limit=3)
        )

        first = projector.project(events)
        second = projector.project(events)

        self.assertEqual(first, second)

    def test_thread_ownership_is_preserved_across_shards(self) -> None:
        events = (
            _evidence(1, "evidence-a1", "thread-a"),
            _evidence(2, "evidence-b1", "thread-b"),
            _evidence(3, "evidence-a2", "thread-a"),
            _evidence(4, "evidence-b2", "thread-b"),
        )
        plan = IntegrationPartitionProjector(
            IntegrationPartitionConfig(shard_count=2, batch_limit=8)
        ).project(events)

        thread_a = plan.for_thread("thread-a")
        thread_b = plan.for_thread("thread-b")

        self.assertEqual(sum(partition.backlog_count for partition in thread_a), 2)
        self.assertEqual(sum(partition.backlog_count for partition in thread_b), 2)
        self.assertTrue(
            all(record.thread_id == "thread-a" for partition in thread_a for record in partition.records)
        )
        self.assertTrue(
            all(record.thread_id == "thread-b" for partition in thread_b for record in partition.records)
        )

    def test_later_disposition_removes_evidence_from_partition_plan(self) -> None:
        events = (
            _evidence(1, "evidence-a", "thread-a"),
            _evidence(2, "evidence-b", "thread-a"),
            _evidence(3, "evidence-c", "thread-a"),
            _event(
                4,
                "INTEGRATION_DISPOSITION_RECORDED",
                thread_id="thread-a",
                reference_ids=("evidence-b",),
                payload={"disposition": "INTEGRATED"},
            ),
        )
        plan = IntegrationPartitionProjector(
            IntegrationPartitionConfig(shard_count=2, batch_limit=8)
        ).project(events)

        self.assertEqual(plan.total_backlog_count, 2)
        all_ids = {
            evidence_id
            for partition in plan.partitions
            for evidence_id in partition.evidence_ids
        }
        self.assertEqual(all_ids, {"evidence-a", "evidence-c"})

    def test_disposition_before_evidence_creation_is_rejected(self) -> None:
        events = (
            _event(
                1,
                "INTEGRATION_DISPOSITION_RECORDED",
                reference_ids=("evidence-a",),
                payload={"disposition": "INTEGRATED"},
            ),
            _evidence(2, "evidence-a", "thread-a"),
        )

        with self.assertRaisesRegex(ValueError, "precedes evidence creation"):
            IntegrationPartitionProjector().project(events)

    def test_duplicate_evidence_identity_is_rejected(self) -> None:
        events = (
            _evidence(1, "evidence-a", "thread-a"),
            _evidence(2, "evidence-a", "thread-a"),
        )
        with self.assertRaisesRegex(ValueError, "duplicate durable evidence ID"):
            IntegrationPartitionProjector().project(events)

    def test_partition_work_uses_only_buffered_partition_records(self) -> None:
        events = tuple(_evidence(index + 1, f"evidence-{index}", "thread-a") for index in range(6))
        plan = IntegrationPartitionProjector(
            IntegrationPartitionConfig(shard_count=2, batch_limit=2)
        ).project(events)
        partition = plan.partitions[0]

        preparation = prepare_partition_integration_work(
            partition,
            revision=plan.revision,
            batch_limit=plan.batch_limit,
        )

        self.assertEqual(preparation.reference_ids, partition.evidence_ids)
        self.assertEqual(preparation.context["context_view"], "SYNTHESIZE")
        self.assertEqual(preparation.context["synthesis_mode"], "INTEGRATION_PARTITION")
        self.assertEqual(
            preparation.context["integration_partition"]["partition_id"],
            partition.partition_id,
        )
        self.assertEqual(
            preparation.context["integration_partition"]["backlog_count"],
            partition.backlog_count,
        )
        self.assertEqual(len(preparation.context["pending_evidence"]), partition.buffered_count)
        self.assertEqual(
            preparation.context["causal_event_ids"],
            list(partition.causal_event_ids),
        )
        self.assertTrue(preparation.constraints["preserve_source_thread_ownership"])

    def test_unknown_disposition_reference_does_not_remove_real_backlog(self) -> None:
        events = (
            _evidence(1, "evidence-a", "thread-a"),
            _event(
                2,
                "INTEGRATION_DISPOSITION_RECORDED",
                reference_ids=("missing-evidence",),
                payload={"disposition": "INVALID"},
            ),
        )
        plan = IntegrationPartitionProjector().project(events)
        self.assertEqual(plan.total_backlog_count, 1)


if __name__ == "__main__":
    unittest.main()
