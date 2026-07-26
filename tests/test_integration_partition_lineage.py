from __future__ import annotations

import unittest

from ai_hypothesis.runtime.contracts import LedgerEvent
from ai_hypothesis.runtime.integration_partition_lineage import (
    PartitionedIntegrationLineageProjector,
)


_SCHEMA = "runtime-event-v0"


def _event(
    sequence: int,
    event_type: str,
    *,
    attempt_id: str | None = None,
    reference_ids: tuple[str, ...] = (),
    payload: dict | None = None,
) -> LedgerEvent:
    return LedgerEvent(
        event_id=f"event-{sequence}",
        event_type=event_type,
        sequence=sequence,
        payload_schema=_SCHEMA,
        thread_id="thread-a",
        attempt_id=attempt_id,
        reference_ids=reference_ids,
        payload=payload or {},
    )


def _evidence(sequence: int, evidence_id: str) -> LedgerEvent:
    return _event(
        sequence,
        "EVIDENCE_ADDED",
        reference_ids=(evidence_id,),
        payload={"evidence_id": evidence_id},
    )


def _decision(sequence: int, *, width: int = 2) -> LedgerEvent:
    return _event(
        sequence,
        "SCHEDULER_DECISION_RECORDED",
        payload={
            "decision_id": "decision-a",
            "action": "SYNTHESIZE",
            "purpose": "SYNTHESIZE",
            "width": width,
            "reason_codes": ["BACKPRESSURE", "PARTITIONED_INTEGRATION"],
            "projection_revision": 4,
            "integration_backpressure": True,
            "max_width": 4,
        },
    )


def _provenance(
    sequence: int,
    *,
    second_refs: tuple[str, ...] = ("e3", "e4"),
    width: int = 2,
) -> LedgerEvent:
    partitions = [
        {
            "partition_id": "partition-a",
            "shard_index": 1,
            "backlog_count": 5,
            "oldest_pending_sequence": 1,
            "evidence_ids": ["e1", "e2"],
        },
    ]
    references = ["partition-a"]
    if width == 2:
        partitions.append(
            {
                "partition_id": "partition-b",
                "shard_index": 6,
                "backlog_count": 4,
                "oldest_pending_sequence": 3,
                "evidence_ids": list(second_refs),
            }
        )
        references.append("partition-b")
    return _event(
        sequence,
        "INTEGRATION_PARTITION_ALLOCATION_RECORDED",
        reference_ids=tuple(references),
        payload={
            "schema": "integration-partition-allocation-v0",
            "decision_id": "decision-a",
            "decision_projection_revision": 4,
            "partition_plan_revision": 5,
            "shard_count": 8,
            "batch_limit": 2,
            "width": width,
            "partitions": partitions,
        },
    )


def _started(
    sequence: int,
    attempt_id: str,
    worker_id: str,
    refs: tuple[str, ...],
) -> LedgerEvent:
    return _event(
        sequence,
        "ATTEMPT_STARTED",
        attempt_id=attempt_id,
        reference_ids=refs,
        payload={
            "work_item_id": f"work-{attempt_id}",
            "worker_id": worker_id,
            "purpose": "SYNTHESIZE",
            "projection_revision": 4,
            "scheduler_decision_id": "decision-a",
            "scope_region_ids": [],
        },
    )


class PartitionedIntegrationLineageProjectorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.projector = PartitionedIntegrationLineageProjector()

    def _base_events(self):
        return (
            _evidence(1, "e1"),
            _evidence(2, "e2"),
            _evidence(3, "e3"),
            _evidence(4, "e4"),
            _decision(5),
        )

    def test_maps_attempts_and_knowledge_to_exact_historical_partitions(self) -> None:
        events = (
            *self._base_events(),
            _provenance(6),
            _started(7, "attempt-a", "worker-a", ("e1", "e2")),
            _started(8, "attempt-b", "worker-b", ("e3", "e4")),
            _event(
                9,
                "KNOWLEDGE_DELTA_RECORDED",
                attempt_id="attempt-a",
                reference_ids=("delta-a", "e1", "e2"),
                payload={
                    "delta_id": "delta-a",
                    "source_reference_ids": ["e1", "e2"],
                },
            ),
            _event(
                10,
                "KNOWLEDGE_DELTA_RECORDED",
                attempt_id="attempt-b",
                reference_ids=("delta-b", "e3"),
                payload={
                    "delta_id": "delta-b",
                    "source_reference_ids": ["e3"],
                },
            ),
            _event(11, "ATTEMPT_COMPLETED", attempt_id="attempt-a", payload={"progress_made": True}),
            _event(12, "ATTEMPT_COMPLETED", attempt_id="attempt-b", payload={"progress_made": True}),
        )

        snapshot = self.projector.project(events)
        self.assertTrue(snapshot.provenance_complete)
        (record,) = snapshot.records
        self.assertTrue(record.provenance_complete)
        self.assertEqual(record.partition_plan_revision, 5)
        self.assertEqual(record.shard_count, 8)
        self.assertEqual(record.batch_limit, 2)
        self.assertEqual(record.unstarted_partition_ids, ())
        self.assertEqual(
            tuple(lineage.partition.partition_id for lineage in record.partition_attempts),
            ("partition-a", "partition-b"),
        )
        self.assertEqual(
            tuple(lineage.attempt.worker_id for lineage in record.partition_attempts),
            ("worker-a", "worker-b"),
        )
        self.assertEqual(record.knowledge_delta_ids, ("delta-a", "delta-b"))
        self.assertEqual(
            record.partition_attempts[0].knowledge_delta_ids,
            ("delta-a",),
        )

    def test_missing_legacy_provenance_is_visible_and_strict_mode_rejects_it(self) -> None:
        events = (
            *self._base_events(),
            _started(6, "attempt-a", "worker-a", ("e1", "e2")),
        )

        snapshot = self.projector.project(events)
        self.assertFalse(snapshot.provenance_complete)
        self.assertEqual(snapshot.missing_provenance_decision_ids, ("decision-a",))
        self.assertFalse(snapshot.records[0].provenance_complete)
        with self.assertRaisesRegex(ValueError, "missing durable allocation provenance"):
            snapshot.require_complete()

    def test_unstarted_partition_remains_visible(self) -> None:
        events = (
            *self._base_events(),
            _provenance(6),
            _started(7, "attempt-a", "worker-a", ("e1", "e2")),
        )

        (record,) = self.projector.project(events).require_complete().records
        self.assertEqual(record.unstarted_partition_ids, ("partition-b",))
        self.assertEqual(len(record.partition_attempts), 1)

    def test_provenance_after_attempt_start_is_rejected(self) -> None:
        events = (
            *self._base_events(),
            _started(6, "attempt-a", "worker-a", ("e1", "e2")),
            _provenance(7),
        )

        with self.assertRaisesRegex(ValueError, "after ATTEMPT_STARTED"):
            self.projector.project(events)

    def test_attempt_input_must_match_one_recorded_partition_exactly(self) -> None:
        events = (
            *self._base_events(),
            _provenance(6),
            _started(7, "attempt-a", "worker-a", ("e1", "e3")),
        )

        with self.assertRaisesRegex(ValueError, "does not match durable partition allocation"):
            self.projector.project(events)

    def test_provenance_width_must_match_scheduler_width(self) -> None:
        events = (
            *self._base_events(),
            _provenance(6, width=1),
        )

        with self.assertRaisesRegex(ValueError, "width does not match scheduler decision"):
            self.projector.project(events)

    def test_overlapping_evidence_authority_in_provenance_is_rejected(self) -> None:
        events = (
            *self._base_events(),
            _provenance(6, second_refs=("e2", "e4")),
        )

        with self.assertRaisesRegex(ValueError, "overlapping evidence authority"):
            self.projector.project(events)


if __name__ == "__main__":
    unittest.main()
