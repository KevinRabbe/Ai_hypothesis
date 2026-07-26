from __future__ import annotations

import unittest

from ai_hypothesis.runtime.contracts import LedgerEvent
from ai_hypothesis.runtime.integration_hierarchy_telemetry import (
    IntegrationHierarchyTelemetryProjector,
)


_SCHEMA = "runtime-event-v0"


def _event(
    sequence: int,
    event_type: str,
    *,
    thread_id: str = "thread-a",
    attempt_id: str | None = None,
    reference_ids: tuple[str, ...] = (),
    payload: dict | None = None,
) -> LedgerEvent:
    return LedgerEvent(
        event_id=f"event-{sequence}",
        event_type=event_type,
        sequence=sequence,
        payload_schema=_SCHEMA,
        thread_id=thread_id,
        attempt_id=attempt_id,
        reference_ids=reference_ids,
        payload=payload or {},
    )


def _history(*, include_provenance: bool = True) -> tuple[LedgerEvent, ...]:
    events = [
        _event(1, "EVIDENCE_ADDED", reference_ids=("e1",), payload={"evidence_id": "e1"}),
        _event(2, "EVIDENCE_ADDED", reference_ids=("e2",), payload={"evidence_id": "e2"}),
        _event(3, "EVIDENCE_ADDED", reference_ids=("e3",), payload={"evidence_id": "e3"}),
        _event(4, "EVIDENCE_ADDED", reference_ids=("e4",), payload={"evidence_id": "e4"}),
        _event(
            5,
            "SCHEDULER_DECISION_RECORDED",
            payload={
                "decision_id": "decision-a",
                "action": "SYNTHESIZE",
                "purpose": "SYNTHESIZE",
                "width": 2,
                "reason_codes": ["BACKPRESSURE", "PARTITIONED_INTEGRATION"],
                "projection_revision": 4,
                "integration_backpressure": True,
                "max_width": 2,
            },
        ),
    ]
    if include_provenance:
        events.append(
            _event(
                6,
                "INTEGRATION_PARTITION_ALLOCATION_RECORDED",
                reference_ids=("partition-a", "partition-b"),
                payload={
                    "schema": "integration-partition-allocation-v0",
                    "decision_id": "decision-a",
                    "decision_projection_revision": 4,
                    "partition_plan_revision": 5,
                    "shard_count": 8,
                    "batch_limit": 2,
                    "width": 2,
                    "partitions": [
                        {
                            "partition_id": "partition-a",
                            "shard_index": 1,
                            "backlog_count": 4,
                            "oldest_pending_sequence": 1,
                            "evidence_ids": ["e1", "e2"],
                        },
                        {
                            "partition_id": "partition-b",
                            "shard_index": 6,
                            "backlog_count": 3,
                            "oldest_pending_sequence": 3,
                            "evidence_ids": ["e3", "e4"],
                        },
                    ],
                },
            )
        )
        offset = 0
    else:
        offset = -1

    events.extend(
        [
            _event(
                7 + offset,
                "ATTEMPT_STARTED",
                attempt_id="attempt-a",
                reference_ids=("e1", "e2"),
                payload={
                    "work_item_id": "work-a",
                    "worker_id": "worker-a",
                    "purpose": "SYNTHESIZE",
                    "projection_revision": 4,
                    "scheduler_decision_id": "decision-a",
                    "scope_region_ids": [],
                },
            ),
            _event(
                8 + offset,
                "ATTEMPT_STARTED",
                attempt_id="attempt-b",
                reference_ids=("e3", "e4"),
                payload={
                    "work_item_id": "work-b",
                    "worker_id": "worker-b",
                    "purpose": "SYNTHESIZE",
                    "projection_revision": 4,
                    "scheduler_decision_id": "decision-a",
                    "scope_region_ids": [],
                },
            ),
            _event(
                9 + offset,
                "KNOWLEDGE_DELTA_RECORDED",
                attempt_id="attempt-a",
                reference_ids=("k1", "e1"),
                payload={
                    "delta_id": "k1",
                    "kind": "INTEGRATION_SUMMARY",
                    "summary": "partition A first",
                    "source_reference_ids": ["e1"],
                    "causal_event_ids": [],
                },
            ),
            _event(
                10 + offset,
                "KNOWLEDGE_DELTA_RECORDED",
                attempt_id="attempt-a",
                reference_ids=("k3", "e2"),
                payload={
                    "delta_id": "k3",
                    "kind": "INTEGRATION_SUMMARY",
                    "summary": "partition A second",
                    "source_reference_ids": ["e2"],
                    "causal_event_ids": [],
                },
            ),
            _event(
                11 + offset,
                "KNOWLEDGE_DELTA_RECORDED",
                attempt_id="attempt-b",
                reference_ids=("k2", "e3", "e4"),
                payload={
                    "delta_id": "k2",
                    "kind": "INTEGRATION_SUMMARY",
                    "summary": "partition B",
                    "source_reference_ids": ["e3", "e4"],
                    "causal_event_ids": [],
                },
            ),
            _event(12 + offset, "ATTEMPT_COMPLETED", attempt_id="attempt-a", payload={"progress_made": True}),
            _event(13 + offset, "ATTEMPT_COMPLETED", attempt_id="attempt-b", payload={"progress_made": True}),
            _event(
                14 + offset,
                "KNOWLEDGE_DELTA_RECORDED",
                reference_ids=("thread-summary", "k1", "k2"),
                payload={
                    "delta_id": "thread-summary",
                    "kind": "THREAD_CONSOLIDATION",
                    "summary": "cross-partition thread knowledge",
                    "source_reference_ids": ["k1", "k2"],
                    "causal_event_ids": [],
                },
            ),
        ]
    )
    return tuple(events)


class IntegrationHierarchyTelemetryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.projector = IntegrationHierarchyTelemetryProjector()

    def test_projects_two_level_information_volume_and_frontier(self) -> None:
        snapshot = self.projector.project(_history())

        self.assertEqual(snapshot.raw_evidence_count, 4)
        self.assertTrue(snapshot.partition_lineage_complete)
        self.assertEqual(snapshot.partition_assignment_count, 2)
        self.assertEqual(snapshot.unique_partition_id_count, 2)
        self.assertEqual(snapshot.started_partition_attempt_count, 2)
        self.assertEqual(snapshot.unstarted_partition_assignment_count, 0)
        self.assertEqual(snapshot.partition_knowledge_delta_count, 3)
        self.assertEqual(snapshot.active_partition_knowledge_delta_count, 3)
        self.assertEqual(snapshot.partition_status_counts["PROVISIONAL"], 3)
        self.assertEqual(snapshot.partition_source_reference_count, 4)
        self.assertEqual(
            snapshot.unique_raw_evidence_referenced_by_partition_knowledge_count,
            4,
        )
        self.assertEqual(snapshot.partition_evidence_reference_fraction, 1.0)
        self.assertEqual(snapshot.partition_knowledge_per_raw_evidence, 0.75)

        self.assertEqual(snapshot.thread_consolidation_delta_count, 1)
        self.assertEqual(snapshot.active_thread_consolidation_delta_count, 1)
        self.assertEqual(snapshot.thread_consolidation_status_counts["PROVISIONAL"], 1)
        self.assertEqual(snapshot.thread_partition_source_reference_count, 2)
        self.assertEqual(snapshot.active_thread_partition_source_reference_count, 2)
        self.assertEqual(snapshot.unique_active_partition_deltas_consumed_count, 2)
        self.assertEqual(snapshot.pending_active_partition_delta_count, 1)
        self.assertEqual(snapshot.active_retracted_partition_source_reference_count, 0)
        self.assertEqual(snapshot.cross_thread_partition_source_reference_count, 0)
        self.assertEqual(snapshot.non_partition_thread_source_reference_count, 0)
        self.assertEqual(snapshot.mean_active_thread_consolidation_fan_in, 2.0)
        self.assertEqual(snapshot.max_active_thread_consolidation_fan_in, 2)

        self.assertEqual(snapshot.active_hierarchy_frontier_count, 2)
        self.assertEqual(snapshot.partition_to_frontier_count_reduction_factor, 1.5)
        self.assertEqual(
            snapshot.consumed_partition_deltas_per_active_thread_consolidation,
            2.0,
        )

    def test_retracted_thread_consolidation_reopens_lower_frontier(self) -> None:
        events = (
            *_history(),
            _event(
                15,
                "KNOWLEDGE_ASSESSMENT_RECORDED",
                reference_ids=("thread-summary",),
                payload={"assessment": "RETRACTED", "reason": "bad synthesis"},
            ),
        )
        snapshot = self.projector.project(events)

        self.assertEqual(snapshot.active_thread_consolidation_delta_count, 0)
        self.assertEqual(snapshot.thread_consolidation_status_counts["RETRACTED"], 1)
        self.assertEqual(snapshot.unique_active_partition_deltas_consumed_count, 0)
        self.assertEqual(snapshot.pending_active_partition_delta_count, 3)
        self.assertEqual(snapshot.active_hierarchy_frontier_count, 3)
        self.assertEqual(snapshot.partition_to_frontier_count_reduction_factor, 1.0)
        self.assertIsNone(snapshot.consumed_partition_deltas_per_active_thread_consolidation)

    def test_retracted_lower_source_is_visible_as_higher_level_integrity_debt(self) -> None:
        events = (
            *_history(),
            _event(
                15,
                "KNOWLEDGE_ASSESSMENT_RECORDED",
                reference_ids=("k1",),
                payload={"assessment": "RETRACTED", "reason": "invalid lower knowledge"},
            ),
        )
        snapshot = self.projector.project(events)

        self.assertEqual(snapshot.active_partition_knowledge_delta_count, 2)
        self.assertEqual(snapshot.active_retracted_partition_source_reference_count, 1)
        self.assertEqual(snapshot.unique_active_partition_deltas_consumed_count, 1)
        self.assertEqual(snapshot.pending_active_partition_delta_count, 1)
        self.assertEqual(snapshot.active_hierarchy_frontier_count, 2)
        self.assertEqual(snapshot.partition_to_frontier_count_reduction_factor, 1.0)

    def test_non_partition_thread_sources_remain_visible_without_becoming_partition_fan_in(self) -> None:
        events = list(_history())
        summary = events[-1]
        events[-1] = _event(
            summary.sequence,
            summary.event_type,
            reference_ids=("thread-summary", "k1", "k2", "document-a"),
            payload={
                "delta_id": "thread-summary",
                "kind": "THREAD_CONSOLIDATION",
                "summary": "cross-partition thread knowledge",
                "source_reference_ids": ["k1", "k2", "document-a"],
                "causal_event_ids": [],
            },
        )
        snapshot = self.projector.project(tuple(events))

        self.assertEqual(snapshot.thread_partition_source_reference_count, 2)
        self.assertEqual(snapshot.non_partition_thread_source_reference_count, 1)
        self.assertEqual(snapshot.mean_active_thread_consolidation_fan_in, 2.0)

    def test_missing_partition_provenance_marks_partition_metrics_incomplete(self) -> None:
        snapshot = self.projector.project(_history(include_provenance=False))

        self.assertFalse(snapshot.partition_lineage_complete)
        self.assertEqual(
            snapshot.missing_partition_provenance_decision_ids,
            ("decision-a",),
        )
        self.assertEqual(snapshot.partition_assignment_count, 0)
        self.assertEqual(snapshot.partition_knowledge_delta_count, 0)
        with self.assertRaisesRegex(ValueError, "missing partition allocation provenance"):
            snapshot.require_complete_partition_lineage()


if __name__ == "__main__":
    unittest.main()
