from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from ai_hypothesis.runtime.indexed_thread_consolidation import (
    IndexedThreadConsolidationPlanner,
    IndexedThreadConsolidationPressureProjector,
)
from ai_hypothesis.runtime.knowledge_index import SQLiteIndexedKnowledgeState
from ai_hypothesis.runtime.ledger import SQLiteResearchLedger
from ai_hypothesis.runtime.partition_knowledge_index import (
    SQLiteIndexedPartitionKnowledgeLineage,
)
from ai_hypothesis.runtime.thread_consolidation import (
    ThreadConsolidationConfig,
    ThreadConsolidationPlanner,
)
from ai_hypothesis.runtime.thread_consolidation_control import (
    ThreadConsolidationPressureProjector,
)


class NoFullReplayLedger(SQLiteResearchLedger):
    def read_all_events(self, *args, **kwargs):
        raise AssertionError("indexed consolidation must not call read_all_events")


def append(
    ledger: SQLiteResearchLedger,
    event_type: str,
    *,
    thread_id: str | None = None,
    attempt_id: str | None = None,
    reference_ids: tuple[str, ...] = (),
    payload: dict | None = None,
):
    return ledger.append_event(
        event_type=event_type,
        thread_id=thread_id,
        attempt_id=attempt_id,
        reference_ids=reference_ids,
        payload=payload or {},
    )


def seed_partition_history(ledger: SQLiteResearchLedger, *, thread_id: str = "thread-a") -> None:
    for evidence_id in ("e1", "e2", "e3"):
        append(
            ledger,
            "EVIDENCE_ADDED",
            thread_id=thread_id,
            reference_ids=(evidence_id,),
            payload={
                "evidence_id": evidence_id,
                "kind": "OBSERVATION",
                "summary": evidence_id,
            },
        )

    append(
        ledger,
        "SCHEDULER_DECISION_RECORDED",
        thread_id=thread_id,
        payload={
            "decision_id": "decision-a",
            "action": "SYNTHESIZE",
            "purpose": "SYNTHESIZE",
            "width": 2,
            "reason_codes": ["BACKPRESSURE", "PARTITIONED_INTEGRATION"],
            "projection_revision": 1,
        },
    )
    append(
        ledger,
        "INTEGRATION_PARTITION_ALLOCATION_RECORDED",
        thread_id=thread_id,
        reference_ids=("partition-0", "partition-1"),
        payload={
            "schema": "integration-partition-allocation-v0",
            "decision_id": "decision-a",
            "decision_projection_revision": 1,
            "partition_plan_revision": ledger.latest_sequence(),
            "shard_count": 4,
            "batch_limit": 32,
            "width": 2,
            "partitions": [
                {
                    "partition_id": "partition-0",
                    "shard_index": 0,
                    "backlog_count": 2,
                    "oldest_pending_sequence": 1,
                    "evidence_ids": ["e1", "e2"],
                },
                {
                    "partition_id": "partition-1",
                    "shard_index": 1,
                    "backlog_count": 1,
                    "oldest_pending_sequence": 3,
                    "evidence_ids": ["e3"],
                },
            ],
        },
    )

    append(
        ledger,
        "ATTEMPT_STARTED",
        thread_id=thread_id,
        attempt_id="attempt-0",
        reference_ids=("e1", "e2"),
        payload={
            "work_item_id": "work-0",
            "worker_id": "worker-0",
            "purpose": "SYNTHESIZE",
            "projection_revision": 1,
            "scheduler_decision_id": "decision-a",
            "scope_region_ids": [],
        },
    )
    append(
        ledger,
        "KNOWLEDGE_DELTA_RECORDED",
        thread_id=thread_id,
        attempt_id="attempt-0",
        reference_ids=("delta-0a", "e1"),
        payload={
            "delta_id": "delta-0a",
            "kind": "PARTITION_SYNTHESIS",
            "summary": "partition zero first",
            "source_reference_ids": ["e1"],
        },
    )
    append(
        ledger,
        "KNOWLEDGE_DELTA_RECORDED",
        thread_id=thread_id,
        attempt_id="attempt-0",
        reference_ids=("delta-0b", "e2"),
        payload={
            "delta_id": "delta-0b",
            "kind": "PARTITION_SYNTHESIS",
            "summary": "partition zero second",
            "source_reference_ids": ["e2"],
        },
    )
    append(
        ledger,
        "ATTEMPT_COMPLETED",
        thread_id=thread_id,
        attempt_id="attempt-0",
        payload={"progress_made": True},
    )

    append(
        ledger,
        "ATTEMPT_STARTED",
        thread_id=thread_id,
        attempt_id="attempt-1",
        reference_ids=("e3",),
        payload={
            "work_item_id": "work-1",
            "worker_id": "worker-1",
            "purpose": "SYNTHESIZE",
            "projection_revision": 1,
            "scheduler_decision_id": "decision-a",
            "scope_region_ids": [],
        },
    )
    append(
        ledger,
        "KNOWLEDGE_DELTA_RECORDED",
        thread_id=thread_id,
        attempt_id="attempt-1",
        reference_ids=("delta-1a", "e3"),
        payload={
            "delta_id": "delta-1a",
            "kind": "PARTITION_SYNTHESIS",
            "summary": "partition one first",
            "source_reference_ids": ["e3"],
        },
    )
    append(
        ledger,
        "ATTEMPT_COMPLETED",
        thread_id=thread_id,
        attempt_id="attempt-1",
        payload={"progress_made": True},
    )


def append_thread_consolidation(
    ledger: SQLiteResearchLedger,
    *,
    delta_id: str = "thread-consolidation-1",
    thread_id: str = "thread-a",
):
    return append(
        ledger,
        "KNOWLEDGE_DELTA_RECORDED",
        thread_id=thread_id,
        reference_ids=(delta_id, "delta-0a", "delta-1a"),
        payload={
            "delta_id": delta_id,
            "kind": "THREAD_CONSOLIDATION",
            "summary": "cross-partition synthesis",
            "source_reference_ids": ["delta-0a", "delta-1a"],
        },
    )


def retract(
    ledger: SQLiteResearchLedger,
    delta_id: str,
    *,
    thread_id: str = "thread-a",
):
    return append(
        ledger,
        "KNOWLEDGE_ASSESSMENT_RECORDED",
        thread_id=thread_id,
        reference_ids=(delta_id,),
        payload={
            "assessment": "RETRACTED",
            "reason": "superseded",
        },
    )


class IndexedThreadConsolidationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        root = Path(self.tempdir.name)
        self.ledger = SQLiteResearchLedger(root / "ledger.sqlite3")
        self.lineage = SQLiteIndexedPartitionKnowledgeLineage(
            self.ledger,
            root / "lineage.sqlite3",
        )
        self.knowledge = SQLiteIndexedKnowledgeState(
            self.ledger,
            root / "knowledge.sqlite3",
        )
        self.config = ThreadConsolidationConfig(
            selection_limit=2,
            minimum_source_deltas=2,
        )
        self.planner = IndexedThreadConsolidationPlanner(
            ledger=self.ledger,
            lineage=self.lineage,
            knowledge=self.knowledge,
            config=self.config,
        )
        self.pressure = IndexedThreadConsolidationPressureProjector(
            ledger=self.ledger,
            lineage=self.lineage,
            knowledge=self.knowledge,
        )

    def tearDown(self) -> None:
        self.knowledge.close()
        self.lineage.close()
        self.ledger.close()
        self.tempdir.cleanup()

    def test_plan_matches_replay_round_robin_selection(self) -> None:
        seed_partition_history(self.ledger)
        history = self.ledger.read_all_events()
        replay = ThreadConsolidationPlanner(self.config).plan(
            history,
            thread_id="thread-a",
        )
        indexed = self.planner.plan(
            sequence=self.ledger.latest_sequence(),
            thread_id="thread-a",
        )
        self.assertEqual(indexed, replay)
        self.assertEqual(
            tuple(source.delta_id for source in indexed.selected_sources),
            ("delta-0a", "delta-1a"),
        )
        self.assertEqual(indexed.pending_source_count, 3)
        self.assertEqual(indexed.pending_partition_count, 2)

    def test_pressure_matches_replay_and_higher_retraction_reopens_sources(self) -> None:
        seed_partition_history(self.ledger)
        replay_projector = ThreadConsolidationPressureProjector()

        before_history = self.ledger.read_all_events()
        before_replay = replay_projector.project(before_history)
        before_indexed = self.pressure.project(sequence=self.ledger.latest_sequence())
        self.assertEqual(before_indexed, before_replay)
        self.assertEqual(before_indexed.pending_source_count["thread-a"], 3)

        append_thread_consolidation(self.ledger)
        active_history = self.ledger.read_all_events()
        active_replay = replay_projector.project(active_history)
        active_indexed = self.pressure.project(sequence=self.ledger.latest_sequence())
        self.assertEqual(active_indexed, active_replay)
        self.assertEqual(active_indexed.pending_source_count["thread-a"], 1)

        retract(self.ledger, "thread-consolidation-1")
        retracted_history = self.ledger.read_all_events()
        retracted_replay = replay_projector.project(retracted_history)
        retracted_indexed = self.pressure.project(sequence=self.ledger.latest_sequence())
        self.assertEqual(retracted_indexed, retracted_replay)
        self.assertEqual(retracted_indexed.pending_source_count["thread-a"], 3)

    def test_plan_uses_exact_historical_revision(self) -> None:
        seed_partition_history(self.ledger)
        before_consolidation = self.ledger.latest_sequence()
        append_thread_consolidation(self.ledger)

        old = self.planner.plan(
            sequence=before_consolidation,
            thread_id="thread-a",
        )
        self.assertEqual(old.pending_source_count, 3)
        self.assertTrue(old.ready)

        current = self.planner.plan(
            sequence=self.ledger.latest_sequence(),
            thread_id="thread-a",
        )
        self.assertEqual(current.pending_source_count, 1)
        self.assertFalse(current.ready)

    def test_missing_selected_thread_provenance_matches_replay_failure(self) -> None:
        append(
            self.ledger,
            "SCHEDULER_DECISION_RECORDED",
            thread_id="thread-a",
            payload={
                "decision_id": "legacy",
                "action": "SYNTHESIZE",
                "purpose": "SYNTHESIZE",
                "width": 1,
                "reason_codes": ["BACKPRESSURE", "PARTITIONED_INTEGRATION"],
                "projection_revision": 0,
            },
        )
        history = self.ledger.read_all_events()
        with self.assertRaisesRegex(ValueError, "without durable provenance"):
            ThreadConsolidationPlanner(self.config).plan(
                history,
                thread_id="thread-a",
            )
        with self.assertRaisesRegex(ValueError, "without durable provenance"):
            self.planner.plan(
                sequence=self.ledger.latest_sequence(),
                thread_id="thread-a",
            )

        indexed_pressure = self.pressure.project(sequence=self.ledger.latest_sequence())
        replay_pressure = ThreadConsolidationPressureProjector().project(history)
        self.assertEqual(indexed_pressure, replay_pressure)
        self.assertIn("thread-a", indexed_pressure.incomplete_thread_ids)
        self.assertEqual(indexed_pressure.pressure_for("thread-a"), 0.0)


class IndexedThreadConsolidationNoReplayTests(unittest.TestCase):
    def test_planning_and_pressure_do_not_read_full_ledger(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            root = Path(tempdir)
            ledger = NoFullReplayLedger(root / "ledger.sqlite3")
            lineage = SQLiteIndexedPartitionKnowledgeLineage(
                ledger,
                root / "lineage.sqlite3",
            )
            knowledge = SQLiteIndexedKnowledgeState(
                ledger,
                root / "knowledge.sqlite3",
            )
            try:
                seed_partition_history(ledger)
                sequence = ledger.latest_sequence()
                planner = IndexedThreadConsolidationPlanner(
                    ledger=ledger,
                    lineage=lineage,
                    knowledge=knowledge,
                    config=ThreadConsolidationConfig(
                        selection_limit=2,
                        minimum_source_deltas=2,
                    ),
                )
                pressure = IndexedThreadConsolidationPressureProjector(
                    ledger=ledger,
                    lineage=lineage,
                    knowledge=knowledge,
                )

                plan = planner.plan(sequence=sequence, thread_id="thread-a")
                overview = pressure.project(sequence=sequence)
                self.assertTrue(plan.ready)
                self.assertGreater(overview.pressure_for("thread-a"), 0.0)
            finally:
                knowledge.close()
                lineage.close()
                ledger.close()


if __name__ == "__main__":
    unittest.main()
