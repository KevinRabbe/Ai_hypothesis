from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from ai_hypothesis.runtime.integration_partition_lineage import (
    PartitionedIntegrationLineageProjector,
)
from ai_hypothesis.runtime.ledger import SQLiteResearchLedger
from ai_hypothesis.runtime.partition_knowledge_index import (
    SQLiteIndexedPartitionKnowledgeLineage,
)


class IncrementalPartitionKnowledgeLineageTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        root = Path(self.tempdir.name)
        self.ledger_path = root / "ledger.sqlite3"
        self.index_path = root / "lineage.sqlite3"
        self.ledger = SQLiteResearchLedger(self.ledger_path)

    def tearDown(self) -> None:
        self.ledger.close()
        self.tempdir.cleanup()

    def _append(
        self,
        event_type: str,
        *,
        thread_id: str | None = None,
        attempt_id: str | None = None,
        reference_ids: tuple[str, ...] = (),
        payload: dict | None = None,
    ):
        return self.ledger.append_event(
            event_type=event_type,
            thread_id=thread_id,
            attempt_id=attempt_id,
            reference_ids=reference_ids,
            payload=payload or {},
        )

    def _evidence(self, evidence_id: str, *, thread_id: str = "thread-a") -> None:
        self._append(
            "EVIDENCE_ADDED",
            thread_id=thread_id,
            reference_ids=(evidence_id,),
            payload={
                "evidence_id": evidence_id,
                "kind": "OBSERVATION",
                "summary": evidence_id,
            },
        )

    def _decision(
        self,
        decision_id: str,
        *,
        width: int,
        thread_id: str = "thread-a",
    ):
        return self._append(
            "SCHEDULER_DECISION_RECORDED",
            thread_id=thread_id,
            payload={
                "decision_id": decision_id,
                "action": "SYNTHESIZE",
                "purpose": "SYNTHESIZE",
                "width": width,
                "reason_codes": ["BACKPRESSURE", "PARTITIONED_INTEGRATION"],
                "projection_revision": 3,
            },
        )

    def _provenance(
        self,
        decision_id: str,
        partitions: tuple[tuple[str, tuple[str, ...]], ...],
        *,
        thread_id: str = "thread-a",
        shard_count: int = 4,
        batch_limit: int = 32,
    ):
        return self._append(
            "INTEGRATION_PARTITION_ALLOCATION_RECORDED",
            thread_id=thread_id,
            reference_ids=tuple(partition_id for partition_id, _ in partitions),
            payload={
                "schema": "integration-partition-allocation-v0",
                "decision_id": decision_id,
                "decision_projection_revision": 3,
                "partition_plan_revision": self.ledger.latest_sequence(),
                "shard_count": shard_count,
                "batch_limit": batch_limit,
                "width": len(partitions),
                "partitions": [
                    {
                        "partition_id": partition_id,
                        "shard_index": ordinal,
                        "backlog_count": len(evidence_ids),
                        "oldest_pending_sequence": 1,
                        "evidence_ids": list(evidence_ids),
                    }
                    for ordinal, (partition_id, evidence_ids) in enumerate(partitions)
                ],
            },
        )

    def _start(
        self,
        attempt_id: str,
        decision_id: str,
        refs: tuple[str, ...],
        *,
        thread_id: str = "thread-a",
    ):
        return self._append(
            "ATTEMPT_STARTED",
            thread_id=thread_id,
            attempt_id=attempt_id,
            reference_ids=refs,
            payload={
                "work_item_id": f"work-{attempt_id}",
                "worker_id": f"worker-{attempt_id}",
                "purpose": "SYNTHESIZE",
                "projection_revision": 3,
                "scheduler_decision_id": decision_id,
                "scope_region_ids": [],
            },
        )

    def _delta(
        self,
        attempt_id: str,
        delta_id: str,
        refs: tuple[str, ...],
        *,
        thread_id: str = "thread-a",
    ):
        return self._append(
            "KNOWLEDGE_DELTA_RECORDED",
            thread_id=thread_id,
            attempt_id=attempt_id,
            reference_ids=(delta_id, *refs),
            payload={
                "delta_id": delta_id,
                "kind": "PARTITION_SYNTHESIS",
                "summary": delta_id,
                "source_reference_ids": list(refs),
            },
        )

    def _terminal(self, attempt_id: str, *, thread_id: str = "thread-a"):
        return self._append(
            "ATTEMPT_COMPLETED",
            thread_id=thread_id,
            attempt_id=attempt_id,
            payload={"progress_made": True},
        )

    def test_matches_replay_partition_mapping_sources_and_unstarted_work(self) -> None:
        for evidence_id in ("e1", "e2", "e3", "e4"):
            self._evidence(evidence_id)
        self._decision("decision-a", width=2)
        self._provenance(
            "decision-a",
            (("partition-0", ("e1", "e2")), ("partition-1", ("e3", "e4"))),
        )
        self._start("attempt-a", "decision-a", ("e1", "e2"))
        self._delta("attempt-a", "delta-a", ("e1", "e2"))
        self._terminal("attempt-a")

        history = self.ledger.read_all_events()
        replay = PartitionedIntegrationLineageProjector().project(history)
        with SQLiteIndexedPartitionKnowledgeLineage(
            self.ledger,
            self.index_path,
        ) as indexed:
            snapshot = indexed.snapshot()

        self.assertEqual(
            snapshot.missing_provenance_decision_ids,
            replay.missing_provenance_decision_ids,
        )
        self.assertEqual(len(snapshot.decisions), len(replay.records))
        indexed_record = snapshot.decisions[0]
        replay_record = replay.records[0]
        self.assertEqual(indexed_record.decision_id, replay_record.allocation.decision_id)
        self.assertEqual(indexed_record.thread_id, replay_record.allocation.thread_id)
        self.assertEqual(indexed_record.width, replay_record.allocation.width)
        self.assertEqual(indexed_record.provenance_event_id, replay_record.provenance_event_id)
        self.assertEqual(indexed_record.provenance_sequence, replay_record.provenance_sequence)
        self.assertEqual(
            indexed_record.partition_plan_revision,
            replay_record.partition_plan_revision,
        )
        self.assertEqual(indexed_record.shard_count, replay_record.shard_count)
        self.assertEqual(indexed_record.batch_limit, replay_record.batch_limit)
        self.assertEqual(
            tuple(
                (
                    partition.partition_id,
                    partition.shard_index,
                    partition.backlog_count,
                    partition.oldest_pending_sequence,
                    partition.evidence_ids,
                )
                for partition in indexed_record.partitions
            ),
            tuple(
                (
                    partition.partition_id,
                    partition.shard_index,
                    partition.backlog_count,
                    partition.oldest_pending_sequence,
                    partition.evidence_ids,
                )
                for partition in replay_record.partitions
            ),
        )
        self.assertEqual(
            indexed_record.unstarted_partition_ids,
            replay_record.unstarted_partition_ids,
        )
        self.assertEqual(
            tuple(source.delta_id for source in indexed_record.sources),
            replay_record.knowledge_delta_ids,
        )
        self.assertEqual(indexed_record.partitions[0].attempt_id, "attempt-a")
        self.assertIsNone(indexed_record.partitions[1].attempt_id)

    def test_legacy_missing_provenance_remains_visible_without_guessing(self) -> None:
        self._decision("legacy", width=1)
        self._start("legacy-attempt", "legacy", ("document-ref",))
        self._delta("legacy-attempt", "legacy-delta", ())
        self._terminal("legacy-attempt")

        history = self.ledger.read_all_events()
        replay = PartitionedIntegrationLineageProjector().project(history)
        with SQLiteIndexedPartitionKnowledgeLineage(self.ledger, self.index_path) as indexed:
            snapshot = indexed.snapshot()
            self.assertEqual(
                snapshot.missing_provenance_decision_ids,
                replay.missing_provenance_decision_ids,
            )
            self.assertEqual(snapshot.missing_provenance_decision_ids, ("legacy",))
            self.assertEqual(snapshot.decisions[0].partitions, ())
            self.assertEqual(snapshot.decisions[0].sources, ())
            with self.assertRaisesRegex(ValueError, "missing durable allocation provenance"):
                snapshot.require_complete()

    def test_late_provenance_after_attempt_is_rejected(self) -> None:
        self._evidence("e1")
        self._decision("decision-a", width=1)
        self._start("attempt-a", "decision-a", ("e1",))
        self._provenance("decision-a", (("partition-0", ("e1",)),))

        history = self.ledger.read_all_events()
        with self.assertRaisesRegex(ValueError, "after ATTEMPT_STARTED"):
            PartitionedIntegrationLineageProjector().project(history)
        with SQLiteIndexedPartitionKnowledgeLineage(self.ledger, self.index_path) as indexed:
            with self.assertRaisesRegex(ValueError, "after ATTEMPT_STARTED"):
                indexed.sync()
            self.assertEqual(indexed.revision, 0)

    def test_attempt_input_must_exactly_match_recorded_partition(self) -> None:
        self._evidence("e1")
        self._evidence("e2")
        self._decision("decision-a", width=1)
        self._provenance("decision-a", (("partition-0", ("e1",)),))
        self._start("attempt-a", "decision-a", ("e2",))

        history = self.ledger.read_all_events()
        with self.assertRaisesRegex(ValueError, "does not match durable partition allocation"):
            PartitionedIntegrationLineageProjector().project(history)
        with SQLiteIndexedPartitionKnowledgeLineage(self.ledger, self.index_path) as indexed:
            with self.assertRaisesRegex(ValueError, "does not match durable partition allocation"):
                indexed.sync()

    def test_future_evidence_reference_is_rejected_even_for_legacy_attempt(self) -> None:
        self._decision("legacy", width=1)
        self._start("attempt-a", "legacy", ("future-evidence",))
        self._evidence("future-evidence")

        history = self.ledger.read_all_events()
        with self.assertRaisesRegex(ValueError, "created after attempt"):
            PartitionedIntegrationLineageProjector().project(history)
        with SQLiteIndexedPartitionKnowledgeLineage(self.ledger, self.index_path) as indexed:
            with self.assertRaisesRegex(ValueError, "created after attempt"):
                indexed.sync()

    def test_knowledge_after_terminal_is_rejected(self) -> None:
        self._evidence("e1")
        self._decision("decision-a", width=1)
        self._provenance("decision-a", (("partition-0", ("e1",)),))
        self._start("attempt-a", "decision-a", ("e1",))
        self._terminal("attempt-a")
        self._delta("attempt-a", "late-delta", ("e1",))

        history = self.ledger.read_all_events()
        with self.assertRaisesRegex(ValueError, "after terminal"):
            PartitionedIntegrationLineageProjector().project(history)
        with SQLiteIndexedPartitionKnowledgeLineage(self.ledger, self.index_path) as indexed:
            with self.assertRaisesRegex(ValueError, "after terminal"):
                indexed.sync()

    def test_repeated_logical_partition_id_across_decisions_is_not_global_identity(self) -> None:
        self._evidence("e1")
        self._evidence("e2")
        self._decision("decision-a", width=1)
        self._provenance("decision-a", (("partition-0", ("e1",)),))
        self._start("attempt-a", "decision-a", ("e1",))
        self._delta("attempt-a", "delta-a", ("e1",))
        self._terminal("attempt-a")

        self._decision("decision-b", width=1)
        self._provenance("decision-b", (("partition-0", ("e2",)),))
        self._start("attempt-b", "decision-b", ("e2",))
        self._delta("attempt-b", "delta-b", ("e2",))
        self._terminal("attempt-b")

        with SQLiteIndexedPartitionKnowledgeLineage(self.ledger, self.index_path) as indexed:
            snapshot = indexed.snapshot()
            self.assertEqual(len(snapshot.decisions), 2)
            self.assertEqual(
                tuple(decision.partitions[0].partition_id for decision in snapshot.decisions),
                ("partition-0", "partition-0"),
            )
            self.assertEqual(
                tuple(source.delta_id for decision in snapshot.decisions for source in decision.sources),
                ("delta-a", "delta-b"),
            )

    def test_exact_snapshot_restart_pagination_and_rebuild(self) -> None:
        self._evidence("e1")
        self._decision("decision-a", width=1)
        provenance = self._provenance("decision-a", (("partition-0", ("e1",)),))
        old_revision = provenance.sequence
        for index in range(1105):
            self._append(
                "OBSERVATION_RECORDED",
                payload={"observation": f"noise-{index}"},
            )
        self._start("attempt-a", "decision-a", ("e1",))
        self._delta("attempt-a", "delta-a", ("e1",))
        self._terminal("attempt-a")

        with SQLiteIndexedPartitionKnowledgeLineage(self.ledger, self.index_path) as first:
            old = first.snapshot_through(old_revision)
            self.assertEqual(first.revision, old_revision)
            self.assertEqual(old.decisions[0].unstarted_partition_ids, ("partition-0",))
            first.sync(page_size=97)
            current = first.snapshot()
            self.assertEqual(current.decisions[0].unstarted_partition_ids, ())
            self.assertEqual(current.decisions[0].sources[0].delta_id, "delta-a")
            final_revision = first.revision

        with SQLiteIndexedPartitionKnowledgeLineage(self.ledger, self.index_path) as second:
            self.assertEqual(second.revision, final_revision)
            before = second.snapshot()
            second.rebuild(page_size=113)
            self.assertEqual(second.snapshot(), before)
            with self.assertRaisesRegex(ValueError, "ahead of requested ledger snapshot"):
                second.snapshot_through(old_revision)

    def test_sidecar_must_not_reuse_canonical_ledger_file(self) -> None:
        with self.assertRaisesRegex(ValueError, "separate from the canonical ledger"):
            SQLiteIndexedPartitionKnowledgeLineage(self.ledger, self.ledger_path)


if __name__ == "__main__":
    unittest.main()
