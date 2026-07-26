from __future__ import annotations

import random
import tempfile
import unittest
from pathlib import Path

from ai_hypothesis.runtime.contracts import AttemptResult, AttemptStatus
from ai_hypothesis.runtime.control import WorkPreparation
from ai_hypothesis.runtime.indexed_control import (
    IndexedRuntimeControlLoop,
    IndexedRuntimeSnapshotProvider,
    IndexedThreadRuntimeState,
)
from ai_hypothesis.runtime.integration_index import SQLiteIndexedIntegrationTracker
from ai_hypothesis.runtime.knowledge_index import SQLiteIndexedKnowledgeState
from ai_hypothesis.runtime.knowledge_verification import KnowledgeVerificationTracker
from ai_hypothesis.runtime.ledger import SQLiteResearchLedger
from ai_hypothesis.runtime.scheduler import SchedulerConfig, SchedulerSignals, SchedulerV0


class NoFullReplayLedger(SQLiteResearchLedger):
    def read_all_events(self, *args, **kwargs):
        raise AssertionError("indexed runtime control must not call read_all_events")


class RecordingBank:
    def __init__(self) -> None:
        self.requests = []

    def execute_batch(self, requests):
        self.requests.extend(requests)
        return tuple(
            AttemptResult(
                attempt_id=request.attempt_id,
                work_item_id=request.work_item.work_item_id,
                thread_id=request.work_item.thread_id,
                worker_id=request.worker_id,
                status=AttemptStatus.COMPLETED,
                progress_made=True,
            )
            for request in requests
        )


class IndexedRuntimeControlTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        root = Path(self.tempdir.name)
        self.ledger = NoFullReplayLedger(root / "ledger.sqlite3")
        self.thread_state = IndexedThreadRuntimeState(
            self.ledger, root / "thread-state.sqlite3"
        )
        self.integration = SQLiteIndexedIntegrationTracker(
            self.ledger, root / "integration.sqlite3"
        )
        self.knowledge = SQLiteIndexedKnowledgeState(
            self.ledger, root / "knowledge.sqlite3"
        )
        self.verification = KnowledgeVerificationTracker(
            self.ledger,
            projector=self.knowledge,
        )
        self.provider = IndexedRuntimeSnapshotProvider(
            ledger=self.ledger,
            thread_state=self.thread_state,
            integration_tracker=self.integration,
            verification_tracker=self.verification,
        )
        self.bank = RecordingBank()
        self.scheduler = SchedulerV0(
            SchedulerConfig(exploration_probability=0.0),
            rng=random.Random(1),
        )
        self.loop = IndexedRuntimeControlLoop(
            ledger=self.ledger,
            scheduler=self.scheduler,
            worker_bank=self.bank,
            worker_ids=("worker-1", "worker-2"),
            snapshot_provider=self.provider,
        )

    def tearDown(self) -> None:
        self.knowledge.close()
        self.integration.close()
        self.thread_state.close()
        self.ledger.close()
        self.tempdir.cleanup()

    @staticmethod
    def _signals(_state):
        return SchedulerSignals(recent_progress=1.0)

    @staticmethod
    def _context(_state, _decision):
        return WorkPreparation(context={"context_view": "PROGRESS"})

    def test_two_control_cycles_run_without_full_history_replay(self) -> None:
        thread_id = self.loop.create_thread(objective="Investigate H1", thread_id="thread-1")
        self.assertEqual(thread_id, "thread-1")

        first = self.loop.run_once(
            signal_provider=self._signals,
            context_provider=self._context,
        )
        self.assertEqual(len(first.assignments), 1)
        self.assertEqual(first.assignments[0].worker_id, "worker-1")
        self.assertIsNotNone(first.assignments[0].work_item.scheduler_decision_id)
        self.assertEqual(
            first.assignments[0].work_item.scheduler_decision_id,
            first.decision.decision_id,
        )

        second = self.loop.run_once(
            signal_provider=self._signals,
            context_provider=self._context,
        )
        self.assertEqual(len(second.assignments), 1)
        self.assertEqual(second.assignments[0].worker_id, "worker-1")
        self.assertEqual(len(self.bank.requests), 2)
        self.assertTrue(
            all(request.work_item.scheduler_decision_id for request in self.bank.requests)
        )

    def test_snapshot_freezes_all_derived_views_at_one_revision(self) -> None:
        self.loop.create_thread(objective="A", thread_id="a")
        self.loop.create_thread(objective="B", thread_id="b")

        snapshot = self.provider.capture()
        self.assertEqual(snapshot.revision, self.ledger.latest_sequence())
        self.assertEqual(
            tuple(state.thread_id for state in snapshot.states),
            ("a", "b"),
        )
        self.assertEqual(
            snapshot.integration_overview.global_snapshot.revision,
            snapshot.revision,
        )
        self.assertEqual(snapshot.verification_overview.revision, snapshot.revision)
        self.assertEqual(snapshot.last_worker_ids, {"a": None, "b": None})

    def test_mutating_thread_graph_uses_index_not_history_replay(self) -> None:
        self.loop.create_thread(objective="Parent", thread_id="p")
        child = self.loop.fork_thread(
            "p",
            objective="Child",
            child_thread_id="c",
        )
        self.assertEqual(child, "c")
        self.loop.add_dependency("c", "p")
        self.loop.remove_dependency("c", "p")
        states = self.thread_state.snapshot_all()
        self.assertEqual(tuple(state.thread_id for state in states), ("p", "c"))
        self.assertEqual(self.thread_state.snapshot(thread_id="p").child_thread_ids, ("c",))

    def test_provider_rejects_replay_based_verification_projector(self) -> None:
        replay_verification = KnowledgeVerificationTracker(self.ledger)
        with self.assertRaisesRegex(ValueError, "SQLiteIndexedKnowledgeState"):
            IndexedRuntimeSnapshotProvider(
                ledger=self.ledger,
                thread_state=self.thread_state,
                integration_tracker=self.integration,
                verification_tracker=replay_verification,
            )


if __name__ == "__main__":
    unittest.main()
