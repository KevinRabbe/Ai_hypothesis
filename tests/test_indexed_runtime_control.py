from __future__ import annotations

import random
import tempfile
import unittest
from pathlib import Path

from ai_hypothesis.runtime.contracts import (
    AttemptResult,
    AttemptStatus,
    EvidenceContribution,
    KnowledgeAssessment,
    KnowledgeAssessmentKind,
    KnowledgeDelta,
)
from ai_hypothesis.runtime.control import WorkPreparation
from ai_hypothesis.runtime.indexed_control import (
    IndexedRuntimeControlLoop,
    IndexedRuntimeIntegrationTracker,
    IndexedRuntimeSnapshotProvider,
    IndexedThreadRuntimeState,
)
from ai_hypothesis.runtime.knowledge_index import SQLiteIndexedKnowledgeState
from ai_hypothesis.runtime.knowledge_verification import KnowledgeVerificationTracker
from ai_hypothesis.runtime.ledger import SQLiteResearchLedger
from ai_hypothesis.runtime.scheduler import SchedulerConfig, SchedulerSignals, SchedulerV0


class NoFullReplayLedger(SQLiteResearchLedger):
    def read_all_events(self, *args, **kwargs):
        raise AssertionError("indexed runtime control must not call read_all_events")


class ProductiveRecordingBank:
    def __init__(self) -> None:
        self.requests = []
        self.next_id = 1

    def execute_batch(self, requests):
        self.requests.extend(requests)
        results = []
        for request in requests:
            suffix = self.next_id
            self.next_id += 1
            evidence_id = f"generated-e-{suffix}"
            delta_id = f"generated-k-{suffix}"
            results.append(
                AttemptResult(
                    attempt_id=request.attempt_id,
                    work_item_id=request.work_item.work_item_id,
                    thread_id=request.work_item.thread_id,
                    worker_id=request.worker_id,
                    status=AttemptStatus.COMPLETED,
                    evidence=(
                        EvidenceContribution(
                            evidence_id=evidence_id,
                            kind="OBSERVATION",
                            summary=f"observation {suffix}",
                        ),
                    ),
                    knowledge_deltas=(
                        KnowledgeDelta(
                            delta_id=delta_id,
                            kind="CLAIM",
                            summary=f"claim {suffix}",
                            reference_ids=(evidence_id,),
                            thread_id=request.work_item.thread_id,
                        ),
                    ),
                    progress_made=True,
                )
            )
        return tuple(results)


class DuplicateEvidenceBank:
    def execute_batch(self, requests):
        return tuple(
            AttemptResult(
                attempt_id=request.attempt_id,
                work_item_id=request.work_item.work_item_id,
                thread_id=request.work_item.thread_id,
                worker_id=request.worker_id,
                status=AttemptStatus.COMPLETED,
                evidence=(
                    EvidenceContribution(
                        evidence_id="existing-evidence",
                        kind="OBSERVATION",
                        summary="collision",
                    ),
                ),
                progress_made=True,
            )
            for request in requests
        )


class AssessmentBank:
    def execute_batch(self, requests):
        return tuple(
            AttemptResult(
                attempt_id=request.attempt_id,
                work_item_id=request.work_item.work_item_id,
                thread_id=request.work_item.thread_id,
                worker_id=request.worker_id,
                status=AttemptStatus.COMPLETED,
                knowledge_assessments=(
                    KnowledgeAssessment(
                        delta_ids=("existing-delta",),
                        assessment=KnowledgeAssessmentKind.VERIFIED,
                        reason="independent verification",
                    ),
                ),
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
        self.integration = IndexedRuntimeIntegrationTracker(
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
        self.bank = ProductiveRecordingBank()
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

    def test_productive_control_cycles_run_without_full_history_replay(self) -> None:
        thread_id = self.loop.create_thread(
            objective="Investigate H1",
            thread_id="thread-1",
        )
        self.assertEqual(thread_id, "thread-1")

        first = self.loop.run_once(
            signal_provider=self._signals,
            context_provider=self._context,
        )
        self.assertEqual(len(first.assignments), 1)
        self.assertEqual(first.assignments[0].worker_id, "worker-1")
        self.assertEqual(len(first.results[0].evidence), 1)
        self.assertEqual(len(first.results[0].knowledge_deltas), 1)
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

        integration = self.integration.snapshot(thread_id="thread-1")
        knowledge = self.knowledge.snapshot(thread_id="thread-1")
        self.assertEqual(integration.evidence_count, 2)
        self.assertEqual(integration.knowledge_delta_count, 2)
        self.assertEqual(len(knowledge.records), 2)

    def test_existing_generated_id_collision_is_detected_without_replay(self) -> None:
        self.loop.create_thread(objective="Collision", thread_id="collision")
        self.ledger.append_event(
            event_type="EVIDENCE_ADDED",
            thread_id="collision",
            reference_ids=("existing-evidence",),
            payload={
                "evidence_id": "existing-evidence",
                "kind": "OBSERVATION",
                "summary": "already durable",
            },
        )
        collision_loop = IndexedRuntimeControlLoop(
            ledger=self.ledger,
            scheduler=self.scheduler,
            worker_bank=DuplicateEvidenceBank(),
            worker_ids=("worker-1", "worker-2"),
            snapshot_provider=self.provider,
        )

        with self.assertRaisesRegex(ValueError, "reused an existing durable"):
            collision_loop.run_once(
                signal_provider=self._signals,
                context_provider=self._context,
            )

    def test_existing_delta_can_be_assessed_without_replay(self) -> None:
        self.loop.create_thread(objective="Verify", thread_id="verify-thread")
        self.ledger.append_event(
            event_type="KNOWLEDGE_DELTA_RECORDED",
            thread_id="verify-thread",
            reference_ids=("existing-delta", "source-1"),
            payload={
                "delta_id": "existing-delta",
                "kind": "CLAIM",
                "summary": "candidate claim",
                "source_reference_ids": ["source-1"],
            },
        )
        verify_loop = IndexedRuntimeControlLoop(
            ledger=self.ledger,
            scheduler=self.scheduler,
            worker_bank=AssessmentBank(),
            worker_ids=("worker-1", "worker-2"),
            snapshot_provider=self.provider,
        )

        step = verify_loop.run_once(
            signal_provider=lambda _state: SchedulerSignals(verification_need=1.0),
            context_provider=lambda _state, _decision: WorkPreparation(
                reference_ids=("existing-delta",),
                context={"context_view": "VERIFY"},
            ),
        )
        self.assertEqual(len(step.results[0].knowledge_assessments), 1)
        self.assertEqual(
            self.knowledge.snapshot(thread_id="verify-thread").records[0].status.value,
            "VERIFIED",
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
        self.assertEqual(
            self.thread_state.snapshot(thread_id="p").child_thread_ids,
            ("c",),
        )

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
