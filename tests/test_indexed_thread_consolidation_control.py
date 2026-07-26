from __future__ import annotations

import random
import tempfile
import unittest
from pathlib import Path

from ai_hypothesis.runtime.contracts import (
    AttemptResult,
    AttemptStatus,
    KnowledgeDelta,
)
from ai_hypothesis.runtime.control import WorkPreparation
from ai_hypothesis.runtime.indexed_control import (
    IndexedRuntimeControlLoop,
    IndexedRuntimeIntegrationTracker,
    IndexedThreadRuntimeState,
)
from ai_hypothesis.runtime.indexed_thread_consolidation import (
    IndexedThreadConsolidationPlanner,
    IndexedThreadConsolidationPressureProjector,
)
from ai_hypothesis.runtime.indexed_thread_consolidation_control import (
    IndexedThreadConsolidationControlAdapter,
    PinnedIndexedRuntimeSnapshotProvider,
)
from ai_hypothesis.runtime.knowledge_index import SQLiteIndexedKnowledgeState
from ai_hypothesis.runtime.ledger import SQLiteResearchLedger
from ai_hypothesis.runtime.partition_knowledge_index import (
    SQLiteIndexedPartitionKnowledgeLineage,
)
from ai_hypothesis.runtime.scheduler import SchedulerConfig, SchedulerSignals, SchedulerV0
from ai_hypothesis.runtime.scheduler_trace import TracingScheduler
from ai_hypothesis.runtime.thread_consolidation import ThreadConsolidationConfig
from ai_hypothesis.runtime.thread_consolidation_control import (
    ThreadConsolidationPressureConfig,
    ThreadConsolidationScheduler,
)


class NoFullReplayLedger(SQLiteResearchLedger):
    def read_all_events(self, *args, **kwargs):
        raise AssertionError("indexed consolidation control must not call read_all_events")


class CountingKnowledgeState(SQLiteIndexedKnowledgeState):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.project_calls = 0

    def project(self, *args, **kwargs):
        self.project_calls += 1
        return super().project(*args, **kwargs)


class ConsolidationBank:
    def __init__(self) -> None:
        self.requests = []

    def execute_batch(self, requests):
        self.requests.extend(requests)
        results = []
        for index, request in enumerate(requests, start=1):
            refs = tuple(request.work_item.reference_ids)
            results.append(
                AttemptResult(
                    attempt_id=request.attempt_id,
                    work_item_id=request.work_item.work_item_id,
                    thread_id=request.work_item.thread_id,
                    worker_id=request.worker_id,
                    status=AttemptStatus.COMPLETED,
                    knowledge_deltas=(
                        KnowledgeDelta(
                            delta_id=f"thread-consolidation-{index}",
                            kind="THREAD_CONSOLIDATION",
                            summary="bounded higher-level synthesis",
                            reference_ids=refs,
                            thread_id=request.work_item.thread_id,
                        ),
                    ),
                    progress_made=True,
                )
            )
        return tuple(results)


def append(ledger, event_type, *, thread_id=None, attempt_id=None, reference_ids=(), payload=None):
    return ledger.append_event(
        event_type=event_type,
        thread_id=thread_id,
        attempt_id=attempt_id,
        reference_ids=tuple(reference_ids),
        payload=payload or {},
    )


def seed_runtime_history(ledger: SQLiteResearchLedger) -> None:
    append(
        ledger,
        "THREAD_CREATED",
        thread_id="thread-a",
        payload={"objective": "integrate findings", "purpose": "PROGRESS", "status": "ACTIVE"},
    )
    for evidence_id in ("e1", "e2"):
        append(
            ledger,
            "EVIDENCE_ADDED",
            thread_id="thread-a",
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
        thread_id="thread-a",
        payload={
            "decision_id": "partition-decision",
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
        thread_id="thread-a",
        reference_ids=("partition-0", "partition-1"),
        payload={
            "schema": "integration-partition-allocation-v0",
            "decision_id": "partition-decision",
            "decision_projection_revision": 1,
            "partition_plan_revision": ledger.latest_sequence(),
            "shard_count": 4,
            "batch_limit": 32,
            "width": 2,
            "partitions": [
                {
                    "partition_id": "partition-0",
                    "shard_index": 0,
                    "backlog_count": 1,
                    "oldest_pending_sequence": 2,
                    "evidence_ids": ["e1"],
                },
                {
                    "partition_id": "partition-1",
                    "shard_index": 1,
                    "backlog_count": 1,
                    "oldest_pending_sequence": 3,
                    "evidence_ids": ["e2"],
                },
            ],
        },
    )
    for ordinal, (partition_id, evidence_id, attempt_id, delta_id) in enumerate(
        (
            ("partition-0", "e1", "partition-attempt-0", "partition-delta-0"),
            ("partition-1", "e2", "partition-attempt-1", "partition-delta-1"),
        )
    ):
        append(
            ledger,
            "ATTEMPT_STARTED",
            thread_id="thread-a",
            attempt_id=attempt_id,
            reference_ids=(evidence_id,),
            payload={
                "work_item_id": f"partition-work-{ordinal}",
                "worker_id": f"partition-worker-{ordinal}",
                "purpose": "SYNTHESIZE",
                "projection_revision": 1,
                "scheduler_decision_id": "partition-decision",
                "scope_region_ids": [],
            },
        )
        append(
            ledger,
            "KNOWLEDGE_DELTA_RECORDED",
            thread_id="thread-a",
            attempt_id=attempt_id,
            reference_ids=(delta_id, evidence_id),
            payload={
                "delta_id": delta_id,
                "kind": "PARTITION_SYNTHESIS",
                "summary": delta_id,
                "source_reference_ids": [evidence_id],
            },
        )
        append(
            ledger,
            "ATTEMPT_COMPLETED",
            thread_id="thread-a",
            attempt_id=attempt_id,
            payload={"progress_made": True},
        )


class IndexedThreadConsolidationControlTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        root = Path(self.tempdir.name)
        self.ledger = NoFullReplayLedger(root / "ledger.sqlite3")
        seed_runtime_history(self.ledger)
        self.thread_state = IndexedThreadRuntimeState(
            self.ledger,
            root / "threads.sqlite3",
        )
        self.integration = IndexedRuntimeIntegrationTracker(
            self.ledger,
            root / "integration.sqlite3",
        )
        self.knowledge = CountingKnowledgeState(
            self.ledger,
            root / "knowledge.sqlite3",
        )
        self.lineage = SQLiteIndexedPartitionKnowledgeLineage(
            self.ledger,
            root / "lineage.sqlite3",
        )
        self.snapshot_provider = PinnedIndexedRuntimeSnapshotProvider(
            ledger=self.ledger,
            thread_state=self.thread_state,
            integration_tracker=self.integration,
            verification_tracker=None,
        )
        self.planner = IndexedThreadConsolidationPlanner(
            ledger=self.ledger,
            lineage=self.lineage,
            knowledge=self.knowledge,
            config=ThreadConsolidationConfig(
                selection_limit=32,
                minimum_source_deltas=2,
            ),
        )
        self.pressure = IndexedThreadConsolidationPressureProjector(
            ledger=self.ledger,
            lineage=self.lineage,
            knowledge=self.knowledge,
            config=ThreadConsolidationPressureConfig(
                full_pressure_count=2,
                minimum_source_deltas=2,
            ),
        )
        self.adapter = IndexedThreadConsolidationControlAdapter(
            ledger=self.ledger,
            revision_provider=self.snapshot_provider,
            signal_fallback=lambda _state: SchedulerSignals(recent_progress=1.0),
            context_fallback=lambda _state, _decision: WorkPreparation(
                context={"context_view": "PROGRESS"}
            ),
            planner=self.planner,
            pressure_projector=self.pressure,
            knowledge=self.knowledge,
        )
        base = SchedulerV0(
            SchedulerConfig(
                exploration_probability=0.0,
                synthesis_threshold=0.65,
            ),
            rng=random.Random(1),
        )
        routed = ThreadConsolidationScheduler(base, control=self.adapter)
        self.scheduler = TracingScheduler(self.ledger, routed)
        self.bank = ConsolidationBank()
        self.loop = IndexedRuntimeControlLoop(
            ledger=self.ledger,
            scheduler=self.scheduler,
            worker_bank=self.bank,
            worker_ids=("worker-a", "worker-b"),
            snapshot_provider=self.snapshot_provider,
        )

    def tearDown(self) -> None:
        self.lineage.close()
        self.knowledge.close()
        self.integration.close()
        self.thread_state.close()
        self.ledger.close()
        self.tempdir.cleanup()

    def test_full_automatic_consolidation_cycle_uses_only_indexed_state(self) -> None:
        before_revision = self.ledger.latest_sequence()
        step = self.loop.run_once(
            signal_provider=self.adapter.signals,
            context_provider=self.adapter.context,
        )

        self.assertEqual(step.decision.action.value, "SYNTHESIZE")
        self.assertIn("SYNTHESIS_NEEDED", step.decision.reason_codes)
        self.assertIn("THREAD_CONSOLIDATION", step.decision.reason_codes)
        self.assertEqual(len(step.assignments), 1)
        preparation = step.assignments[0].work_item
        self.assertEqual(
            preparation.context["synthesis_mode"],
            "THREAD_CONSOLIDATION",
        )
        self.assertEqual(
            preparation.context["consolidation_pressure_revision"],
            before_revision,
        )
        self.assertEqual(
            set(preparation.reference_ids),
            {"partition-delta-0", "partition-delta-1"},
        )
        self.assertEqual(len(step.results[0].knowledge_deltas), 1)
        self.assertEqual(
            step.results[0].knowledge_deltas[0].kind,
            "THREAD_CONSOLIDATION",
        )

        latest = self.ledger.latest_sequence()
        overview = self.pressure.project(sequence=latest)
        self.assertEqual(overview.pending_source_count["thread-a"], 0)
        self.assertEqual(overview.pressure_for("thread-a"), 0.0)

    def test_context_phase_does_not_reproject_after_scheduler_trace_append(self) -> None:
        self.assertEqual(self.knowledge.project_calls, 0)
        step = self.loop.run_once(
            signal_provider=self.adapter.signals,
            context_provider=self.adapter.context,
        )
        # pressure.project -> one global Knowledge projection
        # planner.plan -> one thread Knowledge projection
        # bounded preparation -> one thread Knowledge projection
        self.assertEqual(self.knowledge.project_calls, 3)

        traced_events = self.ledger.read_events(after_sequence=0, limit=10_000)
        trace = [
            event
            for event in traced_events
            if event.event_type == "SCHEDULER_DECISION_RECORDED"
            and event.payload.get("decision_id") == step.decision.decision_id
        ]
        self.assertEqual(len(trace), 1)
        self.assertIn("THREAD_CONSOLIDATION", trace[0].payload["reason_codes"])
        self.assertEqual(
            step.assignments[0].work_item.context["consolidation_pressure_revision"],
            self.snapshot_provider.current_revision,
        )

    def test_forged_route_without_owned_signal_is_rejected(self) -> None:
        snapshot = self.snapshot_provider.capture()
        state = snapshot.states[0]
        from ai_hypothesis.runtime.contracts import SchedulerAction, SchedulerDecision, WorkPurpose

        forged = SchedulerDecision(
            decision_id="forged",
            thread_id=state.thread_id,
            action=SchedulerAction.SYNTHESIZE,
            purpose=WorkPurpose.SYNTHESIZE,
            width=1,
            reason_codes=("SYNTHESIS_NEEDED", "THREAD_CONSOLIDATION"),
            projection_revision=state.revision,
        )
        forged.validate()
        with self.assertRaisesRegex(RuntimeError, "no longer matches"):
            self.adapter.context(state, forged)

    def test_domain_owned_stronger_synthesis_need_is_not_relabelled(self) -> None:
        dominant_adapter = IndexedThreadConsolidationControlAdapter(
            ledger=self.ledger,
            revision_provider=self.snapshot_provider,
            signal_fallback=lambda _state: SchedulerSignals(
                recent_progress=1.0,
                synthesis_need=1.0,
            ),
            context_fallback=lambda _state, _decision: WorkPreparation(
                context={"context_view": "SYNTHESIZE", "domain_owned": True}
            ),
            planner=self.planner,
            pressure_projector=self.pressure,
            knowledge=self.knowledge,
        )
        snapshot = self.snapshot_provider.capture()
        state = snapshot.states[0]
        signals = dominant_adapter.signals(state)
        self.assertEqual(signals.synthesis_need, 1.0)
        self.assertFalse(dominant_adapter.owns_route(state.thread_id, state.revision))


if __name__ == "__main__":
    unittest.main()
