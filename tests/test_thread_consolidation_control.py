from __future__ import annotations

import unittest

from ai_hypothesis.runtime import (
    AttemptResult,
    AttemptStatus,
    KnowledgeDelta,
    KnowledgeStateProjector,
    ProjectedState,
    RuntimeControlLoop,
    SchedulerAction,
    SchedulerConfig,
    SchedulerDecision,
    SchedulerSignals,
    SchedulerV0,
    SQLiteResearchLedger,
    ThreadConsolidationConfig,
    ThreadConsolidationPlanner,
    TracingScheduler,
    WorkPreparation,
    WorkPurpose,
)
from ai_hypothesis.runtime.thread_consolidation_control import (
    ThreadConsolidationControlAdapter,
    ThreadConsolidationPressureConfig,
    ThreadConsolidationPressureProjector,
    ThreadConsolidationScheduler,
)


def _append(
    ledger: SQLiteResearchLedger,
    event_type: str,
    *,
    attempt_id: str | None = None,
    reference_ids: tuple[str, ...] = (),
    payload: dict | None = None,
) -> None:
    ledger.append_event(
        event_type=event_type,
        thread_id="thread-a",
        attempt_id=attempt_id,
        reference_ids=reference_ids,
        payload=payload or {},
    )


def _seed_partition_knowledge(
    ledger: SQLiteResearchLedger,
    *,
    include_thread: bool = True,
    include_provenance: bool = True,
) -> None:
    if include_thread:
        _append(
            ledger,
            "THREAD_CREATED",
            payload={
                "objective": "integrate and consolidate",
                "purpose": "PROGRESS",
                "status": "ACTIVE",
            },
        )
    _append(
        ledger,
        "EVIDENCE_ADDED",
        reference_ids=("e1",),
        payload={"evidence_id": "e1", "kind": "OBSERVATION", "summary": "e1"},
    )
    _append(
        ledger,
        "EVIDENCE_ADDED",
        reference_ids=("e2",),
        payload={"evidence_id": "e2", "kind": "OBSERVATION", "summary": "e2"},
    )
    _append(
        ledger,
        "SCHEDULER_DECISION_RECORDED",
        payload={
            "decision_id": "historical-integration",
            "action": "SYNTHESIZE",
            "purpose": "SYNTHESIZE",
            "width": 2,
            "reason_codes": ["BACKPRESSURE", "PARTITIONED_INTEGRATION"],
            "projection_revision": 2,
            "integration_backpressure": True,
            "max_width": 2,
        },
    )
    if include_provenance:
        _append(
            ledger,
            "INTEGRATION_PARTITION_ALLOCATION_RECORDED",
            reference_ids=("partition-a", "partition-b"),
            payload={
                "schema": "integration-partition-allocation-v0",
                "decision_id": "historical-integration",
                "decision_projection_revision": 2,
                "partition_plan_revision": ledger.latest_sequence(),
                "shard_count": 8,
                "batch_limit": 1,
                "width": 2,
                "partitions": [
                    {
                        "partition_id": "partition-a",
                        "shard_index": 1,
                        "backlog_count": 1,
                        "oldest_pending_sequence": 2,
                        "evidence_ids": ["e1"],
                    },
                    {
                        "partition_id": "partition-b",
                        "shard_index": 6,
                        "backlog_count": 1,
                        "oldest_pending_sequence": 3,
                        "evidence_ids": ["e2"],
                    },
                ],
            },
        )
    _append(
        ledger,
        "ATTEMPT_STARTED",
        attempt_id="historical-attempt-a",
        reference_ids=("e1",),
        payload={
            "work_item_id": "historical-work-a",
            "worker_id": "historical-worker-a",
            "purpose": "SYNTHESIZE",
            "projection_revision": 2,
            "scheduler_decision_id": "historical-integration",
            "scope_region_ids": [],
        },
    )
    _append(
        ledger,
        "ATTEMPT_STARTED",
        attempt_id="historical-attempt-b",
        reference_ids=("e2",),
        payload={
            "work_item_id": "historical-work-b",
            "worker_id": "historical-worker-b",
            "purpose": "SYNTHESIZE",
            "projection_revision": 2,
            "scheduler_decision_id": "historical-integration",
            "scope_region_ids": [],
        },
    )
    _append(
        ledger,
        "KNOWLEDGE_DELTA_RECORDED",
        attempt_id="historical-attempt-a",
        reference_ids=("k1", "e1"),
        payload={
            "delta_id": "k1",
            "kind": "INTEGRATION_SUMMARY",
            "summary": "partition A finding",
            "source_reference_ids": ["e1"],
            "causal_event_ids": [],
        },
    )
    _append(
        ledger,
        "KNOWLEDGE_DELTA_RECORDED",
        attempt_id="historical-attempt-b",
        reference_ids=("k2", "e2"),
        payload={
            "delta_id": "k2",
            "kind": "INTEGRATION_SUMMARY",
            "summary": "partition B finding",
            "source_reference_ids": ["e2"],
            "causal_event_ids": [],
        },
    )
    _append(
        ledger,
        "ATTEMPT_COMPLETED",
        attempt_id="historical-attempt-a",
        payload={"progress_made": True},
    )
    _append(
        ledger,
        "ATTEMPT_COMPLETED",
        attempt_id="historical-attempt-b",
        payload={"progress_made": True},
    )


class ThreadConsolidationPressureTests(unittest.TestCase):
    def test_one_pass_pressure_matches_real_planner_pending_counts(self) -> None:
        ledger = SQLiteResearchLedger(":memory:")
        self.addCleanup(ledger.close)
        _seed_partition_knowledge(ledger)
        events = ledger.read_all_events()
        planner = ThreadConsolidationPlanner(
            ThreadConsolidationConfig(selection_limit=8, minimum_source_deltas=2)
        )
        plan = planner.plan(events, thread_id="thread-a")
        overview = ThreadConsolidationPressureProjector(
            ThreadConsolidationPressureConfig(
                full_pressure_count=2,
                minimum_source_deltas=2,
            )
        ).project(events)

        self.assertEqual(overview.pending_sources_for("thread-a"), plan.pending_source_count)
        self.assertEqual(
            overview.pending_partitions_for("thread-a"),
            plan.pending_partition_count,
        )
        self.assertEqual(overview.pressure_for("thread-a"), 1.0)
        self.assertFalse(overview.is_incomplete("thread-a"))

    def test_active_consolidation_removes_pressure_and_retraction_restores_it(self) -> None:
        ledger = SQLiteResearchLedger(":memory:")
        self.addCleanup(ledger.close)
        _seed_partition_knowledge(ledger)
        projector = ThreadConsolidationPressureProjector(
            ThreadConsolidationPressureConfig(
                full_pressure_count=2,
                minimum_source_deltas=2,
            )
        )
        before = projector.project(ledger.read_all_events())
        self.assertEqual(before.pressure_for("thread-a"), 1.0)

        _append(
            ledger,
            "KNOWLEDGE_DELTA_RECORDED",
            reference_ids=("thread-summary", "k1", "k2"),
            payload={
                "delta_id": "thread-summary",
                "kind": "THREAD_CONSOLIDATION",
                "summary": "thread summary",
                "source_reference_ids": ["k1", "k2"],
                "causal_event_ids": [],
            },
        )
        consumed = projector.project(ledger.read_all_events())
        self.assertEqual(consumed.pending_sources_for("thread-a"), 0)
        self.assertEqual(consumed.pressure_for("thread-a"), 0.0)

        _append(
            ledger,
            "KNOWLEDGE_ASSESSMENT_RECORDED",
            reference_ids=("thread-summary",),
            payload={"assessment": "RETRACTED", "reason": "bad summary"},
        )
        reopened = projector.project(ledger.read_all_events())
        self.assertEqual(reopened.pending_sources_for("thread-a"), 2)
        self.assertEqual(reopened.pressure_for("thread-a"), 1.0)

    def test_missing_partition_provenance_suppresses_pressure_instead_of_guessing(self) -> None:
        ledger = SQLiteResearchLedger(":memory:")
        self.addCleanup(ledger.close)
        _seed_partition_knowledge(ledger, include_provenance=False)
        overview = ThreadConsolidationPressureProjector(
            ThreadConsolidationPressureConfig(
                full_pressure_count=2,
                minimum_source_deltas=2,
            )
        ).project(ledger.read_all_events())

        self.assertTrue(overview.is_incomplete("thread-a"))
        self.assertEqual(overview.pressure_for("thread-a"), 0.0)


class ThreadConsolidationRoutingTests(unittest.TestCase):
    def test_adapter_owns_route_only_when_it_raises_synthesis_need(self) -> None:
        ledger = SQLiteResearchLedger(":memory:")
        self.addCleanup(ledger.close)
        _seed_partition_knowledge(ledger)
        state = ProjectedState(
            revision=ledger.latest_sequence(),
            thread_id="thread-a",
            objective="consolidate",
            status="ACTIVE",
            purpose=WorkPurpose.PROGRESS,
        )
        pressure = ThreadConsolidationPressureProjector(
            ThreadConsolidationPressureConfig(
                full_pressure_count=2,
                minimum_source_deltas=2,
            )
        )
        owned = ThreadConsolidationControlAdapter(
            ledger=ledger,
            signal_fallback=lambda state: SchedulerSignals(recent_progress=1.0),
            context_fallback=lambda state, decision: WorkPreparation(context={"fallback": True}),
            pressure_projector=pressure,
        )
        adjusted = owned.signals(state)
        self.assertEqual(adjusted.synthesis_need, 1.0)
        self.assertTrue(owned.owns_route("thread-a", state.revision))

        domain_owned = ThreadConsolidationControlAdapter(
            ledger=ledger,
            signal_fallback=lambda state: SchedulerSignals(
                recent_progress=1.0,
                synthesis_need=1.0,
            ),
            context_fallback=lambda state, decision: WorkPreparation(context={"fallback": True}),
            pressure_projector=pressure,
        )
        unchanged = domain_owned.signals(state)
        self.assertEqual(unchanged.synthesis_need, 1.0)
        self.assertFalse(domain_owned.owns_route("thread-a", state.revision))

    def test_forged_thread_consolidation_reason_is_rejected(self) -> None:
        ledger = SQLiteResearchLedger(":memory:")
        self.addCleanup(ledger.close)
        _seed_partition_knowledge(ledger)
        state = ProjectedState(
            revision=ledger.latest_sequence(),
            thread_id="thread-a",
            objective="consolidate",
            status="ACTIVE",
            purpose=WorkPurpose.PROGRESS,
        )
        control = ThreadConsolidationControlAdapter(
            ledger=ledger,
            signal_fallback=lambda state: SchedulerSignals(
                recent_progress=1.0,
                synthesis_need=1.0,
            ),
            context_fallback=lambda state, decision: WorkPreparation(context={"fallback": True}),
        )
        control.signals(state)
        decision = SchedulerDecision(
            decision_id="forged",
            thread_id="thread-a",
            action=SchedulerAction.SYNTHESIZE,
            purpose=WorkPurpose.SYNTHESIZE,
            reason_codes=("SYNTHESIS_NEEDED", "THREAD_CONSOLIDATION"),
            projection_revision=state.revision,
        )

        with self.assertRaisesRegex(ValueError, "route was not owned"):
            control.context(state, decision)


class _AutoConsolidationBank:
    def __init__(self) -> None:
        self.calls = 0
        self.contexts = []

    def execute_batch(self, requests):
        self.calls += 1
        results = []
        for request in requests:
            self.contexts.append(dict(request.work_item.context))
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
                            delta_id="auto-thread-summary",
                            kind="THREAD_CONSOLIDATION",
                            summary="automatic cross-partition consolidation",
                            reference_ids=refs,
                            thread_id=request.work_item.thread_id,
                        ),
                    ),
                    progress_made=True,
                )
            )
        return tuple(results)


class AutomaticThreadConsolidationRuntimeTests(unittest.TestCase):
    def test_normal_control_loop_automatically_routes_ready_consolidation(self) -> None:
        ledger = SQLiteResearchLedger(":memory:")
        self.addCleanup(ledger.close)
        _seed_partition_knowledge(ledger)

        pressure_projector = ThreadConsolidationPressureProjector(
            ThreadConsolidationPressureConfig(
                full_pressure_count=2,
                minimum_source_deltas=2,
            )
        )
        planner = ThreadConsolidationPlanner(
            ThreadConsolidationConfig(selection_limit=2, minimum_source_deltas=2)
        )
        adapter = ThreadConsolidationControlAdapter(
            ledger=ledger,
            signal_fallback=lambda state: SchedulerSignals(
                importance=1.0,
                recent_progress=1.0,
            ),
            context_fallback=lambda state, decision: WorkPreparation(context={"fallback": True}),
            pressure_projector=pressure_projector,
            planner=planner,
        )
        base_scheduler = SchedulerV0(
            SchedulerConfig(
                exploration_probability=0.0,
                synthesis_threshold=0.5,
            )
        )
        routed_scheduler = ThreadConsolidationScheduler(
            base_scheduler,
            control=adapter,
        )
        scheduler = TracingScheduler(ledger, routed_scheduler)
        bank = _AutoConsolidationBank()
        loop = RuntimeControlLoop(
            ledger=ledger,
            scheduler=scheduler,
            worker_bank=bank,
            worker_ids=("worker-auto",),
        )

        step = loop.run_once(
            signal_provider=adapter.signals,
            context_provider=adapter.context,
            integration_backpressure=False,
        )

        self.assertEqual(step.decision.action, SchedulerAction.SYNTHESIZE)
        self.assertIn("SYNTHESIS_NEEDED", step.decision.reason_codes)
        self.assertIn("THREAD_CONSOLIDATION", step.decision.reason_codes)
        self.assertEqual(step.decision.width, 1)
        self.assertEqual(bank.calls, 1)
        self.assertEqual(len(step.results), 1)
        self.assertEqual(
            bank.contexts[0]["synthesis_mode"],
            "THREAD_CONSOLIDATION",
        )
        self.assertEqual(
            bank.contexts[0]["synthesis_route"],
            "THREAD_CONSOLIDATION",
        )

        traces = tuple(
            event
            for event in ledger.read_all_events()
            if event.event_type == "SCHEDULER_DECISION_RECORDED"
            and event.payload.get("decision_id") == step.decision.decision_id
        )
        self.assertEqual(len(traces), 1)
        self.assertIn("THREAD_CONSOLIDATION", traces[0].payload["reason_codes"])

        knowledge = KnowledgeStateProjector().project(ledger.read_all_events())
        summary = knowledge.get("auto-thread-summary")
        self.assertIsNotNone(summary)
        self.assertEqual(summary.kind, "THREAD_CONSOLIDATION")
        self.assertEqual(summary.source_reference_ids, ("k1", "k2"))
        self.assertEqual(summary.status.value, "PROVISIONAL")

        next_pressure = pressure_projector.project(ledger.read_all_events())
        self.assertEqual(next_pressure.pending_sources_for("thread-a"), 0)
        self.assertEqual(next_pressure.pressure_for("thread-a"), 0.0)


if __name__ == "__main__":
    unittest.main()
