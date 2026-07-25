"""Regression coverage for the repaired persistent search-loop runtime."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from ai_hypothesis.runtime import (
    AttemptResult,
    AttemptStatus,
    FollowupMaterializer,
    KnowledgeAssessment,
    KnowledgeAssessmentKind,
    KnowledgeStateProjector,
    KnowledgeStatus,
    KnowledgeVerificationConfig,
    KnowledgeVerificationTracker,
    RuntimeControlLoop,
    SchedulerAction,
    SchedulerConfig,
    SchedulerSignals,
    SchedulerV0,
    SchedulableThread,
    SQLiteResearchLedger,
    ThreadStateProjector,
    TracingSchedulerV0,
    WorkGraphContextResolver,
    WorkPreparation,
)


class _NoopBank:
    def execute_batch(self, requests):
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


class _VerificationBank:
    def execute_batch(self, requests):
        results = []
        for request in requests:
            assessments = ()
            if request.work_item.purpose.value == "VERIFY":
                assessments = (
                    KnowledgeAssessment(
                        delta_ids=tuple(request.work_item.reference_ids),
                        assessment=KnowledgeAssessmentKind.VERIFIED,
                        reason="independent verification passed",
                    ),
                )
            results.append(
                AttemptResult(
                    attempt_id=request.attempt_id,
                    work_item_id=request.work_item.work_item_id,
                    thread_id=request.work_item.thread_id,
                    worker_id=request.worker_id,
                    status=AttemptStatus.COMPLETED,
                    knowledge_assessments=assessments,
                    progress_made=True,
                )
            )
        return tuple(results)


def _loop(ledger, *, bank=None, verification_tracker=None, scheduler=None):
    return RuntimeControlLoop(
        ledger=ledger,
        scheduler=scheduler
        or SchedulerV0(SchedulerConfig(exploration_probability=0.0)),
        worker_bank=bank or _NoopBank(),
        worker_ids=("worker-a", "worker-b", "worker-c"),
        verification_tracker=verification_tracker,
    )


class RuntimeSearchLoopV1Tests(unittest.TestCase):
    def test_duplicate_thread_identity_is_rejected_before_write(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with SQLiteResearchLedger(Path(directory) / "ledger.sqlite") as ledger:
                loop = _loop(ledger)
                loop.create_thread(thread_id="same", objective="first")
                before = ledger.latest_sequence()
                with self.assertRaisesRegex(ValueError, "already exists"):
                    loop.create_thread(thread_id="same", objective="second")
                self.assertEqual(ledger.latest_sequence(), before)

    def test_fork_dependency_and_merge_are_rebuildable_graph_state(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with SQLiteResearchLedger(Path(directory) / "ledger.sqlite") as ledger:
                loop = _loop(ledger)
                loop.create_thread(thread_id="root", objective="root")
                loop.fork_thread(
                    "root",
                    child_thread_id="child",
                    objective="child",
                )
                loop.create_thread(thread_id="target", objective="target")
                loop.add_dependency("target", "child")

                states = {
                    state.thread_id: state
                    for state in ThreadStateProjector().project_all(
                        ledger.read_all_events()
                    )
                }
                self.assertEqual(states["root"].child_thread_ids, ("child",))
                self.assertEqual(states["child"].parent_thread_ids, ("root",))
                self.assertEqual(states["target"].dependency_thread_ids, ("child",))

                before = ledger.latest_sequence()
                with self.assertRaisesRegex(ValueError, "dependency cycle"):
                    loop.add_dependency("child", "target")
                self.assertEqual(ledger.latest_sequence(), before)

                loop.merge_threads("target", ("root", "child"))
                merged = {
                    state.thread_id: state
                    for state in ThreadStateProjector().project_all(
                        ledger.read_all_events()
                    )
                }
                self.assertEqual(merged["root"].status, "COMPLETE")
                self.assertEqual(merged["child"].status, "COMPLETE")
                self.assertEqual(merged["root"].merged_into_thread_id, "target")
                self.assertEqual(
                    merged["target"].merged_from_thread_ids,
                    ("root", "child"),
                )

    def test_dependency_blocks_scheduler_until_prerequisite_completes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with SQLiteResearchLedger(Path(directory) / "ledger.sqlite") as ledger:
                loop = _loop(ledger)
                loop.create_thread(thread_id="prerequisite", objective="prerequisite")
                loop.create_thread(thread_id="dependent", objective="dependent")
                loop.add_dependency("dependent", "prerequisite")

                signals = lambda state: SchedulerSignals(
                    importance=1.0 if state.thread_id == "dependent" else 0.1,
                    recent_progress=1.0,
                )
                first = loop.run_once(
                    signal_provider=signals,
                    context_provider=lambda _state, _decision: WorkPreparation(),
                )
                self.assertEqual(first.decision.thread_id, "prerequisite")

                ledger.append_event(
                    event_type="THREAD_COMPLETED",
                    thread_id="prerequisite",
                )
                second = loop.run_once(
                    signal_provider=signals,
                    context_provider=lambda _state, _decision: WorkPreparation(),
                )
                self.assertEqual(second.decision.thread_id, "dependent")

    def test_unresolved_knowledge_routes_verify_then_clears_pressure(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with SQLiteResearchLedger(Path(directory) / "ledger.sqlite") as ledger:
                ledger.append_event(
                    event_type="THREAD_CREATED",
                    thread_id="research",
                    payload={"objective": "research", "purpose": "PROGRESS"},
                )
                ledger.append_event(
                    event_type="KNOWLEDGE_DELTA_RECORDED",
                    thread_id="research",
                    reference_ids=("delta-1", "evidence-1"),
                    payload={
                        "delta_id": "delta-1",
                        "kind": "TEST",
                        "summary": "provisional result",
                        "source_reference_ids": ["evidence-1"],
                        "causal_event_ids": [],
                    },
                )
                tracker = KnowledgeVerificationTracker(
                    ledger,
                    KnowledgeVerificationConfig(full_pressure_count=1),
                )
                loop = _loop(
                    ledger,
                    bank=_VerificationBank(),
                    verification_tracker=tracker,
                )

                first = loop.run_once(
                    signal_provider=lambda _state: SchedulerSignals(
                        importance=1.0,
                        recent_progress=1.0,
                    ),
                    context_provider=lambda state, decision: WorkPreparation(
                        reference_ids=(
                            tracker.pending_delta_ids(
                                thread_id=state.thread_id,
                                limit=8,
                            )
                            if decision.action is SchedulerAction.VERIFY
                            else ()
                        )
                    ),
                )
                self.assertEqual(first.decision.action, SchedulerAction.VERIFY)
                self.assertEqual(tracker.pressure(thread_id="research"), 0.0)

    def test_followups_materialize_idempotent_child_and_graph_context_is_bounded(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with SQLiteResearchLedger(Path(directory) / "ledger.sqlite") as ledger:
                loop = _loop(ledger)
                loop.create_thread(thread_id="parent", objective="parent")
                request = ledger.append_event(
                    event_type="FOLLOWUP_REQUESTED",
                    thread_id="parent",
                    payload={"request": "inspect alternative"},
                )
                materializer = FollowupMaterializer(ledger)
                first = materializer.materialize(limit=1)
                second = materializer.materialize(limit=1)
                self.assertEqual(len(first), 1)
                self.assertEqual(second, ())
                self.assertEqual(
                    first[0],
                    materializer.child_thread_id(request.event_id),
                )

                child_id = first[0]
                ledger.append_event(
                    event_type="KNOWLEDGE_DELTA_RECORDED",
                    thread_id="parent",
                    reference_ids=("delta-parent", "evidence-parent"),
                    payload={
                        "delta_id": "delta-parent",
                        "kind": "TEST",
                        "summary": "parent knowledge",
                        "source_reference_ids": ["evidence-parent"],
                        "causal_event_ids": [],
                    },
                )
                states = ThreadStateProjector().project_all(ledger.read_all_events())
                knowledge = KnowledgeStateProjector().project(ledger.read_all_events())
                resolver = WorkGraphContextResolver(states, knowledge)
                related = resolver.related_knowledge_delta_ids(
                    child_id,
                    statuses=(KnowledgeStatus.PROVISIONAL,),
                    include_parents=True,
                    thread_limit=1,
                    knowledge_limit=1,
                )
                self.assertEqual(related, ("delta-parent",))

    def test_scheduler_trace_records_actual_decision(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with SQLiteResearchLedger(Path(directory) / "ledger.sqlite") as ledger:
                scheduler = TracingSchedulerV0(
                    ledger,
                    SchedulerConfig(exploration_probability=0.0),
                )
                state = ThreadStateProjector().project_all(
                    (
                        ledger.append_event(
                            event_type="THREAD_CREATED",
                            thread_id="thread-1",
                            payload={"objective": "trace", "purpose": "PROGRESS"},
                        ),
                    )
                )[0]
                decision = scheduler.choose(
                    (
                        SchedulableThread(
                            state=state,
                            signals=SchedulerSignals(
                                importance=1.0,
                                recent_progress=1.0,
                            ),
                        ),
                    ),
                    max_width=3,
                )
                traces = [
                    event
                    for event in ledger.read_all_events()
                    if event.event_type == "SCHEDULER_DECISION_RECORDED"
                ]
                self.assertEqual(len(traces), 1)
                self.assertEqual(traces[0].payload["decision_id"], decision.decision_id)
                self.assertEqual(traces[0].payload["width"], decision.width)


if __name__ == "__main__":
    unittest.main()
