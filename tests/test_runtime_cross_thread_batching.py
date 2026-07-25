"""Tests snapshot-isolated cross-thread batching in RuntimeControlLoop."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from ai_hypothesis.runtime import (
    AllocationOutcomeProjector,
    AttemptResult,
    AttemptStatus,
    RuntimeControlLoop,
    SchedulerAction,
    SchedulerDecision,
    SchedulerSignals,
    SQLiteResearchLedger,
    TracingScheduler,
    WorkPreparation,
    WorkPurpose,
)


class _RecordingBank:
    def __init__(self) -> None:
        self.calls: list[tuple[tuple[str, str], ...]] = []

    def execute_batch(self, requests):
        self.calls.append(
            tuple(
                (request.work_item.thread_id, request.worker_id)
                for request in requests
            )
        )
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


class _FirstCandidateScheduler:
    def __init__(self) -> None:
        self.count = 0

    def choose(self, candidates, *, integration_backpressure=False, max_width=1):
        del integration_backpressure
        if max_width <= 0:
            raise ValueError("max_width must be positive")
        self.count += 1
        state = candidates[0].state
        return SchedulerDecision(
            decision_id=f"decision-{self.count}",
            thread_id=state.thread_id,
            action=SchedulerAction.CONTINUE,
            purpose=state.purpose,
            width=1,
            projection_revision=state.revision,
        )


class _WidthTwoScheduler:
    def __init__(self) -> None:
        self.count = 0

    def choose(self, candidates, *, integration_backpressure=False, max_width=1):
        del integration_backpressure
        self.count += 1
        state = candidates[0].state
        width = min(2, max_width)
        return SchedulerDecision(
            decision_id=f"width-decision-{self.count}",
            thread_id=state.thread_id,
            action=(SchedulerAction.ADD_WIDTH if width > 1 else SchedulerAction.CONTINUE),
            purpose=WorkPurpose.EXPLORE,
            width=width,
            projection_revision=state.revision,
        )


class _CompleteDependencyScheduler:
    def __init__(self) -> None:
        self.count = 0

    def choose(self, candidates, *, integration_backpressure=False, max_width=1):
        del integration_backpressure
        self.count += 1
        state = candidates[0].state
        action = (
            SchedulerAction.COMPLETE
            if state.thread_id == "dependency"
            else SchedulerAction.CONTINUE
        )
        return SchedulerDecision(
            decision_id=f"dependency-decision-{self.count}",
            thread_id=state.thread_id,
            action=action,
            purpose=state.purpose,
            width=1,
            projection_revision=state.revision,
        )


class RuntimeCrossThreadBatchingTests(unittest.TestCase):
    def test_three_threads_execute_through_one_worker_bank_call(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with SQLiteResearchLedger(Path(directory) / "ledger.sqlite") as ledger:
                bank = _RecordingBank()
                loop = RuntimeControlLoop(
                    ledger=ledger,
                    scheduler=TracingScheduler(ledger, _FirstCandidateScheduler()),
                    worker_bank=bank,
                    worker_ids=("worker-a", "worker-b", "worker-c"),
                )
                for thread_id in ("thread-a", "thread-b", "thread-c"):
                    loop.create_thread(
                        thread_id=thread_id,
                        objective=f"Work on {thread_id}",
                    )

                batch = loop.run_many(
                    signal_provider=lambda _state: SchedulerSignals(
                        importance=1.0,
                        recent_progress=1.0,
                    ),
                    context_provider=lambda state, _decision: WorkPreparation(
                        scope_region_ids=(f"region-{state.thread_id}",)
                    ),
                    max_threads=3,
                    max_attempts=3,
                )

                self.assertEqual(len(batch.steps), 3)
                self.assertEqual(batch.neural_attempt_count, 3)
                self.assertEqual(len(batch.results), 3)
                self.assertEqual(len(bank.calls), 1)
                self.assertEqual(
                    tuple(thread_id for thread_id, _worker in bank.calls[0]),
                    ("thread-a", "thread-b", "thread-c"),
                )
                self.assertEqual(
                    tuple(step.state.thread_id for step in batch.steps),
                    ("thread-a", "thread-b", "thread-c"),
                )
                self.assertTrue(all(len(step.assignments) == 1 for step in batch.steps))
                self.assertTrue(all(len(step.results) == 1 for step in batch.steps))

                outcomes = AllocationOutcomeProjector().project(ledger.read_all_events())
                self.assertEqual(len(outcomes), 3)
                self.assertTrue(all(outcome.attempt_count == 1 for outcome in outcomes))
                self.assertEqual(
                    {outcome.thread_id for outcome in outcomes},
                    {"thread-a", "thread-b", "thread-c"},
                )

    def test_total_neural_attempt_budget_caps_sum_of_per_thread_widths(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with SQLiteResearchLedger(Path(directory) / "ledger.sqlite") as ledger:
                bank = _RecordingBank()
                loop = RuntimeControlLoop(
                    ledger=ledger,
                    scheduler=TracingScheduler(ledger, _WidthTwoScheduler()),
                    worker_bank=bank,
                    worker_ids=("worker-a", "worker-b", "worker-c", "worker-d"),
                )
                for thread_id in ("thread-a", "thread-b", "thread-c"):
                    loop.create_thread(thread_id=thread_id, objective="work")

                batch = loop.run_many(
                    signal_provider=lambda _state: SchedulerSignals(
                        importance=1.0,
                        recent_progress=1.0,
                    ),
                    context_provider=lambda state, _decision: WorkPreparation(
                        scope_region_ids=(f"region-{state.thread_id}",)
                    ),
                    max_threads=3,
                    max_attempts=3,
                )

                self.assertEqual(batch.neural_attempt_count, 3)
                self.assertEqual(len(bank.calls), 1)
                self.assertEqual(len(bank.calls[0]), 3)
                self.assertEqual(len(batch.steps), 2)
                self.assertEqual(len(batch.steps[0].assignments), 2)
                self.assertEqual(len(batch.steps[1].assignments), 1)
                self.assertEqual(
                    tuple(step.state.thread_id for step in batch.steps),
                    ("thread-a", "thread-b"),
                )

    def test_dependency_unlock_waits_until_next_snapshot(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with SQLiteResearchLedger(Path(directory) / "ledger.sqlite") as ledger:
                bank = _RecordingBank()
                loop = RuntimeControlLoop(
                    ledger=ledger,
                    scheduler=TracingScheduler(ledger, _CompleteDependencyScheduler()),
                    worker_bank=bank,
                    worker_ids=("worker-a", "worker-b"),
                )
                loop.create_thread(thread_id="dependency", objective="finish prerequisite")
                loop.create_thread(thread_id="dependent", objective="wait for prerequisite")
                loop.create_thread(thread_id="unrelated", objective="independent work")
                loop.add_dependency("dependent", "dependency")

                first = loop.run_many(
                    signal_provider=lambda _state: SchedulerSignals(
                        importance=1.0,
                        recent_progress=1.0,
                    ),
                    context_provider=lambda state, _decision: WorkPreparation(
                        scope_region_ids=(f"region-{state.thread_id}",)
                    ),
                    max_threads=2,
                    max_attempts=2,
                )
                self.assertEqual(
                    tuple(step.state.thread_id for step in first.steps),
                    ("dependency", "unrelated"),
                )
                self.assertEqual(first.steps[0].decision.action, SchedulerAction.COMPLETE)
                self.assertEqual(first.neural_attempt_count, 1)
                self.assertEqual(len(bank.calls), 1)
                self.assertEqual(bank.calls[0][0][0], "unrelated")

                second = loop.run_many(
                    signal_provider=lambda _state: SchedulerSignals(
                        importance=1.0,
                        recent_progress=1.0,
                    ),
                    context_provider=lambda state, _decision: WorkPreparation(
                        scope_region_ids=(f"region-{state.thread_id}",)
                    ),
                    max_threads=1,
                    max_attempts=1,
                )
                self.assertEqual(len(second.steps), 1)
                self.assertEqual(second.steps[0].state.thread_id, "dependent")
                self.assertEqual(len(bank.calls), 2)
                self.assertEqual(bank.calls[1][0][0], "dependent")

    def test_run_once_retains_single_decision_surface(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with SQLiteResearchLedger(Path(directory) / "ledger.sqlite") as ledger:
                bank = _RecordingBank()
                loop = RuntimeControlLoop(
                    ledger=ledger,
                    scheduler=TracingScheduler(ledger, _FirstCandidateScheduler()),
                    worker_bank=bank,
                    worker_ids=("worker-a",),
                )
                loop.create_thread(thread_id="thread-a", objective="work")
                loop.create_thread(thread_id="thread-b", objective="other work")
                step = loop.run_once(
                    signal_provider=lambda _state: SchedulerSignals(
                        importance=1.0,
                        recent_progress=1.0,
                    ),
                    context_provider=lambda _state, _decision: WorkPreparation(),
                )
                self.assertEqual(step.state.thread_id, "thread-a")
                self.assertEqual(len(step.assignments), 1)
                self.assertEqual(len(step.results), 1)
                self.assertEqual(len(bank.calls), 1)
                self.assertEqual(len(bank.calls[0]), 1)


if __name__ == "__main__":
    unittest.main()
