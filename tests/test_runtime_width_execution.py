"""Tests that scheduler width decisions create real independent worker attempts."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from ai_hypothesis.runtime import (
    AttemptResult,
    AttemptStatus,
    ControlConfig,
    RuntimeControlLoop,
    SchedulerAction,
    SchedulerConfig,
    SchedulerSignals,
    SchedulerV0,
    SQLiteResearchLedger,
    WorkPreparation,
)


class _RecordingBank:
    def __init__(self) -> None:
        self.batches: list[tuple[str, ...]] = []

    def execute_batch(self, requests):
        self.batches.append(tuple(request.worker_id for request in requests))
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


class WidthExecutionTests(unittest.TestCase):
    def test_add_width_creates_distinct_workers_and_work_items(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with SQLiteResearchLedger(Path(directory) / "ledger.sqlite") as ledger:
                bank = _RecordingBank()
                loop = RuntimeControlLoop(
                    ledger=ledger,
                    scheduler=SchedulerV0(
                        SchedulerConfig(exploration_probability=1.0)
                    ),
                    worker_bank=bank,
                    worker_ids=("worker-a", "worker-b", "worker-c"),
                    config=ControlConfig(add_width_count=2),
                )
                loop.create_thread(
                    thread_id="thread-1",
                    objective="Explore independent alternatives",
                )

                step = loop.run_once(
                    signal_provider=lambda _state: SchedulerSignals(
                        importance=1.0,
                        missing_coverage=1.0,
                        novelty=1.0,
                        recent_progress=1.0,
                    ),
                    context_provider=lambda _state, _decision: WorkPreparation(),
                )

                self.assertEqual(step.decision.action, SchedulerAction.ADD_WIDTH)
                self.assertEqual(len(step.assignments), 2)
                self.assertEqual(len(step.results), 2)
                self.assertEqual(len(step.decision.work_item_ids), 2)
                self.assertEqual(
                    step.decision.work_item_ids,
                    tuple(
                        assignment.work_item.work_item_id
                        for assignment in step.assignments
                    ),
                )
                self.assertEqual(len({a.worker_id for a in step.assignments}), 2)
                self.assertEqual(bank.batches, [("worker-a", "worker-b")])

    def test_width_is_capped_by_available_distinct_workers(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with SQLiteResearchLedger(Path(directory) / "ledger.sqlite") as ledger:
                bank = _RecordingBank()
                loop = RuntimeControlLoop(
                    ledger=ledger,
                    scheduler=SchedulerV0(
                        SchedulerConfig(exploration_probability=1.0)
                    ),
                    worker_bank=bank,
                    worker_ids=("worker-a", "worker-b"),
                    config=ControlConfig(add_width_count=10),
                )
                loop.create_thread(thread_id="thread-1", objective="Explore")

                step = loop.run_once(
                    signal_provider=lambda _state: SchedulerSignals(
                        importance=1.0,
                        missing_coverage=1.0,
                        recent_progress=1.0,
                    ),
                    context_provider=lambda _state, _decision: WorkPreparation(),
                )

                self.assertEqual(len(step.assignments), 2)
                self.assertEqual(len({a.worker_id for a in step.assignments}), 2)


if __name__ == "__main__":
    unittest.main()
