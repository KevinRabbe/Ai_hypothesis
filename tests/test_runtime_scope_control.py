"""Tests width plus scope composition through RuntimeControlLoop."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from ai_hypothesis.runtime import (
    AttemptResult,
    AttemptStatus,
    RuntimeControlLoop,
    SchedulerAction,
    SchedulerDecision,
    SchedulerSignals,
    ScopeCoverageProjector,
    SQLiteResearchLedger,
    WorkPreparation,
    WorkPreparationBatch,
    WorkPurpose,
)


class _WidthTwoScheduler:
    def choose(self, candidates, *, integration_backpressure=False, max_width=1):
        if max_width < 2:
            raise ValueError("test requires two workers")
        state = candidates[0].state
        return SchedulerDecision(
            decision_id="decision-width-two",
            thread_id=state.thread_id,
            action=SchedulerAction.ADD_WIDTH,
            purpose=WorkPurpose.EXPLORE,
            width=2,
            reason_codes=("TEST_SCOPE",),
            projection_revision=state.revision,
        )


class _CompletedBank:
    def __init__(self) -> None:
        self.calls = 0

    def execute_batch(self, requests):
        self.calls += 1
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


class RuntimeScopeControlTests(unittest.TestCase):
    def test_width_two_can_assign_distinct_regions_to_distinct_workers(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with SQLiteResearchLedger(Path(directory) / "ledger.sqlite") as ledger:
                bank = _CompletedBank()
                loop = RuntimeControlLoop(
                    ledger=ledger,
                    scheduler=_WidthTwoScheduler(),
                    worker_bank=bank,
                    worker_ids=("worker-a", "worker-b"),
                )
                loop.create_thread(thread_id="thread-1", objective="Inspect large source")

                step = loop.run_once(
                    signal_provider=lambda _state: SchedulerSignals(importance=1.0),
                    context_provider=lambda _state, _decision: WorkPreparationBatch(
                        items=(
                            WorkPreparation(scope_region_ids=("region-a",)),
                            WorkPreparation(scope_region_ids=("region-b",)),
                        )
                    ),
                )

                self.assertEqual(bank.calls, 1)
                self.assertEqual(len(step.assignments), 2)
                self.assertEqual(
                    tuple(a.work_item.scope_region_ids for a in step.assignments),
                    (("region-a",), ("region-b",)),
                )
                coverage = ScopeCoverageProjector().for_thread(
                    ledger.read_all_events(), "thread-1"
                )
                self.assertEqual(coverage.resolved_region_ids, ("region-a", "region-b"))

    def test_single_preparation_still_means_replication_across_width(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with SQLiteResearchLedger(Path(directory) / "ledger.sqlite") as ledger:
                loop = RuntimeControlLoop(
                    ledger=ledger,
                    scheduler=_WidthTwoScheduler(),
                    worker_bank=_CompletedBank(),
                    worker_ids=("worker-a", "worker-b"),
                )
                loop.create_thread(thread_id="thread-1", objective="Verify one region")

                step = loop.run_once(
                    signal_provider=lambda _state: SchedulerSignals(importance=1.0),
                    context_provider=lambda _state, _decision: WorkPreparation(
                        scope_region_ids=("region-a",)
                    ),
                )

                self.assertEqual(
                    tuple(a.work_item.scope_region_ids for a in step.assignments),
                    (("region-a",), ("region-a",)),
                )
                coverage = ScopeCoverageProjector().for_thread(
                    ledger.read_all_events(), "thread-1"
                )
                self.assertEqual(len(coverage.regions), 1)
                region = coverage.regions[0]
                self.assertEqual(region.started_attempt_count, 2)
                self.assertEqual(region.resolved_attempt_count, 2)
                self.assertEqual(region.distinct_worker_count, 2)

    def test_preparation_batch_must_match_scheduler_width_before_execution(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with SQLiteResearchLedger(Path(directory) / "ledger.sqlite") as ledger:
                bank = _CompletedBank()
                loop = RuntimeControlLoop(
                    ledger=ledger,
                    scheduler=_WidthTwoScheduler(),
                    worker_bank=bank,
                    worker_ids=("worker-a", "worker-b"),
                )
                loop.create_thread(thread_id="thread-1", objective="Inspect")

                with self.assertRaisesRegex(ValueError, "size must match scheduler width"):
                    loop.run_once(
                        signal_provider=lambda _state: SchedulerSignals(importance=1.0),
                        context_provider=lambda _state, _decision: WorkPreparationBatch(
                            items=(WorkPreparation(scope_region_ids=("region-a",)),)
                        ),
                    )
                self.assertEqual(bank.calls, 0)


if __name__ == "__main__":
    unittest.main()
