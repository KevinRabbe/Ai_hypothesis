"""Tests durable source-region coverage derived from worker attempt history."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from ai_hypothesis.runtime.contracts import AttemptResult, AttemptStatus, LedgerEvent, WorkItem, WorkPurpose
from ai_hypothesis.runtime.ledger import SQLiteResearchLedger
from ai_hypothesis.runtime.scope_coverage import ScopeCoverageProjector
from ai_hypothesis.runtime.worker_runtime import WorkerAssignment, WorkerRuntime


class _CompletedBank:
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


class _CrashBank:
    def execute_batch(self, requests):
        raise RuntimeError("simulated crash")


def _event(
    sequence: int,
    event_type: str,
    *,
    attempt_id: str | None = None,
    thread_id: str | None = "thread-1",
    payload: dict | None = None,
) -> LedgerEvent:
    return LedgerEvent(
        event_id=f"event-{sequence}",
        event_type=event_type,
        sequence=sequence,
        payload_schema="v1",
        thread_id=thread_id,
        attempt_id=attempt_id,
        payload=payload or {},
    )


class ScopeCoverageTests(unittest.TestCase):
    def test_worker_runtime_persists_scope_regions_and_completed_coverage(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with SQLiteResearchLedger(Path(directory) / "ledger.sqlite") as ledger:
                runtime = WorkerRuntime(ledger)
                runtime.run_attempt(
                    WorkerAssignment(
                        worker_id="worker-a",
                        work_item=WorkItem(
                            work_item_id="work-1",
                            thread_id="thread-1",
                            objective="Inspect two source regions",
                            purpose=WorkPurpose.EXPLORE,
                            projection_revision=3,
                            scope_region_ids=("region-a", "region-b"),
                        ),
                    ),
                    _CompletedBank(),
                )

                events = ledger.read_all_events()
                started = next(event for event in events if event.event_type == "ATTEMPT_STARTED")
                self.assertEqual(started.payload["scope_region_ids"], ["region-a", "region-b"])

                coverage = ScopeCoverageProjector().for_thread(events, "thread-1")
                self.assertEqual(coverage.attempted_region_ids, ("region-a", "region-b"))
                self.assertEqual(coverage.resolved_region_ids, ("region-a", "region-b"))
                self.assertEqual(coverage.coverage_fraction(("region-a", "region-b", "region-c")), 2 / 3)
                self.assertEqual(
                    coverage.missing_region_ids(("region-a", "region-b", "region-c")),
                    ("region-c",),
                )

    def test_crashed_attempt_is_attempted_but_not_resolved(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with SQLiteResearchLedger(Path(directory) / "ledger.sqlite") as ledger:
                runtime = WorkerRuntime(ledger)
                with self.assertRaisesRegex(RuntimeError, "simulated crash"):
                    runtime.run_attempt(
                        WorkerAssignment(
                            worker_id="worker-a",
                            work_item=WorkItem(
                                work_item_id="work-1",
                                thread_id="thread-1",
                                objective="Inspect region",
                                purpose=WorkPurpose.EXPLORE,
                                projection_revision=0,
                                scope_region_ids=("region-a",),
                            ),
                        ),
                        _CrashBank(),
                    )

                coverage = ScopeCoverageProjector().for_thread(
                    ledger.read_all_events(), "thread-1"
                )
                region = coverage.regions[0]
                self.assertEqual(region.started_attempt_count, 1)
                self.assertEqual(region.crashed_attempt_count, 1)
                self.assertEqual(region.resolved_attempt_count, 0)
                self.assertEqual(coverage.attempted_region_ids, ("region-a",))
                self.assertEqual(coverage.resolved_region_ids, ())
                self.assertEqual(
                    coverage.missing_region_ids(("region-a",), require_resolved=False),
                    (),
                )
                self.assertEqual(coverage.missing_region_ids(("region-a",)), ("region-a",))

    def test_failed_valid_result_counts_as_resolved_inspection(self) -> None:
        events = (
            _event(
                1,
                "ATTEMPT_STARTED",
                attempt_id="attempt-1",
                payload={
                    "worker_id": "worker-a",
                    "scope_region_ids": ["region-a"],
                },
            ),
            _event(2, "ATTEMPT_FAILED", attempt_id="attempt-1"),
        )
        coverage = ScopeCoverageProjector().for_thread(events, "thread-1")
        region = coverage.regions[0]
        self.assertEqual(region.failed_attempt_count, 1)
        self.assertTrue(region.has_resolved_inspection)
        self.assertEqual(coverage.resolved_region_ids, ("region-a",))

    def test_repeated_workers_preserve_redundancy_without_duplicate_regions(self) -> None:
        events = (
            _event(
                1,
                "ATTEMPT_STARTED",
                attempt_id="attempt-1",
                payload={
                    "worker_id": "worker-a",
                    "scope_region_ids": ["region-a"],
                },
            ),
            _event(2, "ATTEMPT_COMPLETED", attempt_id="attempt-1"),
            _event(
                3,
                "ATTEMPT_STARTED",
                attempt_id="attempt-2",
                payload={
                    "worker_id": "worker-b",
                    "scope_region_ids": ["region-a"],
                },
            ),
            _event(4, "ATTEMPT_PARTIAL", attempt_id="attempt-2"),
        )
        coverage = ScopeCoverageProjector().for_thread(events, "thread-1")
        self.assertEqual(len(coverage.regions), 1)
        region = coverage.regions[0]
        self.assertEqual(region.started_attempt_count, 2)
        self.assertEqual(region.completed_attempt_count, 1)
        self.assertEqual(region.partial_attempt_count, 1)
        self.assertEqual(region.distinct_worker_count, 2)
        self.assertEqual(region.worker_ids, ("worker-a", "worker-b"))

    def test_unscoped_legacy_attempts_are_ignored(self) -> None:
        events = (
            _event(
                1,
                "ATTEMPT_STARTED",
                attempt_id="attempt-1",
                payload={"worker_id": "worker-a"},
            ),
            _event(2, "ATTEMPT_COMPLETED", attempt_id="attempt-1"),
        )
        self.assertEqual(ScopeCoverageProjector().project(events), ())

    def test_duplicate_regions_in_one_work_item_are_rejected_before_execution(self) -> None:
        item = WorkItem(
            work_item_id="work-1",
            thread_id="thread-1",
            objective="Inspect",
            purpose=WorkPurpose.EXPLORE,
            projection_revision=0,
            scope_region_ids=("region-a", "region-a"),
        )
        with self.assertRaisesRegex(ValueError, "scope_region_ids must be unique"):
            item.validate()

    def test_multiple_terminal_events_for_scoped_attempt_are_rejected(self) -> None:
        events = (
            _event(
                1,
                "ATTEMPT_STARTED",
                attempt_id="attempt-1",
                payload={
                    "worker_id": "worker-a",
                    "scope_region_ids": ["region-a"],
                },
            ),
            _event(2, "ATTEMPT_COMPLETED", attempt_id="attempt-1"),
            _event(3, "ATTEMPT_FAILED", attempt_id="attempt-1"),
        )
        with self.assertRaisesRegex(ValueError, "multiple terminal events"):
            ScopeCoverageProjector().project(events)


if __name__ == "__main__":
    unittest.main()
