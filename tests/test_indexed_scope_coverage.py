from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from ai_hypothesis.large_scope.evaluate import ScopeWorkerMode
from ai_hypothesis.large_scope.indexed_coverage_planner import IndexedCoverageAwareScopePlanner
from ai_hypothesis.large_scope.relevance import (
    LargeScopeRelevanceConfig,
    generate_large_scope_relevance,
    inspection_order,
)
from ai_hypothesis.large_scope.runtime_bridge import large_scope_region_id
from ai_hypothesis.runtime.ledger import SQLiteResearchLedger
from ai_hypothesis.runtime.scope_coverage import ScopeCoverageProjector
from ai_hypothesis.runtime.scope_coverage_index import SQLiteIndexedScopeCoverage


class NoFullReplayLedger(SQLiteResearchLedger):
    def read_all_events(self, *args, **kwargs):
        raise AssertionError("indexed scope coverage must not call read_all_events")


def append_attempt(
    ledger: SQLiteResearchLedger,
    *,
    attempt_id: str,
    thread_id: str,
    worker_id: str,
    region_ids: tuple[str, ...],
    terminal_event_type: str,
) -> tuple[int, int]:
    started = ledger.append_event(
        event_type="ATTEMPT_STARTED",
        thread_id=thread_id,
        attempt_id=attempt_id,
        payload={
            "worker_id": worker_id,
            "scope_region_ids": list(region_ids),
        },
    )
    terminal = ledger.append_event(
        event_type=terminal_event_type,
        thread_id=thread_id,
        attempt_id=attempt_id,
    )
    return started.sequence, terminal.sequence


class IndexedScopeCoverageTests(unittest.TestCase):
    def test_index_matches_replay_semantics_for_mixed_attempt_history(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with SQLiteResearchLedger(root / "ledger.sqlite") as ledger:
                append_attempt(
                    ledger,
                    attempt_id="a1",
                    thread_id="thread-a",
                    worker_id="worker-1",
                    region_ids=("r1", "r2"),
                    terminal_event_type="ATTEMPT_COMPLETED",
                )
                append_attempt(
                    ledger,
                    attempt_id="a2",
                    thread_id="thread-a",
                    worker_id="worker-2",
                    region_ids=("r1",),
                    terminal_event_type="ATTEMPT_PARTIAL",
                )
                append_attempt(
                    ledger,
                    attempt_id="a3",
                    thread_id="thread-a",
                    worker_id="worker-1",
                    region_ids=("r3",),
                    terminal_event_type="ATTEMPT_FAILED",
                )
                append_attempt(
                    ledger,
                    attempt_id="a4",
                    thread_id="thread-a",
                    worker_id="worker-3",
                    region_ids=("r4",),
                    terminal_event_type="ATTEMPT_CRASHED",
                )
                append_attempt(
                    ledger,
                    attempt_id="a5",
                    thread_id="thread-b",
                    worker_id="worker-9",
                    region_ids=("other",),
                    terminal_event_type="ATTEMPT_INVALID_RESULT",
                )
                ledger.append_event(
                    event_type="ATTEMPT_STARTED",
                    thread_id="thread-a",
                    attempt_id="unscoped",
                    payload={"worker_id": "worker-x"},
                )
                ledger.append_event(
                    event_type="ATTEMPT_COMPLETED",
                    thread_id="thread-a",
                    attempt_id="unscoped",
                )

                replay = ScopeCoverageProjector().for_thread(
                    ledger.read_all_events(),
                    "thread-a",
                )
                with SQLiteIndexedScopeCoverage(
                    ledger,
                    root / "scope.sqlite",
                ) as indexed:
                    self.assertEqual(indexed.for_thread("thread-a"), replay)
                    self.assertEqual(indexed.revision, ledger.latest_sequence())

    def test_exact_forward_snapshot_matches_replay_prefix(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with SQLiteResearchLedger(root / "ledger.sqlite") as ledger:
                started_sequence, terminal_sequence = append_attempt(
                    ledger,
                    attempt_id="a1",
                    thread_id="thread-a",
                    worker_id="worker-1",
                    region_ids=("r1",),
                    terminal_event_type="ATTEMPT_COMPLETED",
                )
                history = ledger.read_all_events()
                replay_started = ScopeCoverageProjector().for_thread(
                    tuple(event for event in history if event.sequence <= started_sequence),
                    "thread-a",
                )
                replay_terminal = ScopeCoverageProjector().for_thread(
                    tuple(event for event in history if event.sequence <= terminal_sequence),
                    "thread-a",
                )
                with SQLiteIndexedScopeCoverage(
                    ledger,
                    root / "scope.sqlite",
                ) as indexed:
                    self.assertEqual(
                        indexed.for_thread("thread-a", sequence=started_sequence),
                        replay_started,
                    )
                    self.assertEqual(
                        indexed.for_thread("thread-a", sequence=terminal_sequence),
                        replay_terminal,
                    )
                    with self.assertRaisesRegex(ValueError, "ahead"):
                        indexed.for_thread("thread-a", sequence=started_sequence)

    def test_persistent_index_resumes_from_checkpoint_and_rebuilds(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            ledger = SQLiteResearchLedger(root / "ledger.sqlite")
            try:
                append_attempt(
                    ledger,
                    attempt_id="a1",
                    thread_id="thread-a",
                    worker_id="worker-1",
                    region_ids=("r1",),
                    terminal_event_type="ATTEMPT_COMPLETED",
                )
                index_path = root / "scope.sqlite"
                with SQLiteIndexedScopeCoverage(ledger, index_path) as first:
                    first_snapshot = first.for_thread("thread-a")
                    first_revision = first.revision

                append_attempt(
                    ledger,
                    attempt_id="a2",
                    thread_id="thread-a",
                    worker_id="worker-2",
                    region_ids=("r2",),
                    terminal_event_type="ATTEMPT_FAILED",
                )
                with SQLiteIndexedScopeCoverage(ledger, index_path) as resumed:
                    self.assertEqual(resumed.revision, first_revision)
                    resumed_snapshot = resumed.for_thread("thread-a")
                    replay = ScopeCoverageProjector().for_thread(
                        ledger.read_all_events(),
                        "thread-a",
                    )
                    self.assertNotEqual(resumed_snapshot, first_snapshot)
                    self.assertEqual(resumed_snapshot, replay)
                    self.assertEqual(resumed.rebuild(page_size=1), ledger.latest_sequence())
                    self.assertEqual(resumed.for_thread("thread-a"), replay)
            finally:
                ledger.close()

    def test_indexed_planner_advances_scope_without_full_history_replay(self) -> None:
        sample = generate_large_scope_relevance(
            52,
            LargeScopeRelevanceConfig(window_count=8),
        )
        order = inspection_order(
            sample.seed,
            sample.config.window_count,
            split=sample.split,
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with NoFullReplayLedger(root / "ledger.sqlite") as ledger:
                with SQLiteIndexedScopeCoverage(
                    ledger,
                    root / "scope.sqlite",
                ) as coverage:
                    planner = IndexedCoverageAwareScopePlanner(
                        coverage,
                        sample,
                        ScopeWorkerMode.DIVERSE_WORKERS,
                    )
                    self.assertEqual(
                        planner.plan("thread-1", 2).selected_window_indices,
                        order[:2],
                    )
                    for offset, window_index in enumerate(order[:2]):
                        append_attempt(
                            ledger,
                            attempt_id=f"attempt-{offset}",
                            thread_id="thread-1",
                            worker_id=f"worker-{offset}",
                            region_ids=(large_scope_region_id(sample, window_index),),
                            terminal_event_type="ATTEMPT_COMPLETED",
                        )
                    second = planner.plan("thread-1", 2)
                    self.assertEqual(second.selected_window_indices, order[2:4])
                    self.assertEqual(second.resolved_region_count, 2)
                    self.assertEqual(second.missing_region_count, 6)
                    self.assertEqual(second.missing_coverage, 0.75)

    def test_index_rejects_multiple_terminal_events_like_replay_projector(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with SQLiteResearchLedger(root / "ledger.sqlite") as ledger:
                ledger.append_event(
                    event_type="ATTEMPT_STARTED",
                    thread_id="thread-a",
                    attempt_id="a1",
                    payload={
                        "worker_id": "worker-1",
                        "scope_region_ids": ["r1"],
                    },
                )
                ledger.append_event(
                    event_type="ATTEMPT_COMPLETED",
                    thread_id="thread-a",
                    attempt_id="a1",
                )
                ledger.append_event(
                    event_type="ATTEMPT_FAILED",
                    thread_id="thread-a",
                    attempt_id="a1",
                )
                with SQLiteIndexedScopeCoverage(
                    ledger,
                    root / "scope.sqlite",
                ) as indexed:
                    with self.assertRaisesRegex(ValueError, "multiple terminal"):
                        indexed.sync()


if __name__ == "__main__":
    unittest.main()
