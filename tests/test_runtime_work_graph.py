"""Tests append-only Work Graph relationships and dependency-aware scheduling."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from ai_hypothesis.runtime import (
    AttemptResult,
    AttemptStatus,
    RuntimeControlLoop,
    SchedulerConfig,
    SchedulerSignals,
    SchedulerV0,
    SQLiteResearchLedger,
    ThreadStateProjector,
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


def _loop(ledger: SQLiteResearchLedger) -> RuntimeControlLoop:
    return RuntimeControlLoop(
        ledger=ledger,
        scheduler=SchedulerV0(SchedulerConfig(exploration_probability=0.0)),
        worker_bank=_NoopBank(),
        worker_ids=("worker-a", "worker-b"),
    )


class WorkGraphTests(unittest.TestCase):
    def test_fork_projects_parent_and_child_relationships(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with SQLiteResearchLedger(Path(directory) / "ledger.sqlite") as ledger:
                loop = _loop(ledger)
                loop.create_thread(thread_id="parent", objective="Investigate root")
                child = loop.fork_thread(
                    "parent",
                    child_thread_id="child",
                    objective="Test alternative branch",
                )

                states = {
                    state.thread_id: state
                    for state in ThreadStateProjector().project_all(
                        ledger.read_all_events()
                    )
                }

                self.assertEqual(child, "child")
                self.assertEqual(states["parent"].child_thread_ids, ("child",))
                self.assertEqual(states["child"].parent_thread_ids, ("parent",))
                self.assertEqual(states["parent"].status, "ACTIVE")
                self.assertEqual(states["child"].status, "ACTIVE")

    def test_dependency_blocks_high_priority_thread_until_prerequisite_completes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with SQLiteResearchLedger(Path(directory) / "ledger.sqlite") as ledger:
                loop = _loop(ledger)
                loop.create_thread(
                    thread_id="prerequisite",
                    objective="Resolve prerequisite",
                )
                loop.create_thread(
                    thread_id="dependent",
                    objective="High-value dependent work",
                )
                loop.add_dependency("dependent", "prerequisite")

                signals = lambda state: SchedulerSignals(
                    importance=1.0 if state.thread_id == "dependent" else 0.1,
                    recent_progress=1.0,
                )
                prepare = lambda _state, _decision: WorkPreparation()

                first = loop.run_once(
                    signal_provider=signals,
                    context_provider=prepare,
                )
                self.assertEqual(first.decision.thread_id, "prerequisite")

                ledger.append_event(
                    event_type="THREAD_COMPLETED",
                    thread_id="prerequisite",
                )
                second = loop.run_once(
                    signal_provider=signals,
                    context_provider=prepare,
                )
                self.assertEqual(second.decision.thread_id, "dependent")

    def test_dependency_cycle_is_rejected_before_event_is_written(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with SQLiteResearchLedger(Path(directory) / "ledger.sqlite") as ledger:
                loop = _loop(ledger)
                for thread_id in ("a", "b", "c"):
                    loop.create_thread(thread_id=thread_id, objective=thread_id)
                loop.add_dependency("a", "b")
                loop.add_dependency("b", "c")
                before = ledger.latest_sequence()

                with self.assertRaisesRegex(ValueError, "create a cycle"):
                    loop.add_dependency("c", "a")

                self.assertEqual(ledger.latest_sequence(), before)

    def test_merge_closes_sources_and_projects_reverse_relationship(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with SQLiteResearchLedger(Path(directory) / "ledger.sqlite") as ledger:
                loop = _loop(ledger)
                for thread_id in ("source-a", "source-b", "target"):
                    loop.create_thread(thread_id=thread_id, objective=thread_id)

                loop.merge_threads(
                    ("source-a", "source-b"),
                    target_thread_id="target",
                )
                states = {
                    state.thread_id: state
                    for state in ThreadStateProjector().project_all(
                        ledger.read_all_events()
                    )
                }

                self.assertEqual(states["source-a"].status, "COMPLETE")
                self.assertEqual(states["source-b"].status, "COMPLETE")
                self.assertEqual(
                    states["source-a"].merged_into_thread_id,
                    "target",
                )
                self.assertEqual(
                    states["source-b"].merged_into_thread_id,
                    "target",
                )
                self.assertEqual(
                    states["target"].merged_from_thread_ids,
                    ("source-a", "source-b"),
                )
                self.assertEqual(states["target"].status, "ACTIVE")

    def test_projector_rejects_missing_dependency_target_from_raw_history(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with SQLiteResearchLedger(Path(directory) / "ledger.sqlite") as ledger:
                ledger.append_event(
                    event_type="THREAD_CREATED",
                    thread_id="a",
                    payload={"objective": "a", "purpose": "EXPLORE"},
                )
                ledger.append_event(
                    event_type="DEPENDENCY_ADDED",
                    thread_id="a",
                    reference_ids=("missing",),
                )

                with self.assertRaisesRegex(ValueError, "missing dependency"):
                    ThreadStateProjector().project_all(ledger.read_all_events())


if __name__ == "__main__":
    unittest.main()
