"""Tests for bounded worker execution and durable attempt lifecycle."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from ai_hypothesis.runtime import (
    AttemptResult,
    AttemptStatus,
    SQLiteResearchLedger,
    ThreadStateProjector,
    WorkerAssignment,
    WorkerRuntime,
    WorkItem,
    WorkPurpose,
)


def make_item(thread_id: str = "thread-1") -> WorkItem:
    return WorkItem(
        work_item_id="work-1",
        thread_id=thread_id,
        objective="Investigate H2",
        purpose=WorkPurpose.EXPLORE,
        projection_revision=1,
        reference_ids=("source-1",),
        resource_budget={"max_steps": 1},
    )


class WorkerRuntimeTests(unittest.TestCase):
    def test_successful_attempt_commits_structured_information(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "ledger.sqlite"
            with SQLiteResearchLedger(path) as ledger:
                ledger.append_event(
                    event_type="THREAD_CREATED",
                    thread_id="thread-1",
                    payload={"objective": "Investigate H2", "purpose": "EXPLORE"},
                )
                runtime = WorkerRuntime(ledger)
                assignment = WorkerAssignment(worker_id="worker-7", work_item=make_item())

                def executor(attempt_id: str, worker_id: str, item: WorkItem) -> AttemptResult:
                    return AttemptResult(
                        attempt_id=attempt_id,
                        work_item_id=item.work_item_id,
                        thread_id=item.thread_id,
                        worker_id=worker_id,
                        status=AttemptStatus.COMPLETED,
                        observations=("Observed local anomaly",),
                        evidence_refs=("evidence-9",),
                        hypotheses_proposed=("H2",),
                        contradictions=("evidence-10",),
                        possibilities_eliminated=("H3",),
                        open_questions=("Does H2 generalize?",),
                        progress_made=True,
                        resource_usage={"steps": 1},
                    )

                result = runtime.run_attempt(assignment, executor)
                events = ledger.read_events()
                event_types = tuple(event.event_type for event in events)

                self.assertEqual(result.status, AttemptStatus.COMPLETED)
                self.assertIn("ATTEMPT_STARTED", event_types)
                self.assertIn("EVIDENCE_ADDED", event_types)
                self.assertIn("HYPOTHESIS_PROPOSED", event_types)
                self.assertIn("CONTRADICTION_FOUND", event_types)
                self.assertIn("POSSIBILITY_ELIMINATED", event_types)
                self.assertIn("OPEN_QUESTION_ADDED", event_types)
                self.assertEqual(event_types[-1], "ATTEMPT_COMPLETED")

                state = ThreadStateProjector().project(events, thread_id="thread-1")
                self.assertEqual(state.hypothesis_ids, ("H2",))
                self.assertEqual(state.contradiction_ids, ("evidence-10",))
                self.assertEqual(state.open_questions, ("Does H2 generalize?",))

    def test_executor_crash_is_durable_and_re_raised(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with SQLiteResearchLedger(Path(directory) / "ledger.sqlite") as ledger:
                runtime = WorkerRuntime(ledger)
                assignment = WorkerAssignment(worker_id="worker-1", work_item=make_item())

                def executor(attempt_id: str, worker_id: str, item: WorkItem) -> AttemptResult:
                    raise RuntimeError("boom")

                with self.assertRaisesRegex(RuntimeError, "boom"):
                    runtime.run_attempt(assignment, executor)

                event_types = tuple(event.event_type for event in ledger.read_events())
                self.assertEqual(event_types, ("ATTEMPT_STARTED", "ATTEMPT_CRASHED"))

    def test_invalid_executor_identity_is_recorded(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with SQLiteResearchLedger(Path(directory) / "ledger.sqlite") as ledger:
                runtime = WorkerRuntime(ledger)
                assignment = WorkerAssignment(worker_id="worker-1", work_item=make_item())

                def executor(attempt_id: str, worker_id: str, item: WorkItem) -> AttemptResult:
                    return AttemptResult(
                        attempt_id="wrong-attempt",
                        work_item_id=item.work_item_id,
                        thread_id=item.thread_id,
                        worker_id=worker_id,
                        status=AttemptStatus.COMPLETED,
                    )

                with self.assertRaisesRegex(ValueError, "mismatched attempt_id"):
                    runtime.run_attempt(assignment, executor)

                event_types = tuple(event.event_type for event in ledger.read_events())
                self.assertEqual(event_types, ("ATTEMPT_STARTED", "ATTEMPT_INVALID_RESULT"))

    def test_partial_attempt_remains_distinct(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with SQLiteResearchLedger(Path(directory) / "ledger.sqlite") as ledger:
                runtime = WorkerRuntime(ledger)
                assignment = WorkerAssignment(worker_id="worker-1", work_item=make_item())

                def executor(attempt_id: str, worker_id: str, item: WorkItem) -> AttemptResult:
                    return AttemptResult(
                        attempt_id=attempt_id,
                        work_item_id=item.work_item_id,
                        thread_id=item.thread_id,
                        worker_id=worker_id,
                        status=AttemptStatus.PARTIAL,
                        open_questions=("Need another pass",),
                    )

                runtime.run_attempt(assignment, executor)
                self.assertEqual(ledger.read_events()[-1].event_type, "ATTEMPT_PARTIAL")

    def test_batch_boundary_can_be_replaced_by_vectorized_execution_later(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with SQLiteResearchLedger(Path(directory) / "ledger.sqlite") as ledger:
                runtime = WorkerRuntime(ledger)
                assignments = (
                    WorkerAssignment(worker_id="worker-1", work_item=make_item("thread-1")),
                    WorkerAssignment(
                        worker_id="worker-2",
                        work_item=WorkItem(
                            work_item_id="work-2",
                            thread_id="thread-2",
                            objective="Investigate H3",
                            purpose=WorkPurpose.EXPLORE,
                            projection_revision=1,
                        ),
                    ),
                )

                def executor(attempt_id: str, worker_id: str, item: WorkItem) -> AttemptResult:
                    return AttemptResult(
                        attempt_id=attempt_id,
                        work_item_id=item.work_item_id,
                        thread_id=item.thread_id,
                        worker_id=worker_id,
                        status=AttemptStatus.COMPLETED,
                    )

                results = runtime.run_batch(assignments, executor)
                self.assertEqual(len(results), 2)
                self.assertEqual(ledger.latest_sequence(), 4)


if __name__ == "__main__":
    unittest.main()
