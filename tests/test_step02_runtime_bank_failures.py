"""Failure-isolation tests for the Step 2 runtime Worker Bank."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch

from ai_hypothesis.runtime import (
    AttemptStatus,
    SQLiteResearchLedger,
    WorkerAssignment,
    WorkerRuntime,
    WorkItem,
    WorkPurpose,
)
from ai_hypothesis.step01.model import LABEL_TO_INDEX, Step01Output
from ai_hypothesis.step01.schema import FEATURE_WIDTH, SEQUENCE_LENGTH, TaskFamily
from ai_hypothesis.step02.population import LoadedCheckpoint
from ai_hypothesis.step02.runtime_bank import Step02RuntimeWorkerBank


class _NumericalFailureBank:
    population_width = 2
    checkpoints = (
        LoadedCheckpoint("healthy.pt", 1, 0.7),
        LoadedCheckpoint("nonfinite.pt", 2, 0.7),
    )
    execution_backend = "fake-population"
    selected_execution_backend = "fake-grouped"
    device = torch.device("cpu")

    def forward_selected(
        self,
        worker_indices: list[int],
        features: torch.Tensor,
        mask: torch.Tensor,
    ) -> Step01Output:
        logits = torch.full((len(worker_indices), 11), -5.0)
        uncertainty = torch.full((len(worker_indices),), -4.0)
        for row, worker_index in enumerate(worker_indices):
            if worker_index == 1:
                logits[row, 0] = float("nan")
            else:
                logits[row, LABEL_TO_INDEX["SIGNAL"]] = 5.0
        return Step01Output(
            label_logits=logits,
            uncertainty_logits=uncertainty,
        )


def _item(
    work_item_id: str,
    thread_id: str,
    *,
    feature_width: int = FEATURE_WIDTH,
) -> WorkItem:
    return WorkItem(
        work_item_id=work_item_id,
        thread_id=thread_id,
        objective="Inspect bounded Step 2 evidence",
        purpose=WorkPurpose.EXPLORE,
        projection_revision=1,
        reference_ids=(f"source:{thread_id}",),
        context={
            "features": torch.zeros((SEQUENCE_LENGTH, feature_width)),
            "mask": torch.ones(SEQUENCE_LENGTH, dtype=torch.bool),
            "task": TaskFamily.PATTERN,
        },
    )


def _create_threads(ledger: SQLiteResearchLedger) -> None:
    for thread_id in ("healthy", "failed"):
        ledger.append_event(
            event_type="THREAD_CREATED",
            thread_id=thread_id,
            payload={
                "objective": "Inspect bounded Step 2 evidence",
                "purpose": "EXPLORE",
            },
        )


class Step02RuntimeWorkerBankFailureTests(unittest.TestCase):
    def test_malformed_context_fails_only_its_attempt(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with SQLiteResearchLedger(Path(directory) / "ledger.sqlite") as ledger:
                _create_threads(ledger)
                adapter = Step02RuntimeWorkerBank(
                    _NumericalFailureBank(),  # type: ignore[arg-type]
                    worker_ids=("healthy-worker", "other-worker"),
                )
                results = WorkerRuntime(ledger).run_batch(
                    (
                        WorkerAssignment(
                            "healthy-worker",
                            _item("healthy-work", "healthy"),
                        ),
                        WorkerAssignment(
                            "other-worker",
                            _item(
                                "failed-work",
                                "failed",
                                feature_width=FEATURE_WIDTH - 1,
                            ),
                        ),
                    ),
                    adapter,
                )

                self.assertEqual(results[0].status, AttemptStatus.COMPLETED)
                self.assertEqual(results[1].status, AttemptStatus.FAILED)
                self.assertEqual(
                    results[1].resource_usage["failure_code"],
                    "STEP02_WORK_ITEM_INVALID",
                )
                self.assertEqual(
                    ledger.read_events(thread_id="healthy")[-1].event_type,
                    "ATTEMPT_COMPLETED",
                )
                self.assertEqual(
                    ledger.read_events(thread_id="failed")[-1].event_type,
                    "ATTEMPT_FAILED",
                )

    def test_nonfinite_worker_output_fails_only_its_attempt(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with SQLiteResearchLedger(Path(directory) / "ledger.sqlite") as ledger:
                _create_threads(ledger)
                adapter = Step02RuntimeWorkerBank(
                    _NumericalFailureBank(),  # type: ignore[arg-type]
                    worker_ids=("healthy-worker", "nonfinite-worker"),
                )
                results = WorkerRuntime(ledger).run_batch(
                    (
                        WorkerAssignment(
                            "healthy-worker",
                            _item("healthy-work", "healthy"),
                        ),
                        WorkerAssignment(
                            "nonfinite-worker",
                            _item("failed-work", "failed"),
                        ),
                    ),
                    adapter,
                )

                self.assertEqual(results[0].status, AttemptStatus.COMPLETED)
                self.assertEqual(results[1].status, AttemptStatus.FAILED)
                self.assertEqual(
                    results[1].resource_usage["failure_code"],
                    "STEP02_WORKER_OUTPUT_INVALID",
                )
                healthy_types = tuple(
                    event.event_type
                    for event in ledger.read_events(thread_id="healthy")
                )
                failed_types = tuple(
                    event.event_type
                    for event in ledger.read_events(thread_id="failed")
                )
                self.assertIn("EVIDENCE_ADDED", healthy_types)
                self.assertEqual(healthy_types[-1], "ATTEMPT_COMPLETED")
                self.assertNotIn("EVIDENCE_ADDED", failed_types)
                self.assertEqual(failed_types[-1], "ATTEMPT_FAILED")


if __name__ == "__main__":
    unittest.main()
