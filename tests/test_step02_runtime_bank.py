"""Tests for heterogeneous-work execution through the Step 2 runtime adapter."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch
from torch.func import stack_module_state

from ai_hypothesis.runtime import (
    AttemptRequest,
    SQLiteResearchLedger,
    ThreadStateProjector,
    WorkerAssignment,
    WorkerRuntime,
    WorkItem,
    WorkPurpose,
)
from ai_hypothesis.step01.model import LABEL_TO_INDEX, Step01Output, Step01Unit, UnitConfig
from ai_hypothesis.step01.schema import FEATURE_WIDTH, SEQUENCE_LENGTH, TaskFamily
from ai_hypothesis.step02.population import HomogeneousWorkerBank, LoadedCheckpoint
from ai_hypothesis.step02.runtime_bank import Step02RuntimeWorkerBank


class _FakeSelectedBank:
    population_width = 2
    checkpoints = (
        LoadedCheckpoint("checkpoint-alpha.pt", 10, 0.71),
        LoadedCheckpoint("checkpoint-beta.pt", 20, 0.73),
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
        for row, worker_index in enumerate(worker_indices):
            label = "SIGNAL" if worker_index == 0 else "NO_SIGNAL"
            logits[row, LABEL_TO_INDEX[label]] = 5.0
        return Step01Output(
            label_logits=logits,
            uncertainty_logits=torch.full((len(worker_indices),), -4.0),
        )


def _item(work_id: str, thread_id: str) -> WorkItem:
    return WorkItem(
        work_item_id=work_id,
        thread_id=thread_id,
        objective="Inspect local pattern evidence",
        purpose=WorkPurpose.EXPLORE,
        projection_revision=1,
        reference_ids=(f"source:{thread_id}",),
        context={
            "features": torch.zeros((SEQUENCE_LENGTH, FEATURE_WIDTH)),
            "mask": torch.ones(SEQUENCE_LENGTH, dtype=torch.bool),
            "task": TaskFamily.PATTERN,
        },
    )


class Step02RuntimeWorkerBankTests(unittest.TestCase):
    def test_forward_selected_matches_direct_worker_execution(self) -> None:
        config = UnitConfig(
            d_model=8,
            block_count=1,
            attention_heads=1,
            feed_forward_width=16,
            dropout=0.0,
        )
        torch.manual_seed(11)
        workers = [Step01Unit(config).eval(), Step01Unit(config).eval()]
        params, buffers = stack_module_state(workers)
        bank = HomogeneousWorkerBank(
            template=workers[0],
            params=params,
            buffers=buffers,
            checkpoints=(
                LoadedCheckpoint("worker-0", None, None),
                LoadedCheckpoint("worker-1", None, None),
            ),
            device=torch.device("cpu"),
            execution_backend="vmap",
        )
        features = torch.randn((3, SEQUENCE_LENGTH, FEATURE_WIDTH))
        mask = torch.ones((3, SEQUENCE_LENGTH), dtype=torch.bool)
        indices = [1, 0, 1]

        actual = bank.forward_selected(indices, features, mask)
        expected = [
            workers[worker_index](
                features[sample_index : sample_index + 1],
                mask[sample_index : sample_index + 1],
            )
            for sample_index, worker_index in enumerate(indices)
        ]
        expected_logits = torch.cat([output.label_logits for output in expected], dim=0)
        expected_uncertainty = torch.cat(
            [output.uncertainty_logits for output in expected], dim=0
        )

        torch.testing.assert_close(actual.label_logits, expected_logits)
        torch.testing.assert_close(actual.uncertainty_logits, expected_uncertainty)

    def test_forward_selected_rejects_wrong_architecture_shape(self) -> None:
        config = UnitConfig(
            d_model=8,
            block_count=1,
            attention_heads=1,
            feed_forward_width=16,
            dropout=0.0,
        )
        workers = [Step01Unit(config).eval()]
        params, buffers = stack_module_state(workers)
        bank = HomogeneousWorkerBank(
            template=workers[0],
            params=params,
            buffers=buffers,
            checkpoints=(LoadedCheckpoint("worker-0", None, None),),
            device=torch.device("cpu"),
            execution_backend="loop",
        )

        with self.assertRaisesRegex(ValueError, "do not match"):
            bank.forward_selected(
                [0],
                torch.zeros((1, SEQUENCE_LENGTH, FEATURE_WIDTH - 1)),
                torch.ones((1, SEQUENCE_LENGTH), dtype=torch.bool),
            )

    def test_adapter_emits_structured_local_evidence_without_voting(self) -> None:
        adapter = Step02RuntimeWorkerBank(
            _FakeSelectedBank(),  # type: ignore[arg-type]
            worker_ids=("alpha", "beta"),
        )
        requests = (
            AttemptRequest("attempt-a", "alpha", _item("work-a", "thread-a")),
            AttemptRequest("attempt-b", "beta", _item("work-b", "thread-b")),
        )

        results = adapter.execute_batch(requests)

        self.assertEqual(results[0].evidence[0].data["top_label"], "SIGNAL")
        self.assertEqual(results[1].evidence[0].data["top_label"], "NO_SIGNAL")
        self.assertGreater(results[0].evidence[0].strength or 0.0, 0.0)
        self.assertLess(results[0].evidence[0].uncertainty or 1.0, 0.1)
        self.assertEqual(results[0].evidence[0].reference_ids, ("source:thread-a",))
        self.assertEqual(results[0].evidence[0].data["worker_index"], 0)
        self.assertEqual(
            results[1].evidence[0].data["checkpoint_path"],
            "checkpoint-beta.pt",
        )
        self.assertEqual(
            results[1].evidence[0].data["checkpoint_step"],
            20,
        )
        self.assertEqual(
            results[0].resource_usage["selected_execution_backend"],
            "fake-grouped",
        )
        self.assertEqual(
            results[0].resource_usage["population_execution_backend"],
            "fake-population",
        )
        self.assertEqual(len(results[0].evidence[0].data["label_logits"]), 11)

    def test_runtime_persists_generated_evidence_as_addressable_state(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with SQLiteResearchLedger(Path(directory) / "ledger.sqlite") as ledger:
                for thread_id in ("thread-a", "thread-b"):
                    ledger.append_event(
                        event_type="THREAD_CREATED",
                        thread_id=thread_id,
                        payload={
                            "objective": "Inspect local pattern evidence",
                            "purpose": "EXPLORE",
                        },
                    )

                adapter = Step02RuntimeWorkerBank(
                    _FakeSelectedBank(),  # type: ignore[arg-type]
                    worker_ids=("alpha", "beta"),
                )
                runtime = WorkerRuntime(ledger)
                results = runtime.run_batch(
                    (
                        WorkerAssignment("alpha", _item("work-a", "thread-a")),
                        WorkerAssignment("beta", _item("work-b", "thread-b")),
                    ),
                    adapter,
                )

                evidence_id = results[0].evidence[0].evidence_id
                evidence_events = [
                    event
                    for event in ledger.read_events(thread_id="thread-a")
                    if event.event_type == "EVIDENCE_ADDED"
                ]
                self.assertEqual(len(evidence_events), 1)
                self.assertEqual(evidence_events[0].payload["evidence_id"], evidence_id)
                self.assertEqual(evidence_events[0].payload["data"]["top_label"], "SIGNAL")
                self.assertEqual(
                    evidence_events[0].payload["data"]["checkpoint_path"],
                    "checkpoint-alpha.pt",
                )

                state = ThreadStateProjector().project(
                    ledger.read_events(), thread_id="thread-a"
                )
                self.assertIn(evidence_id, state.reference_ids)


if __name__ == "__main__":
    unittest.main()
