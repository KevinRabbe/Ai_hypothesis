"""Tests for the first Step 2 population runtime slice."""

from __future__ import annotations

import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path

import torch

from ai_hypothesis.step01.generator import generate_sample
from ai_hypothesis.step01.model import LABEL_TO_INDEX, Step01Unit, UnitConfig
from ai_hypothesis.step01.schema import Difficulty, TaskFamily
from ai_hypothesis.step01.torch_data import collate_samples
from ai_hypothesis.step02.evidence import (
    AggregationConfig,
    aggregate_evidence,
    build_evidence_matrix,
)
from ai_hypothesis.step02.population import HomogeneousWorkerBank, PopulationOutput


class Step02PopulationTests(unittest.TestCase):
    def _write_checkpoint(
        self,
        path: Path,
        config: UnitConfig,
        *,
        seed: int,
    ) -> None:
        torch.manual_seed(seed)
        model = Step01Unit(config)
        torch.save(
            {
                "model_state": model.state_dict(),
                "unit_config": asdict(config),
                "step": 1,
                "validation_metrics": {"macro_task_accuracy": 0.0},
            },
            path,
        )

    def test_evidence_matrix_preserves_strong_minority_signal(self) -> None:
        signal = LABEL_TO_INDEX["SIGNAL"]
        no_signal = LABEL_TO_INDEX["NO_SIGNAL"]
        logits = torch.full((3, 1, 11), -10.0)
        logits[0, 0, signal] = 2.0
        logits[0, 0, no_signal] = 1.0
        logits[1, 0, signal] = 2.0
        logits[1, 0, no_signal] = 1.0
        logits[2, 0, signal] = -2.0
        logits[2, 0, no_signal] = 8.0
        output = PopulationOutput(
            label_logits=logits,
            uncertainty_logits=torch.full((3, 1), -10.0),
        )
        config = AggregationConfig(strong_evidence_threshold=1.0)
        evidence = build_evidence_matrix(output, (TaskFamily.PATTERN,), config)
        summary, _ = aggregate_evidence(evidence, config)

        self.assertTrue(bool(summary.protected_label_mask[0, no_signal]))
        self.assertGreater(
            float(summary.max_evidence_per_label[0, no_signal]),
            float(summary.mean_evidence_per_label[0, no_signal]),
        )

    def test_loop_worker_bank_produces_worker_by_batch_outputs(self) -> None:
        config = UnitConfig(
            d_model=32,
            block_count=1,
            attention_heads=4,
            feed_forward_width=64,
            dropout=0.0,
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = [root / "worker_1.pt", root / "worker_2.pt"]
            self._write_checkpoint(paths[0], config, seed=1)
            self._write_checkpoint(paths[1], config, seed=2)

            bank = HomogeneousWorkerBank.from_checkpoints(
                paths,
                device="cpu",
                execution_backend="loop",
            )
            samples = [
                generate_sample(TaskFamily.PATTERN, Difficulty.EASY, seed=2),
                generate_sample(TaskFamily.CHANGE, Difficulty.MEDIUM, seed=4),
            ]
            batch = collate_samples(samples)
            output = bank(batch["features"], batch["mask"])

            self.assertEqual(tuple(output.label_logits.shape), (2, 2, 11))
            self.assertEqual(tuple(output.uncertainty_logits.shape), (2, 2))

    def test_worker_bank_rejects_mixed_architectures(self) -> None:
        config_a = UnitConfig(
            d_model=32,
            block_count=1,
            attention_heads=4,
            feed_forward_width=64,
            dropout=0.0,
        )
        config_b = UnitConfig(
            d_model=32,
            block_count=2,
            attention_heads=4,
            feed_forward_width=64,
            dropout=0.0,
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path_a = root / "a.pt"
            path_b = root / "b.pt"
            self._write_checkpoint(path_a, config_a, seed=1)
            self._write_checkpoint(path_b, config_b, seed=2)

            with self.assertRaisesRegex(ValueError, "mixed worker architectures"):
                HomogeneousWorkerBank.from_checkpoints(
                    [path_a, path_b],
                    device="cpu",
                    execution_backend="loop",
                )


if __name__ == "__main__":
    unittest.main()
