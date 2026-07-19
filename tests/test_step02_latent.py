"""Tests for Step 2 analysis-only latent extraction."""

from __future__ import annotations

import copy
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path

import torch

from ai_hypothesis.step01.generator import generate_sample
from ai_hypothesis.step01.model import Step01Unit, UnitConfig
from ai_hypothesis.step01.schema import Difficulty, TaskFamily
from ai_hypothesis.step01.torch_data import collate_samples
from ai_hypothesis.step02.latent import (
    extract_pooled_latent,
    summarize_worker_local_scalar_scores,
)
from ai_hypothesis.step02.population import HomogeneousWorkerBank


class Step02LatentExtractionTests(unittest.TestCase):
    def _model_and_batch(self) -> tuple[Step01Unit, dict[str, object]]:
        torch.manual_seed(123)
        model = Step01Unit(
            UnitConfig(
                d_model=32,
                block_count=1,
                attention_heads=4,
                feed_forward_width=64,
                dropout=0.0,
            )
        )
        model.eval()
        samples = [
            generate_sample(TaskFamily.PATTERN, Difficulty.EASY, seed=2),
            generate_sample(TaskFamily.CHANGE, Difficulty.MEDIUM, seed=4),
        ]
        return model, collate_samples(samples)

    def test_forward_outputs_remain_unchanged(self) -> None:
        model, batch = self._model_and_batch()

        with torch.inference_mode():
            before = model(batch["features"], batch["mask"])
            _ = extract_pooled_latent(model, batch["features"], batch["mask"])
            after = model(batch["features"], batch["mask"])

        self.assertTrue(torch.equal(before.label_logits, after.label_logits))
        self.assertTrue(torch.equal(before.uncertainty_logits, after.uncertainty_logits))

    def test_extraction_is_deterministic_and_preserves_model_state(self) -> None:
        model, batch = self._model_and_batch()
        state_before = copy.deepcopy(model.state_dict())

        with torch.inference_mode():
            first = extract_pooled_latent(model, batch["features"], batch["mask"])
            second = extract_pooled_latent(model, batch["features"], batch["mask"])

        self.assertTrue(torch.equal(first, second))
        self.assertEqual(tuple(first.shape), (2, model.config.d_model))
        for name, tensor in model.state_dict().items():
            self.assertTrue(torch.equal(tensor, state_before[name]))

    def test_checkpoint_loads_unchanged_with_latent_extractor_available(self) -> None:
        model, _ = self._model_and_batch()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "worker.pt"
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "unit_config": asdict(model.config),
                    "step": 1,
                    "validation_metrics": {"macro_task_accuracy": 0.0},
                },
                path,
            )

            bank = HomogeneousWorkerBank.from_checkpoints(
                [path],
                device="cpu",
                execution_backend="loop",
            )

        self.assertEqual(bank.population_width, 1)
        self.assertEqual(bank.unit_config, model.config)

    def test_population_summaries_reject_raw_latent_vectors(self) -> None:
        scalar_scores = torch.tensor([[0.1, 0.8], [0.3, 0.6]])
        summaries = summarize_worker_local_scalar_scores(
            scalar_scores,
            torch.tensor([0.5, 0.4]),
        )

        self.assertTrue(torch.allclose(summaries["mean_probability"], torch.tensor([0.2, 0.7])))
        with self.assertRaisesRegex(ValueError, "scalar outputs"):
            summarize_worker_local_scalar_scores(
                torch.zeros(2, 3, 4),
                torch.tensor([0.5, 0.5]),
            )


if __name__ == "__main__":
    unittest.main()
