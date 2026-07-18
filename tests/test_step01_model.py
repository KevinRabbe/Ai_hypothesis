"""Tests for the Step 1 neural unit and PyTorch data path."""

from __future__ import annotations

import unittest

import torch

from ai_hypothesis.step01.generator import generate_dataset, generate_sample
from ai_hypothesis.step01.model import (
    REFERENCE_10M_CONFIG,
    Step01Output,
    Step01Unit,
    UnitConfig,
    decode_predictions,
)
from ai_hypothesis.step01.schema import Difficulty, TaskFamily
from ai_hypothesis.step01.torch_data import (
    Step01TorchDataset,
    collate_samples,
)
from ai_hypothesis.step01.training import compute_loss


class Step01ModelTests(unittest.TestCase):
    def test_reference_configuration_is_about_10m_parameters(self) -> None:
        model = Step01Unit(REFERENCE_10M_CONFIG)
        count = model.trainable_parameter_count()
        self.assertGreaterEqual(count, 9_500_000)
        self.assertLessEqual(count, 10_500_000)

    def test_small_configuration_forward_shapes(self) -> None:
        config = UnitConfig(
            d_model=32,
            block_count=2,
            attention_heads=4,
            feed_forward_width=64,
            dropout=0.0,
        )
        model = Step01Unit(config)
        features = torch.randn(3, 32, 16)
        mask = torch.ones(3, 32, dtype=torch.bool)
        output = model(features, mask)
        self.assertEqual(tuple(output.label_logits.shape), (3, 11))
        self.assertEqual(tuple(output.uncertainty_logits.shape), (3,))

    def test_decode_predictions_respects_uncertainty_head(self) -> None:
        label_logits = torch.zeros(2, 11)
        label_logits[0, 0] = 10.0
        label_logits[1, 1] = 10.0
        output = Step01Output(
            label_logits=label_logits,
            uncertainty_logits=torch.tensor([10.0, -10.0]),
        )
        self.assertEqual(
            decode_predictions(output),
            ["UNCERTAIN", "NO_SIGNAL"],
        )

    def test_torch_dataset_matches_procedural_stream(self) -> None:
        count = 40
        expected = list(generate_dataset("train", count))
        dataset = Step01TorchDataset("train", count)
        actual = [dataset[index] for index in range(count)]
        self.assertEqual(actual, expected)

    def test_collation_marks_uncertain_label_for_ignore_index(self) -> None:
        answerable = generate_sample(TaskFamily.PATTERN, Difficulty.EASY, seed=2)
        uncertain = generate_sample(TaskFamily.PATTERN, Difficulty.AMBIGUOUS, seed=2)
        batch = collate_samples([answerable, uncertain])
        self.assertEqual(tuple(batch["features"].shape), (2, 32, 16))
        self.assertEqual(tuple(batch["mask"].shape), (2, 32))
        self.assertNotEqual(int(batch["label_targets"][0]), -100)
        self.assertEqual(int(batch["label_targets"][1]), -100)
        self.assertEqual(float(batch["uncertainty_targets"][0]), 0.0)
        self.assertEqual(float(batch["uncertainty_targets"][1]), 1.0)

    def test_one_training_step_produces_finite_gradients(self) -> None:
        config = UnitConfig(
            d_model=32,
            block_count=1,
            attention_heads=4,
            feed_forward_width=64,
            dropout=0.0,
        )
        model = Step01Unit(config)
        samples = [
            generate_sample(TaskFamily.PATTERN, Difficulty.EASY, seed=2),
            generate_sample(TaskFamily.CHANGE, Difficulty.MEDIUM, seed=4),
            generate_sample(TaskFamily.CONFLICT, Difficulty.AMBIGUOUS, seed=6),
            generate_sample(TaskFamily.RELATION, Difficulty.HARD, seed=8),
        ]
        batch = collate_samples(samples)
        output = model(batch["features"], batch["mask"])
        loss, _ = compute_loss(
            output,
            batch["label_targets"],
            batch["uncertainty_targets"],
        )
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        gradients = [
            parameter.grad
            for parameter in model.parameters()
            if parameter.requires_grad and parameter.grad is not None
        ]
        self.assertTrue(gradients)
        self.assertTrue(all(torch.isfinite(gradient).all() for gradient in gradients))


if __name__ == "__main__":
    unittest.main()
