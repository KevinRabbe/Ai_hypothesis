"""Tests for Step 2 abstention-layer utilities."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch

from ai_hypothesis.step02.abstention import (
    AbstentionConfig,
    apply_abstention,
    standardize_features,
    validate_inference_feature_names,
)


class Step02AbstentionTests(unittest.TestCase):
    def test_abstention_only_accepts_candidate_or_uncertain(self) -> None:
        labels = ["SIGNAL", "NO_SIGNAL", "CHANGE"]
        scores = torch.tensor([0.1, 0.7, 0.5])

        self.assertEqual(
            apply_abstention(labels, scores, threshold=0.5),
            ["SIGNAL", "UNCERTAIN", "UNCERTAIN"],
        )

    def test_abstention_rejects_shape_mismatch(self) -> None:
        with self.assertRaisesRegex(ValueError, "length must match"):
            apply_abstention(["SIGNAL"], torch.tensor([0.1, 0.2]), threshold=0.5)

    def test_standardize_features_is_deterministic(self) -> None:
        features = torch.tensor([[1.0, 2.0], [3.0, 6.0]])
        mean = torch.tensor([1.0, 2.0])
        std = torch.tensor([2.0, 4.0])

        first = standardize_features(features, mean, std)
        second = standardize_features(features, mean, std)
        self.assertTrue(torch.equal(first, second))
        self.assertTrue(torch.equal(first, torch.tensor([[0.0, 0.0], [1.0, 1.0]])))

    def test_config_serialization_round_trip(self) -> None:
        config = AbstentionConfig(
            version="step2a5-abstention-diagnostic-v0",
            candidate_source="mean_probability_label",
            feature_names=("mean_uncertainty", "candidate_margin"),
            model_type="logistic_regression",
            threshold=0.5,
            weights=(1.0, -1.0),
            bias=0.1,
            feature_mean=(0.2, 0.3),
            feature_std=(0.4, 0.5),
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            config.save(path)
            self.assertEqual(AbstentionConfig.load(path), config)

    def test_feature_names_reject_oracle_leakage(self) -> None:
        with self.assertRaisesRegex(ValueError, "leak"):
            validate_inference_feature_names(("oracle_any_correct",))
        validate_inference_feature_names(("mean_uncertainty", "candidate_margin"))


if __name__ == "__main__":
    unittest.main()
