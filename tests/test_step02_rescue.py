"""Tests for Step 2 minority-rescue candidate and gate utilities."""

from __future__ import annotations

import unittest

import torch

from ai_hypothesis.step01.model import LABEL_TO_INDEX
from ai_hypothesis.step01.schema import TaskFamily
from ai_hypothesis.step02.evidence import (
    AggregationConfig,
    aggregate_evidence,
    build_evidence_matrix,
)
from ai_hypothesis.step02.population import PopulationOutput
from ai_hypothesis.step02.rescue import (
    MINORITY_RESCUE_FEATURE_NAMES,
    apply_rescue_gate,
    build_minority_candidates,
    fit_rescue_gate,
    score_rescue_gate,
    select_rescue_threshold,
)


class Step02RescueTests(unittest.TestCase):
    def test_candidate_builder_keeps_strong_non_primary_signal(self) -> None:
        signal = LABEL_TO_INDEX["SIGNAL"]
        no_signal = LABEL_TO_INDEX["NO_SIGNAL"]
        logits = torch.full((4, 1, 11), -10.0)
        for worker_index in range(3):
            logits[worker_index, 0, signal] = 2.0
            logits[worker_index, 0, no_signal] = 0.0
        logits[3, 0, signal] = 0.0
        logits[3, 0, no_signal] = 5.0
        output = PopulationOutput(
            label_logits=logits,
            uncertainty_logits=torch.full((4, 1), -10.0),
        )
        config = AggregationConfig(strong_evidence_threshold=1.0)
        evidence = build_evidence_matrix(output, (TaskFamily.PATTERN,), config)
        summary, decision = aggregate_evidence(evidence, config)
        candidates = build_minority_candidates(evidence, summary, decision)

        self.assertEqual(int(decision.primary_label_indices[0]), signal)
        self.assertTrue(bool(candidates.candidate_exists[0]))
        self.assertEqual(int(candidates.candidate_label_indices[0]), no_signal)
        self.assertEqual(
            tuple(candidates.features.shape),
            (1, len(MINORITY_RESCUE_FEATURE_NAMES)),
        )

    def test_apply_rescue_gate_switches_only_accepted_candidates(self) -> None:
        primary = torch.tensor([0, 0, 0])
        candidate = torch.tensor([1, 1, 1])
        exists = torch.tensor([True, True, False])
        scores = torch.tensor([0.9, 0.2, 0.99])

        result = apply_rescue_gate(
            primary,
            candidate,
            exists,
            scores,
            threshold=0.5,
        )

        self.assertEqual(result.tolist(), [1, 0, 0])

    def test_threshold_selection_can_rescue_without_harm(self) -> None:
        scores = torch.tensor([0.90, 0.80, 0.20])
        exists = torch.tensor([True, True, True])
        primary_correct = torch.tensor([False, True, True])
        candidate_correct = torch.tensor([True, False, False])

        selection = select_rescue_threshold(
            scores,
            exists,
            primary_correct,
            candidate_correct,
            max_harm_rate=0.0,
            thresholds=(0.0, 0.5, 0.8, 0.95, 1.0),
        )
        selected = selection["selected"]

        self.assertEqual(selected["gain_count"], 1)
        self.assertEqual(selected["harm_count"], 0)
        self.assertGreater(selected["accuracy_delta"], 0.0)

    def test_fitted_gate_separates_simple_development_signal(self) -> None:
        feature_count = len(MINORITY_RESCUE_FEATURE_NAMES)
        features = torch.zeros((8, feature_count), dtype=torch.float32)
        features[:4, 0] = 2.0
        features[4:, 0] = -2.0
        targets = torch.tensor([1, 1, 1, 1, 0, 0, 0, 0], dtype=torch.float32)
        exists = torch.ones(8, dtype=torch.bool)

        config = fit_rescue_gate(
            features,
            targets,
            exists,
            steps=200,
            learning_rate=0.05,
        )
        scores = score_rescue_gate(features, exists, config)

        self.assertGreater(float(scores[:4].mean()), float(scores[4:].mean()))


if __name__ == "__main__":
    unittest.main()
