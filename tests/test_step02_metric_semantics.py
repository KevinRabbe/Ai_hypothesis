"""Metric-semantics tests for Step 2 population diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
import unittest

import torch

from ai_hypothesis.step01.model import LABEL_TO_INDEX, UnitConfig
from ai_hypothesis.step01.schema import TaskFamily
from ai_hypothesis.step02.evaluation import evaluate_population
from ai_hypothesis.step02.population import PopulationOutput


@dataclass(frozen=True, slots=True)
class _Sample:
    label: str
    task: TaskFamily


class _StaticBank:
    def __init__(self, output: PopulationOutput) -> None:
        self.output = output
        self.population_width = output.label_logits.shape[0]
        self.execution_backend = "static-test"
        self.unit_config = UnitConfig(
            d_model=32,
            block_count=1,
            attention_heads=4,
            feed_forward_width=64,
            dropout=0.0,
        )

    def __call__(self, features: torch.Tensor, mask: torch.Tensor) -> PopulationOutput:
        return self.output


def _loader(sample: _Sample) -> list[dict]:
    return [
        {
            "features": torch.zeros((1, 32, 16), dtype=torch.float32),
            "mask": torch.ones((1, 32), dtype=torch.bool),
            "samples": [sample],
        }
    ]


class Step02MetricSemanticsTests(unittest.TestCase):
    def test_w1_rescue_opportunity_zero_denominator_reports_zero_rate(self) -> None:
        signal = LABEL_TO_INDEX["SIGNAL"]
        logits = torch.full((1, 1, 11), -8.0)
        logits[0, 0, signal] = 8.0
        output = PopulationOutput(
            label_logits=logits,
            uncertainty_logits=torch.full((1, 1), -8.0),
        )

        metrics = evaluate_population(
            _StaticBank(output),
            _loader(_Sample(label="SIGNAL", task=TaskFamily.PATTERN)),
        )

        self.assertEqual(metrics["population_width"], 1)
        self.assertEqual(metrics["minority_rescue_opportunity_rate"], 0.0)
        self.assertEqual(metrics["minority_rescue_rate"], 0.0)
        self.assertEqual(metrics["minority_suppression_rate"], 0.0)

    def test_oracle_gap_can_include_reducer_success_without_decoded_worker(self) -> None:
        signal = LABEL_TO_INDEX["SIGNAL"]
        no_signal = LABEL_TO_INDEX["NO_SIGNAL"]
        invalid = LABEL_TO_INDEX["CHANGE"]
        logits = torch.full((1, 1, 11), -8.0)
        logits[0, 0, invalid] = 4.01
        logits[0, 0, signal] = 4.0
        logits[0, 0, no_signal] = 3.0
        output = PopulationOutput(
            label_logits=logits,
            uncertainty_logits=torch.full((1, 1), -8.0),
        )

        metrics = evaluate_population(
            _StaticBank(output),
            _loader(_Sample(label="SIGNAL", task=TaskFamily.PATTERN)),
        )

        self.assertEqual(metrics["oracle_any_correct_coverage"], 0.0)
        self.assertEqual(metrics["evidence_reducer_accuracy"], 1.0)
        self.assertEqual(metrics["evidence_utilization_gap"], -1.0)


if __name__ == "__main__":
    unittest.main()
