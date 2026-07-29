from __future__ import annotations

import math
import unittest

import torch

from ai_hypothesis.population_compute.gate3_v1_development import (
    GATE3_V1_DEVELOPMENT_BOOTSTRAP_SAMPLES,
    GATE3_V1_DEVELOPMENT_EVAL_BATCH_SIZE,
    GATE3_V1_DEVELOPMENT_WORLD_COUNT,
    GATE3_V1_SIGNED_HINT_EVIDENCE,
    Gate3V1TrainingConfig,
    _prefix_targets,
    evaluate_gate3_v1_condition,
    train_gate3_v1_development_model,
)
from ai_hypothesis.population_compute.gate3_v1_model import Gate3V1Scorer
from ai_hypothesis.population_compute.gate3_v1_sparse_active_reserve import Gate3V1ControlMode


class Gate3V1DevelopmentTests(unittest.TestCase):
    def test_frozen_development_defaults(self) -> None:
        config = Gate3V1TrainingConfig()
        self.assertEqual(config.steps, 1200)
        self.assertEqual(config.batch_size, 256)
        self.assertEqual(config.learning_rate, 3e-4)
        self.assertEqual(config.weight_decay, 1e-4)
        self.assertEqual(config.gradient_clip_norm, 1.0)
        self.assertEqual(GATE3_V1_DEVELOPMENT_WORLD_COUNT, 256)
        self.assertEqual(GATE3_V1_DEVELOPMENT_EVAL_BATCH_SIZE, 64)
        self.assertEqual(GATE3_V1_DEVELOPMENT_BOOTSTRAP_SAMPLES, 2000)
        self.assertAlmostEqual(GATE3_V1_SIGNED_HINT_EVIDENCE, math.log(0.7 / 0.3))

    def test_prefix_target_accumulates_signed_hint_evidence(self) -> None:
        targets = _prefix_targets(
            noisy_hints=(1, 0, 1, 0, 1, 0),
            candidate=(1, 1, 1, 0, 0, 0),
            depth=6,
        )
        unit = math.log(0.7 / 0.3) / 6
        self.assertAlmostEqual(targets[0], unit)
        self.assertAlmostEqual(targets[1], 0.0)
        self.assertAlmostEqual(targets[2], unit)
        self.assertAlmostEqual(targets[3], 2 * unit)
        self.assertAlmostEqual(targets[4], unit)
        self.assertAlmostEqual(targets[5], 2 * unit)

    def test_one_step_training_is_finite_and_fixed_parameter(self) -> None:
        model, summary = train_gate3_v1_development_model(
            training_seed=0,
            config=Gate3V1TrainingConfig(steps=1, batch_size=4),
            device="cpu",
        )
        self.assertIsInstance(model, Gate3V1Scorer)
        self.assertEqual(summary.steps, 1)
        self.assertEqual(summary.examples_seen, 4)
        self.assertTrue(math.isfinite(summary.initial_loss))
        self.assertTrue(math.isfinite(summary.final_loss))
        self.assertEqual(summary.learned_parameter_count, 19_649)
        self.assertEqual(len(summary.parameter_fingerprint), 64)

    def test_small_l1_evaluation_is_exact_control_identity(self) -> None:
        torch.manual_seed(88)
        model = Gate3V1Scorer()
        rows = {
            mode: evaluate_gate3_v1_condition(
                model,
                depth=6,
                reserve_capacity=1,
                mode=mode,
                world_count=4,
                evaluation_batch_size=2,
                device="cpu",
            )
            for mode in Gate3V1ControlMode
        }
        stable = rows[Gate3V1ControlMode.STABLE_RESERVE]
        for mode, row in rows.items():
            self.assertEqual(row.covered_by_world, stable.covered_by_world, mode)
            self.assertEqual(row.generated_terminal_count_by_world, stable.generated_terminal_count_by_world, mode)
            self.assertEqual(row.productive_rounds_by_world, stable.productive_rounds_by_world, mode)
            self.assertEqual(row.sink_rounds_by_world, stable.sink_rounds_by_world, mode)
            self.assertEqual(row.total_learned_updates_per_world, 256)
            self.assertEqual(row.parameter_fingerprint, stable.parameter_fingerprint)


if __name__ == "__main__":
    unittest.main()
