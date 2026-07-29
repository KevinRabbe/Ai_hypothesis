from __future__ import annotations

import math
import unittest

import torch

from ai_hypothesis.population_compute.gate3_development import (
    GATE3_DEVELOPMENT_BOOTSTRAP_SAMPLES,
    GATE3_DEVELOPMENT_EVAL_BATCH_SIZE,
    GATE3_DEVELOPMENT_EVAL_WORLD_COUNT,
    GATE3_REVEAL_MISMATCH_PENALTY,
    Gate3TrainingConfig,
    _target_score,
    evaluate_gate3_condition,
    gate3_stable_training_conditions,
    train_gate3_development_model,
)
from ai_hypothesis.population_compute.gate3_hypothesis_model import Gate3HypothesisScorer
from ai_hypothesis.population_compute.gate3_hypothesis_population import (
    Gate3ControlMode,
    generate_gate3_world,
)


class Gate3DevelopmentTests(unittest.TestCase):
    def test_frozen_development_defaults(self) -> None:
        config = Gate3TrainingConfig()
        self.assertEqual(config.steps, 1200)
        self.assertEqual(config.batch_size, 128)
        self.assertEqual(config.learning_rate, 3e-4)
        self.assertEqual(config.weight_decay, 1e-4)
        self.assertEqual(config.gradient_clip_norm, 1.0)
        self.assertEqual(GATE3_DEVELOPMENT_EVAL_WORLD_COUNT, 256)
        self.assertEqual(GATE3_DEVELOPMENT_EVAL_BATCH_SIZE, 64)
        self.assertEqual(GATE3_DEVELOPMENT_BOOTSTRAP_SAMPLES, 2000)
        self.assertEqual(GATE3_REVEAL_MISMATCH_PENALTY, 16.0)

    def test_training_cycle_contains_exactly_the_twelve_stable_depth_width_conditions(self) -> None:
        conditions = gate3_stable_training_conditions()
        self.assertEqual(len(conditions), 12)
        self.assertEqual(
            conditions,
            (
                (4, 1), (4, 4), (4, 16),
                (6, 1), (6, 4), (6, 16), (6, 64),
                (8, 1), (8, 4), (8, 16), (8, 64), (8, 256),
            ),
        )

    def test_target_prefers_hint_consistency_before_reveals(self) -> None:
        world = generate_gate3_world(seed=55, depth=4)
        matching = world.noisy_hints
        flipped = tuple(1 - bit for bit in world.noisy_hints)
        self.assertGreater(
            _target_score(world, matching, phase_index=3),
            _target_score(world, flipped, phase_index=3),
        )

    def test_exact_reveal_penalizes_candidate_mismatch(self) -> None:
        world = generate_gate3_world(seed=56, depth=4)
        correct = world.hidden_path
        wrong = (1 - correct[0],) + correct[1:]
        before_reveal_correct = _target_score(world, correct, phase_index=3)
        before_reveal_wrong = _target_score(world, wrong, phase_index=3)
        after_reveal_correct = _target_score(world, correct, phase_index=4)
        after_reveal_wrong = _target_score(world, wrong, phase_index=4)
        self.assertTrue(math.isfinite(before_reveal_correct))
        self.assertTrue(math.isfinite(before_reveal_wrong))
        self.assertGreater(after_reveal_correct, after_reveal_wrong)
        self.assertLess(after_reveal_wrong - before_reveal_wrong, after_reveal_correct - before_reveal_correct)

    def test_one_step_training_smoke_is_finite_and_keeps_fixed_parameter_count(self) -> None:
        model, summary = train_gate3_development_model(
            training_seed=0,
            config=Gate3TrainingConfig(steps=1, batch_size=2),
            device="cpu",
        )
        self.assertIsInstance(model, Gate3HypothesisScorer)
        self.assertEqual(summary.steps, 1)
        self.assertEqual(summary.examples_seen, 2)
        self.assertTrue(math.isfinite(summary.initial_loss))
        self.assertTrue(math.isfinite(summary.final_loss))
        self.assertEqual(summary.learned_parameter_count, 19_873)
        self.assertEqual(len(summary.parameter_fingerprint), 64)

    def test_small_development_evaluation_preserves_w1_control_identity(self) -> None:
        torch.manual_seed(999)
        model = Gate3HypothesisScorer()
        stable = evaluate_gate3_condition(
            model,
            depth=4,
            width=1,
            mode=Gate3ControlMode.STABLE_DIVERSE,
            world_count=4,
            evaluation_batch_size=2,
            device="cpu",
        )
        collapsed = evaluate_gate3_condition(
            model,
            depth=4,
            width=1,
            mode=Gate3ControlMode.COLLAPSED_DIVERSITY,
            world_count=4,
            evaluation_batch_size=2,
            device="cpu",
        )
        reshuffled = evaluate_gate3_condition(
            model,
            depth=4,
            width=1,
            mode=Gate3ControlMode.RESHUFFLED_CONTINUITY,
            world_count=4,
            evaluation_batch_size=2,
            device="cpu",
        )
        self.assertEqual(stable.solved_by_world, collapsed.solved_by_world)
        self.assertEqual(stable.solved_by_world, reshuffled.solved_by_world)
        self.assertEqual(stable.bit_accuracy_by_world, collapsed.bit_accuracy_by_world)
        self.assertEqual(stable.bit_accuracy_by_world, reshuffled.bit_accuracy_by_world)
        self.assertEqual(stable.learned_updates_per_world, 128)
        self.assertEqual(stable.unique_world_observations_per_world, 8)
        self.assertEqual(stable.parameter_fingerprint, collapsed.parameter_fingerprint)
        self.assertEqual(stable.parameter_fingerprint, reshuffled.parameter_fingerprint)


if __name__ == "__main__":
    unittest.main()
