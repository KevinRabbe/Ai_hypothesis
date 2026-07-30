from __future__ import annotations

import inspect
import subprocess
import sys
import unittest

import torch

from ai_hypothesis.population_compute.gate7_scale_neutral_model_prep import (
    encode_gate7_scale_neutral_child_input,
)
from ai_hypothesis.population_compute.gate7_scale_neutral_transition_training import (
    GATE7_SCALE_NEUTRAL_GRADIENT_CLIP_NORM,
    GATE7_SCALE_NEUTRAL_LEARNING_RATE,
    GATE7_SCALE_NEUTRAL_TRAINING_BATCH_SIZE,
    GATE7_SCALE_NEUTRAL_TRAINING_DEPTHS,
    GATE7_SCALE_NEUTRAL_TRAINING_SEEDS,
    GATE7_SCALE_NEUTRAL_TRAINING_STEPS,
    GATE7_SCALE_NEUTRAL_WEIGHT_DECAY,
    build_gate7_scale_neutral_training_inputs,
    gate7_scale_neutral_training_batch,
)


class Gate7ScaleNeutralTransitionTrainingTests(unittest.TestCase):
    def test_frozen_training_constants(self) -> None:
        self.assertEqual(GATE7_SCALE_NEUTRAL_TRAINING_SEEDS, (0, 1, 2))
        self.assertEqual(GATE7_SCALE_NEUTRAL_TRAINING_DEPTHS, tuple(range(6, 19)))
        self.assertEqual(GATE7_SCALE_NEUTRAL_TRAINING_STEPS, 1_200)
        self.assertEqual(GATE7_SCALE_NEUTRAL_TRAINING_BATCH_SIZE, 256)
        self.assertEqual(GATE7_SCALE_NEUTRAL_LEARNING_RATE, 3e-4)
        self.assertEqual(GATE7_SCALE_NEUTRAL_WEIGHT_DECAY, 1e-4)
        self.assertEqual(GATE7_SCALE_NEUTRAL_GRADIENT_CLIP_NORM, 1.0)

    def test_training_batch_is_deterministic_binary_and_namespace_bound(self) -> None:
        first_hints, first_actions = gate7_scale_neutral_training_batch(
            training_seed=1,
            step=73,
            depth=14,
            batch_size=32,
        )
        second_hints, second_actions = gate7_scale_neutral_training_batch(
            training_seed=1,
            step=73,
            depth=14,
            batch_size=32,
        )
        torch.testing.assert_close(first_hints, second_hints, rtol=0.0, atol=0.0)
        torch.testing.assert_close(first_actions, second_actions, rtol=0.0, atol=0.0)
        self.assertEqual(tuple(first_hints.shape), (32, 14))
        self.assertEqual(tuple(first_actions.shape), (32, 14))
        self.assertTrue(bool(((first_hints == 0) | (first_hints == 1)).all()))
        self.assertTrue(bool(((first_actions == 0) | (first_actions == 1)).all()))

        other_hints, other_actions = gate7_scale_neutral_training_batch(
            training_seed=2,
            step=73,
            depth=14,
            batch_size=32,
        )
        self.assertFalse(torch.equal(first_hints, other_hints) and torch.equal(first_actions, other_actions))

    def test_vectorized_training_encoder_matches_frozen_scalar_encoder(self) -> None:
        hints = torch.tensor([0, 1, 1, 0], dtype=torch.int64)
        actions = torch.tensor([1, 0, 1, 0], dtype=torch.int64)
        vectorized = build_gate7_scale_neutral_training_inputs(
            world_depth=18,
            child_depth=11,
            hints=hints,
            actions=actions,
            device="cpu",
        )
        scalar = torch.stack(
            [
                encode_gate7_scale_neutral_child_input(
                    world_depth=18,
                    child_depth=11,
                    observed_hint=int(hint),
                    branch_action=int(action),
                    sink=False,
                    device="cpu",
                )
                for hint, action in zip(hints.tolist(), actions.tolist(), strict=True)
            ]
        )
        torch.testing.assert_close(vectorized, scalar, rtol=1e-6, atol=1e-7)

    def test_training_input_builder_has_no_cuda_to_python_value_extraction(self) -> None:
        source = inspect.getsource(build_gate7_scale_neutral_training_inputs)
        for forbidden in (".item(", ".cpu(", ".tolist(", "float(", "bool("):
            self.assertNotIn(forbidden, source)

    def test_training_input_builder_rejects_invalid_public_depths(self) -> None:
        hints = torch.tensor([0, 1], dtype=torch.int64)
        actions = torch.tensor([1, 0], dtype=torch.int64)
        with self.assertRaises(ValueError):
            build_gate7_scale_neutral_training_inputs(
                world_depth=19,
                child_depth=1,
                hints=hints,
                actions=actions,
                device="cpu",
            )
        with self.assertRaises(ValueError):
            build_gate7_scale_neutral_training_inputs(
                world_depth=18,
                child_depth=19,
                hints=hints,
                actions=actions,
                device="cpu",
            )

    def test_admitted_training_cli_has_no_scientific_tuning_knobs(self) -> None:
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "ai_hypothesis.population_compute.run_gate7_scale_neutral_transition_training",
                "--help",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        help_text = result.stdout
        self.assertIn("--output-root", help_text)
        for forbidden in (
            "--seed",
            "--depth",
            "--steps",
            "--batch-size",
            "--learning-rate",
            "--k",
            "--population",
            "--bridge",
        ):
            self.assertNotIn(forbidden, help_text)


if __name__ == "__main__":
    unittest.main()
