from __future__ import annotations

import inspect
import unittest

import torch

from ai_hypothesis.population_compute import profile_gate7_high_scale_execution as profile
from ai_hypothesis.population_compute.gate7_high_scale_routing_bandwidth_protocol import (
    GATE7_HIGH_SCALE_POPULATIONS,
    build_gate7_high_scale_tier_plan,
)


class Gate7HighScaleEngineeringProfileTests(unittest.TestCase):
    def test_profile_is_engineering_only_with_fixed_dimensions(self) -> None:
        self.assertTrue(profile.GATE7_HIGH_SCALE_ENGINEERING_PROFILE_ONLY)
        self.assertEqual(profile.GATE7_HIGH_SCALE_ENGINEERING_MODEL_SEED, 71_007)
        self.assertEqual(profile.GATE7_HIGH_SCALE_ENGINEERING_EQUIVALENCE_TOLERANCE, 5e-6)
        self.assertEqual(
            profile.GATE7_HIGH_SCALE_ENGINEERING_CONDITIONS,
            (
                ("global_score", None),
                ("global_hash", None),
                ("bounded_score", 16),
                ("bounded_hash", 16),
                ("bounded_score", 512),
                ("bounded_hash", 512),
            ),
        )

    def test_synthetic_hints_are_deterministic_public_binary_batches(self) -> None:
        for population in GATE7_HIGH_SCALE_POPULATIONS:
            first = profile.gate7_high_scale_engineering_public_hints(population=population)
            second = profile.gate7_high_scale_engineering_public_hints(population=population)
            plan = build_gate7_high_scale_tier_plan(population)
            self.assertEqual(first, second)
            self.assertEqual(len(first), 64)
            self.assertTrue(all(len(row) == plan.world_depth for row in first))
            self.assertTrue(all(value in (0, 1) for row in first for value in row))

    def test_public_seed_batch_is_deterministic_and_population_specific(self) -> None:
        first = profile.gate7_high_scale_engineering_public_seeds(
            population=1024,
            device=torch.device("cpu"),
        )
        repeated = profile.gate7_high_scale_engineering_public_seeds(
            population=1024,
            device=torch.device("cpu"),
        )
        next_population = profile.gate7_high_scale_engineering_public_seeds(
            population=2048,
            device=torch.device("cpu"),
        )
        self.assertEqual(first.shape, (64,))
        self.assertEqual(first.dtype, torch.int64)
        torch.testing.assert_close(first, repeated)
        self.assertFalse(torch.equal(first, next_population))

    def test_largest_tier_memory_arithmetic_records_removed_eightfold_sequence(self) -> None:
        population = 131072
        reference_bytes = 64 * population * 8 * 32 * 4
        bounded_action_bytes = 64 * (population // 2) * 32 * 4
        self.assertEqual(reference_bytes, 8_589_934_592)
        self.assertEqual(bounded_action_bytes, 536_870_912)
        self.assertEqual(reference_bytes // bounded_action_bytes, 16)

    def test_cli_exposes_only_output_root(self) -> None:
        source = inspect.getsource(profile.main)
        self.assertIn('"--output-root"', source)
        for forbidden in (
            "--population",
            "--batch",
            "--k",
            "--checkpoint",
            "--compiler",
            "--mixed-precision",
            "--world",
        ):
            self.assertNotIn(forbidden, source)

    def test_profiler_has_no_checkpoint_or_scientific_execution_surface(self) -> None:
        source = inspect.getsource(profile)
        self.assertNotIn("torch.load", source)
        self.assertNotIn("load_verified", source)
        self.assertNotIn("checkpoint_path", source)
        self.assertNotIn("covered_by_world", source)
        self.assertNotIn("classify_gate7", source)
        self.assertNotIn("generate_gate7_high_scale_world", source)
        self.assertNotIn("hidden_path =", source)


if __name__ == "__main__":
    unittest.main()
