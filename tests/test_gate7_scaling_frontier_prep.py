from __future__ import annotations

import unittest

from ai_hypothesis.population_compute.gate7_scaling_frontier_prep import (
    GATE7_EXISTING_CHECKPOINT_MAX_POPULATION,
    GATE7_GATE6_FIRST_CHECKPOINT_SENSITIVE_K16_POPULATION,
    GATE7_GATE6_LAST_ROBUST_K16_POPULATION,
    GATE7_HIGH_SCALE_LADDER,
    GATE7_K_LADDER,
    GATE7_NONINFERIORITY_MARGIN,
    GATE7_PREPARATION_ONLY,
    GATE7_SCREENING_WORLDS_CANDIDATE,
    build_gate7_scale_plan,
    prepared_gate7_plans,
)


class Gate7ScalingFrontierPreparationTests(unittest.TestCase):
    def test_preparation_boundary_and_ladders(self) -> None:
        self.assertTrue(GATE7_PREPARATION_ONLY)
        self.assertEqual(GATE7_EXISTING_CHECKPOINT_MAX_POPULATION, 512)
        self.assertEqual(
            GATE7_HIGH_SCALE_LADDER,
            (1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072),
        )
        self.assertEqual(GATE7_K_LADDER, (16, 32, 64, 128, 256, 512))
        self.assertEqual(GATE7_GATE6_LAST_ROBUST_K16_POPULATION, 128)
        self.assertEqual(GATE7_GATE6_FIRST_CHECKPOINT_SENSITIVE_K16_POPULATION, 256)
        self.assertEqual(GATE7_SCREENING_WORLDS_CANDIDATE, 64)
        self.assertEqual(GATE7_NONINFERIORITY_MARGIN, 0.05)

    def test_structural_scaling(self) -> None:
        plans = prepared_gate7_plans()
        self.assertEqual(tuple(plan.population for plan in plans), (512, *GATE7_HIGH_SCALE_LADDER))
        self.assertEqual(plans[0].frontier_depth, 9)
        self.assertEqual(plans[0].minimum_world_depth, 10)
        self.assertEqual(plans[-1].frontier_depth, 17)
        self.assertEqual(plans[-1].minimum_world_depth, 18)

        for plan in plans:
            self.assertEqual(plan.stage_a_parent_slots, plan.population - 1)
            self.assertEqual(plan.global_score_observations_nominal, plan.population * 128)
            self.assertEqual(
                plan.valid_k_values,
                tuple(k for k in GATE7_K_LADDER if k < plan.population),
            )
            self.assertEqual(
                plan.bounded_score_observation_upper_bounds,
                tuple((k, k * 128) for k in plan.valid_k_values),
            )

    def test_bounded_visibility_is_sublinear_for_prepared_high_scale_points(self) -> None:
        for population in GATE7_HIGH_SCALE_LADDER:
            plan = build_gate7_scale_plan(population)
            for k in plan.valid_k_values:
                self.assertLess(k, population)
                bound = dict(plan.bounded_score_observation_upper_bounds)[k]
                self.assertEqual(bound, k * 128)
                self.assertLess(bound, plan.global_score_observations_nominal)

    def test_geometric_k_ladder_supports_bandwidth_frontier_localization(self) -> None:
        low = build_gate7_scale_plan(1024)
        high = build_gate7_scale_plan(131072)
        self.assertEqual(low.valid_k_values, GATE7_K_LADDER)
        self.assertEqual(high.valid_k_values, GATE7_K_LADDER)
        self.assertEqual(
            dict(low.bounded_score_observation_upper_bounds)[16],
            dict(high.bounded_score_observation_upper_bounds)[16],
        )
        self.assertEqual(
            high.global_score_observations_nominal // low.global_score_observations_nominal,
            128,
        )

    def test_n512_excludes_k512_as_non_bounded(self) -> None:
        plan = build_gate7_scale_plan(512)
        self.assertEqual(plan.valid_k_values, (16, 32, 64, 128, 256))

    def test_non_power_of_two_rejected(self) -> None:
        with self.assertRaises(ValueError):
            build_gate7_scale_plan(1000)


if __name__ == "__main__":
    unittest.main()
