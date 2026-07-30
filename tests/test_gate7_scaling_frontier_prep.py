from __future__ import annotations

import unittest

from ai_hypothesis.population_compute.gate7_scaling_frontier_prep import (
    GATE7_EXISTING_CHECKPOINT_MAX_POPULATION,
    GATE7_HIGH_SCALE_LADDER,
    GATE7_NONINFERIORITY_MARGIN,
    GATE7_PREPARATION_ONLY,
    GATE7_PRIMARY_K,
    GATE7_SCREENING_WORLDS_CANDIDATE,
    build_gate7_scale_plan,
    prepared_gate7_plans,
)


class Gate7ScalingFrontierPreparationTests(unittest.TestCase):
    def test_preparation_boundary_and_ladder(self) -> None:
        self.assertTrue(GATE7_PREPARATION_ONLY)
        self.assertEqual(GATE7_EXISTING_CHECKPOINT_MAX_POPULATION, 512)
        self.assertEqual(
            GATE7_HIGH_SCALE_LADDER,
            (1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072),
        )
        self.assertEqual(GATE7_PRIMARY_K, 16)
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
            self.assertEqual(plan.k16_score_observations_upper_bound, 16 * 128)
            self.assertEqual(plan.global_score_observations_nominal, plan.population * 128)

    def test_k16_visibility_stays_constant_while_global_scales(self) -> None:
        low = build_gate7_scale_plan(1024)
        high = build_gate7_scale_plan(131072)
        self.assertEqual(low.k16_score_observations_upper_bound, high.k16_score_observations_upper_bound)
        self.assertEqual(high.global_score_observations_nominal // low.global_score_observations_nominal, 128)

    def test_non_power_of_two_rejected(self) -> None:
        with self.assertRaises(ValueError):
            build_gate7_scale_plan(1000)


if __name__ == "__main__":
    unittest.main()
