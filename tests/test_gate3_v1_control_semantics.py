from __future__ import annotations

import unittest

import torch

from ai_hypothesis.population_compute.gate3_v1_model import Gate3V1Scorer, run_gate3_v1_public_world
from ai_hypothesis.population_compute.gate3_v1_sparse_active_reserve import (
    Gate3V1ControlMode,
    generate_gate3_v1_world,
)


class Gate3V1ControlSemanticsTests(unittest.TestCase):
    def test_large_collapsed_reserve_matches_l1_logical_search_path_and_exhaustion(self) -> None:
        torch.manual_seed(404)
        model = Gate3V1Scorer()
        world = generate_gate3_v1_world(seed=8181, depth=6)

        narrow = run_gate3_v1_public_world(
            model,
            world.public,
            reserve_capacity=1,
            mode=Gate3V1ControlMode.STABLE_RESERVE,
            device="cpu",
        )
        collapsed = run_gate3_v1_public_world(
            model,
            world.public,
            reserve_capacity=16,
            mode=Gate3V1ControlMode.COLLAPSED_DIVERSITY,
            device="cpu",
        )

        self.assertEqual(collapsed.generated_terminal_paths, narrow.generated_terminal_paths)
        self.assertEqual(collapsed.telemetry.productive_rounds, narrow.telemetry.productive_rounds)
        self.assertEqual(collapsed.telemetry.sink_rounds, narrow.telemetry.sink_rounds)
        self.assertEqual(collapsed.telemetry.unique_generated_terminal_count, narrow.telemetry.unique_generated_terminal_count)
        self.assertTrue(all(count <= 1 for count in collapsed.telemetry.unique_reserve_population_by_round))
        self.assertEqual(collapsed.telemetry.total_learned_updates, narrow.telemetry.total_learned_updates)


if __name__ == "__main__":
    unittest.main()
