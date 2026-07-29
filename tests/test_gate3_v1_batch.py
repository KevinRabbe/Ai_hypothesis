from __future__ import annotations

import unittest

import torch

from ai_hypothesis.population_compute.gate3_v1_batch import run_gate3_v1_public_world_batch
from ai_hypothesis.population_compute.gate3_v1_model import Gate3V1Scorer, run_gate3_v1_public_world
from ai_hypothesis.population_compute.gate3_v1_sparse_active_reserve import (
    Gate3V1ControlMode,
    generate_gate3_v1_world,
)


class Gate3V1BatchTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(515)
        self.model = Gate3V1Scorer()

    def test_batched_runtime_matches_reference_across_s6_capacities_and_controls(self) -> None:
        worlds = tuple(generate_gate3_v1_world(seed=2000 + index, depth=6) for index in range(4))
        public_worlds = tuple(world.public for world in worlds)

        for capacity in (1, 4, 16):
            for mode in Gate3V1ControlMode:
                batched = run_gate3_v1_public_world_batch(
                    self.model,
                    public_worlds,
                    reserve_capacity=capacity,
                    mode=mode,
                    device="cpu",
                )
                reference = tuple(
                    run_gate3_v1_public_world(
                        self.model,
                        world.public,
                        reserve_capacity=capacity,
                        mode=mode,
                        device="cpu",
                    )
                    for world in worlds
                )
                self.assertEqual(batched.world_seeds, tuple(world.public.seed for world in worlds))
                for batch_row, reference_row in zip(batched.world_results, reference, strict=True):
                    self.assertEqual(batch_row.generated_terminal_paths, reference_row.generated_terminal_paths)
                    self.assertEqual(batch_row.telemetry, reference_row.telemetry)

    def test_batched_round_accounting_is_exact_for_productive_and_sink_worlds(self) -> None:
        worlds = tuple(generate_gate3_v1_world(seed=3000 + index, depth=6).public for index in range(3))
        narrow = run_gate3_v1_public_world_batch(
            self.model,
            worlds,
            reserve_capacity=1,
            mode=Gate3V1ControlMode.STABLE_RESERVE,
            device="cpu",
        )
        wide = run_gate3_v1_public_world_batch(
            self.model,
            worlds,
            reserve_capacity=16,
            mode=Gate3V1ControlMode.STABLE_RESERVE,
            device="cpu",
        )
        for result in narrow.world_results:
            self.assertEqual(result.telemetry.productive_rounds, 6)
            self.assertEqual(result.telemetry.sink_rounds, 10)
            self.assertEqual(result.telemetry.total_learned_updates, 256)
        for result in wide.world_results:
            self.assertEqual(result.telemetry.productive_rounds, 16)
            self.assertEqual(result.telemetry.sink_rounds, 0)
            self.assertEqual(result.telemetry.total_learned_updates, 256)


if __name__ == "__main__":
    unittest.main()
