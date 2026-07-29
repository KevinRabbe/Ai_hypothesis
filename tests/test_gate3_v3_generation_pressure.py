from __future__ import annotations

import unittest

from ai_hypothesis.population_compute.gate3_v1_model import Gate3V1Scorer
from ai_hypothesis.population_compute.gate3_v1_sparse_active_reserve import Gate3V1ControlMode
from ai_hypothesis.population_compute.gate3_v3_generation_pressure import (
    GATE3_V3_DEPTH,
    GATE3_V3_HINT_RELIABILITY,
    GATE3_V3_SCHEDULED_SLOTS,
    GATE3_V3_TOTAL_LEARNED_UPDATES,
    generate_gate3_v3_development_world,
    run_gate3_v3_world_batch,
)


class Gate3V3GenerationPressureTests(unittest.TestCase):
    def test_world_generation_is_deterministic_and_frozen(self) -> None:
        a = generate_gate3_v3_development_world(world_index=17)
        b = generate_gate3_v3_development_world(world_index=17)
        self.assertEqual(a, b)
        self.assertEqual(a.public.depth, GATE3_V3_DEPTH)
        self.assertEqual(len(a.public.noisy_hints), GATE3_V3_DEPTH)
        self.assertEqual(GATE3_V3_HINT_RELIABILITY, 0.70)
        self.assertEqual(GATE3_V3_SCHEDULED_SLOTS, 223)
        self.assertEqual(GATE3_V3_TOTAL_LEARNED_UPDATES, 3568)

    def test_stable_l64_and_l256_have_frozen_pressure_geometry(self) -> None:
        model = Gate3V1Scorer().eval()
        world = generate_gate3_v3_development_world(world_index=0)

        l64 = run_gate3_v3_world_batch(
            model,
            (world,),
            reserve_capacity=64,
            mode=Gate3V1ControlMode.STABLE_RESERVE,
            device="cpu",
        )[0]
        l256 = run_gate3_v3_world_batch(
            model,
            (world,),
            reserve_capacity=256,
            mode=Gate3V1ControlMode.STABLE_RESERVE,
            device="cpu",
        )[0]

        self.assertEqual(l64.telemetry.depth7_preprune_width, 128)
        self.assertEqual(l64.telemetry.depth7_retained_width, 64)
        self.assertEqual(l64.telemetry.depth7_expanded_parents, 64)
        self.assertEqual(l64.telemetry.productive_slots, 191)
        self.assertEqual(l64.telemetry.sink_slots, 32)
        self.assertEqual(l64.telemetry.total_learned_updates, 3568)
        self.assertEqual(l64.telemetry.generated_terminal_count, 128)
        self.assertEqual(l64.telemetry.unique_generated_terminal_count, 128)

        self.assertEqual(l256.telemetry.depth7_preprune_width, 128)
        self.assertEqual(l256.telemetry.depth7_retained_width, 128)
        self.assertEqual(l256.telemetry.depth7_expanded_parents, 96)
        self.assertEqual(l256.telemetry.productive_slots, 223)
        self.assertEqual(l256.telemetry.sink_slots, 0)
        self.assertEqual(l256.telemetry.total_learned_updates, 3568)
        self.assertEqual(l256.telemetry.generated_terminal_count, 192)
        self.assertEqual(l256.telemetry.unique_generated_terminal_count, 192)

    def test_collapsed_control_is_one_logical_hypothesis_per_generation(self) -> None:
        model = Gate3V1Scorer().eval()
        world = generate_gate3_v3_development_world(world_index=1)
        result = run_gate3_v3_world_batch(
            model,
            (world,),
            reserve_capacity=256,
            mode=Gate3V1ControlMode.COLLAPSED_DIVERSITY,
            device="cpu",
        )[0]
        self.assertEqual(result.telemetry.retained_widths, (1, 1, 1, 1, 1, 1, 1))
        self.assertEqual(result.telemetry.unique_retained_widths, (1, 1, 1, 1, 1, 1, 1))
        self.assertEqual(result.telemetry.productive_slots, 8)
        self.assertEqual(result.telemetry.sink_slots, 215)
        self.assertEqual(result.telemetry.total_learned_updates, 3568)


if __name__ == "__main__":
    unittest.main()
