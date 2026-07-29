from __future__ import annotations

import unittest

from ai_hypothesis.population_compute.gate3_v1_sparse_active_reserve import Gate3V1ControlMode
from ai_hypothesis.population_compute.gate3_v2_frontier import (
    GATE3_V2_BOOTSTRAP_SAMPLES,
    GATE3_V2_CONDITIONS,
    GATE3_V2_DEPTH,
    GATE3_V2_EVAL_BATCH_SIZE,
    GATE3_V2_HINT_RELIABILITY,
    GATE3_V2_SEARCH_ROUNDS,
    GATE3_V2_TOTAL_LEARNED_UPDATES,
    GATE3_V2_WORLD_COUNT,
    Gate3V2AmbiguityTier,
    gate3_v2_runtime_seed,
    generate_gate3_v2_development_world,
)


class Gate3V2FrontierMechanicsTest(unittest.TestCase):
    def test_frozen_constants_and_matrix(self) -> None:
        self.assertEqual(GATE3_V2_DEPTH, 10)
        self.assertEqual(GATE3_V2_SEARCH_ROUNDS, 256)
        self.assertEqual(GATE3_V2_TOTAL_LEARNED_UPDATES, 4096)
        self.assertEqual(GATE3_V2_WORLD_COUNT, 256)
        self.assertEqual(GATE3_V2_EVAL_BATCH_SIZE, 64)
        self.assertEqual(GATE3_V2_BOOTSTRAP_SAMPLES, 2000)
        self.assertEqual(GATE3_V2_HINT_RELIABILITY[Gate3V2AmbiguityTier.A60], 0.60)
        self.assertEqual(GATE3_V2_HINT_RELIABILITY[Gate3V2AmbiguityTier.A55], 0.55)
        self.assertEqual(
            GATE3_V2_CONDITIONS,
            (
                (1, Gate3V1ControlMode.STABLE_RESERVE),
                (16, Gate3V1ControlMode.STABLE_RESERVE),
                (64, Gate3V1ControlMode.STABLE_RESERVE),
                (256, Gate3V1ControlMode.STABLE_RESERVE),
                (256, Gate3V1ControlMode.COLLAPSED_DIVERSITY),
                (256, Gate3V1ControlMode.RESHUFFLED_CONTINUITY),
            ),
        )

    def test_world_generation_is_deterministic_and_namespaced(self) -> None:
        first = generate_gate3_v2_development_world(world_index=17, tier=Gate3V2AmbiguityTier.A60)
        second = generate_gate3_v2_development_world(world_index=17, tier=Gate3V2AmbiguityTier.A60)
        self.assertEqual(first, second)
        self.assertEqual(first.public.seed, gate3_v2_runtime_seed(world_index=17, tier=Gate3V2AmbiguityTier.A60))
        self.assertNotEqual(
            gate3_v2_runtime_seed(world_index=17, tier=Gate3V2AmbiguityTier.A60),
            gate3_v2_runtime_seed(world_index=17, tier=Gate3V2AmbiguityTier.A55),
        )

    def test_ambiguity_tiers_share_hidden_paths(self) -> None:
        for world_index in range(GATE3_V2_WORLD_COUNT):
            a60 = generate_gate3_v2_development_world(world_index=world_index, tier=Gate3V2AmbiguityTier.A60)
            a55 = generate_gate3_v2_development_world(world_index=world_index, tier=Gate3V2AmbiguityTier.A55)
            self.assertEqual(a60.hidden_path, a55.hidden_path)

    def test_a55_correct_hint_is_also_correct_in_a60(self) -> None:
        # Both tiers share the same underlying uniform corruption draws.  A55 therefore cannot have
        # a correct hint where A60 has an incorrect one.
        for world_index in range(GATE3_V2_WORLD_COUNT):
            a60 = generate_gate3_v2_development_world(world_index=world_index, tier=Gate3V2AmbiguityTier.A60)
            a55 = generate_gate3_v2_development_world(world_index=world_index, tier=Gate3V2AmbiguityTier.A55)
            for hidden, hint60, hint55 in zip(
                a60.hidden_path,
                a60.public.noisy_hints,
                a55.public.noisy_hints,
                strict=True,
            ):
                if hint55 == hidden:
                    self.assertEqual(hint60, hidden)

    def test_world_index_outside_frozen_domain_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            generate_gate3_v2_development_world(world_index=-1, tier=Gate3V2AmbiguityTier.A60)
        with self.assertRaises(ValueError):
            generate_gate3_v2_development_world(world_index=256, tier=Gate3V2AmbiguityTier.A60)


if __name__ == "__main__":
    unittest.main()
