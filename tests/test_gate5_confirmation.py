from __future__ import annotations

import unittest

from ai_hypothesis.population_compute.gate5_bounded_score_activation import (
    GATE5_NONINFERIORITY_MARGIN,
    GATE5_SCHEDULED_SLOTS,
    GATE5_STAGE_A_SLOTS,
    GATE5_STAGE_B_SLOTS,
    GATE5_TOTAL_LEARNED_UPDATES,
    gate5_runtime_seed,
)
from ai_hypothesis.population_compute.gate5_confirmation import (
    GATE5_CONFIRMATION_BOOTSTRAP_SAMPLES,
    GATE5_CONFIRMATION_CHECKPOINT_INDICES,
    GATE5_CONFIRMATION_CONDITIONS,
    GATE5_CONFIRMATION_EVAL_BATCH_SIZE,
    GATE5_CONFIRMATION_VERSION,
    GATE5_CONFIRMATION_WORLD_COUNT,
    gate5_confirmation_runtime_seed,
)


class Gate5ConfirmationProtocolTest(unittest.TestCase):
    def test_confirmation_constants_are_frozen(self) -> None:
        self.assertEqual(GATE5_CONFIRMATION_VERSION, "gate5-bounded-score-activation-confirmation-v0")
        self.assertEqual(GATE5_CONFIRMATION_WORLD_COUNT, 512)
        self.assertEqual(GATE5_CONFIRMATION_EVAL_BATCH_SIZE, 64)
        self.assertEqual(GATE5_CONFIRMATION_BOOTSTRAP_SAMPLES, 4000)
        self.assertEqual(GATE5_CONFIRMATION_CHECKPOINT_INDICES, (0, 1, 2))
        self.assertEqual(len(GATE5_CONFIRMATION_CONDITIONS), 6)
        self.assertEqual(GATE5_STAGE_A_SLOTS, 63)
        self.assertEqual(GATE5_STAGE_B_SLOTS, 96)
        self.assertEqual(GATE5_SCHEDULED_SLOTS, 159)
        self.assertEqual(GATE5_TOTAL_LEARNED_UPDATES, 2544)
        self.assertEqual(GATE5_NONINFERIORITY_MARGIN, 0.05)

    def test_confirmation_runtime_namespace_is_disjoint_from_development(self) -> None:
        for world_index in (0, 1, 17, 255):
            self.assertNotEqual(
                gate5_confirmation_runtime_seed(world_index=world_index),
                gate5_runtime_seed(world_index=world_index),
            )

    def test_confirmation_runtime_seeds_are_deterministic_and_distinct(self) -> None:
        values = [gate5_confirmation_runtime_seed(world_index=index) for index in range(16)]
        self.assertEqual(values, [gate5_confirmation_runtime_seed(world_index=index) for index in range(16)])
        self.assertEqual(len(set(values)), len(values))


if __name__ == "__main__":
    unittest.main()
