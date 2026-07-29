from __future__ import annotations

import unittest

from ai_hypothesis.population_compute.gate4_adaptive_activation import (
    GATE4_CONDITIONS,
    GATE4_EVAL_BATCH_SIZE,
)
from ai_hypothesis.population_compute.gate4_confirmation import (
    GATE4_CONFIRMATION_BOOTSTRAP_SAMPLES,
    GATE4_CONFIRMATION_CHECKPOINT_INDICES,
    GATE4_CONFIRMATION_CONDITIONS,
    GATE4_CONFIRMATION_EVAL_BATCH_SIZE,
    GATE4_CONFIRMATION_VERSION,
    GATE4_CONFIRMATION_WORLD_COUNT,
)


class Gate4ConfirmationProtocolTests(unittest.TestCase):
    def test_frozen_constants(self) -> None:
        self.assertEqual(GATE4_CONFIRMATION_VERSION, "gate4-adaptive-activation-confirmation-v0")
        self.assertEqual(GATE4_CONFIRMATION_WORLD_COUNT, 512)
        self.assertEqual(GATE4_CONFIRMATION_EVAL_BATCH_SIZE, 64)
        self.assertEqual(GATE4_CONFIRMATION_EVAL_BATCH_SIZE, GATE4_EVAL_BATCH_SIZE)
        self.assertEqual(GATE4_CONFIRMATION_BOOTSTRAP_SAMPLES, 4000)
        self.assertEqual(GATE4_CONFIRMATION_CHECKPOINT_INDICES, (0, 1, 2))
        self.assertEqual(GATE4_CONFIRMATION_CONDITIONS, GATE4_CONDITIONS)
        self.assertEqual(len(GATE4_CONFIRMATION_CONDITIONS), 3)


if __name__ == "__main__":
    unittest.main()
