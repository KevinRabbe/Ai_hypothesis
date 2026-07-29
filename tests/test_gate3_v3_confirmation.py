from __future__ import annotations

import inspect
import unittest

from ai_hypothesis.population_compute import gate3_v3_confirmation as confirmation
from ai_hypothesis.population_compute import gate3_v3_generation_pressure as development


class Gate3V3ConfirmationPreScienceTests(unittest.TestCase):
    def test_frozen_constants(self) -> None:
        self.assertEqual(confirmation.GATE3_V3_CONFIRMATION_WORLD_COUNT, 512)
        self.assertEqual(confirmation.GATE3_V3_CONFIRMATION_EVAL_BATCH_SIZE, 64)
        self.assertEqual(confirmation.GATE3_V3_CONFIRMATION_BOOTSTRAP_SAMPLES, 4000)
        self.assertEqual(
            confirmation.GATE3_V3_CONFIRMATION_CHECKPOINT_INDICES,
            development.GATE3_V3_CHECKPOINT_INDICES,
        )
        self.assertEqual(
            confirmation.GATE3_V3_CONFIRMATION_CONDITIONS,
            development.GATE3_V3_CONDITIONS,
        )

    def test_namespace_is_distinct_without_generating_frozen_worlds(self) -> None:
        source = inspect.getsource(confirmation.generate_gate3_v3_confirmation_world)
        self.assertIn("generation-pressure-confirmation-hidden", source)
        self.assertIn("generation-pressure-confirmation-hints", source)
        self.assertNotIn("generation-pressure-development-hidden", source)
        self.assertNotIn("generation-pressure-development-hints", source)

    def test_no_training_surface(self) -> None:
        names = set(vars(confirmation))
        self.assertFalse(any(name.startswith("train") for name in names))
        self.assertFalse(any("optimizer" in name.lower() for name in names))


if __name__ == "__main__":
    unittest.main()
