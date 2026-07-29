from __future__ import annotations

import unittest

from ai_hypothesis.population_compute.gate3_development import (
    Gate3TrainingConfig,
    train_gate3_development_model,
)
from ai_hypothesis.population_compute.run_gate3_development_v0 import (
    GATE3_FIRST_DEVELOPMENT_TRAINING_SEED,
    train_gate3_development_with_progress,
)


class Gate3DevelopmentRunnerTests(unittest.TestCase):
    def test_progress_training_matches_silent_reference_checkpoint_exactly(self) -> None:
        config = Gate3TrainingConfig(steps=3, batch_size=2)
        silent_model, silent_summary = train_gate3_development_model(
            training_seed=GATE3_FIRST_DEVELOPMENT_TRAINING_SEED,
            config=config,
            device="cpu",
        )
        events: list[tuple[int, int, int, int, float]] = []
        progress_model, progress_summary = train_gate3_development_with_progress(
            training_seed=GATE3_FIRST_DEVELOPMENT_TRAINING_SEED,
            config=config,
            device="cpu",
            progress=lambda done, total, depth, width, loss: events.append(
                (done, total, depth, width, loss)
            ),
        )

        self.assertEqual(progress_model.parameter_fingerprint(), silent_model.parameter_fingerprint())
        self.assertEqual(progress_summary, silent_summary)
        self.assertEqual([event[0] for event in events], [1, 2, 3])
        self.assertTrue(all(event[1] == 3 for event in events))

    def test_admitted_runner_training_seed_is_frozen_to_zero(self) -> None:
        self.assertEqual(GATE3_FIRST_DEVELOPMENT_TRAINING_SEED, 0)
        with self.assertRaises(ValueError):
            train_gate3_development_with_progress(
                training_seed=1,
                config=Gate3TrainingConfig(steps=1, batch_size=1),
                device="cpu",
            )


if __name__ == "__main__":
    unittest.main()
