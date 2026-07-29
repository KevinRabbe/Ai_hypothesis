from __future__ import annotations

import unittest

from ai_hypothesis.population_compute.gate3_v1_development import (
    Gate3V1TrainingConfig,
    train_gate3_v1_development_model,
)
from ai_hypothesis.population_compute.run_gate3_v1_development import (
    GATE3_V1_FIRST_DEVELOPMENT_TRAINING_SEED,
    train_gate3_v1_with_progress,
)


class Gate3V1DevelopmentRunnerTests(unittest.TestCase):
    def test_progress_training_matches_reference_checkpoint_exactly(self) -> None:
        config = Gate3V1TrainingConfig(steps=3, batch_size=4)
        reference_model, reference_summary = train_gate3_v1_development_model(
            training_seed=0,
            config=config,
            device="cpu",
        )
        events: list[tuple[int, int, int, float]] = []
        progress_model, progress_summary = train_gate3_v1_with_progress(
            training_seed=0,
            config=config,
            device="cpu",
            progress=lambda done, total, depth, loss: events.append((done, total, depth, loss)),
        )
        self.assertEqual(progress_model.parameter_fingerprint(), reference_model.parameter_fingerprint())
        self.assertEqual(progress_summary, reference_summary)
        self.assertEqual([event[0] for event in events], [1, 2, 3])
        self.assertEqual([event[2] for event in events], [6, 8, 10])

    def test_admitted_training_seed_is_zero_only(self) -> None:
        self.assertEqual(GATE3_V1_FIRST_DEVELOPMENT_TRAINING_SEED, 0)
        with self.assertRaises(ValueError):
            train_gate3_v1_with_progress(
                training_seed=1,
                config=Gate3V1TrainingConfig(steps=1, batch_size=2),
                device="cpu",
            )


if __name__ == "__main__":
    unittest.main()
