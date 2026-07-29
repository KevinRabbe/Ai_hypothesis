from __future__ import annotations

import unittest
from unittest.mock import patch

import torch

from ai_hypothesis.population_compute.gate2_confirmation import (
    GATE2_CONFIRMATION_TRAINING_SEEDS,
    _primary_confirmation_comparisons,
    _width1_identity_passed,
    evaluate_gate2_confirmation,
    frozen_confirmation_training_config,
    train_gate2_confirmation_model,
)
from ai_hypothesis.population_compute.gate2_development import (
    Gate2TrainingConfig,
    evaluate_gate2_split,
    train_gate2_development_model,
)
from ai_hypothesis.population_compute.gate2_persistent_model import (
    Gate2PersistentModelConfig,
    Gate2PersistentStateModel,
)


class Gate2ConfirmationTests(unittest.TestCase):
    def test_frozen_confirmation_recipe(self) -> None:
        config = frozen_confirmation_training_config()
        self.assertEqual(GATE2_CONFIRMATION_TRAINING_SEEDS, (3, 4, 5))
        self.assertEqual(config.steps, 1_000)
        self.assertEqual(config.batch_size, 32)
        self.assertEqual(config.learning_rate, 3e-4)
        self.assertEqual(config.weight_decay, 1e-4)
        self.assertEqual(config.gradient_clip_norm, 1.0)
        self.assertEqual(config.model.state_width, 64)
        self.assertEqual(config.model.query_width, 24)

    def test_progress_training_path_is_checkpoint_identical_to_silent_training(self) -> None:
        config = Gate2TrainingConfig(
            steps=3,
            batch_size=2,
            learning_rate=3e-4,
            weight_decay=1e-4,
            gradient_clip_norm=1.0,
            model=Gate2PersistentModelConfig(state_width=8, query_width=4),
        )
        silent_model, silent_summary = train_gate2_development_model(
            training_seed=3,
            config=config,
            device="cpu",
        )
        progress_events: list[tuple[int, int, int, int, float]] = []
        progress_model, progress_summary = train_gate2_confirmation_model(
            training_seed=3,
            config=config,
            device="cpu",
            progress=lambda *event: progress_events.append(event),
        )

        self.assertEqual(len(progress_events), config.steps)
        self.assertEqual(silent_model.parameter_fingerprint(), progress_model.parameter_fingerprint())
        self.assertEqual(silent_summary.parameter_fingerprint, progress_summary.parameter_fingerprint)
        self.assertEqual(silent_summary.initial_loss, progress_summary.initial_loss)
        self.assertEqual(silent_summary.final_loss, progress_summary.final_loss)
        self.assertEqual(silent_summary.mean_last_50_loss, progress_summary.mean_last_50_loss)

    def test_progress_evaluation_path_is_identical_to_existing_confirmation_evaluator(self) -> None:
        torch.manual_seed(17)
        model = Gate2PersistentStateModel(Gate2PersistentModelConfig(state_width=8, query_width=4))
        silent_rows, silent_pairs = evaluate_gate2_split(
            model,
            split="confirmation",
            world_count=2,
            batch_size=2,
            device="cpu",
            allow_confirmation=True,
            bootstrap_samples=10,
        )

        events: list[tuple[int, int, int, int, object]] = []
        with (
            patch(
                "ai_hypothesis.population_compute.gate2_confirmation.GATE2_CONFIRMATION_WORLD_COUNT",
                2,
            ),
            patch(
                "ai_hypothesis.population_compute.gate2_confirmation.GATE2_CONFIRMATION_EVALUATION_BATCH_SIZE",
                2,
            ),
            patch(
                "ai_hypothesis.population_compute.gate2_confirmation.GATE2_CONFIRMATION_BOOTSTRAP_SAMPLES",
                10,
            ),
        ):
            progress_rows, progress_pairs = evaluate_gate2_confirmation(
                model,
                world_count=2,
                batch_size=2,
                bootstrap_samples=10,
                device="cpu",
                progress=lambda *event: events.append(event),
            )

        self.assertEqual([row.to_dict() for row in silent_rows], [row.to_dict() for row in progress_rows])
        self.assertEqual([row.to_dict() for row in silent_pairs], [row.to_dict() for row in progress_pairs])
        self.assertEqual(len(events), 36)
        self.assertEqual(events[-1][0:2], (36, 36))
        self.assertTrue(_width1_identity_passed(progress_pairs))
        self.assertEqual(len(_primary_confirmation_comparisons(progress_pairs)), 4)

    def test_confirmation_training_rejects_development_seed(self) -> None:
        tiny = Gate2TrainingConfig(
            steps=1,
            batch_size=1,
            model=Gate2PersistentModelConfig(state_width=4, query_width=4),
        )
        with self.assertRaises(ValueError):
            train_gate2_confirmation_model(training_seed=0, config=tiny, device="cpu")


if __name__ == "__main__":
    unittest.main()
