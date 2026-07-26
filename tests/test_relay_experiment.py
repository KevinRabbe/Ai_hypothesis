from __future__ import annotations

import math
import tempfile
import unittest
from pathlib import Path

from ai_hypothesis.population_compute import (
    DEVELOPMENT_POPULATION_SIZES,
    RELAY_DIFFICULTIES,
    CommunicationMode,
)
from ai_hypothesis.population_compute import evaluate_relay_checkpoint
from ai_hypothesis.population_compute.relay_experiment import (
    RelayTrainingConfig,
    assess_relay_results,
    evaluate_relay_split,
    load_relay_checkpoint,
    save_relay_checkpoint,
    train_relay_model,
    training_world_batch,
)
from ai_hypothesis.population_compute.relay_model import RelayPopulationConfig


class RelayExperimentTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = RelayTrainingConfig(
            steps=2,
            batch_size=2,
            learning_rate=1e-3,
            model=RelayPopulationConfig(state_width=8, message_width=4),
        )

    def test_training_batches_are_complete_at_the_selected_population_threshold(self) -> None:
        difficulty = RELAY_DIFFICULTIES[1]
        worlds = training_world_batch(
            training_seed=0,
            step=3,
            difficulty=difficulty,
            active_workers=16,
            batch_size=4,
        )
        self.assertEqual(len(worlds), 4)
        self.assertTrue(all(world.scope_threshold == 16 for world in worlds))

    def test_tiny_training_produces_finite_summary_and_fixed_checkpoint_identity(self) -> None:
        model, summary = train_relay_model(
            training_seed=0,
            config=self.config,
        )
        self.assertEqual(summary.steps, 2)
        self.assertEqual(summary.examples_seen, 4)
        self.assertTrue(math.isfinite(summary.initial_loss))
        self.assertTrue(math.isfinite(summary.final_loss))
        self.assertEqual(summary.learned_parameter_count, model.trainable_parameter_count())
        self.assertEqual(summary.parameter_fingerprint, model.parameter_fingerprint())

    def test_development_evaluation_reuses_one_checkpoint_and_matched_scope(self) -> None:
        model, summary = train_relay_model(
            training_seed=1,
            config=self.config,
        )
        before = model.parameter_fingerprint()
        results = evaluate_relay_split(
            model,
            training_seed=1,
            split="development",
            world_count=4,
            batch_size=2,
        )
        self.assertEqual(len(results), len(RELAY_DIFFICULTIES) * 2 * len(DEVELOPMENT_POPULATION_SIZES))
        self.assertEqual(model.parameter_fingerprint(), before)
        self.assertEqual(before, summary.parameter_fingerprint)
        self.assertEqual(
            {row.metrics.parameter_fingerprint for row in results},
            {before},
        )
        self.assertEqual(
            {row.metrics.learned_parameter_count for row in results},
            {model.trainable_parameter_count()},
        )

        by_key = {
            (
                row.metrics.difficulty,
                row.metrics.condition.nominal_population_size,
                row.metrics.condition.communication_mode,
            ): row.metrics
            for row in results
        }
        for difficulty in (difficulty.name for difficulty in RELAY_DIFFICULTIES):
            previous_complete = -1
            for population in DEVELOPMENT_POPULATION_SIZES:
                sparse = by_key[
                    (difficulty, population, CommunicationMode.SPARSE_SHARED_V0)
                ]
                control = by_key[
                    (difficulty, population, CommunicationMode.NO_COMMUNICATION)
                ]
                self.assertEqual(
                    sparse.information_complete_count,
                    control.information_complete_count,
                )
                self.assertGreaterEqual(
                    sparse.information_complete_count,
                    previous_complete,
                )
                previous_complete = sparse.information_complete_count
                self.assertEqual(control.messages_emitted, 0)
                self.assertEqual(control.communicated_scalar_count, 0)

        assessments = assess_relay_results(results)
        self.assertEqual(
            tuple(difficulty for difficulty, _assessment in assessments),
            tuple(difficulty.name for difficulty in RELAY_DIFFICULTIES),
        )

    def test_confirmation_is_locked_by_default(self) -> None:
        model, _summary = train_relay_model(
            training_seed=0,
            config=self.config,
        )
        with self.assertRaisesRegex(ValueError, "confirmation split is locked"):
            evaluate_relay_split(
                model,
                training_seed=0,
                split="confirmation",
                world_count=2,
                batch_size=1,
            )

    def test_checkpoint_evaluation_cli_locks_confirmation_before_loading(self) -> None:
        with self.assertRaisesRegex(SystemExit, "Refusing to open frozen confirmation"):
            evaluate_relay_checkpoint.main(
                [
                    "--checkpoint",
                    "does-not-exist.pt",
                    "--evaluation-split",
                    "confirmation",
                    "--output",
                    "unused.json",
                ]
            )

    def test_checkpoint_round_trip_preserves_exact_parameter_fingerprint(self) -> None:
        model, summary = train_relay_model(
            training_seed=2,
            config=self.config,
        )
        with tempfile.TemporaryDirectory() as directory:
            path = save_relay_checkpoint(
                model,
                summary,
                self.config,
                Path(directory) / "model.pt",
            )
            loaded, payload = load_relay_checkpoint(path)
        self.assertEqual(loaded.parameter_fingerprint(), summary.parameter_fingerprint)
        self.assertEqual(payload["parameter_fingerprint"], summary.parameter_fingerprint)


if __name__ == "__main__":
    unittest.main()
