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
from ai_hypothesis.population_compute.relay_experiment_v1 import (
    RELAY_EXPERIMENT_V1,
    RelayTrainingConfigV1,
    assess_relay_results_v1,
    evaluate_relay_split_v1,
    load_relay_checkpoint_v1,
    save_relay_checkpoint_v1,
    train_relay_model_v1,
)
from ai_hypothesis.population_compute.relay_model import RelayPopulationConfig
from ai_hypothesis.population_compute.relay_protocol_v1 import RELAY_PROTOCOL_VERSION


class RelayExperimentV1Tests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = RelayTrainingConfigV1(
            steps=2,
            batch_size=2,
            learning_rate=1e-3,
            model=RelayPopulationConfig(state_width=8, message_width=4),
        )

    def test_tiny_v1_training_is_finite_and_keeps_fixed_identity(self) -> None:
        model, summary = train_relay_model_v1(
            training_seed=0,
            config=self.config,
        )
        self.assertEqual(summary.steps, 2)
        self.assertEqual(summary.examples_seen, 4)
        self.assertTrue(math.isfinite(summary.initial_total_loss))
        self.assertTrue(math.isfinite(summary.final_total_loss))
        self.assertTrue(math.isfinite(summary.final_relay_loss))
        self.assertTrue(math.isfinite(summary.final_gate_loss))
        self.assertEqual(summary.learned_parameter_count, model.trainable_parameter_count())
        self.assertEqual(summary.parameter_fingerprint, model.parameter_fingerprint())

    def test_v1_evaluation_uses_normalized_mode_and_matched_scope(self) -> None:
        model, summary = train_relay_model_v1(
            training_seed=1,
            config=self.config,
        )
        fingerprint = model.parameter_fingerprint()
        rows = evaluate_relay_split_v1(
            model,
            training_seed=1,
            split="development",
            world_count=4,
            batch_size=2,
        )
        self.assertEqual(
            len(rows),
            len(RELAY_DIFFICULTIES) * 2 * len(DEVELOPMENT_POPULATION_SIZES),
        )
        self.assertEqual(model.parameter_fingerprint(), fingerprint)
        self.assertEqual(fingerprint, summary.parameter_fingerprint)

        by_key = {
            (
                row.metrics.difficulty,
                row.metrics.condition.nominal_population_size,
                row.metrics.condition.communication_mode,
            ): row
            for row in rows
        }
        for difficulty in (item.name for item in RELAY_DIFFICULTIES):
            for population in DEVELOPMENT_POPULATION_SIZES:
                repaired = by_key[
                    (difficulty, population, CommunicationMode.SPARSE_SHARED_V1)
                ]
                control = by_key[
                    (difficulty, population, CommunicationMode.NO_COMMUNICATION)
                ]
                repaired.validate()
                control.validate()
                self.assertEqual(
                    repaired.metrics.information_complete_count,
                    control.metrics.information_complete_count,
                )
                self.assertEqual(
                    tuple(
                        (cohort.scope_threshold, cohort.task_count)
                        for cohort in repaired.scope_cohorts
                    ),
                    tuple(
                        (cohort.scope_threshold, cohort.task_count)
                        for cohort in control.scope_cohorts
                    ),
                )
                self.assertEqual(control.metrics.messages_emitted, 0)
                self.assertEqual(control.metrics.communicated_scalar_count, 0)
                self.assertEqual(
                    repaired.metrics.parameter_fingerprint,
                    fingerprint,
                )
                self.assertEqual(
                    repaired.metrics.learned_parameter_count,
                    model.trainable_parameter_count(),
                )

        assessments = assess_relay_results_v1(rows)
        self.assertEqual(
            tuple(difficulty for difficulty, _ in assessments),
            tuple(item.name for item in RELAY_DIFFICULTIES),
        )

    def test_confirmation_is_locked_by_default(self) -> None:
        model, _summary = train_relay_model_v1(
            training_seed=0,
            config=self.config,
        )
        with self.assertRaisesRegex(ValueError, "confirmation split is locked"):
            evaluate_relay_split_v1(
                model,
                training_seed=0,
                split="confirmation",
                world_count=2,
                batch_size=1,
            )

    def test_v1_checkpoint_round_trip_preserves_protocol_and_fingerprint(self) -> None:
        model, summary = train_relay_model_v1(
            training_seed=2,
            config=self.config,
        )
        with tempfile.TemporaryDirectory() as directory:
            path = save_relay_checkpoint_v1(
                model,
                summary,
                self.config,
                Path(directory) / "model-v1.pt",
            )
            loaded, payload = load_relay_checkpoint_v1(path)

        self.assertEqual(payload["experiment_version"], RELAY_EXPERIMENT_V1)
        self.assertEqual(payload["protocol_version"], RELAY_PROTOCOL_VERSION)
        self.assertEqual(
            payload["training_config"]["communication_mode"],
            CommunicationMode.SPARSE_SHARED_V1.value,
        )
        self.assertEqual(
            payload["training_config"]["gate_supervision_weight"],
            self.config.gate_supervision_weight,
        )
        self.assertEqual(
            loaded.parameter_fingerprint(),
            summary.parameter_fingerprint,
        )

    def test_v1_training_config_rejects_invalid_gate_weight(self) -> None:
        with self.assertRaisesRegex(ValueError, "gate_supervision_weight"):
            RelayTrainingConfigV1(gate_supervision_weight=-0.1).validate()


if __name__ == "__main__":
    unittest.main()
