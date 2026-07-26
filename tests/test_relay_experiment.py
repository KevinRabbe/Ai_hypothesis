from __future__ import annotations

import json
import tempfile
import unittest
from collections import Counter
from pathlib import Path
from unittest.mock import patch

import torch

from ai_hypothesis.population_compute import run_relay_scaling
from ai_hypothesis.population_compute.collective_relay import (
    RELAY_DIFFICULTIES,
    relay_scope_thresholds,
)
from ai_hypothesis.population_compute.contract import (
    DEVELOPMENT_POPULATION_SIZES,
    CommunicationMode,
)
from ai_hypothesis.population_compute.relay_experiment import (
    RELAY_TRAINING_PROTOCOL,
    TRAINING_POPULATION_SIZES,
    RelayTrainingConfig,
    RelayTrainingSummary,
    evaluate_relay_development,
    load_relay_checkpoint,
    relay_training_schedule,
    train_relay_checkpoint,
)
from ai_hypothesis.population_compute.relay_model import (
    RelayPopulationConfig,
    RelayPopulationModel,
)


class RelayExperimentTests(unittest.TestCase):
    @staticmethod
    def _summary_for(
        model: RelayPopulationModel,
        training_seed: int = 0,
    ) -> RelayTrainingSummary:
        schedule = relay_training_schedule(1)
        population_counts = Counter(plan.active_workers for plan in schedule)
        difficulty_counts = Counter(plan.difficulty.name for plan in schedule)
        threshold_counts = Counter(plan.scope_threshold for plan in schedule)
        all_thresholds = tuple(
            sorted(
                {
                    threshold
                    for difficulty in RELAY_DIFFICULTIES
                    for threshold in relay_scope_thresholds(difficulty)
                }
            )
        )
        return RelayTrainingSummary(
            training_seed=training_seed,
            steps=1,
            batch_size=1,
            learning_rate=3e-4,
            weight_decay=1e-4,
            first_loss=1.0,
            final_loss=1.0,
            best_loss=1.0,
            elapsed_seconds=0.0,
            learned_parameter_count=model.trainable_parameter_count(),
            parameter_fingerprint=model.parameter_fingerprint(),
            training_protocol=RELAY_TRAINING_PROTOCOL,
            population_batch_counts=tuple(
                (population, population_counts.get(population, 0))
                for population in TRAINING_POPULATION_SIZES
            ),
            difficulty_batch_counts=tuple(
                (difficulty.name, difficulty_counts.get(difficulty.name, 0))
                for difficulty in RELAY_DIFFICULTIES
            ),
            scope_threshold_batch_counts=tuple(
                (threshold, threshold_counts.get(threshold, 0))
                for threshold in all_thresholds
            ),
        )

    def test_training_schedule_balances_population_exposure_and_requires_complete_scope(self) -> None:
        schedule = relay_training_schedule(40)
        counts = Counter(plan.active_workers for plan in schedule)
        self.assertEqual(
            tuple(counts[population] for population in TRAINING_POPULATION_SIZES),
            (10, 10, 10, 10),
        )
        for plan in schedule:
            self.assertGreaterEqual(plan.active_workers, plan.difficulty.hop_count)
            self.assertLessEqual(plan.scope_threshold, plan.active_workers)
            self.assertIn(
                plan.scope_threshold,
                relay_scope_thresholds(plan.difficulty),
            )

    def test_training_changes_checkpoint_and_reload_preserves_exact_identity(self) -> None:
        config = RelayPopulationConfig(state_width=16, message_width=8)
        training = RelayTrainingConfig(
            training_seed=7,
            steps=4,
            batch_size=2,
            learning_rate=1e-3,
        )
        torch.manual_seed(training.training_seed)
        initial = RelayPopulationModel(config)
        initial_fingerprint = initial.parameter_fingerprint()

        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "relay.pt"
            summary = train_relay_checkpoint(
                checkpoint,
                model_config=config,
                training_config=training,
                device="cpu",
            )
            self.assertTrue(checkpoint.is_file())
            self.assertNotEqual(summary.parameter_fingerprint, initial_fingerprint)
            self.assertGreater(summary.learned_parameter_count, 0)
            self.assertEqual(summary.training_protocol, RELAY_TRAINING_PROTOCOL)
            self.assertEqual(
                summary.population_batch_counts,
                ((4, 1), (16, 1), (64, 1), (256, 1)),
            )

            loaded, loaded_summary = load_relay_checkpoint(checkpoint, device="cpu")
            self.assertEqual(loaded_summary, summary)
            self.assertEqual(loaded.parameter_fingerprint(), summary.parameter_fingerprint)
            self.assertEqual(
                loaded.trainable_parameter_count(),
                summary.learned_parameter_count,
            )

    def test_development_curve_uses_one_identity_and_canonical_scope_decomposition(self) -> None:
        torch.manual_seed(1)
        model = RelayPopulationModel(
            RelayPopulationConfig(state_width=16, message_width=8)
        )
        summary = self._summary_for(model)
        fingerprint = model.parameter_fingerprint()
        result = evaluate_relay_development(
            model,
            summary,
            benchmark_seed=3,
            world_count_per_difficulty=12,
            batch_size=4,
            device="cpu",
        )

        self.assertEqual(
            len(result.runs),
            len(RELAY_DIFFICULTIES) * 2 * len(DEVELOPMENT_POPULATION_SIZES),
        )
        self.assertEqual(
            {run.parameter_fingerprint for run in result.runs},
            {fingerprint},
        )
        self.assertEqual(model.parameter_fingerprint(), fingerprint)

        by_scope = {}
        for run in result.runs:
            condition = run.condition
            scope_key = (run.difficulty, condition.nominal_population_size)
            previous_scope = by_scope.setdefault(scope_key, run.information_complete_count)
            self.assertEqual(previous_scope, run.information_complete_count)
            if condition.nominal_population_size == 1:
                self.assertEqual(run.information_complete_count, 0)
                self.assertIsNone(run.solve_rate_given_information_complete)
            if condition.nominal_population_size == 256:
                self.assertEqual(run.information_complete_count, run.task_count)
                self.assertEqual(run.information_complete_rate, 1.0)
            if condition.communication_mode is CommunicationMode.NO_COMMUNICATION:
                self.assertEqual(run.messages_emitted, 0)
                self.assertEqual(run.communicated_scalar_count, 0)
            else:
                self.assertGreater(run.messages_emitted, 0)
                self.assertGreater(run.communicated_scalar_count, 0)

        for difficulty in RELAY_DIFFICULTIES:
            counts = [
                by_scope[(difficulty.name, population)]
                for population in DEVELOPMENT_POPULATION_SIZES
            ]
            self.assertEqual(counts, sorted(counts))
            self.assertGreater(counts[-1], counts[1])

        payload = result.to_dict()
        self.assertEqual(payload["population_sizes"], [1, 4, 16, 64, 256])
        self.assertEqual(
            set(payload["assessments"]),
            {difficulty.name for difficulty in RELAY_DIFFICULTIES},
        )
        self.assertTrue(
            all(row["parameter_fingerprint"] == fingerprint for row in payload["runs"])
        )
        self.assertTrue(
            all("information_complete_rate" in row for row in payload["runs"])
        )
        self.assertTrue(
            all(
                "solve_rate_given_information_complete" in row
                and "solve_rate_given_information_incomplete" in row
                for row in payload["runs"]
            )
        )

    def test_reserved_seed_ranges_reject_cross_split_indices(self) -> None:
        with self.assertRaisesRegex(ValueError, "reserved world-seed range"):
            RelayTrainingConfig(training_seed=10_000).validate()

        model = RelayPopulationModel(
            RelayPopulationConfig(state_width=16, message_width=8)
        )
        with self.assertRaisesRegex(ValueError, "reserved world-seed range"):
            evaluate_relay_development(
                model,
                self._summary_for(model),
                benchmark_seed=10_000,
                world_count_per_difficulty=2,
                batch_size=2,
            )

    def test_cli_smoke_trains_reloads_and_writes_development_result(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "relay.pt"
            output = root / "development.json"
            with patch("builtins.print"):
                exit_code = run_relay_scaling.main(
                    [
                        "--device",
                        "cpu",
                        "--training-seed",
                        "2",
                        "--benchmark-seed",
                        "5",
                        "--train-steps",
                        "4",
                        "--train-batch-size",
                        "2",
                        "--state-width",
                        "16",
                        "--message-width",
                        "8",
                        "--development-world-count",
                        "6",
                        "--eval-batch-size",
                        "2",
                        "--checkpoint",
                        str(checkpoint),
                        "--output",
                        str(output),
                    ]
                )
            self.assertEqual(exit_code, 0)
            self.assertTrue(checkpoint.is_file())
            self.assertTrue(output.is_file())
            payload = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(payload["split"], "development")
            self.assertFalse(payload["provenance"]["confirmation_opened"])
            self.assertEqual(len(payload["runs"]), 30)
            self.assertEqual(
                payload["training"]["training_protocol"],
                RELAY_TRAINING_PROTOCOL,
            )
            self.assertEqual(
                payload["training"]["population_batch_counts"],
                [[4, 1], [16, 1], [64, 1], [256, 1]],
            )
            fingerprints = {row["parameter_fingerprint"] for row in payload["runs"]}
            self.assertEqual(len(fingerprints), 1)
            self.assertEqual(
                next(iter(fingerprints)),
                payload["training"]["parameter_fingerprint"],
            )


if __name__ == "__main__":
    unittest.main()
