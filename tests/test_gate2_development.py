from __future__ import annotations

import unittest

import torch

from ai_hypothesis.population_compute.gate2_development import (
    GATE2_CONFIRMATION_SEED_START,
    GATE2_DEVELOPMENT_EXPERIMENT_VERSION,
    GATE2_DEVELOPMENT_SEED_START,
    Gate2TrainingConfig,
    build_gate2_paired_summaries,
    evaluate_gate2_split,
    gate2_stable_training_conditions,
    generate_gate2_split_worlds,
    run_gate2_development,
    train_gate2_development_model,
)
from ai_hypothesis.population_compute.gate2_persistent_model import (
    Gate2PersistentModelConfig,
    Gate2PersistentStateModel,
)
from ai_hypothesis.population_compute.gate2_persistent_state_capacity import (
    GATE2_ENTITY_COUNTS,
    Gate2ControlMode,
    gate2_population_widths,
)


class Gate2DevelopmentTests(unittest.TestCase):
    def test_training_schedule_covers_every_stable_entity_width_condition_once(self) -> None:
        conditions = gate2_stable_training_conditions()
        expected = tuple(
            (entity_count, width)
            for entity_count in GATE2_ENTITY_COUNTS
            for width in gate2_population_widths(entity_count)
        )
        self.assertEqual(conditions, expected)
        self.assertEqual(len(conditions), 12)

    def test_development_and_confirmation_seed_domains_are_separate(self) -> None:
        development = generate_gate2_split_worlds(
            split="development",
            entity_count=16,
            world_count=3,
        )
        self.assertEqual(
            tuple(world.seed for world in development),
            (GATE2_DEVELOPMENT_SEED_START, GATE2_DEVELOPMENT_SEED_START + 1, GATE2_DEVELOPMENT_SEED_START + 2),
        )
        self.assertTrue(all(world.seed < GATE2_CONFIRMATION_SEED_START for world in development))

        with self.assertRaisesRegex(ValueError, "confirmation worlds are locked"):
            generate_gate2_split_worlds(
                split="confirmation",
                entity_count=16,
                world_count=1,
            )

        confirmation = generate_gate2_split_worlds(
            split="confirmation",
            entity_count=16,
            world_count=2,
            allow_confirmation=True,
        )
        self.assertEqual(
            tuple(world.seed for world in confirmation),
            (GATE2_CONFIRMATION_SEED_START, GATE2_CONFIRMATION_SEED_START + 1),
        )

    def test_confirmation_evaluation_is_locked_by_default(self) -> None:
        model = Gate2PersistentStateModel()
        with self.assertRaisesRegex(ValueError, "confirmation split is locked"):
            evaluate_gate2_split(
                model,
                split="confirmation",
                world_count=1,
                batch_size=1,
                bootstrap_samples=10,
            )

    def test_full_development_matrix_reuses_one_checkpoint_and_world_order(self) -> None:
        torch.manual_seed(17)
        model = Gate2PersistentStateModel()
        expected_count = model.trainable_parameter_count()
        expected_fingerprint = model.parameter_fingerprint()
        rows, paired = evaluate_gate2_split(
            model,
            split="development",
            world_count=2,
            batch_size=2,
            bootstrap_samples=20,
        )

        self.assertEqual(len(rows), 36)
        self.assertEqual(len(paired), 33)
        self.assertTrue(all(row.learned_parameter_count == expected_count for row in rows))
        self.assertTrue(all(row.parameter_fingerprint == expected_fingerprint for row in rows))
        self.assertEqual(model.parameter_fingerprint(), expected_fingerprint)

        for entity_count in GATE2_ENTITY_COUNTS:
            entity_rows = [row for row in rows if row.entity_count == entity_count]
            seed_sets = {row.world_seeds for row in entity_rows}
            self.assertEqual(len(seed_sets), 1)
            self.assertTrue(all(row.inspected_entities_per_world == entity_count for row in entity_rows))
            self.assertTrue(all(row.inspected_observations_per_world == 8 * entity_count for row in entity_rows))
            self.assertTrue(all(row.learned_updates_per_world == 8 * entity_count for row in entity_rows))

    def test_width_one_stable_reshuffled_pair_is_exact_identity(self) -> None:
        torch.manual_seed(19)
        model = Gate2PersistentStateModel()
        rows, paired = evaluate_gate2_split(
            model,
            split="development",
            world_count=3,
            batch_size=3,
            bootstrap_samples=30,
        )
        del rows
        width_one = [
            summary
            for summary in paired
            if summary.comparison == "stable_vs_reshuffled" and summary.treatment_width == 1
        ]
        self.assertEqual(len(width_one), 3)
        for summary in width_one:
            self.assertEqual(summary.exact_solve_delta, 0.0)
            self.assertEqual(summary.treatment_only, 0)
            self.assertEqual(summary.reference_only, 0)
            self.assertEqual(summary.bootstrap_ci_low, 0.0)
            self.assertEqual(summary.bootstrap_ci_high, 0.0)

    def test_paired_builder_rejects_missing_matrix(self) -> None:
        model = Gate2PersistentStateModel()
        rows, _ = evaluate_gate2_split(
            model,
            split="development",
            world_count=1,
            batch_size=1,
            bootstrap_samples=5,
        )
        with self.assertRaises(KeyError):
            build_gate2_paired_summaries(rows[:-1], bootstrap_samples=5)

    def test_tiny_training_smoke_is_finite_and_development_only(self) -> None:
        config = Gate2TrainingConfig(
            steps=2,
            batch_size=2,
            learning_rate=3e-4,
            model=Gate2PersistentModelConfig(state_width=16, query_width=8),
        )
        model, summary = train_gate2_development_model(
            training_seed=5,
            config=config,
            device="cpu",
        )
        self.assertEqual(summary.steps, 2)
        self.assertEqual(summary.examples_seen, 4)
        self.assertEqual(summary.stable_training_condition_count, 12)
        self.assertTrue(torch.isfinite(torch.tensor(summary.initial_loss)))
        self.assertTrue(torch.isfinite(torch.tensor(summary.final_loss)))
        self.assertEqual(summary.learned_parameter_count, model.trainable_parameter_count())
        self.assertEqual(summary.parameter_fingerprint, model.parameter_fingerprint())

        _, result = run_gate2_development(
            training_seed=7,
            training_config=Gate2TrainingConfig(
                steps=1,
                batch_size=1,
                model=Gate2PersistentModelConfig(state_width=8, query_width=4),
            ),
            evaluation_world_count=1,
            evaluation_batch_size=1,
            bootstrap_samples=5,
            device="cpu",
        )
        payload = result.to_dict()
        self.assertEqual(payload["experiment_version"], GATE2_DEVELOPMENT_EXPERIMENT_VERSION)
        self.assertEqual(payload["evaluation_split"], "development")
        self.assertFalse(payload["confirmation_opened"])
        self.assertEqual(payload["scientific_decision"], "DEVELOPMENT_ONLY_NOT_ASSIGNED")
        self.assertEqual(len(payload["conditions"]), 36)
        self.assertEqual(len(payload["paired_summaries"]), 33)


if __name__ == "__main__":
    unittest.main()
