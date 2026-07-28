from __future__ import annotations

import unittest

import torch

from ai_hypothesis.population_compute.gate2_development import (
    GATE2_CONFIRMATION_SEED_START,
    GATE2_DEVELOPMENT_SEED_START,
    Gate2TrainingConfig,
    build_gate2_paired_summaries,
    evaluate_gate2_split,
    gate2_stable_training_conditions,
    generate_gate2_split_worlds,
    train_gate2_development_model,
)
from ai_hypothesis.population_compute.gate2_persistent_model import (
    Gate2PersistentModelConfig,
    Gate2PersistentStateModel,
)
from ai_hypothesis.population_compute.gate2_persistent_state_capacity import (
    GATE2_ENTITY_COUNTS,
    gate2_population_widths,
)


class Gate2DevelopmentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        torch.manual_seed(17)
        cls.matrix_model = Gate2PersistentStateModel(
            Gate2PersistentModelConfig(state_width=8, query_width=4)
        )
        cls.expected_count = cls.matrix_model.trainable_parameter_count()
        cls.expected_fingerprint = cls.matrix_model.parameter_fingerprint()
        cls.rows, cls.paired = evaluate_gate2_split(
            cls.matrix_model,
            split="development",
            world_count=1,
            batch_size=1,
            bootstrap_samples=10,
        )

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
            (
                GATE2_DEVELOPMENT_SEED_START,
                GATE2_DEVELOPMENT_SEED_START + 1,
                GATE2_DEVELOPMENT_SEED_START + 2,
            ),
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
        model = Gate2PersistentStateModel(Gate2PersistentModelConfig(state_width=8, query_width=4))
        with self.assertRaisesRegex(ValueError, "confirmation split is locked"):
            evaluate_gate2_split(
                model,
                split="confirmation",
                world_count=1,
                batch_size=1,
                bootstrap_samples=5,
            )

    def test_full_development_matrix_reuses_one_checkpoint_and_world_order(self) -> None:
        self.assertEqual(len(self.rows), 36)
        self.assertEqual(len(self.paired), 33)
        self.assertTrue(all(row.learned_parameter_count == self.expected_count for row in self.rows))
        self.assertTrue(
            all(row.parameter_fingerprint == self.expected_fingerprint for row in self.rows)
        )
        self.assertEqual(self.matrix_model.parameter_fingerprint(), self.expected_fingerprint)

        for entity_count in GATE2_ENTITY_COUNTS:
            entity_rows = [row for row in self.rows if row.entity_count == entity_count]
            seed_sets = {row.world_seeds for row in entity_rows}
            self.assertEqual(len(seed_sets), 1)
            self.assertTrue(
                all(row.inspected_entities_per_world == entity_count for row in entity_rows)
            )
            self.assertTrue(
                all(row.inspected_observations_per_world == 8 * entity_count for row in entity_rows)
            )
            self.assertTrue(
                all(row.learned_updates_per_world == 8 * entity_count for row in entity_rows)
            )

    def test_width_one_stable_reshuffled_pair_is_exact_identity(self) -> None:
        width_one = [
            summary
            for summary in self.paired
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
        with self.assertRaises(KeyError):
            build_gate2_paired_summaries(self.rows[:-1], bootstrap_samples=5)

    def test_tiny_training_smoke_is_finite_and_keeps_one_checkpoint_identity(self) -> None:
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


if __name__ == "__main__":
    unittest.main()
