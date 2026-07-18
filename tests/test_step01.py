"""Tests for the Step 1 generator and deterministic baselines."""

from __future__ import annotations

import unittest

from ai_hypothesis.step01.baselines import oracle_label, predict_baselines
from ai_hypothesis.step01.generator import SPLIT_BASE_SEEDS, generate_dataset, generate_sample
from ai_hypothesis.step01.schema import (
    DIFFICULTIES,
    FEATURE_WIDTH,
    SEQUENCE_LENGTH,
    TASKS,
    Difficulty,
    VALID_LABELS,
)


class Step01GeneratorTests(unittest.TestCase):
    def test_every_task_and_difficulty_has_valid_shape(self) -> None:
        for task in TASKS:
            for difficulty in DIFFICULTIES:
                sample = generate_sample(task, difficulty, seed=123)
                self.assertEqual(len(sample.features), SEQUENCE_LENGTH)
                self.assertTrue(
                    all(len(row) == FEATURE_WIDTH for row in sample.features)
                )
                self.assertEqual(len(sample.mask), SEQUENCE_LENGTH)
                self.assertIn(sample.label, VALID_LABELS[task])
                sample.validate()

    def test_generation_is_deterministic(self) -> None:
        for task in TASKS:
            for difficulty in DIFFICULTIES:
                first = generate_sample(task, difficulty, seed=987654321)
                second = generate_sample(task, difficulty, seed=987654321)
                self.assertEqual(first, second)

    def test_ambiguous_stratum_requires_abstention(self) -> None:
        for task in TASKS:
            for seed in range(20):
                sample = generate_sample(task, Difficulty.AMBIGUOUS, seed)
                self.assertEqual(sample.label, "UNCERTAIN")

    def test_oracle_reconstructs_all_labels(self) -> None:
        for task in TASKS:
            for difficulty in DIFFICULTIES:
                for seed in range(50):
                    sample = generate_sample(task, difficulty, seed)
                    self.assertEqual(oracle_label(sample), sample.label)

    def test_baselines_return_only_task_valid_labels(self) -> None:
        for task in TASKS:
            for difficulty in DIFFICULTIES:
                for seed in range(20):
                    sample = generate_sample(task, difficulty, seed)
                    for prediction in predict_baselines(sample).values():
                        self.assertIn(prediction, VALID_LABELS[task])

    def test_split_seed_ranges_are_disjoint(self) -> None:
        self.assertEqual(len(set(SPLIT_BASE_SEEDS.values())), len(SPLIT_BASE_SEEDS))
        bases = sorted(SPLIT_BASE_SEEDS.values())
        self.assertGreaterEqual(bases[1] - bases[0], 1_000_000)
        self.assertGreaterEqual(bases[2] - bases[1], 1_000_000)

    def test_dataset_cycle_covers_all_task_difficulty_pairs(self) -> None:
        count = len(TASKS) * len(DIFFICULTIES)
        samples = list(generate_dataset("validation", count))
        pairs = {(sample.task, sample.difficulty) for sample in samples}
        expected = {(task, difficulty) for task in TASKS for difficulty in DIFFICULTIES}
        self.assertEqual(pairs, expected)

    def test_different_splits_do_not_replay_same_samples(self) -> None:
        train = list(generate_dataset("train", 20))
        validation = list(generate_dataset("validation", 20))
        test = list(generate_dataset("test", 20))
        train_keys = {(sample.task, sample.difficulty, sample.seed) for sample in train}
        validation_keys = {
            (sample.task, sample.difficulty, sample.seed) for sample in validation
        }
        test_keys = {(sample.task, sample.difficulty, sample.seed) for sample in test}
        self.assertTrue(train_keys.isdisjoint(validation_keys))
        self.assertTrue(train_keys.isdisjoint(test_keys))
        self.assertTrue(validation_keys.isdisjoint(test_keys))


if __name__ == "__main__":
    unittest.main()
