"""Tests large-scope worlds composed from frozen Step 1 relevance semantics."""

from __future__ import annotations

import unittest

from ai_hypothesis.large_scope import (
    LARGE_SCOPE_SPLIT_SEED_RANGES,
    LargeScopeRelevanceConfig,
    diverse_worker_indices,
    generate_large_scope_dataset,
    generate_large_scope_relevance,
    inspection_order,
    inspection_prefix,
    same_worker_indices,
)
from ai_hypothesis.step01.schema import Difficulty, TaskFamily


class LargeScopeRelevanceTests(unittest.TestCase):
    def test_generation_is_deterministic_and_uses_reserved_window_seeds(self) -> None:
        config = LargeScopeRelevanceConfig(window_count=16)
        first = generate_large_scope_relevance(42, config)
        second = generate_large_scope_relevance(42, config)

        self.assertEqual(first, second)
        self.assertTrue(first.target_present)
        self.assertEqual(first.split, "development")
        self.assertEqual(len(first.windows), 16)
        self.assertEqual(len(set(first.window_seeds)), 16)
        range_start, range_limit = LARGE_SCOPE_SPLIT_SEED_RANGES["development"]
        self.assertTrue(
            all(range_start <= seed < range_limit for seed in first.window_seeds)
        )
        self.assertTrue(all(window.task is TaskFamily.RELEVANCE for window in first.windows))

    def test_large_scope_splits_use_disjoint_local_window_seed_ranges(self) -> None:
        samples = {
            split: generate_large_scope_relevance(10, split=split)
            for split in LARGE_SCOPE_SPLIT_SEED_RANGES
        }
        seed_sets = {split: set(sample.window_seeds) for split, sample in samples.items()}
        self.assertTrue(seed_sets["development"].isdisjoint(seed_sets["confirmation"]))
        self.assertTrue(seed_sets["development"].isdisjoint(seed_sets["test"]))
        self.assertTrue(seed_sets["confirmation"].isdisjoint(seed_sets["test"]))

    def test_dataset_stream_is_seed_ordered_and_balanced_by_parity(self) -> None:
        samples = tuple(generate_large_scope_dataset("development", 4, start_seed=20))
        self.assertEqual(tuple(sample.seed for sample in samples), (20, 21, 22, 23))
        self.assertEqual(
            tuple(sample.target_present for sample in samples),
            (True, False, True, False),
        )

    def test_positive_world_contains_exactly_one_relevant_window(self) -> None:
        sample = generate_large_scope_relevance(
            100,
            LargeScopeRelevanceConfig(
                window_count=32,
                target_difficulty=Difficulty.HARD,
                ambiguous_distractor_fraction=0.25,
            ),
            target_present=True,
        )
        relevant = [
            index for index, window in enumerate(sample.windows) if window.label == "RELEVANT"
        ]
        self.assertEqual(relevant, [sample.target_index])
        self.assertGreater(
            sum(window.label == "UNCERTAIN" for window in sample.windows),
            0,
        )

    def test_negative_world_contains_no_relevant_window(self) -> None:
        sample = generate_large_scope_relevance(
            101,
            LargeScopeRelevanceConfig(window_count=16),
            target_present=False,
        )
        self.assertFalse(sample.target_present)
        self.assertIsNone(sample.target_index)
        self.assertNotIn("RELEVANT", tuple(window.label for window in sample.windows))

    def test_inspection_widths_are_nested_prefixes_without_duplicates(self) -> None:
        sample = generate_large_scope_relevance(24)
        order = inspection_order(
            sample.seed,
            sample.config.window_count,
            split=sample.split,
        )
        width_1 = inspection_prefix(sample, 1)
        width_4 = inspection_prefix(sample, 4)
        width_16 = inspection_prefix(sample, 16)

        self.assertEqual(width_1, order[:1])
        self.assertEqual(width_4, order[:4])
        self.assertEqual(width_16, order)
        self.assertEqual(width_4[:1], width_1)
        self.assertEqual(width_16[:4], width_4)
        self.assertEqual(len(set(width_16)), 16)

    def test_same_worker_and_diverse_worker_controls_use_same_width(self) -> None:
        same_4 = same_worker_indices(seed=7, width=4, population_width=16)
        same_16 = same_worker_indices(seed=7, width=16, population_width=16)
        diverse_4 = diverse_worker_indices(seed=7, width=4, population_width=16)
        diverse_16 = diverse_worker_indices(seed=7, width=16, population_width=16)

        self.assertEqual(len(set(same_16)), 1)
        self.assertEqual(same_16[:4], same_4)
        self.assertEqual(len(set(diverse_16)), 16)
        self.assertEqual(diverse_16[:4], diverse_4)
        self.assertEqual(same_4[0], diverse_4[0])
        self.assertEqual(
            same_worker_indices(seed=7, width=1, population_width=16),
            diverse_worker_indices(seed=7, width=1, population_width=16),
        )

    def test_diverse_width_cannot_exceed_available_population(self) -> None:
        with self.assertRaisesRegex(ValueError, "cannot exceed"):
            diverse_worker_indices(seed=1, width=17, population_width=16)


if __name__ == "__main__":
    unittest.main()
