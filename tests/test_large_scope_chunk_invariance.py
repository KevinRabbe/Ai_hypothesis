"""Tests that execution chunk size cannot change the frozen benchmark corpus."""

from __future__ import annotations

import unittest

from ai_hypothesis.large_scope.relevance import (
    LargeScopeRelevanceConfig,
    generate_large_scope_dataset,
)


class LargeScopeChunkInvarianceTests(unittest.TestCase):
    def test_chunked_generation_matches_one_shot_generation_exactly(self) -> None:
        config = LargeScopeRelevanceConfig(window_count=16)
        split = "development"
        start_seed = 37
        total = 11

        one_shot = tuple(
            generate_large_scope_dataset(
                split,
                total,
                config,
                start_seed=start_seed,
            )
        )

        chunked = []
        offset = 0
        for chunk_size in (2, 1, 5, 3):
            chunked.extend(
                generate_large_scope_dataset(
                    split,
                    chunk_size,
                    config,
                    start_seed=start_seed + offset,
                )
            )
            offset += chunk_size

        self.assertEqual(offset, total)
        self.assertEqual(tuple(chunked), one_shot)
        self.assertEqual(
            tuple(sample.seed for sample in chunked),
            tuple(range(start_seed, start_seed + total)),
        )

    def test_different_chunk_partitions_produce_same_confirmation_worlds(self) -> None:
        config = LargeScopeRelevanceConfig(window_count=8)
        split = "confirmation"
        start_seed = 120
        total = 9

        expected = tuple(
            generate_large_scope_dataset(
                split,
                total,
                config,
                start_seed=start_seed,
            )
        )

        actual = []
        offset = 0
        for chunk_size in (4, 4, 1):
            actual.extend(
                generate_large_scope_dataset(
                    split,
                    chunk_size,
                    config,
                    start_seed=start_seed + offset,
                )
            )
            offset += chunk_size

        self.assertEqual(tuple(actual), expected)


if __name__ == "__main__":
    unittest.main()
