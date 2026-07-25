"""Tests process guards on the large-scope experiment CLI."""

from __future__ import annotations

import unittest

from ai_hypothesis.large_scope.run_relevance import main


class LargeScopeCliTests(unittest.TestCase):
    def test_test_split_requires_explicit_opt_in_before_checkpoint_loading(self) -> None:
        with self.assertRaisesRegex(SystemExit, "Refusing to open the frozen test split"):
            main(
                [
                    "--checkpoints",
                    "does-not-need-to-exist.pt",
                    "--split",
                    "test",
                ]
            )

    def test_invalid_widths_fail_before_checkpoint_loading(self) -> None:
        with self.assertRaisesRegex(SystemExit, "unique and supplied in increasing order"):
            main(
                [
                    "--checkpoints",
                    "does-not-need-to-exist.pt",
                    "--widths",
                    "4",
                    "1",
                ]
            )

    def test_invalid_world_batch_size_fails_before_checkpoint_loading(self) -> None:
        with self.assertRaisesRegex(SystemExit, "--world-batch-size must be positive"):
            main(
                [
                    "--checkpoints",
                    "does-not-need-to-exist.pt",
                    "--world-batch-size",
                    "0",
                ]
            )


if __name__ == "__main__":
    unittest.main()
