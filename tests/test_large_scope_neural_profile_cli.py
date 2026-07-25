"""Tests process guards for the bounded persistent neural profiler."""

from __future__ import annotations

import unittest

from ai_hypothesis.large_scope.run_persistent_neural_profile import main


class PersistentNeuralProfileCliTests(unittest.TestCase):
    def test_test_split_requires_opt_in_before_checkpoint_loading(self) -> None:
        with self.assertRaisesRegex(SystemExit, "Refusing to open the frozen test split"):
            main(
                [
                    "--checkpoints",
                    "does-not-exist.pt",
                    "--split",
                    "test",
                ]
            )

    def test_profile_budget_must_remain_nonredundant_before_checkpoint_loading(self) -> None:
        with self.assertRaisesRegex(SystemExit, "must not exceed"):
            main(
                [
                    "--checkpoints",
                    "does-not-exist.pt",
                    "--window-count",
                    "4",
                    "--step-width",
                    "2",
                    "--rounds",
                    "3",
                ]
            )

    def test_negative_equivalence_tolerance_fails_before_checkpoint_loading(self) -> None:
        with self.assertRaisesRegex(SystemExit, "must be non-negative"):
            main(
                [
                    "--checkpoints",
                    "does-not-exist.pt",
                    "--equivalence-tolerance",
                    "-0.1",
                ]
            )


if __name__ == "__main__":
    unittest.main()
