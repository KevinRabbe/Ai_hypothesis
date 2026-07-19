"""Tests for Step 2 diagnostic metric helpers."""

from __future__ import annotations

import unittest

from ai_hypothesis.step02.metrics import conditional_rate, is_true_minority_opportunity


class Step02DiagnosticMetricHelperTests(unittest.TestCase):
    def test_conditional_rate_uses_none_for_zero_denominator(self) -> None:
        metric = conditional_rate(0, 0)
        self.assertEqual(metric.numerator, 0)
        self.assertEqual(metric.denominator, 0)
        self.assertIsNone(metric.rate)

    def test_conditional_rate_rejects_invalid_counts(self) -> None:
        with self.assertRaisesRegex(ValueError, "numerator cannot exceed"):
            conditional_rate(2, 1)

    def test_w1_has_no_true_minority_opportunity(self) -> None:
        self.assertFalse(
            is_true_minority_opportunity(
                correct_worker_count=1,
                population_width=1,
                majority_is_correct=True,
            )
        )

    def test_w5_true_minority_requires_correct_numerical_minority(self) -> None:
        self.assertTrue(
            is_true_minority_opportunity(
                correct_worker_count=2,
                population_width=5,
                majority_is_correct=False,
            )
        )
        self.assertFalse(
            is_true_minority_opportunity(
                correct_worker_count=3,
                population_width=5,
                majority_is_correct=True,
            )
        )


if __name__ == "__main__":
    unittest.main()
