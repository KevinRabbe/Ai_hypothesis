from __future__ import annotations

import unittest

from ai_hypothesis.population_compute.analyze_gate3_v3_confirmation import (
    CHECKPOINTS,
    _classify,
    _metric_key,
)


class Gate3V3ConfirmationAuditTests(unittest.TestCase):
    def _positive_lows(self) -> dict[str, float]:
        lows: dict[str, float] = {}
        for checkpoint in CHECKPOINTS:
            for comparison in (
                "stable_l256_vs_l64",
                "stable_l64_vs_l16",
                "stable_l256_vs_collapsed",
                "stable_l256_vs_reshuffled",
            ):
                lows[_metric_key(checkpoint, comparison)] = 0.05
        return lows

    def test_all_three_primary_and_controls_positive_confirms(self) -> None:
        self.assertEqual(
            _classify(self._positive_lows()),
            "GATE3_V3_CONFIRMED_GENERATION_PRESSURE_BENEFIT",
        )

    def test_one_primary_miss_does_not_confirm(self) -> None:
        lows = self._positive_lows()
        lows[_metric_key(1, "stable_l256_vs_l64")] = 0.0
        self.assertEqual(
            _classify(lows),
            "GATE3_V3_CONFIRMATION_NOT_ESTABLISHED",
        )

    def test_control_failure_marks_mechanism_failure(self) -> None:
        lows = self._positive_lows()
        lows[_metric_key(2, "stable_l256_vs_reshuffled")] = 0.0
        self.assertEqual(
            _classify(lows),
            "GATE3_V3_CONFIRMATION_INVALID_OR_MECHANISM_FAILED",
        )

    def test_lower_l64_l16_is_secondary_not_acceptance_gate(self) -> None:
        lows = self._positive_lows()
        lows[_metric_key(0, "stable_l64_vs_l16")] = -0.1
        self.assertEqual(
            _classify(lows),
            "GATE3_V3_CONFIRMED_GENERATION_PRESSURE_BENEFIT",
        )


if __name__ == "__main__":
    unittest.main()
