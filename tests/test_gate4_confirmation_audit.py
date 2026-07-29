from __future__ import annotations

import unittest

from ai_hypothesis.population_compute.analyze_gate4_confirmation import classify_confirmation


class Gate4ConfirmationAuditTests(unittest.TestCase):
    @staticmethod
    def _all_positive() -> dict[str, float]:
        values: dict[str, float] = {}
        for checkpoint in (0, 1, 2):
            values[f"c{checkpoint}_adaptive_score_vs_static_generation"] = 0.01
            values[f"c{checkpoint}_adaptive_score_vs_adaptive_hash"] = 0.01
            values[f"c{checkpoint}_static_generation_vs_adaptive_hash"] = 0.0
        return values

    def test_confirmed_requires_both_effects_on_all_checkpoints(self) -> None:
        lows = self._all_positive()
        self.assertEqual(
            classify_confirmation(lows),
            "GATE4_CONFIRMED_ADAPTIVE_ACTIVATION_BENEFIT",
        )

    def test_primary_failure_blocks_confirmation(self) -> None:
        lows = self._all_positive()
        lows["c1_adaptive_score_vs_static_generation"] = 0.0
        self.assertEqual(
            classify_confirmation(lows),
            "GATE4_CONFIRMATION_NOT_ESTABLISHED",
        )

    def test_routing_control_failure_blocks_confirmation(self) -> None:
        lows = self._all_positive()
        lows["c2_adaptive_score_vs_adaptive_hash"] = -0.01
        self.assertEqual(
            classify_confirmation(lows),
            "GATE4_CONFIRMATION_NOT_ESTABLISHED",
        )


if __name__ == "__main__":
    unittest.main()
