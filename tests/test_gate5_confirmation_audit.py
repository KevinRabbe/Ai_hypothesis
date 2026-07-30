from __future__ import annotations

import unittest

from ai_hypothesis.population_compute.analyze_gate5_confirmation import (
    _smallest_noninferior_k,
    classify_gate5_confirmation,
)


def _base_lows() -> dict[str, float]:
    lows: dict[str, float] = {}
    for checkpoint in (0, 1, 2):
        lows[f"c{checkpoint}_bounded_score_k4_vs_global_score"] = -0.08
        lows[f"c{checkpoint}_bounded_score_k8_vs_global_score"] = -0.03
        lows[f"c{checkpoint}_bounded_score_k16_vs_global_score"] = -0.02
        lows[f"c{checkpoint}_bounded_score_k32_vs_global_score"] = -0.01
        lows[f"c{checkpoint}_bounded_score_k16_vs_bounded_hash_k16"] = 0.30
    return lows


class Gate5ConfirmationAuditTest(unittest.TestCase):
    def test_confirmed_requires_both_frozen_primary_conditions_on_all_checkpoints(self) -> None:
        self.assertEqual(
            classify_gate5_confirmation(_base_lows()),
            "GATE5_CONFIRMED_BOUNDED_SCORE_ACTIVATION",
        )

    def test_zero_learned_routing_lower_bound_does_not_pass(self) -> None:
        lows = _base_lows()
        lows["c1_bounded_score_k16_vs_bounded_hash_k16"] = 0.0
        self.assertEqual(
            classify_gate5_confirmation(lows),
            "GATE5_CONFIRMATION_NOT_ESTABLISHED",
        )

    def test_noninferiority_boundary_is_strict(self) -> None:
        lows = _base_lows()
        lows["c2_bounded_score_k16_vs_global_score"] = -0.05
        self.assertEqual(
            classify_gate5_confirmation(lows),
            "GATE5_CONFIRMATION_NOT_ESTABLISHED",
        )

    def test_descriptive_smallest_noninferior_k_can_be_eight(self) -> None:
        self.assertEqual(_smallest_noninferior_k(_base_lows()), 8)

    def test_descriptive_frontier_does_not_rescue_failed_primary_k16(self) -> None:
        lows = _base_lows()
        lows["c0_bounded_score_k16_vs_global_score"] = -0.06
        # K8 can remain descriptively acceptable on all checkpoints, but primary K16 fails.
        self.assertEqual(_smallest_noninferior_k(lows), 8)
        self.assertEqual(
            classify_gate5_confirmation(lows),
            "GATE5_CONFIRMATION_NOT_ESTABLISHED",
        )


if __name__ == "__main__":
    unittest.main()
