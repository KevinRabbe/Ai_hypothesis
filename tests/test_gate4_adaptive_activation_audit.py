from __future__ import annotations

import unittest

from ai_hypothesis.population_compute.analyze_gate4_adaptive_activation import _classify


class Gate4AdaptiveActivationAuditTests(unittest.TestCase):
    def _maps(self) -> tuple[dict[str, float], dict[str, float]]:
        lows: dict[str, float] = {}
        highs: dict[str, float] = {}
        for checkpoint in (0, 1, 2):
            for comparison in (
                "adaptive_score_vs_static_generation",
                "adaptive_score_vs_adaptive_hash",
                "static_generation_vs_adaptive_hash",
            ):
                key = f"c{checkpoint}_{comparison}"
                lows[key] = -0.01
                highs[key] = 0.01
        return lows, highs

    def test_a2_requires_primary_and_routing_positive_for_all_checkpoints(self) -> None:
        lows, highs = self._maps()
        for checkpoint in (0, 1, 2):
            lows[f"c{checkpoint}_adaptive_score_vs_static_generation"] = 0.05
            highs[f"c{checkpoint}_adaptive_score_vs_static_generation"] = 0.10
            lows[f"c{checkpoint}_adaptive_score_vs_adaptive_hash"] = 0.04
            highs[f"c{checkpoint}_adaptive_score_vs_adaptive_hash"] = 0.09
        self.assertEqual(_classify(lows, highs), "G4_A2_ROBUST_ADAPTIVE_ACTIVATION_BENEFIT")

    def test_a1_precedes_a0_when_routing_signal_is_positive(self) -> None:
        lows, highs = self._maps()
        for checkpoint in (0, 1, 2):
            lows[f"c{checkpoint}_adaptive_score_vs_static_generation"] = -0.01
            highs[f"c{checkpoint}_adaptive_score_vs_static_generation"] = 0.02
            lows[f"c{checkpoint}_adaptive_score_vs_adaptive_hash"] = 0.03
            highs[f"c{checkpoint}_adaptive_score_vs_adaptive_hash"] = 0.08
        self.assertEqual(_classify(lows, highs), "G4_A1_ROUTING_SIGNAL_ONLY")

    def test_a3_is_checkpoint_sensitive_primary_effect(self) -> None:
        lows, highs = self._maps()
        lows["c0_adaptive_score_vs_static_generation"] = 0.03
        highs["c0_adaptive_score_vs_static_generation"] = 0.08
        lows["c1_adaptive_score_vs_static_generation"] = -0.01
        highs["c1_adaptive_score_vs_static_generation"] = 0.03
        lows["c2_adaptive_score_vs_static_generation"] = 0.02
        highs["c2_adaptive_score_vs_static_generation"] = 0.07
        self.assertEqual(_classify(lows, highs), "G4_A3_CHECKPOINT_SENSITIVE_ADAPTIVE_EFFECT")

    def test_a0_requires_no_primary_significance_and_no_universal_routing_signal(self) -> None:
        lows, highs = self._maps()
        self.assertEqual(_classify(lows, highs), "G4_A0_NO_ADAPTIVE_ALLOCATION_BENEFIT")

    def test_a4_has_highest_precedence(self) -> None:
        lows, highs = self._maps()
        for checkpoint in (0, 1, 2):
            lows[f"c{checkpoint}_adaptive_score_vs_static_generation"] = 0.05
            highs[f"c{checkpoint}_adaptive_score_vs_static_generation"] = 0.10
            lows[f"c{checkpoint}_adaptive_score_vs_adaptive_hash"] = 0.04
            highs[f"c{checkpoint}_adaptive_score_vs_adaptive_hash"] = 0.09
        lows["c1_adaptive_score_vs_adaptive_hash"] = -0.10
        highs["c1_adaptive_score_vs_adaptive_hash"] = -0.02
        self.assertEqual(_classify(lows, highs), "G4_A4_LEARNED_ROUTING_HARMFUL")


if __name__ == "__main__":
    unittest.main()
