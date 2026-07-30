from __future__ import annotations

import unittest

from ai_hypothesis.population_compute.analyze_gate6_fixed_k_population_scaling import classify_gate6


def _metric(checkpoint: int, population: int, comparison: str) -> str:
    return f"c{checkpoint}_n{population}_{comparison}"


def _maps() -> tuple[dict[str, float], dict[str, float]]:
    lows: dict[str, float] = {}
    highs: dict[str, float] = {}
    for checkpoint in (0, 1, 2):
        for population in (64, 128, 256):
            learned = _metric(checkpoint, population, "bounded_score_k16_vs_bounded_hash_k16")
            global_gap = _metric(checkpoint, population, "bounded_score_k16_vs_global_score")
            descriptive = _metric(checkpoint, population, "bounded_score_k8_vs_global_score")
            lows[learned] = 0.08
            highs[learned] = 0.18
            lows[global_gap] = -0.02
            highs[global_gap] = 0.02
            lows[descriptive] = -0.10
            highs[descriptive] = 0.02
    return lows, highs


class Gate6IndependentClassifierTests(unittest.TestCase):
    def test_s2(self) -> None:
        lows, highs = _maps()
        self.assertEqual(classify_gate6(lows, highs), "G6_S2_ROBUST_FIXED_K_POPULATION_SCALING")

    def test_strict_noninferiority_boundary(self) -> None:
        lows, highs = _maps()
        for checkpoint in (0, 1, 2):
            lows[_metric(checkpoint, 256, "bounded_score_k16_vs_global_score")] = -0.05
        self.assertEqual(classify_gate6(lows, highs), "G6_S1_FIXED_K_DEGRADES_WITH_POPULATION")

    def test_s3_precedes_uniform_degradation_when_checkpoint_mixed(self) -> None:
        lows, highs = _maps()
        lows[_metric(2, 128, "bounded_score_k16_vs_global_score")] = -0.08
        self.assertEqual(classify_gate6(lows, highs), "G6_S3_CHECKPOINT_SENSITIVE_SCALING")

    def test_s4_harmful_precedence(self) -> None:
        lows, highs = _maps()
        key = _metric(1, 256, "bounded_score_k16_vs_bounded_hash_k16")
        lows[key] = -0.15
        highs[key] = -0.01
        self.assertEqual(classify_gate6(lows, highs), "G6_S4_BOUNDED_ROUTING_HARMFUL_AT_SCALE")


if __name__ == "__main__":
    unittest.main()
