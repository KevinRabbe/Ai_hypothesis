from __future__ import annotations

import unittest

from ai_hypothesis.population_compute.analyze_gate5_bounded_score_activation import (
    NONINFERIORITY_MARGIN,
    _metric_key,
    _smallest_noninferior_k,
    classify_gate5,
)


def _maps(*, learned_lows: list[float], learned_highs: list[float], ni_lows: list[float]) -> tuple[dict[str, float], dict[str, float]]:
    lows: dict[str, float] = {}
    highs: dict[str, float] = {}
    for checkpoint in range(3):
        learned_key = _metric_key(checkpoint, "bounded_score_k16_vs_bounded_hash_k16")
        ni_key = _metric_key(checkpoint, "bounded_score_k16_vs_global_score")
        lows[learned_key] = learned_lows[checkpoint]
        highs[learned_key] = learned_highs[checkpoint]
        lows[ni_key] = ni_lows[checkpoint]
        highs[ni_key] = ni_lows[checkpoint] + 0.04
    return lows, highs


class Gate5AuditClassifierTests(unittest.TestCase):
    def test_b2_requires_learned_signal_and_noninferiority_everywhere(self) -> None:
        lows, highs = _maps(
            learned_lows=[0.05, 0.04, 0.03],
            learned_highs=[0.10, 0.09, 0.08],
            ni_lows=[-0.02, -0.01, 0.00],
        )
        self.assertEqual(
            classify_gate5(lows, highs),
            "G5_B2_ROBUST_BOUNDED_SCORE_ACTIVATION",
        )

    def test_b4_has_highest_precedence(self) -> None:
        lows, highs = _maps(
            learned_lows=[-0.12, 0.04, 0.03],
            learned_highs=[-0.01, 0.09, 0.08],
            ni_lows=[-0.02, -0.01, 0.00],
        )
        self.assertEqual(
            classify_gate5(lows, highs),
            "G5_B4_BOUNDED_LEARNED_ROUTING_HARMFUL",
        )

    def test_b3_catches_checkpoint_sensitive_noninferiority(self) -> None:
        lows, highs = _maps(
            learned_lows=[0.05, 0.04, 0.03],
            learned_highs=[0.10, 0.09, 0.08],
            ni_lows=[-0.02, -0.08, -0.01],
        )
        self.assertEqual(
            classify_gate5(lows, highs),
            "G5_B3_CHECKPOINT_SENSITIVE_BOUNDED_EFFECT",
        )

    def test_b1_is_uniform_learned_signal_with_uniform_global_gap(self) -> None:
        lows, highs = _maps(
            learned_lows=[0.05, 0.04, 0.03],
            learned_highs=[0.10, 0.09, 0.08],
            ni_lows=[-0.08, -0.07, -0.06],
        )
        self.assertEqual(
            classify_gate5(lows, highs),
            "G5_B1_LEARNED_SIGNAL_WITH_GLOBAL_GAP",
        )

    def test_b0_is_no_learned_signal(self) -> None:
        lows, highs = _maps(
            learned_lows=[-0.01, -0.02, -0.01],
            learned_highs=[0.04, 0.03, 0.04],
            ni_lows=[-0.03, -0.02, -0.01],
        )
        self.assertEqual(
            classify_gate5(lows, highs),
            "G5_B0_BOUNDED_LEARNED_ROUTING_NOT_ESTABLISHED",
        )

    def test_noninferiority_boundary_is_strict(self) -> None:
        lows, highs = _maps(
            learned_lows=[0.05, 0.04, 0.03],
            learned_highs=[0.10, 0.09, 0.08],
            ni_lows=[-NONINFERIORITY_MARGIN, -0.01, 0.00],
        )
        self.assertEqual(
            classify_gate5(lows, highs),
            "G5_B3_CHECKPOINT_SENSITIVE_BOUNDED_EFFECT",
        )

    def test_smallest_noninferior_k_is_reconstructed(self) -> None:
        lows: dict[str, float] = {}
        for checkpoint in range(3):
            lows[_metric_key(checkpoint, "bounded_score_k4_vs_global_score")] = -0.09
            lows[_metric_key(checkpoint, "bounded_score_k8_vs_global_score")] = -0.06
            lows[_metric_key(checkpoint, "bounded_score_k16_vs_global_score")] = -0.04
            lows[_metric_key(checkpoint, "bounded_score_k32_vs_global_score")] = -0.01
        self.assertEqual(_smallest_noninferior_k(lows), 16)


if __name__ == "__main__":
    unittest.main()
