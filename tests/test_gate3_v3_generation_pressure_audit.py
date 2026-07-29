from __future__ import annotations

import unittest

from ai_hypothesis.population_compute.analyze_gate3_v3_generation_pressure import (
    _classify,
    _metric_key,
    _runtime_seed,
)


class Gate3V3AuditTests(unittest.TestCase):
    def _base_lows(self, frontier: tuple[float, float, float]) -> dict[str, float]:
        lows: dict[str, float] = {}
        for checkpoint in (0, 1, 2):
            lows[_metric_key(checkpoint, "stable_l256_vs_l64")] = frontier[checkpoint]
            lows[_metric_key(checkpoint, "stable_l64_vs_l16")] = 0.01
            lows[_metric_key(checkpoint, "stable_l256_vs_collapsed")] = 0.10
            lows[_metric_key(checkpoint, "stable_l256_vs_reshuffled")] = 0.10
        return lows

    def test_classifier_g0(self) -> None:
        self.assertEqual(
            _classify(self._base_lows((0.0, -0.01, 0.0))),
            "V3_G0_NO_L256_PRESSURE_BENEFIT",
        )

    def test_classifier_g1(self) -> None:
        self.assertEqual(
            _classify(self._base_lows((0.01, 0.02, 0.03))),
            "V3_G1_ROBUST_GENERATION_PRESSURE_BENEFIT",
        )

    def test_classifier_g2(self) -> None:
        self.assertEqual(
            _classify(self._base_lows((0.01, 0.0, 0.02))),
            "V3_G2_CHECKPOINT_SENSITIVE_PRESSURE_BENEFIT",
        )

    def test_classifier_g3_precedes_frontier_classification(self) -> None:
        lows = self._base_lows((0.01, 0.02, 0.03))
        lows[_metric_key(1, "stable_l256_vs_reshuffled")] = 0.0
        self.assertEqual(
            _classify(lows),
            "V3_G3_CONTROL_OR_MECHANISM_DEGRADATION",
        )

    def test_runtime_namespace_is_deterministic(self) -> None:
        self.assertEqual(_runtime_seed(7), _runtime_seed(7))
        self.assertNotEqual(_runtime_seed(7), _runtime_seed(8))


if __name__ == "__main__":
    unittest.main()
