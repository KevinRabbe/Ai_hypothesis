from __future__ import annotations

import unittest

from ai_hypothesis.population_compute.analyze_gate3_v2_frontier import (
    CHECKPOINTS,
    TIERS,
    _classify,
    _metric_key,
)


class Gate3V2FrontierClassifierTest(unittest.TestCase):
    def _base(self) -> tuple[dict[str, float], dict[str, float]]:
        lows: dict[str, float] = {}
        deltas: dict[str, float] = {}
        for checkpoint in CHECKPOINTS:
            for tier in TIERS:
                for comparison in (
                    "stable_l256_vs_l64",
                    "stable_l64_vs_l16",
                    "stable_l256_vs_l1",
                    "stable_l256_vs_collapsed",
                    "stable_l256_vs_reshuffled",
                ):
                    key = _metric_key(checkpoint, tier, comparison)
                    lows[key] = 0.05
                    deltas[key] = 0.10
        return lows, deltas

    def test_f2_clean_extension_both_tiers(self) -> None:
        lows, deltas = self._base()
        self.assertEqual(_classify(lows, deltas), "V2_F2_ROBUST_BEYOND_L64_EXTENSION")

    def test_f1_extension_only_a55(self) -> None:
        lows, deltas = self._base()
        for checkpoint in CHECKPOINTS:
            key = _metric_key(checkpoint, "A60", "stable_l256_vs_l64")
            lows[key] = 0.0
            deltas[key] = 0.0
        self.assertEqual(_classify(lows, deltas), "V2_F1_EXTENSION_AT_A55_ONLY")

    def test_f0_no_extension(self) -> None:
        lows, deltas = self._base()
        for checkpoint in CHECKPOINTS:
            for tier in TIERS:
                key = _metric_key(checkpoint, tier, "stable_l256_vs_l64")
                lows[key] = 0.0
                deltas[key] = 0.0
        self.assertEqual(_classify(lows, deltas), "V2_F0_NO_BEYOND_L64_EXTENSION")

    def test_f3_checkpoint_sensitive_frontier(self) -> None:
        lows, deltas = self._base()
        for checkpoint in CHECKPOINTS:
            key = _metric_key(checkpoint, "A60", "stable_l256_vs_l64")
            lows[key] = 0.0
            deltas[key] = 0.0
        mixed = _metric_key(2, "A55", "stable_l256_vs_l64")
        lows[mixed] = 0.0
        deltas[mixed] = 0.01
        self.assertEqual(_classify(lows, deltas), "V2_F3_CHECKPOINT_SENSITIVE_FRONTIER")

    def test_f4_control_separation_loss_takes_precedence(self) -> None:
        lows, deltas = self._base()
        key = _metric_key(1, "A55", "stable_l256_vs_reshuffled")
        lows[key] = 0.0
        deltas[key] = 0.01
        self.assertEqual(_classify(lows, deltas), "V2_F4_MECHANISM_DEGRADES_UNDER_AMBIGUITY")


if __name__ == "__main__":
    unittest.main()
