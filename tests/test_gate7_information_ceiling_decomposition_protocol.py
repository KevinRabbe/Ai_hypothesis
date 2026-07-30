from __future__ import annotations

import importlib.util
import pathlib
import sys
import unittest


MODULE_PATH = pathlib.Path(
    "ai_hypothesis/population_compute/gate7_information_ceiling_decomposition_protocol.py"
)
SPEC = importlib.util.spec_from_file_location("gate7_information_ceiling_protocol_test", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
p = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = p
SPEC.loader.exec_module(p)


class Gate7InformationCeilingProtocolTests(unittest.TestCase):
    def test_exact_result_and_execution_provenance_is_bound(self) -> None:
        self.assertEqual(
            p.GATE7_INFORMATION_CEILING_BASE_RESULT_HEAD,
            "4591dae55cada819e848ae7f929d5e8f2b8805d6",
        )
        self.assertEqual(
            p.GATE7_INFORMATION_CEILING_EXECUTION_HEAD,
            "19ee6b4e228c56b32a11b11b1c61b35bf640e2c8",
        )
        self.assertEqual(
            p.GATE7_INFORMATION_CEILING_RESULT_SHA256,
            "4921ea99b44156f08271d6fb2b2e0bcba98ef6a646ed0aaf040762d47aa03b36",
        )

    def test_complete_protocol_matrix_is_fixed(self) -> None:
        surface = p.protocol_surface()
        self.assertEqual(surface["populations"], (16_384, 32_768, 65_536, 131_072))
        self.assertEqual(surface["checkpoints"], (0, 1, 2))
        self.assertEqual(surface["world_count"], 512)
        self.assertEqual(surface["evaluation_batch_size"], 64)
        self.assertEqual(surface["bootstrap_samples"], 10_000)
        self.assertEqual(surface["primary_attempts"], 128)
        self.assertEqual(
            surface["attempt_ladder"],
            (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1_024),
        )
        self.assertFalse(surface["training_allowed"])
        self.assertFalse(surface["checkpoint_selection_allowed"])
        self.assertFalse(surface["communication_intervention_allowed"])
        self.assertFalse(surface["adaptive_attempt_exposure_allowed"])
        self.assertFalse(surface["result_reuse_as_fresh_evidence_allowed"])
        self.assertFalse(surface["execution_opened"])

    def test_frontier_depth_is_exact_log2_population(self) -> None:
        self.assertEqual(p.gate7_information_ceiling_frontier_depth(16_384), 14)
        self.assertEqual(p.gate7_information_ceiling_frontier_depth(131_072), 17)
        with self.assertRaisesRegex(ValueError, "power of two"):
            p.gate7_information_ceiling_frontier_depth(12_345)

    def test_exact_bayes_top128_expectations_are_frozen(self) -> None:
        expected = p.expected_primary_ceiling_by_population()
        self.assertAlmostEqual(expected[16_384], 0.1725810781290399, places=15)
        self.assertAlmostEqual(expected[32_768], 0.12944371790375186, places=15)
        self.assertAlmostEqual(expected[65_536], 0.09386607328230152, places=15)
        self.assertAlmostEqual(expected[131_072], 0.06627595867880422, places=15)
        self.assertGreater(expected[16_384], expected[32_768])
        self.assertGreater(expected[32_768], expected[65_536])
        self.assertGreater(expected[65_536], expected[131_072])

    def test_bayes_attempt_curve_is_monotonic(self) -> None:
        for population in p.GATE7_INFORMATION_CEILING_POPULATIONS:
            curve = tuple(
                p.bayes_expected_top_m_coverage(population=population, attempts=attempts)
                for attempts in p.GATE7_INFORMATION_CEILING_ATTEMPT_LADDER
            )
            self.assertEqual(curve, tuple(sorted(curve)))
            self.assertTrue(all(0.0 < value < 1.0 for value in curve))

    def test_prepared_tiers_preserve_all_rankers_and_attempts(self) -> None:
        tiers = p.prepared_information_ceiling_tiers()
        self.assertEqual(tuple(tier.population for tier in tiers), p.GATE7_INFORMATION_CEILING_POPULATIONS)
        for tier in tiers:
            tier.validate()
            self.assertEqual(tier.rankers, p.GATE7_INFORMATION_CEILING_RANKERS)
            self.assertEqual(tier.attempt_ladder, p.GATE7_INFORMATION_CEILING_ATTEMPT_LADDER)

    def _matrix(self, *, low: float, high: float) -> tuple:
        return tuple(
            p.Gate7InformationCeilingComparison(
                checkpoint_index=checkpoint,
                population=population,
                learned_minus_bayes_ci_low=low,
                learned_minus_bayes_ci_high=high,
                learned_minus_hash_ci_low=0.01,
                bayes_minus_hash_ci_low=0.01,
            )
            for population in p.GATE7_INFORMATION_CEILING_POPULATIONS
            for checkpoint in p.GATE7_INFORMATION_CEILING_CHECKPOINT_INDICES
        )

    def test_all_near_ceiling_rows_select_information_ceiling(self) -> None:
        result = p.classify_information_ceiling_campaign(self._matrix(low=-0.01, high=0.01))
        self.assertEqual(result, p.GATE7_INFORMATION_CEILING_DOMINANT)

    def test_all_clear_gap_rows_select_scorer_gap(self) -> None:
        result = p.classify_information_ceiling_campaign(self._matrix(low=-0.05, high=-0.03))
        self.assertEqual(result, p.GATE7_INFORMATION_CEILING_SCORER_GAP)

    def test_mixed_rows_select_mixed_outcome(self) -> None:
        rows = list(self._matrix(low=-0.01, high=0.01))
        first = rows[0]
        rows[0] = p.Gate7InformationCeilingComparison(
            checkpoint_index=first.checkpoint_index,
            population=first.population,
            learned_minus_bayes_ci_low=-0.05,
            learned_minus_bayes_ci_high=-0.03,
            learned_minus_hash_ci_low=0.01,
            bayes_minus_hash_ci_low=0.01,
        )
        self.assertEqual(
            p.classify_information_ceiling_campaign(tuple(rows)),
            p.GATE7_INFORMATION_CEILING_MIXED,
        )

    def test_incomplete_or_reordered_matrix_is_rejected(self) -> None:
        rows = self._matrix(low=-0.01, high=0.01)
        with self.assertRaisesRegex(ValueError, "exact population-major"):
            p.classify_information_ceiling_campaign(rows[:-1])
        with self.assertRaisesRegex(ValueError, "exact population-major"):
            p.classify_information_ceiling_campaign(tuple(reversed(rows)))


if __name__ == "__main__":
    unittest.main()
