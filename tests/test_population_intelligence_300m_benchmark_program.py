from __future__ import annotations

import pathlib
import unittest

from ai_hypothesis.population_language import intelligence_300m_benchmark_program as program
from ai_hypothesis.population_language import intelligence_300m_mechanism_program as mechanisms


ROOT = pathlib.Path(__file__).resolve().parents[1]
DOCUMENT = ROOT / "docs" / "population_intelligence_300m_benchmark_program.md"


class PopulationIntelligence300MBenchmarkProgramContracts(unittest.TestCase):
    def test_program_validates_and_pins_the_mechanism_head(self) -> None:
        snapshot = program.validate_program()
        self.assertTrue(snapshot["valid"])
        self.assertEqual(
            snapshot["source_mechanism_head"],
            "9cc8b1b4c1dc4fc95df346ef82b188e276d97fcb",
        )
        self.assertEqual(
            snapshot["status"],
            "BENCHMARK_PROGRAM_ONLY_NO_GENERATION_OR_PROTECTED_RESULT_ACCESS",
        )
        self.assertEqual(snapshot["minimum_initialization_seeds"], 3)

    def test_split_roles_and_protection_are_exact(self) -> None:
        self.assertEqual(
            program.SPLIT_ROLES,
            (
                ("development", "VISIBLE_FOR_IMPLEMENTATION_AND_DEBUGGING"),
                ("calibration", "VISIBLE_ONLY_FOR_PREDECLARED_SELECTION"),
                ("validation", "PROTECTED_UNTIL_ARCHITECTURE_AND_SELECTION_FREEZE"),
                ("test", "PROTECTED_UNTIL_FINAL_EXECUTION_AUTHORIZATION"),
            ),
        )
        self.assertEqual(program.DESIGN_SPLITS, ("development", "calibration"))
        self.assertEqual(program.PROTECTED_SPLITS, ("validation", "test"))

    def test_capability_modes_match_the_mechanism_program(self) -> None:
        self.assertEqual(program.CAPABILITY_MODES, mechanisms.CAPABILITY_MODES)
        self.assertEqual(
            program.CAPABILITY_MODES,
            (
                "ONE_PASS_CLOSED_BOOK",
                "RECURSIVE_POPULATION",
                "FULL_SYSTEM",
            ),
        )

    def test_benchmark_order_and_mechanism_bindings_are_exact(self) -> None:
        identifiers = tuple(family.identifier for family in program.BENCHMARK_FAMILIES)
        self.assertEqual(identifiers, tuple(f"B{index:02d}" for index in range(1, 9)))
        expected = {
            "B01": ("M01", "M02", "M03", "M05", "M06"),
            "B02": ("M01", "M03", "M07", "M09"),
            "B03": ("M01", "M02", "M03", "M07"),
            "B04": ("M04", "M10"),
            "B05": ("M04", "M10"),
            "B06": ("M05", "M06", "M08"),
            "B07": ("M05", "M08"),
            "B08": ("M01", "M02", "M03", "M09"),
        }
        self.assertEqual(
            {family.identifier: family.mechanism_lanes for family in program.BENCHMARK_FAMILIES},
            expected,
        )
        with self.assertRaises(KeyError):
            program.benchmark_by_id("B09")

    def test_each_benchmark_has_exact_oracle_controls_and_failure_slices(self) -> None:
        for family in program.BENCHMARK_FAMILIES:
            with self.subTest(benchmark=family.identifier):
                family.validate()
                self.assertGreaterEqual(len(family.difficulty_axes), 2)
                self.assertGreaterEqual(len(family.required_metrics), 4)
                self.assertGreaterEqual(len(family.required_baselines), 4)
                self.assertGreaterEqual(len(family.required_failure_slices), 4)
                self.assertTrue(family.exact_oracle)
                self.assertNotEqual(family.allowed_claim, family.forbidden_claim)

    def test_decisive_evidence_rules_block_contamination_and_metric_drift(self) -> None:
        required = {
            "PROCEDURAL_OR_HELD_OUT_INSTANCES_ARE_PRIMARY",
            "PUBLIC_BENCHMARKS_ARE_SUPPORTIVE_NOT_SOLE_DECISION_EVIDENCE",
            "INSTANCE_GENERATION_AND_SCORING_ARE_VERSIONED",
            "DEVELOPMENT_CALIBRATION_VALIDATION_AND_TEST_SEEDS_ARE_DISJOINT",
            "VALIDATION_AND_TEST_LABELS_ARE_UNAVAILABLE_DURING_DESIGN",
            "NO_THRESHOLD_IS_CHANGED_AFTER_PROTECTED_RESULT_ACCESS",
            "ALL_CAPABILITY_MODES_ARE_REPORTED_SEPARATELY",
            "EXACT_ORACLES_OVERRIDE_LEARNED_JUDGES_WHERE_AVAILABLE",
            "FAILURES_TIMEOUTS_AND_INVALID_OUTPUTS_COUNT_AGAINST_RESULTS",
            "NEGATIVE_AND_NULL_RESULTS_ARE_PUBLISHED",
        }
        self.assertEqual(set(program.DECISIVE_EVIDENCE_RULES), required)

    def test_global_result_schema_accounts_for_compute_memory_and_tools(self) -> None:
        self.assertEqual(len(program.GLOBAL_RESULT_FIELDS), 25)
        for field in (
            "total_learned_parameters",
            "active_parameters",
            "worker_count",
            "recurrent_rounds",
            "inference_flops",
            "routed_bytes",
            "retrieved_bytes",
            "persisted_bytes",
            "verifier_calls",
            "tool_calls",
            "peak_vram_bytes",
            "artifact_provenance",
        ):
            self.assertIn(field, program.GLOBAL_RESULT_FIELDS)

    def test_interactive_world_benchmarks_remain_later(self) -> None:
        self.assertEqual(
            program.LATER_INTEGRATION_BENCHMARKS,
            (
                "STRUCTURED_DETERMINISTIC_RPG",
                "SCREEN_AND_EXTRACTED_TEXT_RPG",
                "PIXELS_AND_CONTROLLER_RPG",
                "RANDOMIZED_OR_PROCEDURAL_RPG",
            ),
        )
        benchmark_titles = {family.title for family in program.BENCHMARK_FAMILIES}
        self.assertTrue(all(name not in benchmark_titles for name in program.LATER_INTEGRATION_BENCHMARKS))

    def test_document_mirrors_the_machine_readable_program(self) -> None:
        text = DOCUMENT.read_text(encoding="utf-8")
        for required in (
            "ONE_PASS_CLOSED_BOOK",
            "RECURSIVE_POPULATION",
            "FULL_SYSTEM",
            "B01 — Procedural compositional rule induction",
            "B02 — Algorithmic state tracking",
            "B03 — Graph planning and counterfactual search",
            "B04 — Verified code synthesis",
            "B05 — Code diagnosis and repair",
            "B06 — Changing organization memory and procedure learning",
            "B07 — Conditional memory capacity allocation",
            "B08 — Adaptive compute challenge set",
            "Public benchmarks may provide supportive context",
            "Validation and test labels",
            "Negative and null results",
            "Physical robotics and Population Edge Runtime remain outside",
            "Material changes require a dedicated versioned proposal",
        ):
            with self.subTest(required=required):
                self.assertIn(required, text)

        self.assertLess(text.index("development |"), text.index("calibration |"))
        self.assertLess(text.index("calibration |"), text.index("validation |"))
        self.assertLess(text.index("validation |"), text.index("test |"))


if __name__ == "__main__":
    unittest.main()
