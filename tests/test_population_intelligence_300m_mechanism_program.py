from __future__ import annotations

import pathlib
import unittest

from ai_hypothesis.population_language import intelligence_300m_mechanism_program as program


ROOT = pathlib.Path(__file__).resolve().parents[1]
DOCUMENT = ROOT / "docs" / "population_intelligence_300m_mechanism_program.md"


class PopulationIntelligence300MMechanismProgramContracts(unittest.TestCase):
    def test_program_validates_and_pins_the_roadmap_head(self) -> None:
        snapshot = program.validate_program()
        self.assertTrue(snapshot["valid"])
        self.assertEqual(
            snapshot["source_roadmap_head"],
            "eba20a82710b02092bf5c44b91247c3d277b5694",
        )
        self.assertEqual(snapshot["primary_target_parameters"], 300_000_000)
        self.assertEqual(
            snapshot["status"],
            "MECHANISM_PROGRAM_ONLY_NO_EXPERIMENT_OR_ARCHITECTURE_FREEZE",
        )

    def test_scale_ladder_and_capability_modes_are_exact(self) -> None:
        self.assertEqual(
            program.MODEL_STAGES,
            (
                ("diagnostic", 19_000_000),
                ("integration", 50_000_000),
                ("language_and_code", 100_000_000),
                ("primary", 300_000_000),
            ),
        )
        self.assertEqual(
            program.CAPABILITY_MODES,
            (
                "ONE_PASS_CLOSED_BOOK",
                "RECURSIVE_POPULATION",
                "FULL_SYSTEM",
            ),
        )

    def test_mechanism_lane_order_and_dependencies_are_exact(self) -> None:
        identifiers = tuple(lane.identifier for lane in program.MECHANISM_LANES)
        self.assertEqual(identifiers, tuple(f"M{index:02d}" for index in range(1, 11)))
        expected_dependencies = {
            "M01": (),
            "M02": (),
            "M03": ("M01", "M02"),
            "M04": (),
            "M05": (),
            "M06": ("M05",),
            "M07": ("M02",),
            "M08": ("M05",),
            "M09": ("M01", "M03", "M07"),
            "M10": ("M04",),
        }
        self.assertEqual(
            {lane.identifier: lane.dependencies for lane in program.MECHANISM_LANES},
            expected_dependencies,
        )
        with self.assertRaises(KeyError):
            program.lane_by_id("M11")

    def test_each_lane_has_controls_metrics_and_claim_boundaries(self) -> None:
        for lane in program.MECHANISM_LANES:
            with self.subTest(lane=lane.identifier):
                lane.validate()
                self.assertGreaterEqual(len(lane.required_controls), 4)
                self.assertGreaterEqual(len(lane.primary_metrics), 4)
                self.assertNotEqual(lane.allowed_claim, lane.forbidden_claim)
                self.assertNotIn("UNBOUNDED", lane.allowed_claim)

    def test_promotion_requires_prospective_causal_matched_evidence(self) -> None:
        required = {
            "PROSPECTIVE_PREREGISTRATION",
            "PROTOCOL_SPECIFIC_SUCCESS_THRESHOLD_PASSED",
            "MATCHED_DENSE_OR_RECURRENT_CONTROL",
            "MATCHED_COMPUTE_CONTROL",
            "CAUSAL_ABLATION",
            "END_TO_END_COST_ACCOUNTING",
            "REPLICATION_ACROSS_PREDECLARED_SEEDS",
            "KNOWN_FAILURE_REGION_RECORDED",
            "CLAIM_RESTRICTED_TO_OBSERVED_SCALE_AND_TASKS",
        }
        self.assertEqual(set(program.PROMOTION_REQUIREMENTS), required)
        self.assertIn(
            "THE_100M_RESULT_SELECTS_THE_300M_DESIGN",
            program.ARCHITECTURE_FREEZE_REQUIREMENTS,
        )
        self.assertIn(
            "NO_EDGE_RUNTIME_CONSTRAINT_MAY_SELECT_THE_INTELLIGENCE_ARCHITECTURE",
            program.ARCHITECTURE_FREEZE_REQUIREMENTS,
        )

    def test_interactive_world_and_edge_runtime_remain_deferred(self) -> None:
        self.assertEqual(
            program.LATER_INTEGRATION_BENCHMARK,
            "DETERMINISTIC_INTERACTIVE_WORLD",
        )
        self.assertNotIn(
            program.LATER_INTEGRATION_BENCHMARK,
            program.BENCHMARK_FAMILIES,
        )
        self.assertEqual(program.SEPARATE_LATER_PROJECT, "POPULATION_EDGE_RUNTIME")

    def test_parameter_allocation_is_not_frozen_by_preference(self) -> None:
        self.assertEqual(
            program.PARAMETER_ALLOCATION_DOMAINS,
            (
                "LEXICAL_ENCODER_DECODER",
                "RECURRENT_POPULATION_CORE",
                "ROUTING_AND_COMMUNICATION",
                "VERIFIER_AND_VALUE_SYSTEM",
                "CONDITIONAL_OR_PERSISTENT_MEMORY",
                "POST_TRAINING_PLASTICITY",
            ),
        )
        snapshot = program.validate_program()
        self.assertTrue(
            snapshot["checks"]["parameter_allocation_remains_evidence_driven"]
        )

    def test_document_mirrors_the_machine_readable_program(self) -> None:
        text = DOCUMENT.read_text(encoding="utf-8")
        for required in (
            "smartest population-based language model",
            "approximately 300 million learned parameters",
            "ONE_PASS_CLOSED_BOOK",
            "RECURSIVE_POPULATION",
            "FULL_SYSTEM",
            "M01 — Recurrent latent depth",
            "M02 — Diversity and private deliberation",
            "M03 — Adaptive test-time computation",
            "M04 — Verifier-guided generation and repair",
            "M05 — Memory versus learning separation",
            "M06 — Sequential continual learning",
            "M07 — Hierarchical communication",
            "M08 — Conditional memory allocation",
            "M09 — Scale-stable parameterization",
            "M10 — Verified search distillation",
            "Only promoted mechanisms may enter",
            "The approximately 100M model",
            "Population Edge Runtime",
            "Material changes require a dedicated versioned proposal",
        ):
            with self.subTest(required=required):
                self.assertIn(required, text)

        self.assertLess(
            text.index("approximately 19M diagnostic models"),
            text.index("approximately 50M mechanism integration"),
        )
        self.assertLess(
            text.index("approximately 50M mechanism integration"),
            text.index("approximately 100M realistic language-and-code evidence"),
        )
        self.assertLess(
            text.index("approximately 100M realistic language-and-code evidence"),
            text.index("strongest justified approximately 300M model"),
        )


if __name__ == "__main__":
    unittest.main()
