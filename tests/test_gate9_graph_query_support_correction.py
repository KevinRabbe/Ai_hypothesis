from __future__ import annotations

import importlib.util
import pathlib
import sys
import unittest

CORRECTION_PATH = pathlib.Path(
    "ai_hypothesis/population_compute/gate9_graph_query_support_correction.py"
)
OPERATOR_PATH = pathlib.Path(
    "ai_hypothesis/population_compute/gate9_contextual_operator_contract.py"
)


def _load(path: pathlib.Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


c = _load(CORRECTION_PATH, "gate9_query_policy_correction_test_module")
o = _load(OPERATOR_PATH, "gate9_query_policy_operator_test_module")
p = c.protocol


class Gate9GraphQuerySupportCorrectionTests(unittest.TestCase):
    def test_policy_is_local_only_and_keeps_distribution_unconditioned(self):
        plan = c.corrected_gate9_query_policy_plan()
        self.assertEqual(plan["original_protocol_head"], c.GATE9_ORIGINAL_PROTOCOL_HEAD)
        self.assertEqual(plan["operator_contract_head"], c.GATE9_OPERATOR_CONTRACT_HEAD)
        self.assertTrue(plan["local_query_excludes_support_inputs"])
        self.assertFalse(plan["graph_query_excludes_support_inputs"])
        self.assertTrue(plan["graph_support_hit_report_required"])
        self.assertFalse(plan["graph_operator_rejection_on_support_hit"])
        self.assertFalse(plan["graph_operator_counter_skipping_allowed"])
        self.assertFalse(plan["graph_world_rejection_on_support_hit"])
        self.assertFalse(plan["graph_operator_distribution_conditioned_on_support_hits"])
        self.assertFalse(plan["thresholds_changed"])
        self.assertFalse(plan["condition_matrix_changed"])
        self.assertFalse(plan["operator_ranges_changed"])

    def test_admitted_affine_operator_can_create_a_graph_support_hit(self):
        identity_with_bias_three = o.operator_from_key(3 << 56)
        self.assertEqual(identity_with_bias_three.apply(3), 0)
        self.assertNotIn(3, p.GATE9_SUPPORT_INPUTS)
        self.assertIn(identity_with_bias_three.apply(3), p.GATE9_SUPPORT_INPUTS)
        support = o.public_support_pairs(identity_with_bias_three)
        self.assertEqual(
            o.apply_public_support_oracle(
                support,
                0,
                require_novel_query=False,
            ),
            3,
        )

    def test_support_hit_evidence_requires_exact_world_and_position_counts(self):
        row = c.Gate9GraphSupportHitEvidence(
            population=64,
            depth=8,
            worlds=256,
            total_path_queries=2_048,
            support_hits=36,
            support_hits_by_path_position=(4, 5, 3, 6, 4, 5, 4, 5),
        )
        row.validate()
        self.assertEqual(row.support_hit_rate, 36 / 2_048)
        self.assertEqual(row.to_dict()["support_hit_rate"], 36 / 2_048)

        wrong_total = c.Gate9GraphSupportHitEvidence(
            population=64,
            depth=8,
            worlds=256,
            total_path_queries=2_047,
            support_hits=36,
            support_hits_by_path_position=row.support_hits_by_path_position,
        )
        with self.assertRaisesRegex(ValueError, "query count"):
            wrong_total.validate()

        wrong_positions = c.Gate9GraphSupportHitEvidence(
            population=64,
            depth=8,
            worlds=256,
            total_path_queries=2_048,
            support_hits=36,
            support_hits_by_path_position=(4,) * 7,
        )
        with self.assertRaisesRegex(ValueError, "path-position vector"):
            wrong_positions.validate()

    def test_every_execution_boundary_remains_closed(self):
        plan = c.corrected_gate9_query_policy_plan()
        for key in (
            "operator_generation_admitted",
            "graph_world_generation_admitted",
            "architecture_admitted",
            "training_admitted",
            "checkpoint_loading_admitted",
            "scientific_test_admitted",
            "result_classification_admitted",
        ):
            self.assertFalse(plan[key])

        text = CORRECTION_PATH.read_text(encoding="utf-8")
        for token in (
            "import torch",
            "import numpy",
            "generate_gate9_world(",
            "operator_from_counter(",
            "open(",
            "read_text(",
            "json.load",
        ):
            self.assertNotIn(token, text)


if __name__ == "__main__":
    unittest.main()
