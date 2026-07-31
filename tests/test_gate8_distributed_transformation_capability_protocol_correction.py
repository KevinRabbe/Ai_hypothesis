from __future__ import annotations

import importlib.util
import pathlib
import sys
import unittest


CORRECTION_PATH = pathlib.Path(
    "ai_hypothesis/population_compute/"
    "gate8_distributed_transformation_capability_protocol_correction.py"
)


def _load_correction():
    name = "gate8_distributed_transformation_capability_protocol_correction_test_module"
    spec = importlib.util.spec_from_file_location(name, CORRECTION_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load Gate8 correction contract")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


c = _load_correction()


class Gate8ProtocolCorrectionTests(unittest.TestCase):
    def test_all_21_conditions_preserve_path_and_fraction_invariants(self) -> None:
        rows = c.gate8_relevant_edge_contracts()
        self.assertEqual(len(rows), 21)
        self.assertEqual(
            tuple((row.population, row.depth) for row in rows),
            c.GATE8_CORRECTION_VALID_CONDITIONS,
        )
        for row in rows:
            self.assertEqual(row.relevant_path_edges, row.depth)
            self.assertEqual(row.total_edges, row.population)
            self.assertEqual(row.distractor_edges, row.population - row.depth)
            self.assertLessEqual(row.relevant_fraction, c.GATE8_RELEVANT_EDGE_FRACTION_BOUND)

    def test_bound_is_tight_only_on_equality_conditions(self) -> None:
        tight = tuple(
            (row.population, row.depth)
            for row in c.gate8_relevant_edge_contracts()
            if row.bound_is_tight
        )
        self.assertEqual(
            tight,
            ((32, 4), (64, 8), (128, 16), (256, 32), (512, 64), (1024, 128)),
        )

    def test_invalid_condition_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "outside the frozen matrix"):
            c.Gate8RelevantEdgeContract(population=32, depth=8).validate()

    def test_plan_binds_original_protocol_and_keeps_execution_closed(self) -> None:
        plan = c.gate8_protocol_correction_plan()
        self.assertEqual(
            plan["base_protocol_head"],
            "e73541115e8ddd122f336463dc1a9ffdbf82df46",
        )
        self.assertEqual(c.GATE8_PROTOCOL_CORRECTION_BASE_HEAD, plan["base_protocol_head"])
        self.assertEqual(
            c.GATE8_PROTOCOL_CORRECTION_STATUS,
            "DATA_FROZEN_GATE8_RELEVANT_EDGE_RULE_CORRECTED_EXECUTION_CLOSED",
        )
        self.assertEqual(plan["condition_count"], 21)
        self.assertFalse(plan["execution_admitted"])
        self.assertFalse(plan["generator_admitted"])
        self.assertFalse(plan["training_admitted"])
        self.assertFalse(plan["baseline_execution_admitted"])

    def test_correction_module_has_no_execution_surface(self) -> None:
        text = CORRECTION_PATH.read_text(encoding="utf-8")
        for token in (
            "import torch",
            "import numpy",
            "from .",
            "open(",
            "read_text(",
            "json.load",
            "transformers",
            "AutoModel",
            "generate_world",
            "execution_admitted\": True",
        ):
            self.assertNotIn(token, text)


if __name__ == "__main__":
    unittest.main()
