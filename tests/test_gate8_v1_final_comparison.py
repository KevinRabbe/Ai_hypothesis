from __future__ import annotations

import importlib.util
import pathlib
import sys
import unittest
from collections import OrderedDict

import numpy as np

CONTRACT_PATH = pathlib.Path(
    "ai_hypothesis/population_compute/gate8_v1_final_comparison.py"
)
RUNNER_PATH = pathlib.Path("scripts/run_gate8_v1_final_comparison.py")


def load_contract():
    name = "gate8_v1_final_comparison_test_module"
    spec = importlib.util.spec_from_file_location(name, CONTRACT_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load Gate8 v1 final-comparison contract")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


c = load_contract()


class Gate8V1FinalComparisonTests(unittest.TestCase):
    def test_plan_binds_exact_results_and_keeps_execution_closed(self):
        plan = c.final_comparison_plan()
        self.assertEqual(plan["gemma_result_head"], "1d48ecfd623a2fb9e3a2f846a4d1c49d20d8cadc")
        self.assertEqual(plan["population_result_head"], "14636d219781381853f81036b96c691b7e6997ee")
        self.assertEqual(plan["scientific_protocol_head"], "6bb89111a47713bea0a23bb1cae662ed5ec56b42")
        self.assertEqual(plan["conditions"], [list(row) for row in c.GATE8_V1_VALID_CONDITIONS])
        self.assertEqual(plan["worlds_per_condition"], 512)
        self.assertEqual(plan["population_seeds"], [0, 1, 2])
        self.assertEqual(plan["bootstrap_samples"], 20_000)
        self.assertFalse(plan["model_loading_admitted"])
        self.assertFalse(plan["world_generation_admitted"])
        self.assertFalse(plan["population_execution_admitted"])
        self.assertFalse(plan["reference_inference_admitted"])
        self.assertFalse(plan["training_admitted"])

    def test_condition_comparison_is_paired_across_all_systems(self):
        population = np.ones((3, 512), dtype=np.uint8)
        reference = np.zeros(512, dtype=np.uint8)
        row, vector = c.condition_comparison(
            population=32,
            depth=4,
            population_correctness=population,
            reference_correctness=reference,
            maximum_reference_input_tokens=2801,
        )
        self.assertTrue(np.array_equal(vector, np.ones(512)))
        self.assertEqual(row["population_seed_accuracies"], [1.0, 1.0, 1.0])
        self.assertEqual(row["population_accuracy"], 1.0)
        self.assertEqual(row["reference_accuracy"], 0.0)
        self.assertEqual(row["population_minus_reference_delta"], 1.0)
        self.assertEqual(row["bootstrap_ci_low"], 1.0)
        self.assertEqual(row["bootstrap_ci_high"], 1.0)

    def test_pooled_replicate_uses_exact_equal_condition_weight(self):
        vectors = OrderedDict()
        for index, condition in enumerate(c.GATE8_V1_VALID_CONDITIONS):
            vectors[condition] = np.zeros(512) if index == 0 else np.ones(512)
        pooled = c.pooled_comparison(vectors)
        expected = 20.0 / 21.0
        self.assertAlmostEqual(pooled["population_minus_reference_delta"], expected)
        self.assertAlmostEqual(pooled["bootstrap_ci_low"], expected)
        self.assertAlmostEqual(pooled["bootstrap_ci_high"], expected)
        self.assertEqual(pooled["condition_weight"], 1.0 / 21.0)
        self.assertEqual(pooled["coupling"], c.GATE8_V1_POOLED_COUPLING)

    def _rows(self, population_accuracy: float, reference_accuracy: float, low: float, high: float):
        delta = population_accuracy - reference_accuracy
        return tuple(
            {
                "population": population,
                "depth": depth,
                "population_accuracy": population_accuracy,
                "population_seed_accuracies": [population_accuracy] * 3,
                "reference_accuracy": reference_accuracy,
                "population_minus_reference_delta": delta,
                "bootstrap_ci_low": low,
                "bootstrap_ci_high": high,
                "maximum_reference_input_tokens": 9892,
            }
            for population, depth in c.GATE8_V1_VALID_CONDITIONS
        )

    def test_unchanged_five_outcome_classifier_is_used(self):
        self.assertEqual(
            c.classify(
                self._rows(0.9, 0.1, 0.7, 0.9),
                {"population_minus_reference_delta": 0.8, "bootstrap_ci_low": 0.7, "bootstrap_ci_high": 0.9},
            ),
            "G8_POPULATION_EXCEEDS_1B_REFERENCE",
        )
        self.assertEqual(
            c.classify(
                self._rows(0.89, 0.9, -0.03, 0.01),
                {"population_minus_reference_delta": -0.01, "bootstrap_ci_low": -0.03, "bootstrap_ci_high": 0.01},
            ),
            "G8_POPULATION_NONINFERIOR_TO_1B_REFERENCE",
        )
        self.assertEqual(
            c.classify(
                self._rows(0.1, 0.9, -0.9, -0.7),
                {"population_minus_reference_delta": -0.8, "bootstrap_ci_low": -0.9, "bootstrap_ci_high": -0.7},
            ),
            "G8_1B_REFERENCE_SUPERIOR",
        )

        mixed = list(self._rows(0.5, 0.5, -0.1, 0.1))
        mixed[0] = {**mixed[0], "population_accuracy": 0.8, "reference_accuracy": 0.2,
                    "population_seed_accuracies": [0.8] * 3,
                    "population_minus_reference_delta": 0.6,
                    "bootstrap_ci_low": 0.2, "bootstrap_ci_high": 0.8}
        mixed[1] = {**mixed[1], "population_accuracy": 0.1, "reference_accuracy": 0.9,
                    "population_seed_accuracies": [0.1] * 3,
                    "population_minus_reference_delta": -0.8,
                    "bootstrap_ci_low": -0.9, "bootstrap_ci_high": -0.7}
        self.assertEqual(
            c.classify(
                tuple(mixed),
                {"population_minus_reference_delta": 0.0, "bootstrap_ci_low": -0.2, "bootstrap_ci_high": 0.2},
            ),
            "G8_1B_REFERENCE_MIXED",
        )

        self.assertEqual(
            c.classify(
                self._rows(0.85, 0.9, -0.08, 0.0),
                {"population_minus_reference_delta": -0.05, "bootstrap_ci_low": -0.05, "bootstrap_ci_high": 0.0},
            ),
            "G8_1B_REFERENCE_COMPARISON_INCONCLUSIVE",
        )

    def test_shape_binary_matrix_and_condition_order_fail_closed(self):
        with self.assertRaisesRegex(ValueError, "shape"):
            c.world_delta_vector(np.ones((2, 512)), np.zeros(512))
        bad = np.zeros((3, 512), dtype=np.uint8)
        bad[0, 0] = 2
        with self.assertRaisesRegex(ValueError, "binary"):
            c.world_delta_vector(bad, np.zeros(512))
        reversed_vectors = OrderedDict(
            (condition, np.ones(512))
            for condition in reversed(c.GATE8_V1_VALID_CONDITIONS)
        )
        with self.assertRaisesRegex(ValueError, "exact ordered"):
            c.pooled_comparison(reversed_vectors)

    def test_executor_contains_no_model_world_or_training_surface(self):
        contract = CONTRACT_PATH.read_text(encoding="utf-8")
        runner = RUNNER_PATH.read_text(encoding="utf-8")
        combined = contract + runner
        forbidden = (
            "import torch",
            "from transformers",
            "from_pretrained(",
            "model.generate(",
            "generate_gate8_world(",
            "torch.load(",
            "optimizer",
            "backward(",
        )
        for token in forbidden:
            self.assertNotIn(token, combined)
        self.assertIn("classify_gate8_v1_reference_comparison", contract)
        self.assertIn("source_ledgers_read_only", runner)


if __name__ == "__main__":
    unittest.main()
