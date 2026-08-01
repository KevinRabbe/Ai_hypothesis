from __future__ import annotations

import importlib.util
import pathlib
import sys
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
PROTOCOL_PATH = ROOT / (
    "ai_hypothesis/population_compute/"
    "gate9_contextual_failure_decomposition_protocol.py"
)
OPERATOR_PATH = ROOT / (
    "ai_hypothesis/population_compute/gate9_contextual_operator_contract.py"
)


def load(path: pathlib.Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


protocol = load(PROTOCOL_PATH, "gate9d_protocol_test_dependency")
operators = load(OPERATOR_PATH, "gate9d_operator_test_dependency")


class Gate9FailureDecompositionProtocolTests(unittest.TestCase):
    def test_exact_bindings_stage_order_and_query_domain(self) -> None:
        self.assertEqual(
            protocol.GATE9D_FINAL_RESULT_HEAD,
            "33f2860795a1b70e5fbe20998f4fe8a2a6fc8452",
        )
        self.assertEqual(
            protocol.GATE9D_OPERATOR_CONTRACT_HEAD,
            "be6451e1af82b18749bd0313a9c02ca62c4eee5c",
        )
        self.assertEqual(
            protocol.GATE9D_TRAINING_PROTOCOL_HEAD,
            "1228c19cbf85da4ab738c3355c58f946cd6a965c",
        )
        architecture_head = protocol.GATE9D_ARCHITECTURE_HEAD
        self.assertEqual(len(architecture_head), 40)
        self.assertTrue(all(character in "0123456789abcdef" for character in architecture_head))
        self.assertEqual(protocol.GATE9D_LEARNED_PARAMETER_COUNT, 19_649)
        self.assertEqual(
            tuple(stage.name for stage in protocol.GATE9D_STAGES),
            (
                "single_operator_query_fit",
                "paired_operator_context_collision",
                "held_in_multi_operator_fit",
                "unseen_operator_generalization",
            ),
        )
        self.assertEqual(
            set(protocol.GATE9D_QUERY_VALUES),
            set(range(256)) - set(protocol.GATE9D_SUPPORT_INPUTS),
        )
        self.assertEqual(len(protocol.GATE9D_QUERY_VALUES), 247)

    def test_collision_pair_is_exact_and_forces_context(self) -> None:
        observed = tuple(
            operators.operator_from_counter(counter)
            for counter in protocol.GATE9D_COLLISION_OPERATOR_COUNTERS
        )
        self.assertEqual(
            tuple(operator.key for operator in observed),
            protocol.GATE9D_COLLISION_OPERATOR_KEYS,
        )
        self.assertEqual(observed[0].matrix_rows, observed[1].matrix_rows)
        self.assertEqual((observed[0].bias, observed[1].bias), (0, 255))
        for query in protocol.GATE9D_QUERY_VALUES:
            self.assertEqual(
                observed[0].apply(query) ^ observed[1].apply(query),
                255,
            )

    def test_operator_namespaces_are_pairwise_and_v0_disjoint(self) -> None:
        groups = (
            set(protocol.GATE9D_SINGLE_OPERATOR_RANGE.counters()),
            set(protocol.GATE9D_COLLISION_OPERATOR_COUNTERS),
            set(protocol.GATE9D_HELD_IN_OPERATOR_RANGE.counters()),
            set(protocol.GATE9D_UNSEEN_TRAIN_OPERATOR_RANGE.counters()),
            set(protocol.GATE9D_UNSEEN_EVAL_OPERATOR_RANGE.counters()),
        )
        for index, left in enumerate(groups):
            for right in groups[index + 1 :]:
                self.assertFalse(left & right)
        for frozen in protocol.GATE9D_FROZEN_V0_RANGES:
            for group in groups:
                self.assertFalse(
                    any(frozen.start <= counter < frozen.stop for counter in group)
                )

    def test_stage_sizes_schedules_and_unseen_split(self) -> None:
        expected = (
            (1, 1, 247, 247, 1_024, 247),
            (2, 2, 494, 494, 2_048, 494),
            (16, 16, 3_952, 3_952, 4_096, 512),
            (256, 64, 63_232, 15_808, 8_192, 512),
        )
        for stage, required in zip(protocol.GATE9D_STAGES, expected, strict=True):
            self.assertEqual(
                (
                    len(stage.train_operator_counters),
                    len(stage.evaluation_operator_counters),
                    stage.train_examples,
                    stage.evaluation_examples,
                    stage.steps,
                    stage.batch_size,
                ),
                required,
            )
        self.assertFalse(protocol.GATE9D_STAGES[0].requires_context_causality)
        self.assertTrue(
            all(stage.requires_context_causality for stage in protocol.GATE9D_STAGES[1:])
        )
        final = protocol.GATE9D_STAGES[-1]
        self.assertTrue(final.unseen_operator_evaluation)
        self.assertFalse(
            set(final.train_operator_counters)
            & set(final.evaluation_operator_counters)
        )

    def test_thresholds_are_strict_and_stage_aware(self) -> None:
        single, contextual = protocol.GATE9D_STAGES[:2]
        single_pass = protocol.SeedStageEvidence(
            exact_accuracy=0.995,
            bit_accuracy=0.999,
            full_minus_shuffled=-1.0,
            full_minus_query_only=-1.0,
            oracle_accuracy=1.0,
        )
        self.assertTrue(single_pass.passes(single))
        self.assertFalse(single_pass.passes(contextual))
        boundary = protocol.SeedStageEvidence(
            exact_accuracy=0.995,
            bit_accuracy=0.999,
            full_minus_shuffled=0.50,
            full_minus_query_only=0.50,
            oracle_accuracy=1.0,
        )
        self.assertFalse(boundary.passes(contextual))
        above = protocol.SeedStageEvidence(
            exact_accuracy=0.995,
            bit_accuracy=0.999,
            full_minus_shuffled=0.5000000000000001,
            full_minus_query_only=0.5000000000000001,
            oracle_accuracy=1.0,
        )
        self.assertTrue(above.passes(contextual))

    def test_classifier_stops_at_first_nonpassing_stage(self) -> None:
        names = [stage.name for stage in protocol.GATE9D_STAGES]
        self.assertEqual(protocol.classify_diagnostic({}), "G9D_DIAGNOSTIC_INCOMPLETE")
        self.assertEqual(
            protocol.classify_diagnostic({names[0]: (False, False, False)}),
            "G9D_BASIC_QUERY_MAPPING_FAILED",
        )
        self.assertEqual(
            protocol.classify_diagnostic({names[0]: (True, False, True)}),
            "G9D_DIAGNOSTIC_INCONCLUSIVE",
        )
        results = {names[0]: (True, True, True)}
        self.assertEqual(protocol.classify_diagnostic(results), "G9D_DIAGNOSTIC_INCOMPLETE")
        expected_failures = (
            "G9D_CONTEXTUAL_CONTROL_FAILED",
            "G9D_HELD_IN_OPERATOR_FIT_FAILED",
            "G9D_UNSEEN_OPERATOR_GENERALIZATION_FAILED",
        )
        for index, expected in enumerate(expected_failures, 1):
            results[names[index]] = (False, False, False)
            self.assertEqual(protocol.classify_diagnostic(results), expected)
            results[names[index]] = (True, True, True)
        self.assertEqual(protocol.classify_diagnostic(results), "G9D_V0_FAILURE_NOT_LOCALIZED")

    def test_plan_and_source_keep_execution_closed(self) -> None:
        plan = protocol.gate9d_protocol_plan()
        self.assertEqual(plan["status"], "G9D_PROTOCOL_FROZEN_EXECUTION_CLOSED")
        self.assertEqual(plan["initialization_seeds"], [910900, 910901, 910902])
        self.assertTrue(plan["decision_rule"]["stop_after_first_nonpassing_stage"])
        self.assertTrue(plan["boundaries"]["protocol_only"])
        for key, value in plan["boundaries"].items():
            if key != "protocol_only":
                self.assertFalse(value, key)
        source = PROTOCOL_PATH.read_text(encoding="utf-8")
        for token in (
            "import torch",
            "torch.",
            "optimizer.step(",
            "torch.load(",
            "torch.save(",
            "operator_from_counter(",
            "generate_gate9_test_world(",
            "scientific_assignment_key",
        ):
            self.assertNotIn(token, source)


if __name__ == "__main__":
    unittest.main()
