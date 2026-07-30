from __future__ import annotations

import importlib.util
import pathlib
import sys
import unittest


PROTOCOL_PATH = pathlib.Path(
    "ai_hypothesis/population_compute/"
    "gate8_distributed_transformation_capability_protocol.py"
)


def _load_protocol():
    name = "gate8_distributed_transformation_capability_protocol_test_module"
    spec = importlib.util.spec_from_file_location(name, PROTOCOL_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load Gate8 capability protocol")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


p = _load_protocol()


class Gate8DistributedTransformationCapabilityProtocolTests(unittest.TestCase):
    def _frontiers(self, depths=(4, 8, 16, 32, 64, 128)):
        return tuple(
            p.Gate8CapabilityFrontierRow(
                population=population,
                max_solved_depth=depth,
                frontier_accuracy=0.93,
                frontier_ci_low=0.90,
                frontier_ci_high=0.95,
                active_workers=population,
                communicated_bits=population * depth * 8,
                recurrent_updates=population * depth,
            )
            for population, depth in zip(p.GATE8_POPULATIONS, depths, strict=True)
        )

    def _ablations(self, *, pass_guard=True):
        low = 0.25 if pass_guard else 0.10
        return tuple(
            p.Gate8AblationRow(
                population=population,
                depth=depth,
                full_accuracy=0.93,
                no_communication_accuracy=0.20,
                shuffled_worker_accuracy=0.18,
                full_minus_no_communication_ci_low=low,
                full_minus_shuffled_worker_ci_low=low,
            )
            for population, depth in p.GATE8_CAUSAL_ABLATION_CONDITIONS
        )

    def _reference_rows(self, *, delta=-0.01, low=-0.03, high=0.01):
        return tuple(
            p.Gate8ReferenceConditionRow(
                population=population,
                depth=depth,
                population_accuracy=0.89,
                reference_accuracy=0.90,
                population_minus_reference_delta=delta,
                bootstrap_ci_low=low,
                bootstrap_ci_high=high,
                reference_input_tokens=2_000,
            )
            for population, depth in p.GATE8_VALID_CONDITIONS
        )

    def test_plan_freezes_exact_capability_and_reference_contract(self) -> None:
        plan = p.gate8_protocol_plan()
        self.assertFalse(plan["execution_admitted"])
        self.assertFalse(plan["training_admitted"])
        self.assertFalse(plan["baseline_execution_admitted"])
        self.assertEqual(plan["learned_parameter_count"], 19_649)
        self.assertEqual(plan["populations"], [32, 64, 128, 256, 512, 1_024])
        self.assertEqual(plan["depths"], [4, 8, 16, 32, 64, 128])
        self.assertEqual(plan["message_bits"], 8)
        self.assertEqual(plan["reference_model_id"], "google/gemma-3-1b-it")
        self.assertEqual(plan["reference_parameter_class"], "1.0B")
        self.assertEqual(plan["reference_context_tokens"], 32_768)
        self.assertEqual(plan["reference_max_input_tokens"], 24_576)
        self.assertFalse(plan["reference_task_specific_weight_updates"])
        self.assertEqual(plan["reference_demonstrations"], 8)
        self.assertFalse(plan["brute_force_candidate_search"])

    def test_valid_condition_matrix_is_population_major(self) -> None:
        expected = tuple(
            (population, depth)
            for population in p.GATE8_POPULATIONS
            for depth in p.GATE8_DEPTHS
            if depth * 8 <= population
        )
        self.assertEqual(p.GATE8_VALID_CONDITIONS, expected)
        self.assertEqual(len(expected), 21)

    def test_positive_scaling_requires_depth_growth_and_causal_guards(self) -> None:
        outcome = p.classify_gate8_population_scaling(
            frontiers=self._frontiers(),
            ablations=self._ablations(),
        )
        self.assertEqual(outcome, p.GATE8_SCALING_POSITIVE)

        blocked = p.classify_gate8_population_scaling(
            frontiers=self._frontiers(),
            ablations=self._ablations(pass_guard=False),
        )
        self.assertEqual(blocked, p.GATE8_SCALING_INCONCLUSIVE)

    def test_flat_and_negative_scaling_are_distinct(self) -> None:
        flat = self._frontiers(depths=(4, 4, 4, 4, 4, 4))
        self.assertEqual(
            p.classify_gate8_population_scaling(
                frontiers=flat,
                ablations=self._ablations(),
            ),
            p.GATE8_SCALING_FLAT,
        )

        negative = self._frontiers(depths=(4, 8, 16, 8, 16, 32))
        self.assertEqual(
            p.classify_gate8_population_scaling(
                frontiers=negative,
                ablations=self._ablations(),
            ),
            p.GATE8_SCALING_NEGATIVE,
        )

    def test_reference_noninferiority_and_superiority_are_frozen(self) -> None:
        noninferior = p.classify_gate8_reference_comparison(
            conditions=self._reference_rows(),
            pooled=p.Gate8ReferencePooledRow(
                population_minus_reference_delta=-0.01,
                bootstrap_ci_low=-0.03,
                bootstrap_ci_high=0.01,
            ),
        )
        self.assertEqual(noninferior, p.GATE8_REFERENCE_NONINFERIOR)

        superior = p.classify_gate8_reference_comparison(
            conditions=self._reference_rows(delta=-0.10, low=-0.13, high=-0.07),
            pooled=p.Gate8ReferencePooledRow(
                population_minus_reference_delta=-0.10,
                bootstrap_ci_low=-0.12,
                bootstrap_ci_high=-0.08,
            ),
        )
        self.assertEqual(superior, p.GATE8_REFERENCE_SUPERIOR)

    def test_reference_matrix_and_token_budget_fail_closed(self) -> None:
        rows = self._reference_rows()
        with self.assertRaisesRegex(ValueError, "exact condition matrix"):
            p.classify_gate8_reference_comparison(
                conditions=tuple(reversed(rows)),
                pooled=p.Gate8ReferencePooledRow(-0.01, -0.03, 0.01),
            )

        bad = list(rows)
        first = bad[0]
        bad[0] = p.Gate8ReferenceConditionRow(
            population=first.population,
            depth=first.depth,
            population_accuracy=first.population_accuracy,
            reference_accuracy=first.reference_accuracy,
            population_minus_reference_delta=first.population_minus_reference_delta,
            bootstrap_ci_low=first.bootstrap_ci_low,
            bootstrap_ci_high=first.bootstrap_ci_high,
            reference_input_tokens=24_577,
        )
        with self.assertRaisesRegex(ValueError, "input budget"):
            p.classify_gate8_reference_comparison(
                conditions=tuple(bad),
                pooled=p.Gate8ReferencePooledRow(-0.01, -0.03, 0.01),
            )

    def test_protocol_has_no_execution_or_artifact_surface(self) -> None:
        text = PROTOCOL_PATH.read_text(encoding="utf-8")
        forbidden = (
            "import torch",
            "torch.",
            "open(",
            "read_text(",
            "json.load",
            "from_pretrained(",
            "execution_admitted\": True",
            "training_admitted\": True",
            "baseline_execution_admitted\": True",
        )
        for token in forbidden:
            self.assertNotIn(token, text)


if __name__ == "__main__":
    unittest.main()
