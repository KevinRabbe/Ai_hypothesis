from __future__ import annotations

import importlib.util
import pathlib
import sys
import unittest

PROTOCOL_PATH = pathlib.Path(
    "ai_hypothesis/population_compute/"
    "gate8_v1_scientific_evaluation_protocol.py"
)


def _load_protocol():
    name = "gate8_v1_scientific_evaluation_protocol_test_module"
    spec = importlib.util.spec_from_file_location(name, PROTOCOL_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load Gate8 v1 scientific-evaluation protocol")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


p = _load_protocol()
base = p.capability


class Gate8V1ScientificEvaluationProtocolTests(unittest.TestCase):
    def _conditions(self, frontiers=(4, 8, 16, 32, 64, 128)):
        frontier_by_population = dict(zip(base.GATE8_POPULATIONS, frontiers, strict=True))
        rows = []
        for population, depth in p.GATE8_V1_VALID_CONDITIONS:
            solved = depth <= frontier_by_population[population]
            accuracy = 0.94 if solved else 0.20
            rows.append(
                p.Gate8V1ConditionEvidence(
                    population=population,
                    depth=depth,
                    accuracy=accuracy,
                    bootstrap_ci_low=0.90 if solved else 0.15,
                    bootstrap_ci_high=0.97 if solved else 0.25,
                    seed_accuracies=(accuracy, accuracy, accuracy),
                    mean_active_workers=float(min(population, depth * 3)),
                    mean_communicated_bits=float(depth * 24),
                    mean_recurrent_updates=float(depth * 3),
                )
            )
        return tuple(rows)

    def _ablations(self, low=0.25):
        return tuple(
            p.Gate8V1AblationEvidence(
                population=population,
                depth=depth,
                full_accuracy=0.94,
                no_communication_accuracy=0.20,
                shuffled_worker_accuracy=0.18,
                full_seed_accuracies=(0.94, 0.94, 0.94),
                no_communication_seed_accuracies=(0.20, 0.20, 0.20),
                shuffled_worker_seed_accuracies=(0.18, 0.18, 0.18),
                full_minus_no_communication_ci_low=low,
                full_minus_shuffled_worker_ci_low=low,
            )
            for population, depth in p.GATE8_V1_CAUSAL_ABLATION_CONDITIONS
        )

    def _reference(self, population_accuracy=0.89, reference_accuracy=0.90):
        delta = population_accuracy - reference_accuracy
        return tuple(
            p.Gate8V1ReferenceEvidence(
                population=population,
                depth=depth,
                population_accuracy=population_accuracy,
                reference_accuracy=reference_accuracy,
                population_seed_accuracies=(
                    population_accuracy,
                    population_accuracy,
                    population_accuracy,
                ),
                population_minus_reference_delta=delta,
                bootstrap_ci_low=-0.03,
                bootstrap_ci_high=0.01,
                maximum_reference_input_tokens=9_843,
            )
            for population, depth in p.GATE8_V1_VALID_CONDITIONS
        )

    def test_plan_binds_three_exact_checkpoints_and_keeps_execution_closed(self):
        plan = p.gate8_v1_scientific_evaluation_plan()
        self.assertEqual(plan["base_result_head"], p.GATE8_V1_BASE_RESULT_HEAD)
        self.assertEqual(plan["checkpoint_seeds"], [0, 1, 2])
        self.assertEqual(
            [row["selected_checkpoint_sha256"] for row in plan["checkpoint_bindings"]],
            [
                "3005369a4830c12baee8ffa7fedc1bed0f1888784e1043bd88f4afd2b7cddde9",
                "cbcae487dd7f4c695e1d6a83a61926cd43f5ccf6add1a7469c16a15697d22d07",
                "e1e35b3864354e8f3398497a897b6a759dfa3454a33d866de63784a323f461e4",
            ],
        )
        self.assertEqual(plan["condition_count"], 21)
        self.assertEqual(plan["test_split"], "test")
        self.assertEqual(plan["test_seed"], 0)
        self.assertEqual(plan["test_world_index_start"], 0)
        self.assertEqual(plan["test_world_index_end_inclusive"], 511)
        self.assertEqual(plan["bootstrap_samples"], 20_000)
        self.assertEqual(
            plan["bootstrap_unit"],
            "world_index_shared_across_all_three_checkpoint_seeds",
        )
        self.assertFalse(plan["execution_admitted"])
        self.assertFalse(plan["checkpoint_loading_admitted"])
        self.assertFalse(plan["scientific_test_generation_admitted"])
        self.assertFalse(plan["reference_weight_binding_admitted"])
        self.assertFalse(plan["reference_inference_admitted"])
        self.assertFalse(plan["training_admitted"])
        self.assertFalse(plan["test_answers_exposed"])

    def test_checkpoint_bindings_fail_closed_on_order_hash_and_budget(self):
        p.validate_gate8_v1_checkpoint_bindings()
        reversed_bindings = tuple(reversed(p.GATE8_V1_CHECKPOINT_BINDINGS))
        with self.assertRaisesRegex(ValueError, "ordered seeds"):
            p.validate_gate8_v1_checkpoint_bindings(reversed_bindings)

        first = p.GATE8_V1_CHECKPOINT_BINDINGS[0]
        malformed = p.Gate8V1CheckpointBinding(
            seed=0,
            result_head=first.result_head,
            result_sha256=first.result_sha256,
            selected_checkpoint_sha256="x" * 64,
            source_manifest_sha256=first.source_manifest_sha256,
        )
        with self.assertRaisesRegex(ValueError, "malformed SHA-256"):
            malformed.validate()

        wrong_budget = p.Gate8V1CheckpointBinding(
            seed=0,
            result_head=first.result_head,
            result_sha256=first.result_sha256,
            selected_checkpoint_sha256=first.selected_checkpoint_sha256,
            source_manifest_sha256=first.source_manifest_sha256,
            learned_parameter_count=19_650,
        )
        with self.assertRaisesRegex(ValueError, "budget changed"):
            wrong_budget.validate()

    def test_frontier_builder_and_original_positive_classifier_are_preserved(self):
        frontiers = p.build_gate8_v1_population_frontiers(self._conditions())
        self.assertEqual(
            tuple(row.max_solved_depth for row in frontiers),
            (4, 8, 16, 32, 64, 128),
        )
        outcome = p.classify_gate8_v1_population_scaling(
            conditions=self._conditions(),
            ablations=self._ablations(),
        )
        self.assertEqual(outcome, base.GATE8_SCALING_POSITIVE)

        blocked = p.classify_gate8_v1_population_scaling(
            conditions=self._conditions(),
            ablations=self._ablations(low=0.20),
        )
        self.assertEqual(blocked, base.GATE8_SCALING_INCONCLUSIVE)

    def test_flat_negative_and_unsolved_evidence_remain_distinct(self):
        flat = p.classify_gate8_v1_population_scaling(
            conditions=self._conditions(frontiers=(4, 4, 4, 4, 4, 4)),
            ablations=self._ablations(),
        )
        self.assertEqual(flat, base.GATE8_SCALING_FLAT)

        negative = p.classify_gate8_v1_population_scaling(
            conditions=self._conditions(frontiers=(4, 8, 16, 8, 16, 32)),
            ablations=self._ablations(),
        )
        self.assertEqual(negative, base.GATE8_SCALING_NEGATIVE)

        unsolved_rows = tuple(
            p.Gate8V1ConditionEvidence(
                population=row.population,
                depth=row.depth,
                accuracy=0.10,
                bootstrap_ci_low=0.05,
                bootstrap_ci_high=0.15,
                seed_accuracies=(0.10, 0.10, 0.10),
                mean_active_workers=row.mean_active_workers,
                mean_communicated_bits=row.mean_communicated_bits,
                mean_recurrent_updates=row.mean_recurrent_updates,
            )
            for row in self._conditions()
        )
        unsolved = p.classify_gate8_v1_population_scaling(
            conditions=unsolved_rows,
            ablations=self._ablations(),
        )
        self.assertEqual(unsolved, base.GATE8_SCALING_INCONCLUSIVE)

    def test_equal_seed_weighting_and_exact_matrix_are_enforced(self):
        rows = list(self._conditions())
        first = rows[0]
        rows[0] = p.Gate8V1ConditionEvidence(
            population=first.population,
            depth=first.depth,
            accuracy=0.50,
            bootstrap_ci_low=first.bootstrap_ci_low,
            bootstrap_ci_high=first.bootstrap_ci_high,
            seed_accuracies=(0.90, 0.90, 0.90),
            mean_active_workers=first.mean_active_workers,
            mean_communicated_bits=first.mean_communicated_bits,
            mean_recurrent_updates=first.mean_recurrent_updates,
        )
        with self.assertRaisesRegex(ValueError, "equal-seed weighted"):
            p.build_gate8_v1_population_frontiers(tuple(rows))

        with self.assertRaisesRegex(ValueError, "exact 21-condition matrix"):
            p.build_gate8_v1_population_frontiers(tuple(reversed(self._conditions())))

        bad_ablation = list(self._ablations())
        first_ablation = bad_ablation[0]
        bad_ablation[0] = p.Gate8V1AblationEvidence(
            population=first_ablation.population,
            depth=first_ablation.depth,
            full_accuracy=0.50,
            no_communication_accuracy=0.20,
            shuffled_worker_accuracy=0.18,
            full_seed_accuracies=(0.94, 0.94, 0.94),
            no_communication_seed_accuracies=(0.20, 0.20, 0.20),
            shuffled_worker_seed_accuracies=(0.18, 0.18, 0.18),
            full_minus_no_communication_ci_low=0.25,
            full_minus_shuffled_worker_ci_low=0.25,
        )
        with self.assertRaisesRegex(ValueError, "equal-seed weighted"):
            p.build_gate8_v1_ablation_rows(tuple(bad_ablation))

    def test_reference_wrapper_preserves_equal_seed_pooling_and_classifier(self):
        rows = self._reference()
        base_rows = p.build_gate8_v1_reference_rows(rows)
        self.assertEqual(
            tuple((row.population, row.depth) for row in base_rows),
            p.GATE8_V1_VALID_CONDITIONS,
        )
        outcome = p.classify_gate8_v1_reference_comparison(
            conditions=rows,
            pooled=base.Gate8ReferencePooledRow(
                population_minus_reference_delta=-0.01,
                bootstrap_ci_low=-0.03,
                bootstrap_ci_high=0.01,
            ),
        )
        self.assertEqual(outcome, base.GATE8_REFERENCE_NONINFERIOR)

        first = rows[0]
        bad = p.Gate8V1ReferenceEvidence(
            population=first.population,
            depth=first.depth,
            population_accuracy=0.50,
            reference_accuracy=first.reference_accuracy,
            population_seed_accuracies=(0.89, 0.89, 0.89),
            population_minus_reference_delta=-0.40,
            bootstrap_ci_low=first.bootstrap_ci_low,
            bootstrap_ci_high=first.bootstrap_ci_high,
            maximum_reference_input_tokens=first.maximum_reference_input_tokens,
        )
        with self.assertRaisesRegex(ValueError, "equal-seed weighted"):
            bad.validate()

    def test_protocol_has_no_execution_or_artifact_surface(self):
        text = PROTOCOL_PATH.read_text(encoding="utf-8")
        forbidden = (
            "import torch",
            "torch.",
            "generate_gate8_world(",
            "open(",
            "read_text(",
            "json.load",
            "from_pretrained(",
            "torch.load(",
            "execution_admitted\": True",
            "checkpoint_loading_admitted\": True",
            "scientific_test_generation_admitted\": True",
            "reference_weight_binding_admitted\": True",
            "reference_inference_admitted\": True",
            "training_admitted\": True",
        )
        for token in forbidden:
            self.assertNotIn(token, text)


if __name__ == "__main__":
    unittest.main()
