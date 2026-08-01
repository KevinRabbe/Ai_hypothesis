from __future__ import annotations

import importlib.util
import pathlib
import sys
import unittest

PROTOCOL_PATH = pathlib.Path(
    "ai_hypothesis/population_compute/gate9_contextual_operator_induction_protocol.py"
)


def _load_protocol():
    name = "gate9_contextual_operator_induction_protocol_test_module"
    spec = importlib.util.spec_from_file_location(name, PROTOCOL_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load Gate9 protocol")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


p = _load_protocol()


class Gate9ProtocolTests(unittest.TestCase):
    def _splits(self):
        return p.Gate9OperatorSplitEvidence(
            train_operators=p.GATE9_TRAIN_OPERATOR_COUNT,
            validation_operators=p.GATE9_VALIDATION_OPERATOR_COUNT,
            local_test_operators=p.GATE9_LOCAL_TEST_OPERATOR_COUNT,
            graph_test_operators=p.GATE9_GRAPH_TEST_OPERATOR_COUNT,
            train_validation_intersection=0,
            train_scientific_intersection=0,
            validation_scientific_intersection=0,
            local_graph_test_intersection=0,
            injective_counter_to_operator_mapping_proven=True,
            operator_keys_exposed_to_model=False,
        )

    def _local(self, *, full=1.0, shuffled=0.01, query_only=0.01, ci_low=0.999):
        return tuple(
            p.Gate9LocalInductionEvidence(
                checkpoint_seed=seed,
                accuracy=full,
                bootstrap_ci_low=ci_low,
                bootstrap_ci_high=1.0,
                shuffled_context_accuracy=shuffled,
                query_only_accuracy=query_only,
                full_minus_shuffled_context_ci_low=full - shuffled - 0.01,
                full_minus_query_only_ci_low=full - query_only - 0.01,
            )
            for seed in p.GATE9_CHECKPOINT_SEEDS
        )

    def _conditions(self, frontiers=(4, 8, 16, 32, 64, 128)):
        by_population = dict(zip(p.GATE9_POPULATIONS, frontiers, strict=True))
        rows = []
        for population, depth in p.GATE9_VALID_CONDITIONS:
            solved = depth <= by_population[population]
            accuracy = 0.96 if solved else 0.20
            rows.append(
                p.Gate9ConditionEvidence(
                    population=population,
                    depth=depth,
                    accuracy=accuracy,
                    bootstrap_ci_low=0.92 if solved else 0.15,
                    bootstrap_ci_high=0.99 if solved else 0.25,
                    seed_accuracies=(accuracy, accuracy, accuracy),
                    mean_active_workers=float(min(population, depth * 4)),
                    mean_communicated_bits=float(depth * 32),
                    mean_context_examples_read=float(depth * 36),
                    mean_worker_updates=float(depth * 4),
                )
            )
        return tuple(rows)

    def _causal(self, *, context_low=0.80, communication_low=0.80):
        return tuple(
            p.Gate9CausalEvidence(
                population=population,
                depth=depth,
                full_accuracy=0.96,
                no_communication_accuracy=0.01,
                shuffled_context_accuracy=0.01,
                query_only_accuracy=0.01,
                full_minus_no_communication_ci_low=communication_low,
                full_minus_shuffled_context_ci_low=context_low,
                full_minus_query_only_ci_low=context_low,
            )
            for population, depth in p.GATE9_CAUSAL_CONDITIONS
        )

    def test_plan_freezes_novel_operator_and_execution_boundaries(self):
        plan = p.gate9_protocol_plan()
        self.assertEqual(plan["gate8_final_result_head"], p.GATE9_GATE8_FINAL_RESULT_HEAD)
        self.assertEqual(plan["learned_parameter_budget"], 19_649)
        self.assertEqual(plan["operator_key_bits"], 64)
        self.assertEqual(plan["operator_family_size"], 1 << 64)
        self.assertEqual(plan["support_examples"], 9)
        self.assertEqual(plan["condition_count"], 21)
        self.assertEqual(plan["worlds_per_condition"], 256)
        self.assertEqual(plan["operator_counts"]["graph_test"], 2_629_632)
        self.assertFalse(plan["operator_id_visible_to_model"])
        for key in (
            "execution_admitted",
            "operator_generation_admitted",
            "world_generation_admitted",
            "architecture_admitted",
            "training_admitted",
            "checkpoint_loading_admitted",
            "scientific_test_admitted",
            "reference_inference_admitted",
            "result_classification_admitted",
        ):
            self.assertFalse(plan[key])

    def test_operator_splits_fail_closed_on_overlap_key_exposure_and_count(self):
        self._splits().validate()

        overlap = p.Gate9OperatorSplitEvidence(
            **{
                **self._splits().to_dict(),
                "train_scientific_intersection": 1,
            }
        )
        with self.assertRaisesRegex(ValueError, "overlap"):
            overlap.validate()

        exposed = p.Gate9OperatorSplitEvidence(
            **{
                **self._splits().to_dict(),
                "operator_keys_exposed_to_model": True,
            }
        )
        with self.assertRaisesRegex(ValueError, "operator key"):
            exposed.validate()

        wrong_count = p.Gate9OperatorSplitEvidence(
            **{
                **self._splits().to_dict(),
                "graph_test_operators": p.GATE9_GRAPH_TEST_OPERATOR_COUNT - 1,
            }
        )
        with self.assertRaisesRegex(ValueError, "counts drifted"):
            wrong_count.validate()

    def test_positive_classifier_requires_novel_induction_context_and_communication(self):
        outcome = p.classify_gate9(
            operator_splits=self._splits(),
            local=self._local(),
            conditions=self._conditions(),
            causal=self._causal(),
        )
        self.assertEqual(outcome, p.GATE9_POSITIVE)

        failed_local = p.classify_gate9(
            operator_splits=self._splits(),
            local=self._local(full=0.90, ci_low=0.88),
            conditions=self._conditions(),
            causal=self._causal(),
        )
        self.assertEqual(failed_local, p.GATE9_LOCAL_INDUCTION_FAILED)

        context_not_causal = p.classify_gate9(
            operator_splits=self._splits(),
            local=self._local(),
            conditions=self._conditions(),
            causal=self._causal(context_low=0.20),
        )
        self.assertEqual(context_not_causal, p.GATE9_CONTEXT_NOT_CAUSAL)

        communication_inconclusive = p.classify_gate9(
            operator_splits=self._splits(),
            local=self._local(),
            conditions=self._conditions(),
            causal=self._causal(communication_low=0.20),
        )
        self.assertEqual(communication_inconclusive, p.GATE9_INCONCLUSIVE)

    def test_flat_negative_and_unsolved_results_remain_distinct(self):
        flat = p.classify_gate9(
            operator_splits=self._splits(),
            local=self._local(),
            conditions=self._conditions(frontiers=(4, 4, 4, 4, 4, 4)),
            causal=self._causal(),
        )
        self.assertEqual(flat, p.GATE9_PRESENT_NO_SCALING)

        negative = p.classify_gate9(
            operator_splits=self._splits(),
            local=self._local(),
            conditions=self._conditions(frontiers=(4, 8, 16, 8, 16, 32)),
            causal=self._causal(),
        )
        self.assertEqual(negative, p.GATE9_NEGATIVE)

        rows = tuple(
            p.Gate9ConditionEvidence(
                population=row.population,
                depth=row.depth,
                accuracy=0.10,
                bootstrap_ci_low=0.05,
                bootstrap_ci_high=0.15,
                seed_accuracies=(0.10, 0.10, 0.10),
                mean_active_workers=row.mean_active_workers,
                mean_communicated_bits=row.mean_communicated_bits,
                mean_context_examples_read=row.mean_context_examples_read,
                mean_worker_updates=row.mean_worker_updates,
            )
            for row in self._conditions()
        )
        inconclusive = p.classify_gate9(
            operator_splits=self._splits(),
            local=self._local(),
            conditions=rows,
            causal=self._causal(),
        )
        self.assertEqual(inconclusive, p.GATE9_INCONCLUSIVE)

    def test_exact_order_equal_seed_weighting_and_threshold_strictness(self):
        with self.assertRaisesRegex(ValueError, "exact 21-condition matrix"):
            p.build_gate9_frontiers(tuple(reversed(self._conditions())))

        rows = list(self._conditions())
        first = rows[0]
        rows[0] = p.Gate9ConditionEvidence(
            population=first.population,
            depth=first.depth,
            accuracy=0.5,
            bootstrap_ci_low=first.bootstrap_ci_low,
            bootstrap_ci_high=first.bootstrap_ci_high,
            seed_accuracies=(0.96, 0.96, 0.96),
            mean_active_workers=first.mean_active_workers,
            mean_communicated_bits=first.mean_communicated_bits,
            mean_context_examples_read=first.mean_context_examples_read,
            mean_worker_updates=first.mean_worker_updates,
        )
        with self.assertRaisesRegex(ValueError, "equal-seed weighted"):
            p.build_gate9_frontiers(tuple(rows))

        boundary_local = self._local(
            full=0.995,
            shuffled=0.495,
            query_only=0.495,
            ci_low=0.990,
        )
        self.assertFalse(boundary_local[0].induction_passes())

        causal = self._causal(context_low=0.20)
        self.assertFalse(causal[0].context_guard_passes())

    def test_protocol_has_no_generator_model_training_or_artifact_reader(self):
        text = PROTOCOL_PATH.read_text(encoding="utf-8")
        forbidden = (
            "import torch",
            "import numpy",
            "open(",
            "read_text(",
            "json.load",
            "generate_operator(",
            "generate_world(",
            "torch.load(",
            "torch.optim",
            ".backward(",
            "execution_admitted\": True",
            "training_admitted\": True",
            "scientific_test_admitted\": True",
        )
        for token in forbidden:
            self.assertNotIn(token, text)


if __name__ == "__main__":
    unittest.main()
