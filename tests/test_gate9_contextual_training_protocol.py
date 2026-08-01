from __future__ import annotations

import importlib.util
import math
import pathlib
import sys
import unittest

PROTOCOL_PATH = pathlib.Path(
    "ai_hypothesis/population_compute/gate9_contextual_training_protocol.py"
)


def _load_protocol():
    name = "gate9_contextual_training_protocol_test_module"
    spec = importlib.util.spec_from_file_location(name, PROTOCOL_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load Gate9 training protocol")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


p = _load_protocol()


class Gate9TrainingProtocolTests(unittest.TestCase):
    def _row(
        self,
        seed: int,
        *,
        accuracy: float = 0.999,
        shuffled: float = 0.01,
        query_only: float = 0.01,
    ):
        return p.Gate9CheckpointValidationEvidence(
            seed=seed,
            initialization_seed=p.GATE9_INITIALIZATION_SEEDS[seed],
            checkpoint_step=512,
            train_episodes=262_144,
            unique_train_operators=262_144,
            validation_episodes=32_768,
            unique_validation_operators=32_768,
            learned_parameter_count=19_649,
            tensor_count=17,
            checkpoint_sha256=f"{seed + 1:064x}",
            parameters_finite=True,
            final_train_loss=0.001,
            validation_exact_accuracy=accuracy,
            validation_bit_accuracy=0.9999,
            shuffled_context_accuracy=shuffled,
            query_only_accuracy=query_only,
            oracle_accuracy=1.0,
        )

    def test_plan_freezes_exact_data_optimizer_checkpoint_and_closed_boundaries(self):
        plan = p.gate9_training_protocol_plan()
        self.assertEqual(plan["architecture_head"], p.GATE9_ARCHITECTURE_HEAD)
        self.assertEqual(plan["checkpoint_seeds"], [0, 1, 2])
        self.assertEqual(plan["learned_parameter_count"], 19_649)
        self.assertEqual(plan["state_tensor_count"], 17)
        self.assertEqual(plan["train"]["operator_count"], 262_144)
        self.assertEqual(plan["train"]["episodes_per_seed"], 262_144)
        self.assertEqual(plan["train"]["batch_size"], 512)
        self.assertEqual(plan["train"]["steps"], 512)
        self.assertEqual(plan["validation"]["operator_count"], 32_768)
        self.assertEqual(plan["validation"]["batches"], 64)
        self.assertEqual(plan["optimizer"]["name"], "AdamW")
        self.assertEqual(plan["checkpoint"]["selected_step"], 512)
        self.assertEqual(
            plan["checkpoint"]["selection"],
            "fixed_final_step_only_no_best_checkpoint_selection",
        )
        self.assertFalse(plan["checkpoint"]["retraining_after_failure_allowed"])
        for key in (
            "local_test_operator_access",
            "graph_test_operator_access",
            "scientific_assignment_key_access",
            "operator_generation_admitted",
            "training_execution_admitted",
            "optimizer_instantiation_admitted",
            "checkpoint_serialization_admitted",
            "checkpoint_loading_admitted",
            "scientific_test_generation_admitted",
            "scientific_execution_admitted",
            "result_classification_admitted",
        ):
            self.assertFalse(plan[key])

    def test_train_and_validation_orders_are_exact_bijections(self):
        for seed in p.GATE9_CHECKPOINT_SEEDS:
            observed = {
                p.training_operator_ordinal(seed, index)
                for index in range(p.GATE9_TRAIN_EPISODES)
            }
            self.assertEqual(len(observed), p.GATE9_TRAIN_OPERATOR_COUNT)
            for index in (0, 1, 17, 100_000, p.GATE9_TRAIN_EPISODES - 1):
                ordinal = p.training_operator_ordinal(seed, index)
                self.assertEqual(
                    p.inverse_training_episode_index(seed, ordinal), index
                )
        validation = {
            p.validation_operator_ordinal(index)
            for index in range(p.GATE9_VALIDATION_EPISODES)
        }
        self.assertEqual(len(validation), p.GATE9_VALIDATION_OPERATOR_COUNT)
        for index in (0, 1, 12_345, p.GATE9_VALIDATION_EPISODES - 1):
            ordinal = p.validation_operator_ordinal(index)
            self.assertEqual(p.inverse_validation_episode_index(ordinal), index)

    def test_queries_are_novel_and_balanced(self):
        support = set(p.GATE9_SUPPORT_INPUTS)
        for seed in p.GATE9_CHECKPOINT_SEEDS:
            counts = {value: 0 for value in p.GATE9_NOVEL_QUERY_VALUES}
            for ordinal in range(p.GATE9_TRAIN_OPERATOR_COUNT):
                query = p.training_query(seed, ordinal)
                self.assertNotIn(query, support)
                counts[query] += 1
            self.assertLessEqual(max(counts.values()) - min(counts.values()), 1)
        counts = {value: 0 for value in p.GATE9_NOVEL_QUERY_VALUES}
        for ordinal in range(p.GATE9_VALIDATION_OPERATOR_COUNT):
            query = p.validation_query(ordinal)
            self.assertNotIn(query, support)
            counts[query] += 1
        self.assertLessEqual(max(counts.values()) - min(counts.values()), 1)

    def test_learning_rate_schedule_hits_exact_boundaries(self):
        self.assertAlmostEqual(p.learning_rate_at_step(1), 1e-3 / 16)
        self.assertAlmostEqual(p.learning_rate_at_step(16), 1e-3)
        self.assertLess(p.learning_rate_at_step(17), 1e-3)
        self.assertAlmostEqual(p.learning_rate_at_step(512), 1e-4)
        values = [p.learning_rate_at_step(step) for step in range(16, 513)]
        self.assertTrue(
            all(left >= right for left, right in zip(values, values[1:]))
        )
        with self.assertRaisesRegex(ValueError, "outside"):
            p.learning_rate_at_step(0)

    def test_checkpoint_admission_is_all_seed_fixed_step_and_context_guarded(self):
        rows = tuple(self._row(seed) for seed in p.GATE9_CHECKPOINT_SEEDS)
        self.assertEqual(
            p.classify_gate9_checkpoint_admission(rows),
            p.GATE9_CHECKPOINTS_ADMITTED,
        )
        failed = list(rows)
        failed[1] = self._row(1, accuracy=0.994)
        self.assertEqual(
            p.classify_gate9_checkpoint_admission(tuple(failed)),
            p.GATE9_CHECKPOINT_ADMISSION_FAILED,
        )
        shortcut = list(rows)
        shortcut[2] = self._row(2, shuffled=0.60)
        self.assertEqual(
            p.classify_gate9_checkpoint_admission(tuple(shortcut)),
            p.GATE9_CHECKPOINT_ADMISSION_FAILED,
        )
        with self.assertRaisesRegex(ValueError, "ordered seeds"):
            p.classify_gate9_checkpoint_admission(tuple(reversed(rows)))

        duplicate_payload = {
            field: getattr(rows[1], field) for field in rows[1].__dataclass_fields__
        }
        duplicate_payload["checkpoint_sha256"] = rows[0].checkpoint_sha256
        duplicate = (
            rows[0],
            p.Gate9CheckpointValidationEvidence(**duplicate_payload),
            rows[2],
        )
        with self.assertRaisesRegex(ValueError, "identities must be distinct"):
            p.classify_gate9_checkpoint_admission(duplicate)

    def test_evidence_fails_closed_on_step_coverage_budget_hash_and_finiteness(self):
        row = self._row(0)
        for replacement, pattern in (
            ({"checkpoint_step": 511}, "fixed final"),
            ({"unique_train_operators": 262_143}, "training coverage"),
            ({"learned_parameter_count": 19_650}, "parameter count"),
            ({"tensor_count": 16}, "tensor count"),
            ({"checkpoint_sha256": "x" * 64}, "SHA-256"),
            ({"parameters_finite": False}, "non-finite"),
            ({"final_train_loss": math.inf}, "loss"),
        ):
            payload = {
                field: getattr(row, field) for field in row.__dataclass_fields__
            }
            payload.update(replacement)
            with self.assertRaisesRegex(ValueError, pattern):
                p.Gate9CheckpointValidationEvidence(**payload).validate()

    def test_protocol_has_no_tensor_optimizer_training_or_artifact_surface(self):
        text = PROTOCOL_PATH.read_text(encoding="utf-8")
        for token in (
            "import torch",
            "import numpy",
            "torch.optim",
            "AdamW(",
            "open(",
            "read_text(",
            "json.load",
            "torch.save(",
            "torch.load(",
        ):
            self.assertNotIn(token, text)


if __name__ == "__main__":
    unittest.main()
