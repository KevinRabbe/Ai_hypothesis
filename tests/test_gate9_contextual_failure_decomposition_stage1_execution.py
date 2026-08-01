from __future__ import annotations

import importlib.util
import json
import math
import pathlib
import sys
import tempfile
import unittest

import torch
import torch.nn.functional as F

ROOT = pathlib.Path(__file__).resolve().parents[1]
RUNTIME_PATH = ROOT / (
    "ai_hypothesis/population_compute/"
    "gate9_contextual_failure_decomposition_stage1_runtime.py"
)
CLI_PATH = ROOT / "scripts/run_gate9_contextual_failure_decomposition_stage1.py"
WRAPPER_PATH = ROOT / "scripts/run_gate9_contextual_failure_decomposition_stage1.ps1"


def load_runtime():
    name = "gate9d_stage1_test_runtime"
    spec = importlib.util.spec_from_file_location(name, RUNTIME_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load Gate9D stage-1 runtime")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


runtime = load_runtime()


class Gate9FailureDecompositionStage1ExecutionTests(unittest.TestCase):
    def test_exact_stage_identity_and_schedule(self) -> None:
        self.assertEqual(
            runtime.GATE9D_STAGE1_EXECUTION_VERSION,
            "gate9-contextual-failure-decomposition-stage1-execution-v0",
        )
        self.assertEqual(
            runtime.GATE9D_STAGE1_EXECUTION_BRANCH,
            "agent/gate9-contextual-failure-decomposition-stage1-execution-v0",
        )
        self.assertEqual(
            runtime.GATE9D_PROTOCOL_HEAD,
            "8deca15aef78d8636b07570aff044f9b7ae31928",
        )
        self.assertEqual(runtime.GATE9D_STAGE_NAME, "single_operator_query_fit")
        self.assertEqual(runtime.GATE9D_STAGE.order, 1)
        self.assertFalse(runtime.GATE9D_STAGE.requires_context_causality)
        self.assertFalse(runtime.GATE9D_STAGE.unseen_operator_evaluation)
        self.assertEqual(runtime.GATE9D_TRAIN_STEPS, 1_024)
        self.assertEqual(runtime.GATE9D_BATCH_SIZE, 247)
        self.assertEqual(runtime.GATE9D_TRAIN_EXAMPLES, 247)

    def test_exact_material_and_public_oracle(self) -> None:
        first = runtime.stage1_material()
        second = runtime.stage1_material()
        self.assertEqual(first, second)
        self.assertEqual(
            first["operator_counter"],
            runtime.protocol.GATE9D_DIAGNOSTIC_COUNTER_BASE,
        )
        self.assertEqual(len(first["support"]), 9)
        self.assertEqual(
            tuple(source for source, _ in first["support"]),
            tuple(runtime.architecture.GATE9_SUPPORT_ORDER),
        )
        self.assertEqual(len(first["queries"]), 247)
        self.assertEqual(len(set(first["queries"])), 247)
        self.assertFalse(
            set(first["queries"]) & set(runtime.protocol.GATE9D_SUPPORT_INPUTS)
        )
        self.assertEqual(first["targets"], first["oracle_targets"])
        digest = runtime.stage1_dataset_sha256(first)
        self.assertEqual(len(digest), 64)
        self.assertEqual(digest, runtime.stage1_dataset_sha256(second))

    def test_learning_rate_schedule_is_fixed(self) -> None:
        self.assertEqual(runtime.learning_rate_at_step(1), 0.001 / 16)
        self.assertEqual(runtime.learning_rate_at_step(16), 0.001)
        self.assertEqual(runtime.learning_rate_at_step(1_024), 0.0001)
        self.assertGreater(runtime.learning_rate_at_step(17), 0.0001)
        self.assertLess(runtime.learning_rate_at_step(1_023), 0.001)
        with self.assertRaises(ValueError):
            runtime.learning_rate_at_step(0)
        with self.assertRaises(ValueError):
            runtime.learning_rate_at_step(1_025)

    def test_cpu_forward_backward_and_checkpoint_contract(self) -> None:
        initialization_seed = runtime.configure_determinism(0)
        self.assertEqual(initialization_seed, 910_900)
        material = runtime.stage1_material()
        device = torch.device("cpu")
        support_inputs, support_outputs, queries, targets = runtime.tensor_batch(
            material, device
        )
        self.assertEqual(tuple(support_inputs.shape), (247, 9))
        self.assertEqual(tuple(support_outputs.shape), (247, 9))
        self.assertEqual(tuple(queries.shape), (247,))
        self.assertEqual(tuple(targets.shape), (247,))

        model = runtime.architecture.Gate9ContextualWorker().to(
            device=device, dtype=torch.float32
        )
        logits = model(support_inputs, support_outputs, queries)
        loss = F.binary_cross_entropy_with_logits(
            logits, runtime.target_bits(targets)
        )
        self.assertTrue(bool(torch.isfinite(loss)))
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), runtime.protocol.GATE9D_GRADIENT_CLIP_NORM
        )
        self.assertTrue(bool(torch.isfinite(gradient_norm)))

        state = runtime.validate_model_state(model)
        self.assertEqual(state["tensor_count"], 17)
        self.assertEqual(state["learned_parameter_count"], 19_649)
        payload = runtime._checkpoint_payload(
            model,
            seed_index=0,
            initialization_seed=initialization_seed,
            dataset_sha256=runtime.stage1_dataset_sha256(material),
        )
        self.assertEqual(
            set(payload),
            {
                "experiment_version",
                "protocol_head",
                "architecture_head",
                "operator_contract_head",
                "stage",
                "seed_index",
                "initialization_seed",
                "step",
                "unique_train_examples",
                "dataset_sha256",
                "learned_parameter_count",
                "tensor_count",
                "state_dict",
            },
        )
        self.assertEqual(payload["step"], 1_024)
        self.assertEqual(payload["unique_train_examples"], 247)
        self.assertEqual(payload["tensor_count"], 17)
        self.assertEqual(payload["learned_parameter_count"], 19_649)

    def test_evaluation_ledger_is_complete_and_oracle_exact(self) -> None:
        runtime.configure_determinism(1)
        material = runtime.stage1_material()
        model = runtime.architecture.Gate9ContextualWorker().to(
            device="cpu", dtype=torch.float32
        )
        with tempfile.TemporaryDirectory() as directory:
            path = pathlib.Path(directory) / "evaluation.jsonl"
            evidence = runtime.evaluate_stage1(
                model=model,
                material=material,
                device=torch.device("cpu"),
                output_path=path,
            )
            rows = [json.loads(line) for line in path.read_text().splitlines()]
        self.assertEqual(len(rows), 247)
        self.assertEqual(
            [row["episode_index"] for row in rows], list(range(247))
        )
        self.assertEqual(len({row["query"] for row in rows}), 247)
        self.assertTrue(all(row["oracle_correct"] for row in rows))
        self.assertEqual(evidence["rows"], 247)
        self.assertEqual(evidence["oracle_correct"], 247)
        self.assertEqual(evidence["oracle_accuracy"], 1.0)
        self.assertTrue(0.0 <= evidence["exact_accuracy"] <= 1.0)
        self.assertTrue(0.0 <= evidence["bit_accuracy"] <= 1.0)
        self.assertTrue(0.0 <= evidence["query_only_accuracy"] <= 1.0)
        expected_pass = (
            evidence["exact_accuracy"]
            >= runtime.protocol.GATE9D_EXACT_ACCURACY_MIN
            and evidence["bit_accuracy"]
            >= runtime.protocol.GATE9D_BIT_ACCURACY_MIN
        )
        self.assertIs(evidence["stage_passes"], expected_pass)

    def test_source_and_wrapper_keep_later_stages_closed(self) -> None:
        runtime_source = RUNTIME_PATH.read_text(encoding="utf-8")
        cli_source = CLI_PATH.read_text(encoding="utf-8")
        wrapper_bytes = WRAPPER_PATH.read_bytes()
        wrapper_source = wrapper_bytes.decode("ascii")

        self.assertIn("GATE9D_STAGES[GATE9D_STAGE_INDEX]", runtime_source)
        self.assertIn("GATE9D_STAGE_INDEX = 0", runtime_source)
        for token in (
            "GATE9D_COLLISION_OPERATOR_COUNTERS",
            "GATE9D_HELD_IN_OPERATOR_RANGE",
            "GATE9D_UNSEEN_TRAIN_OPERATOR_RANGE",
            "GATE9D_UNSEEN_EVAL_OPERATOR_RANGE",
            "generate_gate9_test_world(",
            "scientific_assignment_key",
            "classify_diagnostic(",
        ):
            self.assertNotIn(token, runtime_source)
            self.assertNotIn(token, cli_source)
        self.assertIn("GATE9D_STAGE1_WRAPPER_SMOKE", wrapper_source)
        self.assertNotIn("Stage 2", wrapper_source)
        self.assertNotIn("Stage 3", wrapper_source)
        self.assertNotIn("Stage 4", wrapper_source)


if __name__ == "__main__":
    unittest.main()
