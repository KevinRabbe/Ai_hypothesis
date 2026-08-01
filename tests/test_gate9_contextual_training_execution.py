from __future__ import annotations

import hashlib
import importlib.util
import pathlib
import sys
import tempfile
import unittest

import numpy as np
import torch

RUNTIME_PATH = pathlib.Path(
    "ai_hypothesis/population_compute/gate9_contextual_training_runtime.py"
)
DATA_PATH = pathlib.Path(
    "ai_hypothesis/population_compute/gate9_contextual_training_data.py"
)
SCRIPT_PATH = pathlib.Path("scripts/run_gate9_contextual_training.py")
WRAPPER_PATH = pathlib.Path("scripts/run_gate9_contextual_training.ps1")


def _load(path: pathlib.Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


r = _load(RUNTIME_PATH, "gate9_training_execution_test_runtime")
s = _load(SCRIPT_PATH, "gate9_training_execution_test_script")
p = r.protocol


class Gate9TrainingExecutionTests(unittest.TestCase):
    def test_fast_material_matches_public_support_oracle(self):
        counters = (0, 1, 2, 17, 2**32, 2**40, 2**48, 2**60, 2**64 - 1)
        queries = (3, 5, 6, 7, 9, 10, 11, 12, 255)
        identities = set()
        for counter, query in zip(counters, queries, strict=True):
            support, target = r.data.fast_operator_material(counter, query)
            self.assertEqual(
                tuple(source for source, _ in support),
                r.data.operators.GATE9_GLOBAL_SUPPORT_ORDER,
            )
            self.assertEqual(
                r.data.operators.apply_public_support_oracle(
                    support, query, require_novel_query=True
                ),
                target,
            )
            identities.add(
                hashlib.sha256(repr(support).encode("ascii")).hexdigest()
            )
        self.assertEqual(len(identities), len(counters))

    def test_training_batches_follow_exact_protocol_without_overlap(self):
        seen = set()
        for step in (1, 2, 511, 512):
            ordinals, counters, arrays = r.data.training_batch_arrays(0, step)
            self.assertEqual(ordinals.shape, (512,))
            self.assertEqual(counters.shape, (512,))
            self.assertEqual(arrays[0].shape, (512, 9))
            self.assertEqual(arrays[1].shape, (512, 9))
            self.assertEqual(arrays[2].shape, (512,))
            self.assertEqual(arrays[3].shape, (512,))
            self.assertTrue(
                np.all(counters < p.GATE9_VALIDATION_OPERATOR_COUNTER_START)
            )
            self.assertTrue(
                all(
                    int(value) not in p.GATE9_SUPPORT_INPUTS
                    for value in arrays[2]
                )
            )
            self.assertFalse(seen.intersection(int(value) for value in ordinals))
            seen.update(int(value) for value in ordinals)
        self.assertEqual(len(seen), 2_048)

    def test_validation_shuffle_is_complete_derangement_and_batches_are_exact(self):
        mapped = {
            r.data.validation_shuffled_episode_index(index)
            for index in range(p.GATE9_VALIDATION_EPISODES)
        }
        self.assertEqual(len(mapped), p.GATE9_VALIDATION_EPISODES)
        self.assertTrue(
            all(
                r.data.validation_shuffled_episode_index(index) != index
                for index in range(p.GATE9_VALIDATION_EPISODES)
            )
        )
        (
            indices,
            ordinals,
            counters,
            arrays,
            shuffled_ordinals,
            shuffled_outputs,
        ) = r.data.validation_batch_arrays(0)
        self.assertEqual(indices, tuple(range(512)))
        self.assertEqual(arrays[0].shape, (512, 9))
        self.assertTrue(np.array_equal(arrays[3], arrays[4]))
        self.assertTrue(
            np.all(counters >= p.GATE9_VALIDATION_OPERATOR_COUNTER_START)
        )
        self.assertTrue(
            np.all(counters < p.GATE9_LOCAL_TEST_OPERATOR_COUNTER_START)
        )
        self.assertFalse(np.array_equal(ordinals, shuffled_ordinals))
        self.assertEqual(shuffled_outputs.shape, (512, 9))

    def test_contract_smoke_optimizer_step_and_state_contract(self):
        r.data.configure_determinism(0)
        model = r.architecture.Gate9ContextualWorker()
        r.data.validate_model_state(model)
        counters = tuple((1 << 60) + index for index in range(32))
        queries = tuple(p.GATE9_NOVEL_QUERY_VALUES[index] for index in range(32))
        arrays = r.data.batch_arrays(
            counters=counters,
            queries=queries,
            verify_public_oracle=True,
        )
        tensors = r.data.tensor_batch(arrays, torch.device("cpu"))
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=p.GATE9_BASE_LEARNING_RATE,
            betas=p.GATE9_ADAM_BETAS,
            eps=p.GATE9_ADAM_EPSILON,
            weight_decay=p.GATE9_WEIGHT_DECAY,
        )
        logits = model(tensors[0], tensors[1], tensors[2])
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            logits, r.data.target_bits(tensors[3])
        )
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), p.GATE9_GRADIENT_CLIP_NORM
        )
        self.assertTrue(torch.isfinite(norm))
        optimizer.step()
        r.data.validate_model_state(model)

    def test_checkpoint_payload_is_fixed_final_model_only(self):
        model = r.architecture.Gate9ContextualWorker()
        payload = r._checkpoint_payload(model, 1)
        self.assertEqual(set(payload), set(p.GATE9_CHECKPOINT_REQUIRED_FIELDS))
        self.assertEqual(payload["seed"], 1)
        self.assertEqual(payload["initialization_seed"], 900_901)
        self.assertEqual(payload["step"], 512)
        self.assertEqual(payload["train_episodes"], 262_144)
        self.assertEqual(payload["learned_parameter_count"], 19_649)
        self.assertEqual(payload["tensor_count"], 17)
        self.assertEqual(set(payload["state_dict"]), set(p.GATE9_STATE_TENSOR_SHAPES))
        self.assertNotIn("optimizer", payload)

    def test_manifest_is_recursive_sorted_and_excludes_itself(self):
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            (root / "b.txt").write_text("b\n", encoding="utf-8")
            (root / "nested").mkdir()
            (root / "nested/a.txt").write_text("a\n", encoding="utf-8")
            manifest = s._write_manifest(root)
            lines = manifest.read_text(encoding="ascii").splitlines()
            self.assertEqual(lines, sorted(lines))
            self.assertEqual(len(lines), 2)
            self.assertNotIn("manifest.sha256", manifest.read_text(encoding="ascii"))

    def test_execution_sources_keep_scientific_ranges_and_assignment_closed(self):
        runtime = RUNTIME_PATH.read_text(encoding="utf-8")
        data_source = DATA_PATH.read_text(encoding="utf-8")
        script = SCRIPT_PATH.read_text(encoding="utf-8")
        wrapper = WRAPPER_PATH.read_text(encoding="utf-8")
        for token in (
            "GATE9_LOCAL_TEST_OPERATOR_COUNTER_START",
            "GATE9_GRAPH_TEST_OPERATOR_COUNTER_START",
            "graph_test_operator_counter(",
            "generate_gate9_test_world(",
            "test_assignment_key",
            "classify_gate9(",
        ):
            self.assertNotIn(token, runtime)
            self.assertNotIn(token, data_source)
        self.assertIn(
            'EXPECTED_BRANCH = "agent/gate9-contextual-training-execution-v0"',
            script,
        )
        self.assertIn("GATE9_CONTEXTUAL_TRAINING_WRAPPER_SMOKE", wrapper)
        self.assertIn("Scientific test:   CLOSED", wrapper)


if __name__ == "__main__":
    unittest.main()
