from __future__ import annotations

import dataclasses
import importlib.util
import pathlib
import sys
import unittest

WORLD_PATH = pathlib.Path(
    "ai_hypothesis/population_compute/gate9_contextual_graph_world_contract.py"
)


def _load_worlds():
    name = "gate9_contextual_graph_world_contract_test_module"
    spec = importlib.util.spec_from_file_location(name, WORLD_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load Gate9 graph-world contract")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


w = _load_worlds()
p = w.protocol


class Gate9GraphWorldContractTests(unittest.TestCase):
    def test_contract_worlds_are_deterministic_valid_and_public_oracle_exact(self):
        conditions = ((32, 4), (128, 16), (512, 64), (1024, 128))
        identities = set()
        saw_support_hit = False
        saw_noncontiguous_path_order = False
        for population, depth in conditions:
            for world_index in (0, 1, 7):
                public, truth = w.generate_gate9_contract_world(
                    population=population,
                    depth=depth,
                    world_index=world_index,
                )
                repeated = w.generate_gate9_contract_world(
                    population=population,
                    depth=depth,
                    world_index=world_index,
                )
                self.assertEqual((public, truth), repeated)
                public.validate()
                truth.validate(public)
                answer, hits = w.gate9_public_support_path_oracle(public)
                self.assertEqual(answer, truth.answer_symbol)
                self.assertEqual(hits, truth.support_hits_by_path_position)
                self.assertEqual(len(truth.path_worker_indices), depth)
                self.assertEqual(len(set(truth.operator_counters_by_worker)), population)
                identities.add(public.world_id)
                saw_support_hit |= any(hits)
                saw_noncontiguous_path_order |= truth.path_worker_indices != tuple(
                    range(depth)
                )
        self.assertEqual(len(identities), len(conditions) * 3)
        self.assertTrue(saw_support_hit)
        self.assertTrue(saw_noncontiguous_path_order)

    def test_complete_contract_operator_allocation_is_one_exact_permutation(self):
        observed = set()
        for population, depth in p.GATE9_VALID_CONDITIONS:
            for world_index in range(w.GATE9_CONTRACT_WORLDS_PER_CONDITION):
                for edge_index in range(population):
                    observed.add(
                        w._contract_operator_counter(
                            population=population,
                            depth=depth,
                            world_index=world_index,
                            canonical_edge_index=edge_index,
                        )
                    )
        expected = set(
            range(
                w.GATE9_CONTRACT_OPERATOR_COUNTER_START,
                w.GATE9_CONTRACT_OPERATOR_COUNTER_START
                + w.GATE9_CONTRACT_OPERATOR_COUNT,
            )
        )
        self.assertEqual(observed, expected)

    def test_test_assignment_is_bijective_but_generation_key_remains_unbound(self):
        plan = w.gate9_graph_world_contract_plan()
        self.assertFalse(plan["test_assignment_key_bound"])
        self.assertIsNone(plan["test_assignment_multiplier"])
        self.assertFalse(plan["scientific_test_world_generation_admitted"])
        self.assertTrue(plan["test_operator_allocation_bijective_after_key_binding"])

        key = "1" * 64
        size = p.GATE9_GRAPH_TEST_OPERATOR_COUNT
        samples = (0, 1, 2, size // 2, size - 2, size - 1)
        mapped = tuple(w.permute_operator_ordinal(value, size, key) for value in samples)
        self.assertEqual(len(mapped), len(set(mapped)))
        self.assertEqual(
            tuple(w.invert_operator_ordinal(value, size, key) for value in mapped),
            samples,
        )
        counters = {
            w.graph_test_operator_counter(
                population=population,
                depth=depth,
                world_index=world_index,
                canonical_edge_index=edge,
                assignment_key=key,
            )
            for population, depth, world_index, edge in (
                (32, 4, 0, 0),
                (64, 8, 255, 63),
                (1024, 128, 255, 1023),
            )
        }
        self.assertEqual(len(counters), 3)
        self.assertTrue(
            all(
                p.GATE9_GRAPH_TEST_OPERATOR_COUNTER_START
                <= value
                < p.GATE9_GRAPH_TEST_OPERATOR_COUNTER_START
                + p.GATE9_GRAPH_TEST_OPERATOR_COUNT
                for value in counters
            )
        )
        with self.assertRaisesRegex(ValueError, "cannot reuse contract"):
            w.graph_test_operator_counter(
                population=32,
                depth=4,
                world_index=0,
                canonical_edge_index=0,
                assignment_key=w.GATE9_CONTRACT_ASSIGNMENT_KEY,
            )

    def test_public_schema_exposes_no_counter_key_truth_or_canonical_index(self):
        public, truth = w.generate_gate9_contract_world(
            population=64, depth=8, world_index=3
        )
        public_fields = {
            field.name for field in dataclasses.fields(w.Gate9GraphWorkerPublic)
        }
        self.assertEqual(
            public_fields,
            {"worker_index", "source_node", "target_node", "support_pairs"},
        )
        serialized = public.to_dict()
        text = repr(serialized).lower()
        for token in ("counter", "operator_key", "canonical", "truth", "answer"):
            self.assertNotIn(token, text)
        self.assertNotEqual(public.world_id, truth.world_id + "x")

    def test_tampering_topology_support_truth_and_identity_fails_closed(self):
        public, truth = w.generate_gate9_contract_world(
            population=64, depth=8, world_index=4
        )
        workers = list(public.workers)
        workers[1] = dataclasses.replace(
            workers[1], target_node=workers[0].target_node
        )
        with self.assertRaisesRegex(ValueError, "multiple incoming"):
            dataclasses.replace(public, workers=tuple(workers)).validate()

        support = list(public.workers[0].support_pairs)
        support[-1] = support[0]
        bad_worker = dataclasses.replace(
            public.workers[0], support_pairs=tuple(support)
        )
        with self.assertRaisesRegex(ValueError, "duplicate"):
            bad_worker.validate()

        with self.assertRaisesRegex(ValueError, "identity drifted"):
            dataclasses.replace(public, world_id="0" * 64).validate()

        with self.assertRaisesRegex(ValueError, "oracle"):
            dataclasses.replace(truth, answer_symbol=truth.answer_symbol ^ 1).validate(
                public
            )

    def test_contract_plan_keeps_science_model_and_training_closed(self):
        plan = w.gate9_graph_world_contract_plan()
        self.assertEqual(plan["query_policy_correction_head"], w.GATE9_QUERY_POLICY_CORRECTION_HEAD)
        self.assertEqual(plan["operator_contract_head"], w.GATE9_OPERATOR_CONTRACT_HEAD)
        self.assertTrue(plan["contract_world_generation_admitted"])
        self.assertTrue(plan["support_hit_reporting_required"])
        self.assertFalse(plan["public_operator_counter_exposed"])
        self.assertFalse(plan["public_operator_key_exposed"])
        for key in (
            "scientific_test_world_generation_admitted",
            "architecture_admitted",
            "training_admitted",
            "checkpoint_loading_admitted",
            "scientific_execution_admitted",
            "result_classification_admitted",
        ):
            self.assertFalse(plan[key])

        source = WORLD_PATH.read_text(encoding="utf-8")
        for token in (
            "import torch",
            "import numpy",
            "torch.load(",
            "torch.optim",
            ".backward(",
            "generate_gate9_test_world(",
            "from_pretrained(",
            "json.load",
        ):
            self.assertNotIn(token, source)


if __name__ == "__main__":
    unittest.main()
