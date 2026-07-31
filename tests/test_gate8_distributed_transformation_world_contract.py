from __future__ import annotations

import importlib.util
import pathlib
import sys
import unittest


WORLD_PATH = pathlib.Path(
    "ai_hypothesis/population_compute/gate8_distributed_transformation_worlds.py"
)


def _load_worlds():
    name = "gate8_distributed_transformation_worlds_test_module"
    spec = importlib.util.spec_from_file_location(name, WORLD_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load Gate8 world contract")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


w = _load_worlds()


def _keys(value):
    if isinstance(value, dict):
        result = set(value)
        for nested in value.values():
            result.update(_keys(nested))
        return result
    if isinstance(value, list):
        result = set()
        for nested in value:
            result.update(_keys(nested))
        return result
    return set()


class Gate8WorldContractTests(unittest.TestCase):
    def test_transform_library_is_bijective_unique_and_pairwise_noncommuting(self) -> None:
        w.validate_gate8_transform_library()
        expected = tuple(range(16))
        self.assertEqual(len(w.GATE8_TRANSFORM_PERMUTATIONS), 8)
        self.assertEqual(len(set(w.GATE8_TRANSFORM_PERMUTATIONS)), 8)
        for transform in w.GATE8_TRANSFORM_PERMUTATIONS:
            self.assertEqual(tuple(sorted(transform)), expected)
        for left_index in range(8):
            for right_index in range(left_index + 1, 8):
                left = w.GATE8_TRANSFORM_PERMUTATIONS[left_index]
                right = w.GATE8_TRANSFORM_PERMUTATIONS[right_index]
                left_after_right = tuple(left[right[symbol]] for symbol in range(16))
                right_after_left = tuple(right[left[symbol]] for symbol in range(16))
                self.assertNotEqual(left_after_right, right_after_left)

    def test_all_21_conditions_generate_valid_contract_worlds(self) -> None:
        self.assertEqual(len(w.GATE8_VALID_CONDITIONS), 21)
        for population, depth in w.GATE8_VALID_CONDITIONS:
            generated = w.generate_gate8_world(
                split="contract",
                seed=0,
                world_index=0,
                population=population,
                depth=depth,
            )
            generated.validate()
            public = generated.public
            truth = generated.truth
            self.assertEqual(len(public.workers), population)
            self.assertEqual(len(truth.relevant_worker_indices), depth)
            self.assertLessEqual(depth / population, 1 / 8)
            self.assertEqual(
                tuple(worker.worker_index for worker in public.workers),
                tuple(range(population)),
            )
            transform_counts = tuple(
                sum(worker.transform_id == transform_id for worker in public.workers)
                for transform_id in range(8)
            )
            self.assertEqual(transform_counts, (population // 8,) * 8)
            ordered = sorted(truth.relevant_worker_indices)
            self.assertNotEqual(ordered, list(range(ordered[0], ordered[0] + depth)))

    def test_generation_is_deterministic_and_world_namespaces_are_disjoint(self) -> None:
        left = w.generate_gate8_world(
            split="contract", seed=0, world_index=7, population=128, depth=16
        )
        right = w.generate_gate8_world(
            split="contract", seed=0, world_index=7, population=128, depth=16
        )
        other_world = w.generate_gate8_world(
            split="contract", seed=0, world_index=8, population=128, depth=16
        )
        other_seed = w.generate_gate8_world(
            split="contract", seed=1, world_index=7, population=128, depth=16
        )
        self.assertEqual(left, right)
        self.assertNotEqual(left.public.world_id, other_world.public.world_id)
        self.assertNotEqual(left.public.world_id, other_seed.public.world_id)
        left_labels = {
            label
            for edge in left.public.workers
            for label in (edge.source_node, edge.target_node)
        }
        other_labels = {
            label
            for edge in other_world.public.workers
            for label in (edge.source_node, edge.target_node)
        }
        self.assertTrue(left_labels.isdisjoint(other_labels))

    def test_public_surface_contains_no_answer_or_path_metadata(self) -> None:
        generated = w.generate_gate8_world(
            split="contract", seed=0, world_index=3, population=64, depth=8
        )
        payload = generated.public.to_dict()
        forbidden = {
            "answer",
            "answer_symbol",
            "truth",
            "relevant",
            "relevant_worker_indices",
            "path",
            "path_worker_indices",
            "path_transform_ids",
        }
        self.assertTrue(forbidden.isdisjoint(_keys(payload)))
        self.assertEqual(
            set(payload["workers"][0]),
            {"worker_index", "source_node", "target_node", "transform_id"},
        )

    def test_exact_oracle_uses_public_graph_and_preserves_order(self) -> None:
        generated = w.generate_gate8_world(
            split="contract", seed=0, world_index=11, population=256, depth=32
        )
        oracle = w.gate8_exact_symbolic_oracle(generated.public)
        self.assertEqual(oracle.answer_symbol, generated.truth.answer_symbol)
        self.assertEqual(
            oracle.path_worker_indices,
            generated.truth.relevant_worker_indices,
        )
        self.assertEqual(len(oracle.path_transform_ids), generated.public.depth)
        symbol = generated.public.query.root_symbol
        for transform_id in oracle.path_transform_ids:
            symbol = w.apply_gate8_transform(transform_id, symbol)
        self.assertEqual(symbol, oracle.answer_symbol)

    def test_graph_is_one_rooted_tree_with_unique_root_target_path(self) -> None:
        generated = w.generate_gate8_world(
            split="contract", seed=0, world_index=4, population=1024, depth=128
        )
        public = generated.public
        incoming = {}
        nodes = {public.query.root_node}
        for edge in public.workers:
            nodes.update((edge.source_node, edge.target_node))
            self.assertNotIn(edge.target_node, incoming)
            incoming[edge.target_node] = edge.source_node
        self.assertEqual(len(nodes), public.population + 1)
        self.assertNotIn(public.query.root_node, incoming)
        self.assertEqual(set(incoming), nodes - {public.query.root_node})
        current = public.query.target_node
        distance = 0
        while current != public.query.root_node:
            current = incoming[current]
            distance += 1
        self.assertEqual(distance, public.depth)

    def test_frozen_split_bounds_reject_out_of_range_indices_before_generation(self) -> None:
        with self.assertRaisesRegex(ValueError, "test world index"):
            w.generate_gate8_world(
                split="test", seed=0, world_index=512, population=32, depth=4
            )
        with self.assertRaisesRegex(ValueError, "training world index"):
            w.generate_gate8_world(
                split="train", seed=0, world_index=262_144, population=32, depth=4
            )
        with self.assertRaisesRegex(ValueError, "demonstration index"):
            w.generate_gate8_world(
                split="demonstration", seed=0, world_index=8, population=32, depth=4
            )

    def test_world_contract_binds_protocol_and_correction_heads(self) -> None:
        self.assertEqual(
            w.GATE8_WORLD_CONTRACT_PROTOCOL_HEAD,
            "e73541115e8ddd122f336463dc1a9ffdbf82df46",
        )
        self.assertEqual(
            w.GATE8_WORLD_CONTRACT_CORRECTION_HEAD,
            "124065691d257d483a37be4200452f1f7ca50063",
        )
        self.assertEqual(
            w.GATE8_WORLD_CONTRACT_STATUS,
            "GATE8_WORLD_GENERATOR_AND_SYMBOLIC_ORACLE_ADMITTED_EXECUTION_CLOSED",
        )

    def test_contract_module_has_no_model_training_or_encoder_surface(self) -> None:
        text = WORLD_PATH.read_text(encoding="utf-8")
        for token in (
            "import torch",
            "import numpy",
            "transformers",
            "AutoModel",
            "AutoTokenizer",
            "optimizer",
            "backward(",
            "tokenizer",
            "prompt_template",
            "bfloat16",
            "cuda",
        ):
            self.assertNotIn(token, text)


if __name__ == "__main__":
    unittest.main()
