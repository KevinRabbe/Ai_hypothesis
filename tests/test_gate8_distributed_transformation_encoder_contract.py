from __future__ import annotations

import importlib.util
import pathlib
import sys
import unittest


WORLD_PATH = pathlib.Path(
    "ai_hypothesis/population_compute/gate8_distributed_transformation_worlds.py"
)
ENCODER_PATH = pathlib.Path(
    "ai_hypothesis/population_compute/gate8_distributed_transformation_encoder.py"
)


def _load(path: pathlib.Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


w = _load(WORLD_PATH, "gate8_worlds_encoder_test_module")
e = _load(ENCODER_PATH, "gate8_encoder_test_module")


def _demonstrations():
    return tuple(
        w.generate_gate8_world(
            split="demonstration",
            seed=0,
            world_index=world_index,
            population=32,
            depth=4,
        )
        for world_index in range(8)
    )


class Gate8EncoderContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.demonstrations = _demonstrations()

    def test_encoder_transform_table_exactly_matches_world_library(self) -> None:
        observed = tuple(
            tuple(int(symbol, 16) for symbol in encoded)
            for encoded in e.GATE8_TRANSFORM_HEX
        )
        self.assertEqual(observed, w.GATE8_TRANSFORM_PERMUTATIONS)

    def test_aliases_are_fixed_width_unique_and_topology_neutral(self) -> None:
        generated = w.generate_gate8_world(
            split="contract", seed=0, world_index=7, population=1024, depth=128
        )
        aliases = e.gate8_node_aliases(generated.public)
        self.assertEqual(len(aliases), 1025)
        self.assertEqual(len(set(aliases.values())), 1025)
        self.assertTrue(all(len(alias) == 2 for alias in aliases.values()))
        self.assertEqual(
            tuple(aliases[label] for label in sorted(aliases)),
            tuple(e._base36_fixed(index) for index in range(1025)),
        )

    def test_public_graph_encoder_is_deterministic_complete_and_ascii(self) -> None:
        generated = w.generate_gate8_world(
            split="contract", seed=0, world_index=2, population=256, depth=32
        )
        left = e.encode_gate8_public_graph(generated.public)
        right = e.encode_gate8_public_graph(generated.public)
        self.assertEqual(left, right)
        left.encode("ascii")
        lines = left.splitlines()
        self.assertEqual(len(lines), 257)
        self.assertTrue(lines[0].startswith("Q|P=256|D=32|R="))
        self.assertEqual(sum(">" in line and ":" in line for line in lines[1:]), 256)

    def test_worker_observation_contains_one_edge_only(self) -> None:
        generated = w.generate_gate8_world(
            split="contract", seed=0, world_index=3, population=128, depth=16
        )
        observation = e.encode_gate8_worker_observation(generated.public, 17)
        observation.encode("ascii")
        self.assertTrue(observation.startswith("G8W0|P=128|D=16|"))
        self.assertEqual(observation.count("|E="), 1)
        self.assertEqual(observation.count(">"), 1)
        self.assertIn("|I=0H|", observation)

    def test_reference_prompt_freezes_eight_public_demonstrations(self) -> None:
        target = w.generate_gate8_world(
            split="contract", seed=0, world_index=5, population=512, depth=64
        )
        prompt = e.encode_gate8_reference_prompt(target.public, self.demonstrations)
        self.assertEqual(sum(line in {f"D{index}" for index in range(8)} for line in prompt.splitlines()), 8)
        self.assertTrue(prompt.endswith("\nY="))
        self.assertEqual(prompt.splitlines()[-1], "Y=")
        self.assertEqual(prompt.count("\nX\n"), 1)
        self.assertEqual(prompt, e.encode_gate8_reference_prompt(target.public, self.demonstrations))
        self.assertEqual(len(e.gate8_reference_prompt_sha256(prompt)), 64)

    def test_all_21_target_conditions_fit_the_frozen_ascii_budget(self) -> None:
        observed = []
        for population, depth in w.GATE8_VALID_CONDITIONS:
            target = w.generate_gate8_world(
                split="contract",
                seed=0,
                world_index=0,
                population=population,
                depth=depth,
            )
            prompt = e.encode_gate8_reference_prompt(target.public, self.demonstrations)
            budget = e.validate_gate8_reference_prompt_budget(prompt)
            observed.append((population, depth, budget.ascii_bytes))
            self.assertLessEqual(
                budget.ascii_bytes,
                e.GATE8_REFERENCE_CONTENT_ASCII_BYTE_LIMIT,
            )
            self.assertFalse(budget.exact_tokenizer_bound_proven)
        largest = max(observed, key=lambda row: row[2])
        self.assertEqual(largest[:2], (1024, 128))
        self.assertLess(largest[2], 12_000)

    def test_reference_prompt_rejects_noncanonical_demonstrations(self) -> None:
        target = w.generate_gate8_world(
            split="contract", seed=0, world_index=0, population=32, depth=4
        )
        with self.assertRaisesRegex(ValueError, "exactly eight"):
            e.encode_gate8_reference_prompt(target.public, self.demonstrations[:-1])
        reordered = tuple(reversed(self.demonstrations))
        with self.assertRaisesRegex(ValueError, "ordered 0..7"):
            e.encode_gate8_reference_prompt(target.public, reordered)
        wrong_split = tuple(
            w.generate_gate8_world(
                split="contract",
                seed=0,
                world_index=index,
                population=32,
                depth=4,
            )
            for index in range(8)
        )
        with self.assertRaisesRegex(ValueError, "demonstration split"):
            e.encode_gate8_reference_prompt(target.public, wrong_split)

    def test_parser_accepts_exactly_one_hex_symbol(self) -> None:
        self.assertEqual(e.parse_gate8_reference_answer(" a\n"), 10)
        self.assertEqual(e.parse_gate8_reference_answer("F"), 15)
        for invalid in ("", "10", "Y=A", "A because", "G"):
            with self.assertRaises(ValueError):
                e.parse_gate8_reference_answer(invalid)

    def test_budget_contract_is_explicitly_pre_tokenizer(self) -> None:
        plan = e.gate8_encoder_contract_plan()
        self.assertEqual(
            plan["world_contract_head"],
            "722c646eacfd05c51fb9d1e8887fe1620d53672c",
        )
        self.assertTrue(plan["encoder_admitted"])
        self.assertFalse(plan["tokenizer_bound"])
        self.assertFalse(plan["model_bound"])
        self.assertFalse(plan["training_admitted"])
        self.assertFalse(plan["baseline_execution_admitted"])
        self.assertFalse(plan["scientific_execution_admitted"])
        self.assertEqual(
            plan["content_ascii_byte_limit"]
            + plan["template_and_special_token_reserve"],
            plan["reference_max_input_tokens"],
        )

    def test_encoder_has_no_tokenizer_model_training_or_execution_surface(self) -> None:
        text = ENCODER_PATH.read_text(encoding="utf-8")
        for token in (
            "import torch",
            "import numpy",
            "transformers",
            "AutoTokenizer",
            "AutoModel",
            "from_pretrained",
            "optimizer",
            "backward(",
            ".generate(",
            "cuda",
            "bfloat16",
        ):
            self.assertNotIn(token, text)


if __name__ == "__main__":
    unittest.main()
