from __future__ import annotations

import ast
import importlib.util
import pathlib
import sys
import unittest

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
CONTRACT_PATH = REPO_ROOT / "ai_hypothesis/population_compute/gate8_v1_gemma_reference_execution.py"
RUNNER_PATH = REPO_ROOT / "scripts/run_gate8_v1_gemma_reference.py"
WRAPPER_PATH = REPO_ROOT / "scripts/run_gate8_v1_gemma_reference.ps1"


def load_contract():
    name = "gate8_v1_gemma_reference_execution_test_contract"
    spec = importlib.util.spec_from_file_location(name, CONTRACT_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load Gate8 v1 reference contract")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


contract = load_contract()


class Gate8V1GemmaReferenceExecutionTests(unittest.TestCase):
    def test_exact_plan(self) -> None:
        plan = contract.gate8_v1_gemma_reference_execution_plan()
        self.assertEqual(plan["population_result_head"], "14636d219781381853f81036b96c691b7e6997ee")
        self.assertEqual(plan["reference_repo_id"], "google/gemma-3-1b-it")
        self.assertEqual(plan["reference_revision"], "dcc83ea841ab6100d6b47a070329e1ba4cf78752")
        self.assertEqual(plan["reference_parameter_count"], 999_885_952)
        self.assertEqual(plan["reference_max_input_tokens"], 24_576)
        self.assertEqual(plan["reference_max_new_tokens"], 64)
        self.assertEqual(plan["reference_demonstrations"], 8)
        self.assertEqual(plan["reference_decoding"], "greedy_temperature_0")
        self.assertEqual(plan["reference_batch_size"], 1)
        self.assertEqual(plan["condition_count"], 21)
        self.assertEqual(plan["reference_rows"], 10_752)
        self.assertTrue(plan["prompt_index_completed_before_model_load"])
        self.assertTrue(plan["transactional_resume"])
        self.assertFalse(plan["training_admitted"])
        self.assertFalse(plan["population_execution_admitted"])
        self.assertFalse(plan["joint_reference_comparison_admitted"])
        self.assertFalse(plan["final_classifier_admitted"])

    def test_frozen_condition_order_and_sequences(self) -> None:
        self.assertEqual(len(contract.GATE8_V1_VALID_CONDITIONS), 21)
        observed = []
        for population, depth in contract.GATE8_V1_VALID_CONDITIONS:
            for world_index in range(512):
                observed.append(
                    contract.gate8_v1_reference_sequence(
                        population, depth, world_index
                    )
                )
        self.assertEqual(observed, list(range(10_752)))
        with self.assertRaises(ValueError):
            contract.gate8_v1_reference_sequence(32, 8, 0)
        with self.assertRaises(ValueError):
            contract.gate8_v1_reference_sequence(32, 4, 512)

    def test_exact_answer_parser(self) -> None:
        self.assertEqual(contract.parse_gate8_v1_reference_answer("a\n"), 10)
        self.assertEqual(contract.parse_gate8_v1_reference_answer(" F "), 15)
        for invalid in ("", "10", "A.", "answer=A", "G", "A B"):
            with self.assertRaises(ValueError):
                contract.parse_gate8_v1_reference_answer(invalid)

    def test_prompt_and_result_contracts(self) -> None:
        prompt = contract.Gate8V1ReferencePromptRow(
            sequence=0,
            population=32,
            depth=4,
            world_index=0,
            world_id="g8_" + "0" * 24,
            prompt_sha256="1" * 64,
            ascii_bytes=3000,
            input_tokens=2800,
            answer_symbol=10,
        )
        prompt.validate()
        valid = contract.Gate8V1ReferenceResultRow(
            **prompt.to_dict(),
            generated_text="A\n",
            output_token_ids=(10, 1),
            predicted_symbol=10,
            parse_status="valid",
            correct=True,
            wall_seconds=0.5,
            peak_device_bytes=100,
        )
        valid.validate()
        invalid = contract.Gate8V1ReferenceResultRow(
            **prompt.to_dict(),
            generated_text="The answer is A",
            output_token_ids=(1, 2, 3),
            predicted_symbol=None,
            parse_status="invalid",
            correct=False,
            wall_seconds=0.5,
            peak_device_bytes=100,
        )
        invalid.validate()
        self.assertEqual(
            contract.validate_gate8_v1_result_prefix((valid,), (prompt,)),
            (valid,),
        )

    def test_exact_external_file_bindings(self) -> None:
        self.assertEqual(
            tuple(contract.GATE8_V1_REQUIRED_TOKENIZER_FILE_SHA256),
            (
                "added_tokens.json",
                "config.json",
                "special_tokens_map.json",
                "tokenizer.json",
                "tokenizer.model",
                "tokenizer_config.json",
            ),
        )
        self.assertEqual(
            tuple(contract.GATE8_V1_REQUIRED_MODEL_FILE_SHA256),
            ("config.json", "generation_config.json", "model.safetensors"),
        )
        self.assertEqual(
            contract.GATE8_V1_REQUIRED_MODEL_FILE_SHA256["model.safetensors"],
            "3d4ef8d71c14db7e448a09ebe891cfb6bf32c57a9b44499ae0d1c098e48516b6",
        )
        self.assertEqual(
            contract.GATE8_V1_REQUIRED_SOFTWARE,
            {
                "python": "3.11.9",
                "torch": "2.9.1+cu130",
                "transformers": "4.57.6",
                "tokenizers": "0.22.2",
                "numpy": "2.3.5",
                "huggingface-hub": "0.36.2",
            },
        )

    def test_runner_keeps_closed_scientific_boundaries(self) -> None:
        source = RUNNER_PATH.read_text(encoding="utf-8")
        self.assertIn("prompt_index_completed_before_model_load", CONTRACT_PATH.read_text(encoding="utf-8"))
        self.assertIn("sqlite3", source)
        self.assertIn("PRAGMA synchronous=FULL", source)
        self.assertIn("do_sample=False", source)
        self.assertIn("num_beams=1", source)
        self.assertIn("max_new_tokens=64", source)
        self.assertIn('split="test"', source)
        self.assertNotIn("classify_gate8_v1_reference_comparison", source)
        self.assertNotIn("run_gate8_v1_scientific_population", source)
        self.assertNotIn("optimizer", source.lower())
        tree = ast.parse(source)
        top_level_imports = {
            alias.name
            for node in tree.body
            if isinstance(node, (ast.Import, ast.ImportFrom))
            for alias in node.names
        }
        self.assertNotIn("torch", top_level_imports)
        self.assertNotIn("transformers", top_level_imports)
        self.assertNotIn("numpy", top_level_imports)

    def test_wrapper_smoke_precedes_snapshot_and_package_access(self) -> None:
        source = WRAPPER_PATH.read_text(encoding="utf-8")
        smoke = source.index("if ($wrapperSmoke)")
        snapshot = source.index("Resolve-Path -LiteralPath $TokenizerSnapshot")
        preflight = source.index("import importlib.metadata")
        self.assertLess(smoke, snapshot)
        self.assertLess(smoke, preflight)
        self.assertIn("--resume", source)
        self.assertIn("Joint classifier: CLOSED", source)


if __name__ == "__main__":
    unittest.main()
