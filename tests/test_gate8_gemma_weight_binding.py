from __future__ import annotations

import hashlib
import importlib.util
import json
import pathlib
import struct
import sys
import tempfile
import unittest


BINDING_PATH = pathlib.Path(
    "ai_hypothesis/population_compute/gate8_gemma_weight_binding.py"
)
RUNNER_PATH = pathlib.Path("scripts/bind_gate8_gemma_weights.py")
WRAPPER_PATH = pathlib.Path("scripts/bind_gate8_gemma_weights.ps1")


def _load_binding():
    name = "gate8_gemma_weight_binding_test_module"
    spec = importlib.util.spec_from_file_location(name, BINDING_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load Gate8 Gemma weight binding")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


b = _load_binding()


def _write_json(path: pathlib.Path, payload: dict) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _valid_config() -> dict:
    return {
        "architectures": ["Gemma3ForCausalLM"],
        "model_type": "gemma3_text",
        "torch_dtype": "bfloat16",
        "hidden_size": 1_152,
        "intermediate_size": 6_912,
        "num_hidden_layers": 26,
        "num_attention_heads": 4,
        "num_key_value_heads": 1,
        "max_position_embeddings": 32_768,
        "vocab_size": 262_144,
    }


def _valid_generation_config() -> dict:
    return {
        "bos_token_id": 2,
        "eos_token_id": [1, 106],
        "pad_token_id": 0,
        "cache_implementation": "hybrid",
        "do_sample": True,
        "top_k": 64,
        "top_p": 0.95,
    }


def _write_safetensors(
    path: pathlib.Path,
    tensors: list[tuple[str, str, list[int], list[int]]],
    *,
    metadata: dict[str, str] | None = None,
    data_bytes: bytes | None = None,
) -> None:
    header: dict[str, object] = {}
    for name, dtype, shape, offsets in tensors:
        header[name] = {
            "dtype": dtype,
            "shape": shape,
            "data_offsets": offsets,
        }
    if metadata is not None:
        header["__metadata__"] = metadata
    raw = json.dumps(header, separators=(",", ":")).encode("utf-8")
    raw += b" " * ((-len(raw)) % 8)
    maximum_end = max((offsets[1] for _, _, _, offsets in tensors), default=0)
    body = bytes(maximum_end) if data_bytes is None else data_bytes
    path.write_bytes(struct.pack("<Q", len(raw)) + raw + body)


class Gate8GemmaWeightBindingTests(unittest.TestCase):
    def test_plan_freezes_exact_repo_revision_files_and_closed_boundaries(self):
        plan = b.gate8_gemma_weight_binding_plan()
        self.assertEqual(plan["scientific_protocol_head"], b.GATE8_GEMMA_WEIGHT_BINDING_PROTOCOL_HEAD)
        self.assertEqual(plan["tokenizer_result_head"], b.GATE8_GEMMA_WEIGHT_BINDING_TOKENIZER_RESULT_HEAD)
        self.assertEqual(plan["repo_id"], "google/gemma-3-1b-it")
        self.assertEqual(
            plan["revision"],
            "dcc83ea841ab6100d6b47a070329e1ba4cf78752",
        )
        self.assertEqual(
            plan["required_model_files"],
            ["config.json", "generation_config.json", "model.safetensors"],
        )
        self.assertEqual(
            plan["qualified_config_sha256"],
            "19cb5d28c97778271ba2b3c3df47bf76bdd6706724777a2318b3522230afe91e",
        )
        self.assertTrue(plan["model_file_binding_admitted"])
        self.assertTrue(plan["model_file_download_admitted"])
        self.assertFalse(plan["model_instantiation_admitted"])
        self.assertFalse(plan["tokenizer_loading_admitted"])
        self.assertFalse(plan["training_admitted"])
        self.assertFalse(plan["inference_admitted"])
        self.assertFalse(plan["scientific_test_worlds_admitted"])

    def test_config_and_generation_semantics_are_frozen(self):
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            config_path = root / "config.json"
            generation_path = root / "generation_config.json"
            _write_json(config_path, _valid_config())
            _write_json(generation_path, _valid_generation_config())

            semantics = b.validate_gate8_gemma_config(config_path)
            self.assertEqual(semantics["architectures"], ["Gemma3ForCausalLM"])
            self.assertEqual(semantics["hidden_size"], 1_152)
            self.assertEqual(semantics["num_hidden_layers"], 26)
            self.assertEqual(semantics["vocab_size"], 262_144)

            generation = b.validate_gate8_gemma_generation_config(generation_path)
            self.assertEqual(generation["bos_token_id"], 2)
            self.assertEqual(generation["eos_token_id"], [1, 106])
            self.assertEqual(generation["pad_token_id"], 0)
            self.assertEqual(
                generation["scientific_decoding_override"],
                "greedy_temperature_0",
            )
            self.assertEqual(generation["scientific_max_new_tokens"], 64)

            changed = _valid_config()
            changed["hidden_size"] = 1_153
            _write_json(config_path, changed)
            with self.assertRaisesRegex(ValueError, "config semantics changed"):
                b.validate_gate8_gemma_config(config_path)

    def test_safetensors_header_and_exact_byte_ledger(self):
        with tempfile.TemporaryDirectory() as directory:
            path = pathlib.Path(directory) / "model.safetensors"
            _write_safetensors(
                path,
                [
                    ("layer.weight", "BF16", [3, 2], [0, 12]),
                    ("layer.bias", "BF16", [2], [12, 16]),
                ],
                metadata={"format": "pt"},
            )
            summary = b.inspect_gate8_safetensors(path)
            self.assertEqual(summary.tensor_count, 2)
            self.assertEqual(summary.parameter_count, 8)
            self.assertEqual(summary.storage_bytes, 16)
            self.assertEqual(summary.data_bytes, 16)
            self.assertEqual(summary.dtype_parameter_counts, {"BF16": 8})
            self.assertEqual(summary.dtype_storage_bytes, {"BF16": 16})
            self.assertEqual(summary.metadata, {"format": "pt"})
            self.assertEqual(
                [row.name for row in summary.tensors],
                ["layer.weight", "layer.bias"],
            )
            self.assertEqual(summary.tensors[0].shape, (3, 2))
            summary.validate()

    def test_safetensors_rejects_gap_overlap_wrong_size_and_unknown_dtype(self):
        cases = (
            (
                [
                    ("a", "BF16", [2], [0, 4]),
                    ("b", "BF16", [1], [6, 8]),
                ],
                "gap or overlap",
            ),
            (
                [
                    ("a", "BF16", [2], [0, 4]),
                    ("b", "BF16", [1], [3, 5]),
                ],
                "gap or overlap",
            ),
            (
                [("a", "BF16", [2], [0, 5])],
                "data span is inconsistent",
            ),
            (
                [("a", "F128", [1], [0, 16])],
                "dtype is unsupported",
            ),
        )
        for tensors, pattern in cases:
            with self.subTest(pattern=pattern):
                with tempfile.TemporaryDirectory() as directory:
                    path = pathlib.Path(directory) / "model.safetensors"
                    _write_safetensors(path, tensors)
                    with self.assertRaisesRegex(ValueError, pattern):
                        b.inspect_gate8_safetensors(path)

    def test_snapshot_requires_exact_three_files_and_qualified_config_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            _write_json(root / "config.json", _valid_config())
            _write_json(
                root / "generation_config.json",
                _valid_generation_config(),
            )
            _write_safetensors(
                root / "model.safetensors",
                [("weight", "BF16", [2], [0, 4])],
            )
            config_hash = hashlib.sha256(
                (root / "config.json").read_bytes()
            ).hexdigest()
            original_hash = b.GATE8_GEMMA_CONFIG_SHA256
            b.GATE8_GEMMA_CONFIG_SHA256 = config_hash
            try:
                hashes, sizes = b.validate_gate8_gemma_model_snapshot(root)
            finally:
                b.GATE8_GEMMA_CONFIG_SHA256 = original_hash
            self.assertEqual(tuple(hashes), b.GATE8_GEMMA_REQUIRED_MODEL_FILES)
            self.assertEqual(tuple(sizes), b.GATE8_GEMMA_REQUIRED_MODEL_FILES)
            self.assertEqual(sizes["model.safetensors"], (root / "model.safetensors").stat().st_size)

            (root / "README.md").write_text("extra", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "file set changed"):
                b.validate_gate8_gemma_model_snapshot(root)

    def test_summary_requires_bf16_and_one_billion_parameter_class(self):
        with tempfile.TemporaryDirectory() as directory:
            path = pathlib.Path(directory) / "model.safetensors"
            _write_safetensors(
                path,
                [("weight", "BF16", [4], [0, 8])],
            )
            summary = b.inspect_gate8_safetensors(path)
            original_minimum = b.GATE8_GEMMA_PARAMETER_COUNT_MINIMUM
            original_maximum = b.GATE8_GEMMA_PARAMETER_COUNT_MAXIMUM
            b.GATE8_GEMMA_PARAMETER_COUNT_MINIMUM = 1
            b.GATE8_GEMMA_PARAMETER_COUNT_MAXIMUM = 10
            try:
                summary.validate_gemma_weight_contract()
            finally:
                b.GATE8_GEMMA_PARAMETER_COUNT_MINIMUM = original_minimum
                b.GATE8_GEMMA_PARAMETER_COUNT_MAXIMUM = original_maximum

            with tempfile.TemporaryDirectory() as other_directory:
                other = pathlib.Path(other_directory) / "model.safetensors"
                _write_safetensors(
                    other,
                    [("weight", "F32", [4], [0, 16])],
                )
                float_summary = b.inspect_gate8_safetensors(other)
                b.GATE8_GEMMA_PARAMETER_COUNT_MINIMUM = 1
                b.GATE8_GEMMA_PARAMETER_COUNT_MAXIMUM = 10
                try:
                    with self.assertRaisesRegex(ValueError, "exclusively BF16"):
                        float_summary.validate_gemma_weight_contract()
                finally:
                    b.GATE8_GEMMA_PARAMETER_COUNT_MINIMUM = original_minimum
                    b.GATE8_GEMMA_PARAMETER_COUNT_MAXIMUM = original_maximum

    def test_duplicate_json_and_header_keys_fail_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            duplicate_config = root / "config.json"
            duplicate_config.write_text(
                '{"model_type":"gemma3_text","model_type":"changed"}',
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "duplicate JSON key"):
                b.validate_gate8_gemma_config(duplicate_config)

            header = (
                b'{"a":{"dtype":"BF16","shape":[1],"data_offsets":[0,2]},'
                b'"a":{"dtype":"BF16","shape":[1],"data_offsets":[0,2]}}'
            )
            header += b" " * ((-len(header)) % 8)
            path = root / "model.safetensors"
            path.write_bytes(struct.pack("<Q", len(header)) + header + b"\0\0")
            with self.assertRaisesRegex(ValueError, "duplicate Safetensors header key"):
                b.inspect_gate8_safetensors(path)

    def test_sources_have_no_model_tokenizer_inference_or_test_surface(self):
        contract = BINDING_PATH.read_text(encoding="utf-8")
        runner = RUNNER_PATH.read_text(encoding="utf-8")
        wrapper = WRAPPER_PATH.read_text(encoding="utf-8")
        for text in (contract, runner):
            for forbidden in (
                "import torch",
                "torch.",
                "transformers",
                "AutoModel",
                "AutoTokenizer",
                "from_pretrained(",
                "generate_gate8_world(",
                "model.generate(",
            ):
                self.assertNotIn(forbidden, text)
        self.assertIn("hf_hub_download", runner)
        self.assertIn("GATE8_GEMMA_REQUIRED_MODEL_FILES", runner)
        self.assertNotIn("snapshot_download", runner)
        self.assertIn("GATE8_GEMMA_WEIGHT_BINDING_WRAPPER_SMOKE", wrapper)
        self.assertIn("HuggingFaceLicenseAndAccessAttested", wrapper)
        self.assertIn("scientific_test_worlds_generated = $false", wrapper)


if __name__ == "__main__":
    unittest.main()
