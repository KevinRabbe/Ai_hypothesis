from __future__ import annotations

import importlib.util
import pathlib
import sys
import tempfile
import unittest


BINDING_PATH = pathlib.Path(
    "ai_hypothesis/population_compute/gate8_gemma_tokenizer_binding.py"
)
RUNNER_PATH = pathlib.Path("scripts/bind_gate8_gemma_tokenizer.py")
WRAPPER_PATH = pathlib.Path("scripts/bind_gate8_gemma_tokenizer.ps1")


def _load_binding():
    name = "gate8_gemma_tokenizer_binding_test_module"
    spec = importlib.util.spec_from_file_location(name, BINDING_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load Gate8 tokenizer binding contract")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


b = _load_binding()


class FakeTokenizer:
    chat_template = "<start>{{ messages[0]['content'] }}<assistant>"

    def apply_chat_template(
        self,
        messages,
        *,
        add_generation_prompt,
        tokenize,
        return_dict,
    ):
        assert add_generation_prompt is True
        assert tokenize is True
        assert return_dict is True
        assert messages == [{"role": "user", "content": "ABC"}]
        return {"input_ids": [2, 10, 11, 12, 106]}


class NestedFakeTokenizer(FakeTokenizer):
    def apply_chat_template(self, *args, **kwargs):
        return {"input_ids": [[2, 10, 11, 12, 106]]}


class Gate8GemmaTokenizerBindingTests(unittest.TestCase):
    def _snapshot(self, root: pathlib.Path) -> pathlib.Path:
        root.mkdir()
        for index, filename in enumerate(b.GATE8_GEMMA_REQUIRED_TOKENIZER_FILES):
            (root / filename).write_bytes(f"fixture-{index}-{filename}".encode("ascii"))
        return root

    def _rows(self, tokens: int = 100):
        return tuple(
            b.Gate8TokenizerConditionCount(
                population=population,
                depth=depth,
                prompt_sha256=(f"{index + 1:064x}"[-64:]),
                ascii_bytes=1000 + index,
                input_tokens=tokens + index,
            )
            for index, (population, depth) in enumerate(b.GATE8_VALID_CONDITIONS)
        )

    def test_frozen_official_identity_and_file_allowlist(self) -> None:
        self.assertEqual(b.GATE8_GEMMA_REPO_ID, "google/gemma-3-1b-it")
        self.assertEqual(
            b.GATE8_GEMMA_REVISION,
            "dcc83ea841ab6100d6b47a070329e1ba4cf78752",
        )
        b.validate_gate8_gemma_revision(b.GATE8_GEMMA_REVISION)
        self.assertEqual(
            b.GATE8_GEMMA_REQUIRED_TOKENIZER_FILES,
            (
                "added_tokens.json",
                "config.json",
                "special_tokens_map.json",
                "tokenizer.json",
                "tokenizer.model",
                "tokenizer_config.json",
            ),
        )
        self.assertNotIn("model.safetensors", b.GATE8_GEMMA_REQUIRED_TOKENIZER_FILES)

    def test_snapshot_hashes_exact_six_files_and_ignores_hub_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = self._snapshot(pathlib.Path(temporary) / "snapshot")
            metadata = root / ".cache/huggingface"
            metadata.mkdir(parents=True)
            (metadata / "metadata.json").write_text("{}", encoding="utf-8")
            hashes = b.validate_gate8_tokenizer_snapshot(root)
            self.assertEqual(tuple(hashes), b.GATE8_GEMMA_REQUIRED_TOKENIZER_FILES)
            self.assertTrue(all(len(value) == 64 for value in hashes.values()))

    def test_snapshot_rejects_missing_extra_or_model_files(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            base = pathlib.Path(temporary)
            missing = self._snapshot(base / "missing")
            (missing / "tokenizer.model").unlink()
            with self.assertRaisesRegex(ValueError, "file set changed"):
                b.validate_gate8_tokenizer_snapshot(missing)

            extra = self._snapshot(base / "extra")
            (extra / "notes.txt").write_text("unexpected", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "file set changed"):
                b.validate_gate8_tokenizer_snapshot(extra)

            model = self._snapshot(base / "model")
            (model / "model.safetensors").write_bytes(b"forbidden")
            with self.assertRaisesRegex(ValueError, "file set changed"):
                b.validate_gate8_tokenizer_snapshot(model)

    def test_chat_template_hash_and_token_count_are_deterministic(self) -> None:
        tokenizer = FakeTokenizer()
        self.assertEqual(
            b.gate8_prompt_token_ids(tokenizer, "ABC"),
            (2, 10, 11, 12, 106),
        )
        self.assertEqual(
            b.gate8_prompt_token_ids(NestedFakeTokenizer(), "ABC"),
            (2, 10, 11, 12, 106),
        )
        self.assertEqual(len(b.gate8_chat_template_sha256(tokenizer)), 64)

    def test_complete_condition_matrix_and_maximum_are_frozen(self) -> None:
        rows = self._rows()
        validated = b.validate_gate8_tokenizer_condition_matrix(rows)
        self.assertEqual(len(validated), 21)
        maximum = b.gate8_maximum_token_condition(rows)
        self.assertEqual((maximum.population, maximum.depth), (1024, 128))
        with self.assertRaisesRegex(ValueError, "population-major"):
            b.validate_gate8_tokenizer_condition_matrix(tuple(reversed(rows)))

    def test_token_limit_is_hard(self) -> None:
        rows = list(self._rows())
        final = rows[-1]
        rows[-1] = b.Gate8TokenizerConditionCount(
            population=final.population,
            depth=final.depth,
            prompt_sha256=final.prompt_sha256,
            ascii_bytes=final.ascii_bytes,
            input_tokens=24_577,
        )
        with self.assertRaisesRegex(ValueError, "input-token limit"):
            b.validate_gate8_tokenizer_condition_matrix(tuple(rows))

    def test_binding_summary_is_tokenizer_bound_but_model_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = self._snapshot(pathlib.Path(temporary) / "snapshot")
            hashes = b.validate_gate8_tokenizer_snapshot(root)
            summary = b.Gate8TokenizerBindingSummary(
                repo_id=b.GATE8_GEMMA_REPO_ID,
                revision=b.GATE8_GEMMA_REVISION,
                tokenizer_class="GemmaTokenizerFast",
                transformers_version="fixture-transformers",
                tokenizers_version="fixture-tokenizers",
                huggingface_hub_version="fixture-hub",
                chat_template_sha256="a" * 64,
                tokenizer_file_sha256=hashes,
                conditions=self._rows(),
            )
            payload = summary.to_dict()
            self.assertTrue(payload["tokenizer_bound"])
            self.assertFalse(payload["model_bound"])
            self.assertFalse(payload["model_weights_downloaded"])
            self.assertFalse(payload["inference_performed"])
            self.assertFalse(payload["scientific_test_worlds_generated"])

    def test_plan_keeps_training_model_and_inference_closed(self) -> None:
        plan = b.gate8_tokenizer_binding_plan()
        self.assertEqual(
            plan["encoder_head"],
            "9882256ae0152bc266dc4d96cab3bbeb0c4ef95b",
        )
        self.assertTrue(plan["tokenizer_binding_admitted"])
        self.assertFalse(plan["model_binding_admitted"])
        self.assertFalse(plan["model_weights_downloaded"])
        self.assertFalse(plan["training_admitted"])
        self.assertFalse(plan["inference_admitted"])
        self.assertFalse(plan["scientific_test_worlds_admitted"])

    def test_runner_and_wrapper_exclude_model_and_scientific_execution(self) -> None:
        runner = RUNNER_PATH.read_text(encoding="utf-8")
        wrapper = WRAPPER_PATH.read_text(encoding="utf-8")
        self.assertIn("allow_patterns=list(binding.GATE8_GEMMA_REQUIRED_TOKENIZER_FILES)", runner)
        self.assertIn('split="contract"', runner)
        self.assertIn('split="demonstration"', runner)
        self.assertNotIn('split="test"', runner)
        for token in (
            "AutoModel",
            "ForCausalLM",
            "model.generate",
            "model.safetensors",
            "training_step",
            "optimizer",
            "backward(",
        ):
            self.assertNotIn(token, runner)
        self.assertIn("HuggingFaceLicenseAndAccessAttested", wrapper)
        self.assertIn("Model weights:   FORBIDDEN", wrapper)


if __name__ == "__main__":
    unittest.main()
