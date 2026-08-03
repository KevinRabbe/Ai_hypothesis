from __future__ import annotations

import pathlib
import tempfile
import unittest

import torch

from ai_hypothesis.population_language import l0_reference_preflight as preflight
from ai_hypothesis.population_language.l0_data import materialize_batch


def _success(model: str, microbatch: int) -> dict[str, object]:
    return {
        "model": model,
        "microbatch": microbatch,
        "success": True,
        "loss": 4.0,
        "gradient_norm_before_clip": 2.0,
        "seconds": 0.1,
        "peak_allocated_bytes": 1000,
        "peak_reserved_bytes": 2000,
    }


def _oom(model: str, microbatch: int) -> dict[str, object]:
    return {
        "model": model,
        "microbatch": microbatch,
        "success": False,
        "failure": "CUDA_OUT_OF_MEMORY",
        "error": "synthetic OOM",
    }


class PopulationLanguageL0ReferencePreflightContract(unittest.TestCase):
    def test_cache_state_estimates(self) -> None:
        self.assertEqual(
            preflight.transformer_kv_cache_bytes(sequence_length=32),
            393_216,
        )
        self.assertEqual(
            preflight.organism_state_bytes(worker_count=16),
            4_096,
        )
        self.assertEqual(
            preflight.organism_state_bytes(worker_count=256),
            65_536,
        )
        with self.assertRaises(ValueError):
            preflight.transformer_kv_cache_bytes(sequence_length=0)
        with self.assertRaises(ValueError):
            preflight.organism_state_bytes(worker_count=0)

    def test_full_next_token_loss_is_finite(self) -> None:
        batch = materialize_batch("train", (0, 1))
        logits = torch.zeros(
            (*batch.target_ids.shape, 64), dtype=torch.float32
        )
        loss = preflight.full_next_token_loss(logits, batch)
        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(float(loss.item()), 0.0)
        with self.assertRaises(ValueError):
            preflight.full_next_token_loss(logits[:, :-1], batch)

    def test_classifier_selects_largest_common_microbatch(self) -> None:
        rows = [
            _success(model, microbatch)
            for model in ("transformer", "population")
            for microbatch in preflight.MICROBATCH_CANDIDATES
        ]
        diagnosis, recommended = preflight.classify(rows)
        self.assertEqual(diagnosis, preflight.PASS)
        self.assertEqual(recommended, 8)
        self.assertEqual(preflight.GLOBAL_BATCH_SIZE // recommended, 32)

    def test_classifier_handles_partial_oom_prefixes(self) -> None:
        rows = [
            _success("transformer", 1),
            _success("transformer", 2),
            _success("transformer", 4),
            _oom("transformer", 8),
            _success("population", 1),
            _success("population", 2),
            _oom("population", 4),
        ]
        diagnosis, recommended = preflight.classify(rows)
        self.assertEqual(diagnosis, preflight.FAIL)
        self.assertEqual(recommended, 2)

    def test_classifier_rejects_malformed_rows(self) -> None:
        with self.assertRaises(ValueError):
            preflight.classify(
                [
                    _success("transformer", 2),
                    _success("population", 1),
                ]
            )
        with self.assertRaises(ValueError):
            preflight.classify(
                [
                    _oom("transformer", 1),
                    _success("transformer", 2),
                    _success("population", 1),
                ]
            )
        with self.assertRaises(ValueError):
            preflight.classify(
                [
                    _success("transformer", 1),
                    _success("unknown", 1),
                    _success("population", 1),
                ]
            )

    def test_nonfinite_success_row_fails_closed(self) -> None:
        rows = [
            _success(model, microbatch)
            for model in ("transformer", "population")
            for microbatch in preflight.MICROBATCH_CANDIDATES
        ]
        rows[-1]["loss"] = float("nan")
        diagnosis, recommended = preflight.classify(rows)
        self.assertEqual(diagnosis, preflight.FAIL)
        self.assertIsNone(recommended)

    def test_existing_output_and_locked_candidates_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(FileExistsError):
                preflight.run(
                    pathlib.Path(directory),
                    "0" * 40,
                )
        with tempfile.TemporaryDirectory() as directory:
            output = pathlib.Path(directory) / "new-output"
            with self.assertRaises(ValueError):
                preflight.run(
                    output,
                    "0" * 40,
                    candidates=(1,),
                )
            self.assertFalse(output.exists())

    def test_malformed_head_fails_before_cuda_boundary(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = pathlib.Path(directory) / "new-output"
            with self.assertRaises(ValueError):
                preflight.run(output, "not-a-head")
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
