from __future__ import annotations

from collections import OrderedDict
import hashlib
import os
import pathlib
import tempfile
import unittest

import torch

from ai_hypothesis.population_language import l0_protocol
from ai_hypothesis.population_language.l0_models import PopulationLanguageOrganism
from ai_hypothesis.population_language.l0_reference_training import (
    OPTIMIZER_STEPS,
    POPULATION_COMMUNICATION_ROUNDS,
    POPULATION_TOP_K,
    VERSION as REFERENCE_TRAINING_VERSION,
    canonical_state_sha256,
)
from ai_hypothesis.population_language.post_training_learning_l0_adapter import (
    AdapterConfig,
    BoundedPopulationAdapter,
)
from ai_hypothesis.population_language.post_training_learning_l0_execution import (
    save_adaptation_artifact_create_once,
)
from ai_hypothesis.population_language.post_training_learning_l0_fresh_process import (
    FRESH_PROCESS_COMPLETE,
    MAX_PROBE_BATCH,
    MAX_PROBE_SEQUENCE,
    MAX_PROBE_WORKERS,
    REQUEST_KEYS,
    RESULT_KEYS,
    run_fresh_process_probe,
    validate_fresh_process_contract,
    validate_request,
    write_fresh_process_request_create_once,
)


class FreshProcessContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.temporary = tempfile.TemporaryDirectory()
        cls.root = pathlib.Path(cls.temporary.name)
        cls.checkpoint_path = cls.root / "population-seed-120100.pt"
        cls.adapter_path = cls.root / "adapter-rank-1.bin"

        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(120100)
            base = PopulationLanguageOrganism(
                communication_rounds=POPULATION_COMMUNICATION_ROUNDS,
                top_k=POPULATION_TOP_K,
            )
        cls.canonical_sha256 = canonical_state_sha256(base)
        state = OrderedDict(
            (name, value.detach().cpu().contiguous())
            for name, value in base.state_dict().items()
        )
        torch.save(
            {
                "version": REFERENCE_TRAINING_VERSION,
                "model": "population",
                "seed": 120100,
                "optimizer_step": OPTIMIZER_STEPS,
                "state_dict": state,
            },
            cls.checkpoint_path,
        )
        checkpoint_payload = cls.checkpoint_path.read_bytes()
        cls.checkpoint_file_sha256 = hashlib.sha256(checkpoint_payload).hexdigest()

        adapted = BoundedPopulationAdapter(
            base,
            model_seed=120100,
            config=AdapterConfig(rank=1),
        )
        adaptation_state = adapted.adaptation_state_dict()
        adaptation_state["value_logit_bias"][0] = 0.25
        artifact = save_adaptation_artifact_create_once(
            cls.adapter_path,
            adaptation_state,
        )
        cls.adapter_file_sha256 = artifact.sha256
        ids = l0_protocol.TOKEN_TO_ID
        cls.input_ids = [[
            ids["<bos>"],
            ids["<query>"],
            ids["dax"],
            ids["red"],
            ids["<answer>"],
        ]]
        del adapted, base, state, adaptation_state, checkpoint_payload

    @classmethod
    def tearDownClass(cls) -> None:
        cls.temporary.cleanup()

    def _write_request(
        self,
        name: str,
        *,
        adapter_sha256: str | None = None,
    ):
        return write_fresh_process_request_create_once(
            self.root / f"{name}-request.json",
            checkpoint_path=self.checkpoint_path,
            checkpoint_seed=120100,
            checkpoint_file_sha256=self.checkpoint_file_sha256,
            checkpoint_canonical_sha256=self.canonical_sha256,
            adapter_path=self.adapter_path,
            adapter_file_sha256=(
                self.adapter_file_sha256
                if adapter_sha256 is None
                else adapter_sha256
            ),
            adapter_rank=1,
            worker_count=1,
            input_ids=self.input_ids,
            result_path=self.root / f"{name}-result.json",
        )

    def test_static_contract_is_bounded(self) -> None:
        report = validate_fresh_process_contract()
        self.assertTrue(report["valid"])
        self.assertEqual(len(REQUEST_KEYS), 13)
        self.assertEqual(len(RESULT_KEYS), 18)
        self.assertEqual(MAX_PROBE_BATCH, 8)
        self.assertEqual(MAX_PROBE_SEQUENCE, 8)
        self.assertEqual(MAX_PROBE_WORKERS, 256)

    def test_separate_process_loads_base_and_adapter_create_once(self) -> None:
        request = self._write_request("success")
        result = run_fresh_process_probe(
            pathlib.Path(request.path),
            expected_request_sha256=request.sha256,
            timeout_seconds=180,
        )
        self.assertEqual(result.parent_pid, os.getpid())
        self.assertNotEqual(result.child_pid, result.parent_pid)
        self.assertEqual(len(result.child_start_nonce), 64)
        self.assertEqual(result.checkpoint_seed, 120100)
        self.assertEqual(result.checkpoint_file_sha256, self.checkpoint_file_sha256)
        self.assertEqual(result.checkpoint_canonical_sha256, self.canonical_sha256)
        self.assertEqual(result.adapter_file_sha256, self.adapter_file_sha256)
        self.assertEqual(result.adapter_rank, 1)
        self.assertEqual(result.worker_count, 1)
        self.assertEqual(len(result.final_argmax_token_ids), 1)
        self.assertEqual(len(result.logits_sha256), 64)
        self.assertTrue(pathlib.Path(request.result_path).is_file())

        with self.assertRaises(FileExistsError):
            run_fresh_process_probe(
                pathlib.Path(request.path),
                expected_request_sha256=request.sha256,
            )

    def test_request_hash_and_adapter_hash_tampering_fail_closed(self) -> None:
        tampered = self._write_request("tampered-request")
        path = pathlib.Path(tampered.path)
        path.write_bytes(path.read_bytes() + b" ")
        with self.assertRaises(ValueError):
            run_fresh_process_probe(
                path,
                expected_request_sha256=tampered.sha256,
            )
        self.assertFalse(pathlib.Path(tampered.result_path).exists())

        wrong_adapter = self._write_request(
            "wrong-adapter",
            adapter_sha256="0" * 64,
        )
        with self.assertRaises(RuntimeError):
            run_fresh_process_probe(
                pathlib.Path(wrong_adapter.path),
                expected_request_sha256=wrong_adapter.sha256,
                timeout_seconds=180,
            )
        self.assertFalse(pathlib.Path(wrong_adapter.result_path).exists())

    def test_request_schema_rejects_labels_paths_and_non_adapter_inputs(self) -> None:
        request = self._write_request("schema")
        raw = pathlib.Path(request.path).read_text(encoding="utf-8")
        import json

        decoded = json.loads(raw)
        self.assertEqual(tuple(decoded), REQUEST_KEYS)
        decoded["input_ids"] = [[0, 1, 2, 3, 4]]
        with self.assertRaises(ValueError):
            validate_request(decoded)

        decoded = json.loads(raw)
        decoded["checkpoint_path"] = "relative.pt"
        with self.assertRaises(ValueError):
            validate_request(decoded)

        decoded = json.loads(raw)
        decoded["target_ids"] = [1]
        with self.assertRaises(ValueError):
            validate_request(decoded)

        decoded = json.loads(raw)
        decoded["status"] = FRESH_PROCESS_COMPLETE
        with self.assertRaises(ValueError):
            validate_request(decoded)


if __name__ == "__main__":
    unittest.main()
