from __future__ import annotations

import hashlib
import json
import pathlib
from types import SimpleNamespace
import tempfile
import unittest
from unittest import mock

from ai_hypothesis.population_language import l0_protocol as protocol
from ai_hypothesis.population_language import l0_reference_manifest as manifest
from ai_hypothesis.population_language import l0_reference_training as training


EXECUTION_HEAD = "f7d997e828e2a8791592c060973080e3fe3c43bd"


def _evaluation(
    split: str,
    episodes: int,
    checkpoint_sha256: str,
    *,
    answer_exact: float = 0.96,
    answer_nll: float = 0.2,
    worker: int | None = None,
) -> dict[str, object]:
    row: dict[str, object] = {
        "split": split,
        "episodes": episodes,
        "checkpoint_sha256": checkpoint_sha256,
        "next_token_nll": 0.3,
        "perplexity": 1.35,
        "answer_span_nll": answer_nll,
        "answer_exact_accuracy": answer_exact,
        "color_token_accuracy": 0.98,
        "shape_token_accuracy": 0.98,
        "relation_token_accuracy": 0.99,
        "swapped_definition_answer_exact_accuracy": answer_exact,
        "definition_order_answer_token_agreement": 0.99,
        "estimated_forward_flops_per_episode": 1_000_000_000,
        "estimated_greedy_answer_flops_per_episode": 5_000_000_000,
        "answer_exact_accuracy_per_gigaflop": answer_exact / 5.0,
        "answer_exact_decoding": "AUTOREGRESSIVE_GREEDY_FIVE_TOKEN",
        "seconds": 1.0,
    }
    if worker is not None:
        row.update(
            {
                "worker_count": worker,
                "routed_messages": 1_000,
                "routed_messages_per_processed_token": 4.0,
                "routed_messages_per_episode": 124.0,
                "persistent_state_bytes_per_episode_bf16": worker * 256,
                "routing": {
                    "router_decisions": 100,
                    "selected_messages": 400,
                    "mean_router_entropy_nats": 1.0,
                    "normalized_router_entropy": 0.8,
                    "selected_sender_coverage": 1.0,
                    "effective_worker_utilization": 0.9,
                    "sender_selection_coefficient_of_variation": 0.2,
                },
            }
        )
    return row


def _write_json(path: pathlib.Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _seed_rows(root: pathlib.Path) -> list[dict[str, object]]:
    schedule = training.training_schedule_sha256()
    rows: list[dict[str, object]] = []
    for seed in protocol.INITIALIZATION_SEEDS:
        transformer_hash = f"{seed:064x}"[-64:]
        population_hash = f"{seed + 10:064x}"[-64:]
        model_rows: dict[str, dict[str, object]] = {}
        for model, canonical_hash in (
            ("transformer", transformer_hash),
            ("population", population_hash),
        ):
            relative = f"checkpoints/{model}-seed-{seed}.pt"
            checkpoint_path = root / relative
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            checkpoint_path.write_bytes(f"fixture:{model}:{seed}".encode("ascii"))
            file_hash = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
            model_rows[model] = {
                "parameter_count": (
                    protocol.transformer_parameter_count()
                    if model == "transformer"
                    else protocol.population_parameter_count()
                ),
                "optimizer_steps": training.OPTIMIZER_STEPS,
                "global_batch_size": training.GLOBAL_BATCH_SIZE,
                "microbatch": 8,
                "gradient_accumulation_steps": 32,
                "training_schedule_sha256": schedule,
                "training_tokens": (
                    training.OPTIMIZER_STEPS
                    * training.GLOBAL_BATCH_SIZE
                    * training.EXPECTED_TARGET_TOKENS_PER_EPISODE
                ),
                "estimated_training_flops": 1,
                "seconds": 1.0,
                "peak_allocated_bytes": 1,
                "peak_reserved_bytes": 1,
                "canonical_checkpoint_sha256": canonical_hash,
                "checkpoint_file": relative,
                "checkpoint_file_sha256": file_hash,
                "curves": [],
            }

        transformer = model_rows["transformer"]
        transformer["validation"] = _evaluation(
            "validation",
            protocol.REFERENCE_VALIDATION_EPISODES,
            transformer_hash,
        )
        transformer["test"] = _evaluation(
            "test",
            protocol.REFERENCE_TEST_EPISODES,
            transformer_hash,
            answer_exact=0.97,
        )

        validation_by_workers: dict[str, object] = {}
        test_by_workers: dict[str, object] = {}
        for worker in protocol.EVAL_WORKERS:
            progress = (worker - 16) / 240
            accuracy = 0.70 + 0.08 * progress
            answer_nll = 0.4 - 0.08 * progress
            validation_by_workers[str(worker)] = _evaluation(
                "validation",
                protocol.REFERENCE_VALIDATION_EPISODES,
                population_hash,
                answer_exact=accuracy,
                answer_nll=answer_nll,
                worker=worker,
            )
            test_by_workers[str(worker)] = _evaluation(
                "test",
                protocol.REFERENCE_TEST_EPISODES,
                population_hash,
                answer_exact=accuracy,
                answer_nll=answer_nll,
                worker=worker,
            )
        population = model_rows["population"]
        population["validation_by_workers"] = validation_by_workers
        population["test_by_workers"] = test_by_workers
        rows.append({"seed": seed, "transformer": transformer, "population": population})
    return rows


def _build_output(root: pathlib.Path) -> dict[str, object]:
    root.mkdir(parents=True)
    rows = _seed_rows(root)
    contract = training.validate_contract(8, 8).__dict__
    fingerprints = {
        split: protocol.dataset_fingerprint(split, 256)
        for split in ("train", "validation", "test")
    }
    start = {
        "status": training.STATUS,
        "phase": "TRAINING",
        "version": training.VERSION,
        "branch": training.BRANCH,
        "base_head": training.BASE_HEAD,
        "execution_head": EXECUTION_HEAD,
        "training_schedule_sha256": training.training_schedule_sha256(),
        "contract": contract,
        "seeds": list(protocol.INITIALIZATION_SEEDS),
        "dataset_fingerprints_first_256": fingerprints,
        "dataset_cache_build_seconds": 1.0,
        "dataset_cache_resident_bytes": 1,
    }
    _write_json(root / "run-start.json", start)

    for row in rows:
        seed = row["seed"]
        _write_json(root / f"seed-{seed}.json", row)
        for model in ("transformer", "population"):
            trained = row[model]
            _write_json(
                root / "progress" / f"{model}-seed-{seed}.json",
                {
                    "status": "COMPLETE",
                    "version": training.VERSION,
                    "model": model,
                    "seed": seed,
                    "last_completed_optimizer_step": training.OPTIMIZER_STEPS,
                    "canonical_checkpoint_sha256": trained["canonical_checkpoint_sha256"],
                    "checkpoint_file_sha256": trained["checkpoint_file_sha256"],
                    "curves": trained["curves"],
                },
            )

    diagnosis = training.classify(rows)
    summary: dict[str, object] = {
        "status": training.STATUS,
        "version": training.VERSION,
        "branch": training.BRANCH,
        "base_head": training.BASE_HEAD,
        "execution_head": EXECUTION_HEAD,
        "diagnosis": diagnosis,
        "training_schedule_sha256": training.training_schedule_sha256(),
        "contract": contract,
        "dataset_cache_build_seconds": 1.0,
        "dataset_cache_resident_bytes": 1,
        "dataset_fingerprints_first_256": fingerprints,
        "cuda": {
            "device_name": "fixture",
            "device_capability": [8, 9],
            "total_memory_bytes": 1,
            "torch_version": "fixture",
            "cuda_version": "fixture",
            "bf16_supported": True,
        },
        "seed_rows": rows,
        "population_scaling": training.population_scaling_summary(rows),
        "boundaries": {
            "full_next_token_objective_only": True,
            "answer_span_training_weighted": False,
            "fixed_final_checkpoint_used": True,
            "test_used_for_checkpoint_selection": False,
            "population_trained_only_at_32_workers": True,
            "same_population_checkpoint_used_at_all_worker_counts": True,
            "worker_specific_learned_parameters_used": False,
            "gate9_evidence_modified": False,
        },
    }
    _write_json(root / "summary.json", summary)
    return summary


class ReferenceManifestContracts(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = pathlib.Path(self.temporary.name).resolve() / "reference-output"
        self.summary = _build_output(self.root)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _fake_checkpoint_load(
        self,
        path: pathlib.Path,
        *,
        expected_seed: int,
        expected_file_sha256: str,
        expected_canonical_sha256: str,
    ) -> SimpleNamespace:
        return SimpleNamespace(
            model=object(),
            path=str(path),
            seed=expected_seed,
            file_bytes=path.stat().st_size,
            file_sha256=expected_file_sha256,
            canonical_state_sha256=expected_canonical_sha256,
        )

    def _verify(self) -> manifest.ReferenceOutputManifest:
        with mock.patch.object(
            manifest.checkpoint,
            "load_reference_checkpoint",
            side_effect=self._fake_checkpoint_load,
        ):
            return manifest.verify_reference_output(
                self.root,
                expected_execution_head=EXECUTION_HEAD,
            )

    def test_static_inventory_contract_is_exact(self) -> None:
        report = manifest.validate_reference_manifest_contract()
        self.assertTrue(report["valid"])
        self.assertEqual(len(manifest.expected_relative_files()), 17)
        self.assertEqual(manifest.EXPECTED_FILE_COUNT, 17)

    def test_complete_output_is_recomputed_and_checkpoint_linked(self) -> None:
        result = self._verify()
        self.assertEqual(result.execution_head, EXECUTION_HEAD)
        self.assertEqual(result.diagnosis, training.PASS)
        self.assertTrue(result.post_training_base_eligible)
        self.assertEqual(len(result.population_checkpoints), 3)
        self.assertEqual(
            [record.seed for record in result.population_checkpoints],
            list(protocol.INITIALIZATION_SEEDS),
        )
        self.assertEqual(len(result.summary_sha256), 64)

    def test_inventory_seed_and_summary_drift_fail_closed(self) -> None:
        (self.root / "unexpected.txt").write_text("x", encoding="utf-8")
        with self.assertRaises(ValueError):
            self._verify()
        (self.root / "unexpected.txt").unlink()

        seed_path = self.root / f"seed-{protocol.INITIALIZATION_SEEDS[0]}.json"
        seed_payload = json.loads(seed_path.read_text(encoding="utf-8"))
        seed_payload["seed"] = 999
        _write_json(seed_path, seed_payload)
        with self.assertRaises(ValueError):
            self._verify()

    def test_checkpoint_and_execution_identity_drift_fail_closed(self) -> None:
        checkpoint_path = (
            self.root
            / "checkpoints"
            / f"population-seed-{protocol.INITIALIZATION_SEEDS[0]}.pt"
        )
        checkpoint_path.write_bytes(checkpoint_path.read_bytes() + b"tamper")
        with self.assertRaises(ValueError):
            self._verify()

        with self.assertRaises(ValueError):
            manifest.verify_reference_output(
                self.root,
                expected_execution_head="0" * 40,
            )

    def test_recomputed_diagnosis_cannot_be_overridden(self) -> None:
        summary_path = self.root / "summary.json"
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        summary["diagnosis"] = training.INVALID
        _write_json(summary_path, summary)
        with self.assertRaises(ValueError):
            self._verify()


if __name__ == "__main__":
    unittest.main()
