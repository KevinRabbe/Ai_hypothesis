from __future__ import annotations

import importlib.util
import json
import pathlib
import sys
import tempfile
import unittest

import torch

AUDITOR_PATH = pathlib.Path(
    "ai_hypothesis/population_compute/gate9_contextual_seed_audit.py"
)


def _load_auditor():
    name = "gate9_contextual_seed_audit_test_module"
    spec = importlib.util.spec_from_file_location(name, AUDITOR_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load Gate9 seed auditor")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


a = _load_auditor()


def _write_json(path: pathlib.Path, payload) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


class Gate9ContextualSeedAuditTests(unittest.TestCase):
    def _write_training(self, path: pathlib.Path) -> None:
        with path.open("w", encoding="utf-8", newline="\n") as handle:
            for step in range(1, 513):
                row = {
                    "seed": 0,
                    "step": step,
                    "episodes_seen": step * 512,
                    "learning_rate": a.learning_rate_at_step(step),
                    "loss": 0.7 - step / 100000.0,
                    "pre_clip_gradient_norm": 0.1 + step / 10000.0,
                    "batch_operator_ordinal_sha256": f"{step:064x}",
                    "batch_query_sha256": f"{step + 1000:064x}",
                    "wall_seconds": 0.01,
                }
                handle.write(json.dumps(row, sort_keys=True) + "\n")

    def _write_validation(self, path: pathlib.Path) -> None:
        # Reconstruct the exact seed-0 aggregate counts printed by the run:
        # full=109, shuffled=112, query-only=117, oracle=32768,
        # bit-correct=131200 of 262144.
        with path.open("w", encoding="utf-8", newline="\n") as handle:
            for index in range(32768):
                answer = 0
                if index < 109:
                    full = 0
                elif index < 109 + 308:
                    full = 31  # 3 correct bits against zero
                else:
                    full = 15  # 4 correct bits against zero
                shuffled = 0 if index < 112 else 1
                query_only = 0 if index < 117 else 1
                row = {
                    "episode_index": index,
                    "operator_ordinal": index,
                    "operator_counter": (1 << 32) + index,
                    "query": 3,
                    "answer": answer,
                    "full_prediction": full,
                    "shuffled_context_operator_ordinal": (index + 1) % 32768,
                    "shuffled_context_prediction": shuffled,
                    "query_only_prediction": query_only,
                    "oracle_prediction": answer,
                    "full_correct": full == answer,
                    "shuffled_context_correct": shuffled == answer,
                    "query_only_correct": query_only == answer,
                    "oracle_correct": True,
                }
                handle.write(json.dumps(row, sort_keys=True) + "\n")

    def _checkpoint_payload(self):
        state = {
            name: torch.zeros(shape, dtype=torch.float32)
            for name, shape in a.STATE_TENSOR_SHAPES.items()
        }
        return {
            "experiment_version": a.EXPECTED_EXPERIMENT_VERSION,
            "architecture_head": a.EXPECTED_ARCHITECTURE_HEAD,
            "training_protocol_head": a.EXPECTED_TRAINING_PROTOCOL_HEAD,
            "seed": 0,
            "initialization_seed": 900900,
            "step": 512,
            "train_episodes": 262144,
            "learned_parameter_count": 19649,
            "tensor_count": 17,
            "state_dict": state,
        }

    def _build_artifact(self, root: pathlib.Path) -> dict[str, str]:
        seed_root = root / "seed-0"
        seed_root.mkdir(parents=True)
        training_path = seed_root / "train-steps.jsonl"
        validation_path = seed_root / "validation-per-episode.jsonl"
        checkpoint_path = seed_root / "selected-checkpoint.pt"
        summary_path = seed_root / "summary.json"
        self._write_training(training_path)
        self._write_validation(validation_path)
        torch.save(self._checkpoint_payload(), checkpoint_path)

        checkpoint_hash = a.sha256_file(checkpoint_path)
        final_loss = 0.7 - 512 / 100000.0
        summary = {
            "experiment_version": a.EXPECTED_EXPERIMENT_VERSION,
            "scientific_status": a.EXPECTED_SEED_STATUS,
            "execution_head": a.EXPECTED_EXECUTION_HEAD,
            "training_protocol_head": a.EXPECTED_TRAINING_PROTOCOL_HEAD,
            "architecture_head": a.EXPECTED_ARCHITECTURE_HEAD,
            "seed": 0,
            "validation_evidence": {
                "seed": 0,
                "initialization_seed": 900900,
                "checkpoint_step": 512,
                "train_episodes": 262144,
                "unique_train_operators": 262144,
                "validation_episodes": 32768,
                "unique_validation_operators": 32768,
                "learned_parameter_count": 19649,
                "tensor_count": 17,
                "checkpoint_sha256": checkpoint_hash,
                "parameters_finite": True,
                "final_train_loss": final_loss,
                "validation_exact_accuracy": 109 / 32768,
                "validation_bit_accuracy": 131200 / 262144,
                "shuffled_context_accuracy": 112 / 32768,
                "query_only_accuracy": 117 / 32768,
                "oracle_accuracy": 1.0,
                "admission_passes": False,
            },
            "artifacts": {
                "train_steps": "train-steps.jsonl",
                "validation_per_episode": "validation-per-episode.jsonl",
                "selected_checkpoint": "selected-checkpoint.pt",
                "selected_checkpoint_sha256": checkpoint_hash,
            },
            "boundaries": {
                "training_performed": True,
                "validation_performed": True,
                "checkpoint_serialized": True,
                "local_test_operator_accessed": False,
                "graph_test_operator_accessed": False,
                "scientific_assignment_key_accessed": False,
                "scientific_test_generated": False,
                "scientific_execution_performed": False,
                "result_classification_performed": False,
            },
        }
        _write_json(summary_path, summary)
        (root / "git-head.txt").write_text(
            a.EXPECTED_EXECUTION_HEAD + "\n",
            encoding="ascii",
            newline="\n",
        )
        (root / "git-status.txt").write_text(
            "", encoding="utf-8", newline="\n"
        )
        _write_json(
            root / "run-config.json",
            {
                "experiment_version": a.EXPECTED_EXPERIMENT_VERSION,
                "execution_head": a.EXPECTED_EXECUTION_HEAD,
                "branch": a.EXPECTED_EXECUTION_BRANCH,
                "seed": 0,
                "training_protocol_head": a.EXPECTED_TRAINING_PROTOCOL_HEAD,
                "architecture_head": a.EXPECTED_ARCHITECTURE_HEAD,
                "python": "3.11.9",
                "torch": "2.9.1+cu130",
                "numpy": "2.3.5",
                "output_root": str(root.resolve()),
                "local_test_operator_access": False,
                "graph_test_operator_access": False,
                "scientific_assignment_key_access": False,
            },
        )
        rows = []
        for path in sorted(item for item in root.rglob("*") if item.is_file()):
            if path.name == "manifest.sha256":
                continue
            rows.append(
                f"{a.sha256_file(path)}  {path.relative_to(root).as_posix()}"
            )
        manifest = root / "manifest.sha256"
        manifest.write_text(
            "\n".join(rows) + "\n", encoding="ascii", newline="\n"
        )
        return {
            "summary": a.sha256_file(summary_path),
            "validation": a.sha256_file(validation_path),
            "checkpoint": checkpoint_hash,
            "manifest": a.sha256_file(manifest),
        }

    def test_full_failed_seed_artifact_reconstructs_exact_terminal_metrics(self):
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            hashes = self._build_artifact(root)
            report = a.audit_seed_artifact(
                root,
                seed=0,
                expected_summary_sha256=hashes["summary"],
                expected_validation_sha256=hashes["validation"],
                expected_checkpoint_sha256=hashes["checkpoint"],
                expected_manifest_sha256=hashes["manifest"],
            )
            self.assertEqual(
                report["seed_outcome"],
                "G9_CONTEXTUAL_SEED_CHECKPOINT_ADMISSION_FAILED",
            )
            self.assertFalse(report["all_seed_admission_still_possible"])
            self.assertFalse(report["scientific_test_generation_allowed"])
            self.assertEqual(report["training"]["rows"], 512)
            self.assertEqual(report["validation"]["rows"], 32768)
            self.assertEqual(report["validation"]["full_correct"], 109)
            self.assertEqual(report["validation"]["shuffled_correct"], 112)
            self.assertEqual(report["validation"]["query_only_correct"], 117)
            self.assertEqual(report["validation"]["oracle_correct"], 32768)
            self.assertEqual(report["validation"]["bit_accuracy"], 0.50048828125)

    def test_manifest_tampering_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            self._build_artifact(root)
            with (root / "seed-0/train-steps.jsonl").open("a", encoding="utf-8") as handle:
                handle.write("{}\n")
            with self.assertRaisesRegex(a.AuditError, "manifest digest mismatch"):
                a.verify_manifest(root)

    def test_checkpoint_shape_and_finiteness_fail_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            path = root / "checkpoint.pt"
            payload = self._checkpoint_payload()
            payload["state_dict"]["output_scale"] = torch.tensor(float("nan"))
            torch.save(payload, path)
            with self.assertRaisesRegex(a.AuditError, "non-finite"):
                a.audit_checkpoint(path, 0)

    def test_auditor_has_no_trainer_operator_or_scientific_runtime_import(self):
        text = AUDITOR_PATH.read_text(encoding="utf-8")
        for token in (
            "gate9_contextual_training_runtime",
            "gate9_contextual_training_data",
            "gate9_contextual_worker_architecture",
            "gate9_contextual_operator_contract",
            "generate_gate9_test_world",
            "operator_from_counter",
            "torch.optim",
        ):
            self.assertNotIn(token, text)


if __name__ == "__main__":
    unittest.main()
