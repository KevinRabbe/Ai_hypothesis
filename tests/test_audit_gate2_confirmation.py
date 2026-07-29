from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from ai_hypothesis.population_compute.audit_gate2_confirmation import (
    CONFIRMATION_WORLD_START,
    ENTITY_COUNTS,
    MODES,
    TRAINING_SEEDS,
    WIDTHS,
    audit_confirmation_root,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _paired_rows(*, failing_seed: bool = False) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for c in ENTITY_COUNTS:
        widths = WIDTHS[c]
        for width in widths[1:]:
            is_primary = (c, width) in {(64, 64), (256, 256)}
            low = 0.02 if is_primary else 0.0
            if failing_seed and (c, width) == (256, 256):
                low = -0.01
            rows.append(
                {
                    "comparison": "stable_width_vs_width1",
                    "entity_count": c,
                    "treatment_width": width,
                    "reference_width": 1,
                    "treatment_mode": "stable_persistent",
                    "reference_mode": "stable_persistent",
                    "world_count": 512,
                    "treatment_only": 20 if is_primary else 0,
                    "reference_only": 5 if is_primary else 0,
                    "both_solved": 10,
                    "neither_solved": 477,
                    "exact_solve_delta": 0.03 if is_primary else 0.0,
                    "bootstrap_ci_low": low,
                    "bootstrap_ci_high": 0.06 if is_primary else 0.0,
                }
            )
        for width in widths:
            for comparison, reference_mode in (
                ("stable_vs_reshuffled", "reshuffled_locality"),
                ("stable_vs_reset", "reset_state"),
            ):
                identity = width == 1 and comparison == "stable_vs_reshuffled"
                is_primary = c == 256 and width == 256
                low = 0.02 if is_primary else 0.0
                rows.append(
                    {
                        "comparison": comparison,
                        "entity_count": c,
                        "treatment_width": width,
                        "reference_width": width,
                        "treatment_mode": "stable_persistent",
                        "reference_mode": reference_mode,
                        "world_count": 512,
                        "treatment_only": 0 if identity else (20 if is_primary else 0),
                        "reference_only": 0 if identity else (5 if is_primary else 0),
                        "both_solved": 10,
                        "neither_solved": 502 if identity else 477,
                        "exact_solve_delta": 0.0 if identity else (0.03 if is_primary else 0.0),
                        "bootstrap_ci_low": 0.0 if identity else low,
                        "bootstrap_ci_high": 0.0 if identity else (0.06 if is_primary else 0.0),
                    }
                )
    assert len(rows) == 33
    return rows


def _primary_key(row: dict[str, object]) -> bool:
    comparison = row["comparison"]
    c = row["entity_count"]
    w = row["treatment_width"]
    return (
        comparison == "stable_width_vs_width1" and (c, w) in {(64, 64), (256, 256)}
    ) or (
        comparison in {"stable_vs_reshuffled", "stable_vs_reset"}
        and c == 256
        and w == 256
    )


def _build_root(root: Path, *, failing_seed: int | None = None) -> None:
    run_config = {
        "protocol": "gate2-persistent-state-confirmation-v0",
        "scientific_status": "FROZEN_CONFIRMATION",
        "training_seeds": [3, 4, 5],
        "steps": 1000,
        "training_batch_size": 32,
        "state_width": 64,
        "query_width": 24,
        "learning_rate": 3e-4,
        "weight_decay": 1e-4,
        "gradient_clip_norm": 1.0,
        "evaluation_world_count": 512,
        "evaluation_batch_size": 64,
        "bootstrap_samples": 2000,
        "device": "cuda",
        "idle_machine_attested": True,
        "git_head": "test-head",
    }
    (root / "run-config.json").write_text(json.dumps(run_config), encoding="utf-8")

    suite_seeds: list[dict[str, object]] = []
    all_pass = True
    world_seeds = list(range(CONFIRMATION_WORLD_START, CONFIRMATION_WORLD_START + 512))

    for seed in TRAINING_SEEDS:
        seed_root = root / f"seed_{seed}"
        seed_root.mkdir(parents=True)
        checkpoint = seed_root / "gate2-confirmation-checkpoint.pt"
        checkpoint.write_bytes(f"checkpoint-{seed}".encode("ascii"))
        checkpoint_sha = _sha256(checkpoint)
        fingerprint = f"fingerprint-{seed}"

        conditions: list[dict[str, object]] = []
        for c in ENTITY_COUNTS:
            for width in WIDTHS[c]:
                for mode in MODES:
                    conditions.append(
                        {
                            "entity_count": c,
                            "width": width,
                            "mode": mode,
                            "world_count": 512,
                            "exact_solve_rate": 0.1,
                            "bit_accuracy": 0.6,
                            "collision_load": c // width,
                            "learned_updates_per_world": 8 * c,
                            "inspected_entities_per_world": c,
                            "inspected_observations_per_world": 8 * c,
                            "learned_parameter_count": 21580,
                            "parameter_fingerprint": fingerprint,
                            "world_seeds": world_seeds,
                            "solved_by_world": [False] * 512,
                        }
                    )
        assert len(conditions) == 36

        paired = _paired_rows(failing_seed=seed == failing_seed)
        primary = []
        for row in paired:
            if not _primary_key(row):
                continue
            primary.append(
                {
                    "comparison": row["comparison"],
                    "entity_count": row["entity_count"],
                    "treatment_width": row["treatment_width"],
                    "exact_solve_delta": row["exact_solve_delta"],
                    "bootstrap_ci_low": row["bootstrap_ci_low"],
                    "bootstrap_ci_high": row["bootstrap_ci_high"],
                    "passed": float(row["bootstrap_ci_low"]) > 0.0,
                }
            )
        seed_passed = all(bool(row["passed"]) for row in primary)
        all_pass = all_pass and seed_passed
        result = {
            "experiment_version": "gate2-persistent-state-confirmation-v0",
            "evaluation_split": "confirmation",
            "confirmation_opened": True,
            "training": {
                "training_seed": seed,
                "steps": 1000,
                "examples_seen": 32000,
                "initial_loss": 0.7,
                "final_loss": 0.6,
                "mean_last_50_loss": 0.6,
                "learned_parameter_count": 21580,
                "parameter_fingerprint": fingerprint,
                "stable_training_condition_count": 12,
            },
            "training_config": {
                "steps": 1000,
                "batch_size": 32,
                "learning_rate": 3e-4,
                "weight_decay": 1e-4,
                "gradient_clip_norm": 1.0,
                "model": {"state_width": 64, "query_width": 24},
            },
            "evaluation_world_count": 512,
            "evaluation_batch_size": 64,
            "bootstrap_samples": 2000,
            "conditions": conditions,
            "paired_summaries": paired,
            "primary_comparisons": primary,
            "width1_identity_passed": True,
            "seed_passed": seed_passed,
            "scientific_status": "CONFIRMATION_SEED_RESULT",
            "gate2_verdict": "NOT_ASSIGNED_UNTIL_ALL_SEEDS_AND_RESOURCE_PROTOCOL_COMPLETE",
            "checkpoint_file": checkpoint.name,
            "checkpoint_sha256": checkpoint_sha,
        }
        result_path = seed_root / "gate2-confirmation.json"
        result_path.write_text(json.dumps(result), encoding="utf-8")
        runtime = {
            "scientific_status": "FROZEN_GATE2_CONFIRMATION_SEED",
            "checkpoint_sha256": checkpoint_sha,
            "seed_passed": seed_passed,
        }
        (seed_root / "runtime.json").write_text(json.dumps(runtime), encoding="utf-8")
        suite_seeds.append(
            {
                "training_seed": seed,
                "seed_passed": seed_passed,
                "width1_identity_passed": True,
                "result_sha256": _sha256(result_path),
                "checkpoint_sha256": checkpoint_sha,
                "parameter_fingerprint": fingerprint,
                "primary_comparisons": primary,
            }
        )

    suite = {
        "protocol": "gate2-persistent-state-confirmation-v0",
        "confirmation_training_seeds": [3, 4, 5],
        "capability_confirmation_passed": all_pass,
        "gate2_overall_verdict": "NOT_ASSIGNED_UNTIL_RESOURCE_PROTOCOL_COMPLETE",
        "seeds": suite_seeds,
    }
    (root / "confirmation-suite.json").write_text(json.dumps(suite), encoding="utf-8")


class Gate2ConfirmationAuditTests(unittest.TestCase):
    def test_valid_positive_confirmation_is_audited_as_pass(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _build_root(root)
            audit = audit_confirmation_root(root)
            self.assertTrue(audit.artifact_valid, audit.errors)
            self.assertTrue(audit.capability_confirmation_passed)
            self.assertEqual(audit.seed_passes, {3: True, 4: True, 5: True})

    def test_valid_scientific_failure_remains_a_valid_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _build_root(root, failing_seed=4)
            audit = audit_confirmation_root(root)
            self.assertTrue(audit.artifact_valid, audit.errors)
            self.assertFalse(audit.capability_confirmation_passed)
            self.assertFalse(audit.seed_passes[4])

    def test_checkpoint_tamper_is_invalid_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _build_root(root)
            (root / "seed_3" / "gate2-confirmation-checkpoint.pt").write_bytes(b"tampered")
            audit = audit_confirmation_root(root)
            self.assertFalse(audit.artifact_valid)
            self.assertIsNone(audit.capability_confirmation_passed)
            self.assertTrue(any("checkpoint SHA-256 mismatch" in error for error in audit.errors))

    def test_world_seed_tamper_is_invalid_mechanics(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _build_root(root)
            result_path = root / "seed_5" / "gate2-confirmation.json"
            result = json.loads(result_path.read_text(encoding="utf-8"))
            result["conditions"][0]["world_seeds"][0] += 1
            result_path.write_text(json.dumps(result), encoding="utf-8")
            # Keep suite hash aligned so the auditor reaches the world-seed check.
            suite_path = root / "confirmation-suite.json"
            suite = json.loads(suite_path.read_text(encoding="utf-8"))
            for row in suite["seeds"]:
                if row["training_seed"] == 5:
                    row["result_sha256"] = _sha256(result_path)
            suite_path.write_text(json.dumps(suite), encoding="utf-8")
            audit = audit_confirmation_root(root)
            self.assertFalse(audit.artifact_valid)
            self.assertTrue(any("confirmation world seeds mismatch" in error for error in audit.errors))


if __name__ == "__main__":
    unittest.main()
