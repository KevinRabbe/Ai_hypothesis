from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from ai_hypothesis.population_compute.audit_gate2_resource_frontier import (
    BATCH_SIZES,
    ENTITY_WIDTHS,
    RESOURCE_WORLD_SEED_START,
    audit_resource_root,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _timing(schedule: str, *, c: int, w: int, b: int, median: float) -> dict[str, object]:
    raw = [median] * 50
    return {
        "schedule": schedule,
        "raw_latency_ms": raw,
        "median_latency_ms": median,
        "p25_latency_ms": median,
        "p75_latency_ms": median,
        "min_latency_ms": median,
        "max_latency_ms": median,
        "samples_per_second": b / (median / 1000.0),
        "peak_allocated_bytes": 1000,
        "peak_reserved_bytes": 2000,
        "learned_updates_per_sample": 8 * c,
        "peak_simultaneous_updates_per_sample": w if schedule == "parallel_persistent" else 1,
        "persistent_state_vectors_per_sample": w,
        "collision_load": c // w,
    }


def _build_root(root: Path, *, fail_endpoint: bool = False) -> None:
    checkpoint = root / "seed3-checkpoint.pt"
    checkpoint.write_bytes(b"seed3-checkpoint")
    checkpoint_sha = _sha256(checkpoint)

    confirmation_audit = {
        "artifact_valid": True,
        "capability_confirmation_passed": True,
        "seed_passes": {"3": True, "4": True, "5": True},
        "primary_ci_lows": {},
        "errors": [],
    }
    (root / "confirmation-audit.json").write_text(json.dumps(confirmation_audit), encoding="utf-8")
    (root / "git-head.txt").write_text("resource-head\n", encoding="utf-8")
    (root / "git-status.txt").write_text("", encoding="utf-8")
    (root / "nvidia-smi-before.txt").write_text("before\n", encoding="utf-8")
    (root / "nvidia-smi-after.txt").write_text("after\n", encoding="utf-8")

    config = {
        "protocol": "gate2-persistent-state-resource-frontier-v0",
        "scientific_status": "FROZEN_RESOURCE_TIMING",
        "confirmation_measurement_head": "c2a26a17a94746ca88f29950197131689405917b",
        "resource_runner_head": "resource-head",
        "confirmation_root": str(root / "confirmation"),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": checkpoint_sha,
        "checkpoint_training_seed": 3,
        "entity_widths": {"64": [1, 4, 16, 64], "256": [1, 4, 16, 64, 256]},
        "batch_sizes": [1, 64],
        "warmup_iterations": 10,
        "timed_iterations": 50,
        "resource_world_seed_start": 4 << 30,
        "execution_mode": "eager_cuda",
        "compiler_enabled": False,
        "idle_machine_attested": True,
        "gpu": "NVIDIA GeForce RTX 4060 Ti",
        "torch_version": "test",
        "cuda_runtime": "test",
    }
    (root / "run-config.json").write_text(json.dumps(config), encoding="utf-8")

    cells: list[dict[str, object]] = []
    endpoint_passes: dict[str, bool] = {}
    for c, widths in ENTITY_WIDTHS.items():
        for w in widths:
            for b in BATCH_SIZES:
                parallel_median = 1.0
                serial_median = 2.0
                if fail_endpoint and (c, w, b) == (256, 256, 64):
                    serial_median = 0.8
                parallel = _timing("parallel_persistent", c=c, w=w, b=b, median=parallel_median)
                serial = _timing("serial_persistent", c=c, w=w, b=b, median=serial_median)
                cells.append(
                    {
                        "entity_count": c,
                        "width": w,
                        "batch_size": b,
                        "preflight": {
                            "entity_count": c,
                            "width": w,
                            "batch_size": b,
                            "world_seeds": list(
                                range(
                                    RESOURCE_WORLD_SEED_START + c * 1000,
                                    RESOURCE_WORLD_SEED_START + c * 1000 + b,
                                )
                            ),
                            "decoded_identity": True,
                            "learned_update_identity": True,
                            "state_bank_identity": True,
                            "max_abs_logit_drift": 1e-6,
                            "max_abs_final_state_drift": 1e-6,
                        },
                        "parallel": parallel,
                        "serial": serial,
                        "serial_over_parallel_median_speedup": serial_median / parallel_median,
                    }
                )
                if (c, w, b) in {(64, 64, 1), (64, 64, 64), (256, 256, 1), (256, 256, 64)}:
                    endpoint_passes[f"c{c}_w{w}_b{b}"] = parallel_median < serial_median

    resource_pass = all(endpoint_passes.values())
    result = {
        "experiment_version": "gate2-persistent-state-resource-frontier-v0",
        "checkpoint_path": str(checkpoint),
        "checkpoint_training_seed": 3,
        "checkpoint_parameter_fingerprint": "fingerprint",
        "learned_parameter_count": 21580,
        "device": "cuda",
        "cuda_device_name": "NVIDIA GeForce RTX 4060 Ti",
        "warmup_iterations": 10,
        "timed_iterations": 50,
        "cells": cells,
        "all_preflights_passed": True,
        "decision_endpoint_passes": endpoint_passes,
        "resource_frontier_passed": resource_pass,
        "scientific_status": "FROZEN_GATE2_RESOURCE_RESULT",
        "gate2_overall_verdict": "NOT_ASSIGNED_BY_RESOURCE_RUNNER",
        "checkpoint_sha256": checkpoint_sha,
        "wall_seconds": 123.0,
    }
    (root / "gate2-resource-frontier.json").write_text(json.dumps(result), encoding="utf-8")
    summary = {
        "protocol": "gate2-persistent-state-resource-frontier-v0",
        "resource_frontier_passed": resource_pass,
        "all_preflights_passed": True,
        "decision_endpoint_passes": endpoint_passes,
        "capability_confirmation_passed": True,
        "overall_gate2_v0_passed": resource_pass,
        "overall_gate2_verdict": "POSITIVE_V0" if resource_pass else "NOT_POSITIVE_V0_RESOURCE_HALF_FAILED",
    }
    (root / "gate2-v0-summary.json").write_text(json.dumps(summary), encoding="utf-8")

    names = [
        "gate2-resource-frontier.json",
        "gate2-v0-summary.json",
        "confirmation-audit.json",
        "run-config.json",
        "git-head.txt",
        "git-status.txt",
        "nvidia-smi-before.txt",
        "nvidia-smi-after.txt",
    ]
    lines = [f"{_sha256(root / name)}  {name}" for name in names]
    (root / "result-manifest.sha256").write_text("\n".join(lines) + "\n", encoding="ascii")


class Gate2ResourceAuditTests(unittest.TestCase):
    def test_positive_resource_result_audits_as_positive_gate2_v0(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _build_root(root)
            audit = audit_resource_root(root)
            self.assertTrue(audit.artifact_valid, audit.errors)
            self.assertTrue(audit.capability_confirmation_passed)
            self.assertTrue(audit.resource_frontier_passed)
            self.assertTrue(audit.overall_gate2_v0_positive)
            self.assertEqual(len(audit.endpoint_passes), 4)

    def test_negative_resource_endpoint_is_valid_scientific_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _build_root(root, fail_endpoint=True)
            audit = audit_resource_root(root)
            self.assertTrue(audit.artifact_valid, audit.errors)
            self.assertFalse(audit.resource_frontier_passed)
            self.assertFalse(audit.overall_gate2_v0_positive)
            self.assertFalse(audit.endpoint_passes["c256_w256_b64"])

    def test_raw_timing_tamper_is_invalid(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _build_root(root)
            result_path = root / "gate2-resource-frontier.json"
            result = json.loads(result_path.read_text(encoding="utf-8"))
            result["cells"][0]["parallel"]["raw_latency_ms"][0] = -1.0
            result_path.write_text(json.dumps(result), encoding="utf-8")
            # Keep manifest aligned so the numerical integrity check is reached.
            manifest_path = root / "result-manifest.sha256"
            lines = manifest_path.read_text(encoding="ascii").splitlines()
            lines = [
                f"{_sha256(result_path)}  gate2-resource-frontier.json"
                if line.endswith("gate2-resource-frontier.json") else line
                for line in lines
            ]
            manifest_path.write_text("\n".join(lines) + "\n", encoding="ascii")
            audit = audit_resource_root(root)
            self.assertFalse(audit.artifact_valid)
            self.assertTrue(any("invalid latency samples" in error for error in audit.errors))


if __name__ == "__main__":
    unittest.main()
