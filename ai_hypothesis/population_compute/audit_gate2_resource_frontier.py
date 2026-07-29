"""Independent auditor for frozen Gate-2 resource-frontier artifacts.

A resource result may be scientifically negative and still be a valid artifact.  Structural,
provenance, timing-corpus, or rule-recomputation mismatches make the audit invalid.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


PROTOCOL = "gate2-persistent-state-resource-frontier-v0"
EXPERIMENT_VERSION = "gate2-persistent-state-resource-frontier-v0"
ENTITY_WIDTHS = {64: (1, 4, 16, 64), 256: (1, 4, 16, 64, 256)}
BATCH_SIZES = (1, 64)
WARMUP_ITERATIONS = 10
TIMED_ITERATIONS = 50
RESOURCE_WORLD_SEED_START = 4 << 30
EXPECTED_CELL_COUNT = 18
PRIMARY_GPU = "NVIDIA GeForce RTX 4060 Ti"
DECISION_ENDPOINTS = ((64, 64, 1), (64, 64, 64), (256, 256, 1), (256, 256, 64))


@dataclass(frozen=True, slots=True)
class ResourceAudit:
    artifact_valid: bool
    capability_confirmation_passed: bool | None
    resource_frontier_passed: bool | None
    overall_gate2_v0_positive: bool | None
    endpoint_passes: dict[str, bool]
    errors: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain one JSON object")
    return payload


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _quantile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    index = q * (len(ordered) - 1)
    low = math.floor(index)
    high = math.ceil(index)
    if low == high:
        return ordered[low]
    fraction = index - low
    return ordered[low] * (1.0 - fraction) + ordered[high] * fraction


def _close(a: Any, b: float, *, rel: float = 1e-9, abs_: float = 1e-9) -> bool:
    return isinstance(a, (int, float)) and math.isclose(float(a), b, rel_tol=rel, abs_tol=abs_)


def audit_resource_root(root: Path) -> ResourceAudit:
    root = root.resolve()
    errors: list[str] = []
    endpoint_passes: dict[str, bool] = {}

    required = {
        "result": root / "gate2-resource-frontier.json",
        "summary": root / "gate2-v0-summary.json",
        "confirmation_audit": root / "confirmation-audit.json",
        "config": root / "run-config.json",
        "git_head": root / "git-head.txt",
        "git_status": root / "git-status.txt",
        "nvidia_before": root / "nvidia-smi-before.txt",
        "nvidia_after": root / "nvidia-smi-after.txt",
        "manifest": root / "result-manifest.sha256",
    }
    for name, path in required.items():
        if not path.is_file():
            errors.append(f"missing required resource artifact: {name} ({path.name})")
    if errors:
        return ResourceAudit(False, None, None, None, endpoint_passes, tuple(errors))

    try:
        result = _load_json(required["result"])
        summary = _load_json(required["summary"])
        confirmation_audit = _load_json(required["confirmation_audit"])
        config = _load_json(required["config"])
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return ResourceAudit(False, None, None, None, endpoint_passes, (str(exc),))

    # Manifest verifies the bytes written by the admitted runner.
    manifest_lines = required["manifest"].read_text(encoding="ascii").splitlines()
    manifest: dict[str, str] = {}
    for line in manifest_lines:
        parts = line.strip().split(maxsplit=1)
        if len(parts) != 2:
            errors.append(f"malformed manifest line: {line!r}")
            continue
        manifest[parts[1]] = parts[0]
    for name in (
        "gate2-resource-frontier.json",
        "gate2-v0-summary.json",
        "confirmation-audit.json",
        "run-config.json",
        "git-head.txt",
        "git-status.txt",
        "nvidia-smi-before.txt",
        "nvidia-smi-after.txt",
    ):
        path = root / name
        expected = manifest.get(name)
        if expected is None:
            errors.append(f"manifest missing {name}")
        elif expected != _sha256(path):
            errors.append(f"manifest SHA-256 mismatch for {name}")

    if required["git_status"].read_text(encoding="utf-8-sig").strip():
        errors.append("resource runner recorded a dirty Git working tree")

    if confirmation_audit.get("artifact_valid") is not True:
        errors.append("embedded capability confirmation audit is not structurally valid")
    capability_passed = confirmation_audit.get("capability_confirmation_passed")
    if capability_passed is not True:
        errors.append("resource timing was admitted without a passed capability confirmation")

    expected_config = {
        "protocol": PROTOCOL,
        "scientific_status": "FROZEN_RESOURCE_TIMING",
        "confirmation_measurement_head": "c2a26a17a94746ca88f29950197131689405917b",
        "checkpoint_training_seed": 3,
        "batch_sizes": [1, 64],
        "warmup_iterations": WARMUP_ITERATIONS,
        "timed_iterations": TIMED_ITERATIONS,
        "resource_world_seed_start": RESOURCE_WORLD_SEED_START,
        "execution_mode": "eager_cuda",
        "compiler_enabled": False,
        "idle_machine_attested": True,
        "gpu": PRIMARY_GPU,
    }
    for key, expected in expected_config.items():
        if config.get(key) != expected:
            errors.append(f"run-config {key!r} expected {expected!r}, got {config.get(key)!r}")
    if config.get("entity_widths") != {"64": [1, 4, 16, 64], "256": [1, 4, 16, 64, 256]}:
        errors.append("run-config entity_widths mismatch")

    checkpoint_path = Path(str(config.get("checkpoint", "")))
    if not checkpoint_path.is_file():
        errors.append("resource run-config checkpoint path is missing locally")
    else:
        checkpoint_sha = _sha256(checkpoint_path)
        if config.get("checkpoint_sha256") != checkpoint_sha:
            errors.append("run-config checkpoint SHA-256 mismatch")
        if result.get("checkpoint_sha256") != checkpoint_sha:
            errors.append("resource result checkpoint SHA-256 mismatch")

    if result.get("experiment_version") != EXPERIMENT_VERSION:
        errors.append("resource experiment version mismatch")
    if result.get("checkpoint_training_seed") != 3:
        errors.append("resource result is not bound to confirmation seed 3")
    if result.get("cuda_device_name") != PRIMARY_GPU:
        errors.append("resource result GPU mismatch")
    if result.get("device") != "cuda":
        errors.append("resource result device must be cuda")
    if result.get("warmup_iterations") != WARMUP_ITERATIONS:
        errors.append("resource result warmup count mismatch")
    if result.get("timed_iterations") != TIMED_ITERATIONS:
        errors.append("resource result timed-iteration count mismatch")
    if result.get("scientific_status") != "FROZEN_GATE2_RESOURCE_RESULT":
        errors.append("resource scientific_status mismatch")
    if result.get("gate2_overall_verdict") != "NOT_ASSIGNED_BY_RESOURCE_RUNNER":
        errors.append("resource result assigned an overall verdict internally")

    cells = result.get("cells")
    if not isinstance(cells, list) or len(cells) != EXPECTED_CELL_COUNT:
        errors.append(f"expected {EXPECTED_CELL_COUNT} resource cells")
        cells = []
    index: dict[tuple[int, int, int], dict[str, Any]] = {}
    for cell in cells:
        if not isinstance(cell, dict):
            errors.append("resource cell must be an object")
            continue
        key = (cell.get("entity_count"), cell.get("width"), cell.get("batch_size"))
        if key in index:
            errors.append(f"duplicate resource cell {key}")
        index[key] = cell
    expected_keys = {
        (c, w, b)
        for c, widths in ENTITY_WIDTHS.items()
        for w in widths
        for b in BATCH_SIZES
    }
    if set(index) != expected_keys:
        errors.append("resource matrix is not the frozen 18-cell matrix")

    all_preflights = True
    for c, w, b in sorted(expected_keys):
        cell = index.get((c, w, b))
        if cell is None:
            all_preflights = False
            continue
        preflight = cell.get("preflight", {})
        if not isinstance(preflight, dict):
            errors.append(f"C{c}/W{w}/B{b}: missing preflight")
            all_preflights = False
            continue
        expected_worlds = list(range(RESOURCE_WORLD_SEED_START + c * 1000, RESOURCE_WORLD_SEED_START + c * 1000 + b))
        if preflight.get("world_seeds") != expected_worlds:
            errors.append(f"C{c}/W{w}/B{b}: resource world seed corpus mismatch")
        for key in ("decoded_identity", "learned_update_identity", "state_bank_identity"):
            if preflight.get(key) is not True:
                errors.append(f"C{c}/W{w}/B{b}: preflight {key} failed")
                all_preflights = False
        for key in ("max_abs_logit_drift", "max_abs_final_state_drift"):
            value = preflight.get(key)
            if not isinstance(value, (int, float)) or not math.isfinite(float(value)) or float(value) < 0.0:
                errors.append(f"C{c}/W{w}/B{b}: invalid {key}")

        timings: dict[str, dict[str, Any]] = {}
        for schedule_key, expected_schedule, peak_updates in (
            ("parallel", "parallel_persistent", w),
            ("serial", "serial_persistent", 1),
        ):
            timing = cell.get(schedule_key)
            if not isinstance(timing, dict):
                errors.append(f"C{c}/W{w}/B{b}: missing {schedule_key} timing")
                continue
            timings[schedule_key] = timing
            if timing.get("schedule") != expected_schedule:
                errors.append(f"C{c}/W{w}/B{b}: {schedule_key} schedule label mismatch")
            raw = timing.get("raw_latency_ms")
            if not isinstance(raw, list) or len(raw) != TIMED_ITERATIONS:
                errors.append(f"C{c}/W{w}/B{b}: {schedule_key} raw timing count mismatch")
                continue
            if not all(isinstance(value, (int, float)) and math.isfinite(float(value)) and float(value) > 0 for value in raw):
                errors.append(f"C{c}/W{w}/B{b}: {schedule_key} contains invalid latency samples")
                continue
            raw_f = [float(value) for value in raw]
            median = float(statistics.median(raw_f))
            recomputed = {
                "median_latency_ms": median,
                "p25_latency_ms": _quantile(raw_f, 0.25),
                "p75_latency_ms": _quantile(raw_f, 0.75),
                "min_latency_ms": min(raw_f),
                "max_latency_ms": max(raw_f),
                "samples_per_second": b / (median / 1000.0),
            }
            for metric, expected in recomputed.items():
                if not _close(timing.get(metric), expected, rel=1e-8, abs_=1e-8):
                    errors.append(f"C{c}/W{w}/B{b}: {schedule_key} {metric} mismatch")
            if timing.get("learned_updates_per_sample") != 8 * c:
                errors.append(f"C{c}/W{w}/B{b}: {schedule_key} learned update count mismatch")
            if timing.get("persistent_state_vectors_per_sample") != w:
                errors.append(f"C{c}/W{w}/B{b}: {schedule_key} state-bank size mismatch")
            if timing.get("peak_simultaneous_updates_per_sample") != peak_updates:
                errors.append(f"C{c}/W{w}/B{b}: {schedule_key} peak simultaneous updates mismatch")
            if timing.get("collision_load") != c // w:
                errors.append(f"C{c}/W{w}/B{b}: {schedule_key} collision load mismatch")
            for memory_key in ("peak_allocated_bytes", "peak_reserved_bytes"):
                memory = timing.get(memory_key)
                if not isinstance(memory, int) or memory < 0:
                    errors.append(f"C{c}/W{w}/B{b}: {schedule_key} invalid {memory_key}")

        if set(timings) == {"parallel", "serial"}:
            p = timings["parallel"].get("median_latency_ms")
            s = timings["serial"].get("median_latency_ms")
            if isinstance(p, (int, float)) and isinstance(s, (int, float)) and float(p) > 0:
                speedup = float(s) / float(p)
                if not _close(cell.get("serial_over_parallel_median_speedup"), speedup, rel=1e-8, abs_=1e-8):
                    errors.append(f"C{c}/W{w}/B{b}: derived speedup mismatch")

    if result.get("all_preflights_passed") is not all_preflights:
        errors.append("resource all_preflights_passed differs from independent recomputation")

    for c, w, b in DECISION_ENDPOINTS:
        key = f"c{c}_w{w}_b{b}"
        cell = index.get((c, w, b))
        passed = False
        if cell:
            parallel = cell.get("parallel", {})
            serial = cell.get("serial", {})
            p = parallel.get("median_latency_ms") if isinstance(parallel, dict) else None
            s = serial.get("median_latency_ms") if isinstance(serial, dict) else None
            passed = isinstance(p, (int, float)) and isinstance(s, (int, float)) and float(p) < float(s)
        endpoint_passes[key] = passed

    declared_endpoints = result.get("decision_endpoint_passes")
    if declared_endpoints != endpoint_passes:
        errors.append("declared resource endpoint decisions differ from independent recomputation")
    recomputed_resource_pass = all_preflights and len(endpoint_passes) == 4 and all(endpoint_passes.values())
    if result.get("resource_frontier_passed") is not recomputed_resource_pass:
        errors.append("resource_frontier_passed differs from independent recomputation")

    if summary.get("protocol") != PROTOCOL:
        errors.append("Gate-2 v0 summary protocol mismatch")
    if summary.get("capability_confirmation_passed") is not True:
        errors.append("Gate-2 v0 summary lost capability-confirmation pass")
    if summary.get("resource_frontier_passed") is not recomputed_resource_pass:
        errors.append("Gate-2 v0 summary resource pass mismatch")
    if summary.get("decision_endpoint_passes") != endpoint_passes:
        errors.append("Gate-2 v0 summary endpoint decisions mismatch")
    if summary.get("overall_gate2_v0_passed") is not recomputed_resource_pass:
        errors.append("Gate-2 v0 summary overall pass mismatch")
    expected_verdict = "POSITIVE_V0" if recomputed_resource_pass else "NOT_POSITIVE_V0_RESOURCE_HALF_FAILED"
    if summary.get("overall_gate2_verdict") != expected_verdict:
        errors.append("Gate-2 v0 summary verdict mismatch")

    if errors:
        return ResourceAudit(False, None, None, None, endpoint_passes, tuple(errors))
    return ResourceAudit(
        artifact_valid=True,
        capability_confirmation_passed=True,
        resource_frontier_passed=recomputed_resource_pass,
        overall_gate2_v0_positive=recomputed_resource_pass,
        endpoint_passes=endpoint_passes,
        errors=(),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    audit = audit_resource_root(args.root)
    text = json.dumps(audit.to_dict(), indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(text, encoding="utf-8")
    print(text, end="")
    return 0 if audit.artifact_valid else 2


if __name__ == "__main__":
    raise SystemExit(main())
