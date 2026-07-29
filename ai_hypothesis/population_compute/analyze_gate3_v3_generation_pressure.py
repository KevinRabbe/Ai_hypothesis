"""Independent auditor for Gate-3 v3 generation-pressure development artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

EXPERIMENT_VERSION = "gate3-v3-generation-pressure-v0"
CHECKPOINTS = {
    0: {
        "sha256": "e63748a1182472d03c540f9123c3442ce44e130519e2176681034568826db590",
        "fingerprint": "e4f9990f08e85393a155637cfa50b5831d779770447ec3716fc9e67946992afc",
    },
    1: {
        "sha256": "8ce384627f5543fa4fe78498f9990f8214cf2a4afed0c9b734e86761ad13c989",
        "fingerprint": "2a57984f755ee1404fd828c2df36a5efff8f13a541b7d7c93a891014fbf4897c",
    },
    2: {
        "sha256": "103826bbd9451b965eced5134d1674cb8e893f5d3b378cb828312ffdb6fc9a37",
        "fingerprint": "8afaf956f200f41ea914eafdd1b5f151dd303cae1552165c74e12bb8c945af02",
    },
}
CONDITIONS = (
    (16, "stable_reserve"),
    (64, "stable_reserve"),
    (256, "stable_reserve"),
    (256, "collapsed_diversity"),
    (256, "reshuffled_continuity"),
)
PAIR_SPECS = (
    ("stable_l256_vs_l64", 256, "stable_reserve", 64, "stable_reserve"),
    ("stable_l64_vs_l16", 64, "stable_reserve", 16, "stable_reserve"),
    ("stable_l256_vs_collapsed", 256, "stable_reserve", 256, "collapsed_diversity"),
    ("stable_l256_vs_reshuffled", 256, "stable_reserve", 256, "reshuffled_continuity"),
)
WORLD_COUNT = 256
BATCH_SIZE = 64
BOOTSTRAP_SAMPLES = 2_000
DEPTH = 8
HINT_RELIABILITY = 0.70
SCHEDULED_SLOTS = 223
TOTAL_UPDATES = 3_568
PARAMETER_COUNT = 19_649
EXPECTED_CONDITIONS = 15
EXPECTED_PAIRS = 12


@dataclass(frozen=True, slots=True)
class Gate3V3Audit:
    artifact_valid: bool
    scientific_status: str
    directional_outcome: str | None
    primary_deltas: dict[str, float]
    primary_ci_lows: dict[str, float]
    primary_ci_highs: dict[str, float]
    errors: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _load(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise ValueError("Gate-3 v3 result must be one JSON object")
    return payload


def _seed_from_parts(*parts: object) -> int:
    digest = hashlib.sha256(":".join(str(part) for part in parts).encode("ascii")).digest()
    return int.from_bytes(digest[:8], "big")


def _runtime_seed(world_index: int) -> int:
    return _seed_from_parts("gate3-v3-generation-pressure-development-runtime", world_index, DEPTH)


def _bootstrap_ci(
    differences: tuple[int, ...], *, checkpoint: int, comparison: str
) -> tuple[float, float]:
    rng = random.Random(
        _seed_from_parts("gate3-v3-generation-pressure-bootstrap", checkpoint, comparison)
    )
    estimates = sorted(
        sum(differences[rng.randrange(WORLD_COUNT)] for _ in range(WORLD_COUNT)) / WORLD_COUNT
        for _ in range(BOOTSTRAP_SAMPLES)
    )
    return (
        estimates[int(math.floor(0.025 * (BOOTSTRAP_SAMPLES - 1)))],
        estimates[int(math.ceil(0.975 * (BOOTSTRAP_SAMPLES - 1)))],
    )


def _float_equal(expected: float, observed: Any) -> bool:
    try:
        return math.isclose(expected, float(observed), rel_tol=0.0, abs_tol=1e-12)
    except (TypeError, ValueError):
        return False


def _metric_key(checkpoint: int, comparison: str) -> str:
    return f"c{checkpoint}_{comparison}"


def _recompute_pair(
    *,
    comparison: str,
    checkpoint: int,
    treatment: dict[str, Any],
    reference: dict[str, Any],
) -> dict[str, Any]:
    if treatment.get("world_indices") != reference.get("world_indices"):
        raise ValueError("paired Gate-3 v3 conditions use different world indices")
    a = treatment.get("covered_by_world")
    b = reference.get("covered_by_world")
    if not isinstance(a, list) or not isinstance(b, list) or len(a) != WORLD_COUNT or len(b) != WORLD_COUNT:
        raise ValueError("paired Gate-3 v3 coverage vectors must contain 256 worlds")
    pairs = tuple(zip(a, b, strict=True))
    treatment_only = sum(int(bool(x) and not bool(y)) for x, y in pairs)
    reference_only = sum(int(bool(y) and not bool(x)) for x, y in pairs)
    both = sum(int(bool(x) and bool(y)) for x, y in pairs)
    neither = WORLD_COUNT - treatment_only - reference_only - both
    differences = tuple(int(bool(x)) - int(bool(y)) for x, y in pairs)
    low, high = _bootstrap_ci(differences, checkpoint=checkpoint, comparison=comparison)
    return {
        "comparison": comparison,
        "checkpoint_index": checkpoint,
        "treatment_capacity": treatment["reserve_capacity"],
        "treatment_mode": treatment["mode"],
        "reference_capacity": reference["reserve_capacity"],
        "reference_mode": reference["mode"],
        "world_count": WORLD_COUNT,
        "treatment_only": treatment_only,
        "reference_only": reference_only,
        "both_covered": both,
        "neither_covered": neither,
        "coverage_delta": sum(differences) / WORLD_COUNT,
        "bootstrap_ci_low": low,
        "bootstrap_ci_high": high,
    }


def _classify(lows: dict[str, float]) -> str:
    control_keys = [
        _metric_key(checkpoint, comparison)
        for checkpoint in CHECKPOINTS
        for comparison in ("stable_l256_vs_collapsed", "stable_l256_vs_reshuffled")
    ]
    if any(lows[key] <= 0.0 for key in control_keys):
        return "V3_G3_CONTROL_OR_MECHANISM_DEGRADATION"

    frontier = [_metric_key(checkpoint, "stable_l256_vs_l64") for checkpoint in CHECKPOINTS]
    significant = [lows[key] > 0.0 for key in frontier]
    if all(significant):
        return "V3_G1_ROBUST_GENERATION_PRESSURE_BENEFIT"
    if not any(significant):
        return "V3_G0_NO_L256_PRESSURE_BENEFIT"
    return "V3_G2_CHECKPOINT_SENSITIVE_PRESSURE_BENEFIT"


def _validate_constant_vector(
    row: dict[str, Any], name: str, expected: int, errors: list[str], key: tuple[Any, ...]
) -> None:
    values = row.get(name)
    if not isinstance(values, list) or len(values) != WORLD_COUNT or any(value != expected for value in values):
        errors.append(f"condition {key} violates frozen {name}={expected}")


def audit_gate3_v3_generation_pressure(path: Path) -> Gate3V3Audit:
    errors: list[str] = []
    try:
        payload = _load(path)
    except Exception as exc:  # noqa: BLE001
        return Gate3V3Audit(False, "INVALID_ARTIFACT", None, {}, {}, {}, (str(exc),))

    if payload.get("experiment_version") != EXPERIMENT_VERSION:
        errors.append("unexpected experiment version")
    if payload.get("scientific_status") != "DEVELOPMENT_ONLY_NO_GATE_VERDICT":
        errors.append("unexpected scientific status")
    if payload.get("scientific_decision") != "DEVELOPMENT_ONLY_NOT_ASSIGNED":
        errors.append("development artifact attempted to assign a Gate verdict")
    if payload.get("confirmation_opened") is not False:
        errors.append("confirmation must remain closed")
    if payload.get("training_performed") is not False:
        errors.append("Gate-3 v3 must perform no training")
    if payload.get("world_count") != WORLD_COUNT:
        errors.append("world count differs from frozen 256")
    if payload.get("evaluation_batch_size") != BATCH_SIZE:
        errors.append("evaluation batch size differs from frozen 64")
    if payload.get("bootstrap_samples") != BOOTSTRAP_SAMPLES:
        errors.append("bootstrap sample count differs from frozen 2000")
    if payload.get("depth") != DEPTH:
        errors.append("depth differs from frozen depth 8")
    if not _float_equal(HINT_RELIABILITY, payload.get("hint_reliability")):
        errors.append("hint reliability differs from frozen 0.70")
    if payload.get("scheduled_slots") != SCHEDULED_SLOTS:
        errors.append("scheduled slot budget differs from frozen 223")
    if payload.get("active_child_lanes") != 2 or payload.get("recurrent_updates_per_child") != 8:
        errors.append("active neural lane/update identity differs from frozen values")
    if payload.get("total_learned_updates_per_world") != TOTAL_UPDATES:
        errors.append("top-level learned-work total differs from 3568")

    checkpoints = payload.get("checkpoints")
    if not isinstance(checkpoints, list) or len(checkpoints) != 3:
        errors.append("artifact must identify exactly three frozen checkpoints")
        checkpoints = []
    seen_checkpoints: set[int] = set()
    for row in checkpoints:
        if not isinstance(row, dict) or row.get("checkpoint_index") not in CHECKPOINTS:
            errors.append("invalid checkpoint identity row")
            continue
        checkpoint = int(row["checkpoint_index"])
        if checkpoint in seen_checkpoints:
            errors.append(f"duplicate checkpoint identity {checkpoint}")
            continue
        seen_checkpoints.add(checkpoint)
        expected = CHECKPOINTS[checkpoint]
        if str(row.get("checkpoint_sha256", "")).lower() != expected["sha256"]:
            errors.append(f"checkpoint {checkpoint} SHA256 differs from frozen identity")
        if row.get("parameter_fingerprint") != expected["fingerprint"]:
            errors.append(f"checkpoint {checkpoint} fingerprint differs from frozen identity")
        if row.get("learned_parameter_count") != PARAMETER_COUNT:
            errors.append(f"checkpoint {checkpoint} parameter count differs from 19649")
    if seen_checkpoints != set(CHECKPOINTS):
        errors.append("checkpoint identity set is incomplete")

    conditions = payload.get("conditions")
    if not isinstance(conditions, list) or len(conditions) != EXPECTED_CONDITIONS:
        errors.append("result must contain exactly 15 conditions")
        conditions = []
    index: dict[tuple[int, int, str], dict[str, Any]] = {}
    expected_world_indices = list(range(WORLD_COUNT))
    expected_runtime_seeds = [_runtime_seed(index) for index in range(WORLD_COUNT)]

    for row in conditions:
        if not isinstance(row, dict):
            errors.append("condition is not an object")
            continue
        key = (row.get("checkpoint_index"), row.get("reserve_capacity"), row.get("mode"))
        if key in index:
            errors.append(f"duplicate condition {key}")
            continue
        index[key] = row
        checkpoint, capacity, mode = key
        if checkpoint not in CHECKPOINTS or (capacity, mode) not in CONDITIONS:
            errors.append(f"unexpected condition {key}")
            continue
        if row.get("world_count") != WORLD_COUNT or row.get("world_indices") != expected_world_indices:
            errors.append(f"condition {key} does not use frozen world indices")
        if row.get("runtime_seeds") != expected_runtime_seeds:
            errors.append(f"condition {key} runtime seeds differ from frozen namespace")
        covered = row.get("covered_by_world")
        if not isinstance(covered, list) or len(covered) != WORLD_COUNT:
            errors.append(f"condition {key} has invalid coverage vector")
        else:
            rate = sum(int(bool(value)) for value in covered) / WORLD_COUNT
            if not _float_equal(rate, row.get("coverage_rate")):
                errors.append(f"condition {key} coverage rate differs from raw vector")
        if row.get("total_learned_updates_per_world") != TOTAL_UPDATES:
            errors.append(f"condition {key} violates frozen learned-work total")
        if row.get("learned_parameter_count") != PARAMETER_COUNT:
            errors.append(f"condition {key} parameter count differs from 19649")
        if checkpoint in CHECKPOINTS and row.get("parameter_fingerprint") != CHECKPOINTS[checkpoint]["fingerprint"]:
            errors.append(f"condition {key} parameter fingerprint mismatch")

        productive = row.get("productive_slots_by_world")
        sink = row.get("sink_slots_by_world")
        if not isinstance(productive, list) or not isinstance(sink, list) or len(productive) != WORLD_COUNT or len(sink) != WORLD_COUNT:
            errors.append(f"condition {key} has invalid productive/sink vectors")
        else:
            if any(int(a) + int(b) != SCHEDULED_SLOTS for a, b in zip(productive, sink, strict=True)):
                errors.append(f"condition {key} violates 223-slot accounting")

        for field in (
            "preprune_widths_by_world",
            "retained_widths_by_world",
            "unique_retained_widths_by_world",
            "binding_by_generation_by_world",
        ):
            values = row.get(field)
            if not isinstance(values, list) or len(values) != WORLD_COUNT or any(not isinstance(v, list) or len(v) != 7 for v in values):
                errors.append(f"condition {key} has invalid seven-generation telemetry in {field}")

        if mode == "stable_reserve" and capacity == 64:
            _validate_constant_vector(row, "depth7_preprune_width_by_world", 128, errors, key)
            _validate_constant_vector(row, "depth7_retained_width_by_world", 64, errors, key)
            _validate_constant_vector(row, "depth7_expanded_parents_by_world", 64, errors, key)
            _validate_constant_vector(row, "productive_slots_by_world", 191, errors, key)
            _validate_constant_vector(row, "sink_slots_by_world", 32, errors, key)
        if mode == "stable_reserve" and capacity == 256:
            _validate_constant_vector(row, "depth7_preprune_width_by_world", 128, errors, key)
            _validate_constant_vector(row, "depth7_retained_width_by_world", 128, errors, key)
            _validate_constant_vector(row, "depth7_expanded_parents_by_world", 96, errors, key)
            _validate_constant_vector(row, "productive_slots_by_world", 223, errors, key)
            _validate_constant_vector(row, "sink_slots_by_world", 0, errors, key)

    expected_keys = {
        (checkpoint, capacity, mode)
        for checkpoint in CHECKPOINTS
        for capacity, mode in CONDITIONS
    }
    if set(index) != expected_keys:
        errors.append("condition matrix differs from frozen 3x5 design")

    pair_rows = payload.get("paired_summaries")
    if not isinstance(pair_rows, list) or len(pair_rows) != EXPECTED_PAIRS:
        errors.append("result must contain exactly 12 paired summaries")
        pair_rows = []
    observed_pairs = {
        (
            row.get("comparison"),
            row.get("checkpoint_index"),
            row.get("treatment_capacity"),
            row.get("treatment_mode"),
            row.get("reference_capacity"),
            row.get("reference_mode"),
        ): row
        for row in pair_rows
        if isinstance(row, dict)
    }

    deltas: dict[str, float] = {}
    lows: dict[str, float] = {}
    highs: dict[str, float] = {}
    for checkpoint in CHECKPOINTS:
        for comparison, t_cap, t_mode, r_cap, r_mode in PAIR_SPECS:
            try:
                recomputed = _recompute_pair(
                    comparison=comparison,
                    checkpoint=checkpoint,
                    treatment=index[(checkpoint, t_cap, t_mode)],
                    reference=index[(checkpoint, r_cap, r_mode)],
                )
            except Exception as exc:  # noqa: BLE001
                errors.append(f"could not reconstruct C{checkpoint} {comparison}: {exc}")
                continue
            pair_key = (comparison, checkpoint, t_cap, t_mode, r_cap, r_mode)
            observed = observed_pairs.get(pair_key)
            if observed is None:
                errors.append(f"missing paired summary {pair_key}")
                continue
            for field in (
                "world_count",
                "treatment_only",
                "reference_only",
                "both_covered",
                "neither_covered",
            ):
                if observed.get(field) != recomputed[field]:
                    errors.append(f"paired summary {pair_key} differs in {field}")
            for field in ("coverage_delta", "bootstrap_ci_low", "bootstrap_ci_high"):
                if not _float_equal(recomputed[field], observed.get(field)):
                    errors.append(f"paired summary {pair_key} differs in {field}")
            metric = _metric_key(checkpoint, comparison)
            deltas[metric] = recomputed["coverage_delta"]
            lows[metric] = recomputed["bootstrap_ci_low"]
            highs[metric] = recomputed["bootstrap_ci_high"]

    outcome = None if errors else _classify(lows)
    return Gate3V3Audit(
        artifact_valid=not errors,
        scientific_status="DEVELOPMENT_ONLY_NO_GATE_VERDICT" if not errors else "INVALID_ARTIFACT",
        directional_outcome=outcome,
        primary_deltas=deltas,
        primary_ci_lows=lows,
        primary_ci_highs=highs,
        errors=tuple(errors),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    audit = audit_gate3_v3_generation_pressure(args.result)
    text = json.dumps(audit.to_dict(), indent=2, sort_keys=True) + "\n"
    print(text, end="")
    if args.output is not None:
        args.output.write_text(text, encoding="utf-8")
    return 0 if audit.artifact_valid else 2


if __name__ == "__main__":
    raise SystemExit(main())
