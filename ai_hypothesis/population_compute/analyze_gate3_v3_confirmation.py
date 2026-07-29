"""Independent auditor for Gate-3 v3 generation-pressure confirmation artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

EXPERIMENT_VERSION = "gate3-v3-generation-pressure-confirmation-v0"
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
WORLD_COUNT = 512
BATCH_SIZE = 64
BOOTSTRAP_SAMPLES = 4_000
DEPTH = 8
HINT_RELIABILITY = 0.70
SCHEDULED_SLOTS = 223
TOTAL_UPDATES = 3_568
PARAMETER_COUNT = 19_649
EXPECTED_CONDITIONS = 15
EXPECTED_PAIRS = 12


@dataclass(frozen=True, slots=True)
class Gate3V3ConfirmationAudit:
    artifact_valid: bool
    scientific_status: str
    confirmation_outcome: str | None
    primary_deltas: dict[str, float]
    primary_ci_lows: dict[str, float]
    primary_ci_highs: dict[str, float]
    errors: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _load(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise ValueError("Gate-3 v3 confirmation result must be one JSON object")
    return payload


def _seed_from_parts(*parts: object) -> int:
    digest = hashlib.sha256(":".join(str(part) for part in parts).encode("ascii")).digest()
    return int.from_bytes(digest[:8], "big")


def _runtime_seed(world_index: int) -> int:
    return _seed_from_parts(
        "gate3-v3-generation-pressure-confirmation-runtime",
        world_index,
        DEPTH,
    )


def _bootstrap_ci(
    differences: tuple[int, ...], *, checkpoint: int, comparison: str
) -> tuple[float, float]:
    rng = random.Random(
        _seed_from_parts(
            "gate3-v3-generation-pressure-confirmation-bootstrap",
            checkpoint,
            comparison,
        )
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
        raise ValueError("paired confirmation conditions use different world indices")
    a = treatment.get("covered_by_world")
    b = reference.get("covered_by_world")
    if not isinstance(a, list) or not isinstance(b, list) or len(a) != WORLD_COUNT or len(b) != WORLD_COUNT:
        raise ValueError("paired confirmation coverage vectors must each contain 512 worlds")
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
        return "GATE3_V3_CONFIRMATION_INVALID_OR_MECHANISM_FAILED"

    primary_keys = [_metric_key(checkpoint, "stable_l256_vs_l64") for checkpoint in CHECKPOINTS]
    if all(lows[key] > 0.0 for key in primary_keys):
        return "GATE3_V3_CONFIRMED_GENERATION_PRESSURE_BENEFIT"
    return "GATE3_V3_CONFIRMATION_NOT_ESTABLISHED"


def _validate_constant_vector(
    row: dict[str, Any], field: str, expected: int, errors: list[str], key: tuple[Any, ...]
) -> None:
    values = row.get(field)
    if not isinstance(values, list) or len(values) != WORLD_COUNT or any(value != expected for value in values):
        errors.append(f"condition {key} violates frozen {field}={expected}")


def audit_gate3_v3_confirmation(path: Path) -> Gate3V3ConfirmationAudit:
    errors: list[str] = []
    try:
        payload = _load(path)
    except Exception as exc:  # noqa: BLE001
        return Gate3V3ConfirmationAudit(False, "INVALID_ARTIFACT", None, {}, {}, {}, (str(exc),))

    if payload.get("experiment_version") != EXPERIMENT_VERSION:
        errors.append("unexpected confirmation experiment version")
    if payload.get("scientific_status") != "CONFIRMATION_DATA_COMPLETE_PENDING_INDEPENDENT_AUDIT":
        errors.append("unexpected confirmation scientific status")
    if payload.get("scientific_decision") != "NOT_ASSIGNED_UNTIL_INDEPENDENT_CONFIRMATION_AUDIT":
        errors.append("confirmation result assigned a decision before independent audit")
    if payload.get("confirmation_opened") is not True:
        errors.append("confirmation artifact must record confirmation_opened=true")
    if payload.get("training_performed") is not False:
        errors.append("Gate-3 v3 confirmation must perform no training")
    if payload.get("world_count") != WORLD_COUNT:
        errors.append("confirmation world count differs from frozen 512")
    if payload.get("evaluation_batch_size") != BATCH_SIZE:
        errors.append("confirmation batch size differs from frozen 64")
    if payload.get("bootstrap_samples") != BOOTSTRAP_SAMPLES:
        errors.append("confirmation bootstrap count differs from frozen 4000")
    if payload.get("depth") != DEPTH:
        errors.append("confirmation depth differs from 8")
    if not _float_equal(HINT_RELIABILITY, payload.get("hint_reliability")):
        errors.append("confirmation hint reliability differs from 0.70")
    if payload.get("scheduled_slots") != SCHEDULED_SLOTS:
        errors.append("confirmation scheduled-slot budget differs from 223")
    if payload.get("active_child_lanes") != 2 or payload.get("recurrent_updates_per_child") != 8:
        errors.append("confirmation active neural lane/update identity differs from frozen values")
    if payload.get("total_learned_updates_per_world") != TOTAL_UPDATES:
        errors.append("confirmation learned-work total differs from 3568")

    checkpoints = payload.get("checkpoints")
    if not isinstance(checkpoints, list) or len(checkpoints) != 3:
        errors.append("confirmation must identify exactly three frozen checkpoints")
        checkpoints = []
    seen_checkpoints: set[int] = set()
    for row in checkpoints:
        if not isinstance(row, dict) or row.get("checkpoint_index") not in CHECKPOINTS:
            errors.append("invalid confirmation checkpoint identity row")
            continue
        checkpoint = int(row["checkpoint_index"])
        if checkpoint in seen_checkpoints:
            errors.append(f"duplicate confirmation checkpoint identity {checkpoint}")
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
        errors.append("confirmation checkpoint identity set is incomplete")

    conditions = payload.get("conditions")
    if not isinstance(conditions, list) or len(conditions) != EXPECTED_CONDITIONS:
        errors.append("confirmation result must contain exactly 15 conditions")
        conditions = []

    index: dict[tuple[int, int, str], dict[str, Any]] = {}
    expected_world_indices = list(range(WORLD_COUNT))
    expected_runtime_seeds = [_runtime_seed(i) for i in range(WORLD_COUNT)]

    for row in conditions:
        if not isinstance(row, dict):
            errors.append("confirmation condition is not an object")
            continue
        key = (row.get("checkpoint_index"), row.get("reserve_capacity"), row.get("mode"))
        if key in index:
            errors.append(f"duplicate confirmation condition {key}")
            continue
        index[key] = row
        checkpoint, capacity, mode = key
        if checkpoint not in CHECKPOINTS or (capacity, mode) not in CONDITIONS:
            errors.append(f"unexpected confirmation condition {key}")
            continue
        if row.get("world_count") != WORLD_COUNT or row.get("world_indices") != expected_world_indices:
            errors.append(f"condition {key} does not use frozen 512 world indices")
        if row.get("runtime_seeds") != expected_runtime_seeds:
            errors.append(f"condition {key} runtime seeds differ from untouched confirmation namespace")
        covered = row.get("covered_by_world")
        if not isinstance(covered, list) or len(covered) != WORLD_COUNT:
            errors.append(f"condition {key} has invalid confirmation coverage vector")
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
        elif any(int(a) + int(b) != SCHEDULED_SLOTS for a, b in zip(productive, sink, strict=True)):
            errors.append(f"condition {key} violates 223-slot accounting")

        for field in (
            "preprune_widths_by_world",
            "retained_widths_by_world",
            "unique_retained_widths_by_world",
            "binding_by_generation_by_world",
        ):
            values = row.get(field)
            if not isinstance(values, list) or len(values) != WORLD_COUNT or any(not isinstance(v, list) or len(v) != 7 for v in values):
                errors.append(f"condition {key} has malformed {field}")

        if capacity == 64 and mode == "stable_reserve":
            _validate_constant_vector(row, "depth7_preprune_width_by_world", 128, errors, key)
            _validate_constant_vector(row, "depth7_retained_width_by_world", 64, errors, key)
            _validate_constant_vector(row, "depth7_expanded_parents_by_world", 64, errors, key)
            _validate_constant_vector(row, "productive_slots_by_world", 191, errors, key)
            _validate_constant_vector(row, "sink_slots_by_world", 32, errors, key)
        if capacity == 256 and mode in ("stable_reserve", "reshuffled_continuity"):
            _validate_constant_vector(row, "depth7_preprune_width_by_world", 128, errors, key)
            _validate_constant_vector(row, "depth7_retained_width_by_world", 128, errors, key)
            _validate_constant_vector(row, "depth7_expanded_parents_by_world", 96, errors, key)
            _validate_constant_vector(row, "productive_slots_by_world", 223, errors, key)
            _validate_constant_vector(row, "sink_slots_by_world", 0, errors, key)

    if set(index) != {
        (checkpoint, capacity, mode)
        for checkpoint in CHECKPOINTS
        for capacity, mode in CONDITIONS
    }:
        errors.append("confirmation condition matrix is incomplete")

    provided_pairs = payload.get("paired_summaries")
    if not isinstance(provided_pairs, list) or len(provided_pairs) != EXPECTED_PAIRS:
        errors.append("confirmation result must contain exactly 12 paired summaries")
        provided_pairs = []
    provided_index = {
        (row.get("checkpoint_index"), row.get("comparison")): row
        for row in provided_pairs
        if isinstance(row, dict)
    }

    deltas: dict[str, float] = {}
    lows: dict[str, float] = {}
    highs: dict[str, float] = {}
    if len(index) == EXPECTED_CONDITIONS:
        for checkpoint in CHECKPOINTS:
            for comparison, t_cap, t_mode, r_cap, r_mode in PAIR_SPECS:
                try:
                    reconstructed = _recompute_pair(
                        comparison=comparison,
                        checkpoint=checkpoint,
                        treatment=index[(checkpoint, t_cap, t_mode)],
                        reference=index[(checkpoint, r_cap, r_mode)],
                    )
                except Exception as exc:  # noqa: BLE001
                    errors.append(f"could not reconstruct c{checkpoint} {comparison}: {exc}")
                    continue
                provided = provided_index.get((checkpoint, comparison))
                if not isinstance(provided, dict):
                    errors.append(f"missing provided confirmation pair c{checkpoint} {comparison}")
                    continue
                for field in (
                    "treatment_capacity",
                    "treatment_mode",
                    "reference_capacity",
                    "reference_mode",
                    "world_count",
                    "treatment_only",
                    "reference_only",
                    "both_covered",
                    "neither_covered",
                ):
                    if provided.get(field) != reconstructed[field]:
                        errors.append(f"pair c{checkpoint} {comparison} differs in {field}")
                for field in ("coverage_delta", "bootstrap_ci_low", "bootstrap_ci_high"):
                    if not _float_equal(reconstructed[field], provided.get(field)):
                        errors.append(f"pair c{checkpoint} {comparison} differs in {field}")
                key = _metric_key(checkpoint, comparison)
                deltas[key] = float(reconstructed["coverage_delta"])
                lows[key] = float(reconstructed["bootstrap_ci_low"])
                highs[key] = float(reconstructed["bootstrap_ci_high"])

    if errors:
        return Gate3V3ConfirmationAudit(
            artifact_valid=False,
            scientific_status="INVALID_ARTIFACT",
            confirmation_outcome=None,
            primary_deltas=deltas,
            primary_ci_lows=lows,
            primary_ci_highs=highs,
            errors=tuple(errors),
        )

    outcome = _classify(lows)
    return Gate3V3ConfirmationAudit(
        artifact_valid=True,
        scientific_status="FINAL_GATE3_V3_CONFIRMATION_EVIDENCE",
        confirmation_outcome=outcome,
        primary_deltas=deltas,
        primary_ci_lows=lows,
        primary_ci_highs=highs,
        errors=(),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    audit = audit_gate3_v3_confirmation(args.result)
    args.output.write_text(json.dumps(audit.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(audit.to_dict(), indent=2, sort_keys=True))
    return 0 if audit.artifact_valid else 1


if __name__ == "__main__":
    raise SystemExit(main())
