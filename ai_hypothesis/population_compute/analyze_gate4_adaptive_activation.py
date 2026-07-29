"""Independent auditor for Gate-4 v0 adaptive-activation development artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

EXPERIMENT_VERSION = "gate4-adaptive-activation-v0"
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
MODES = ("adaptive_score", "static_generation", "adaptive_hash")
PAIR_SPECS = (
    ("adaptive_score_vs_static_generation", "adaptive_score", "static_generation"),
    ("adaptive_score_vs_adaptive_hash", "adaptive_score", "adaptive_hash"),
    ("static_generation_vs_adaptive_hash", "static_generation", "adaptive_hash"),
)
WORLD_COUNT = 256
BATCH_SIZE = 64
BOOTSTRAP_SAMPLES = 2_000
DEPTH = 8
HINT_RELIABILITY = 0.70
RESERVE_CAPACITY = 256
SCHEDULED_SLOTS = 159
TOTAL_UPDATES = 2_544
PARAMETER_COUNT = 19_649
EXPECTED_CONDITIONS = 9
EXPECTED_PAIRS = 9
STATIC_DEPTH_COUNTS = [1, 2, 4, 8, 16, 32, 64, 32]
STATIC_TERMINAL_COUNT = 64


@dataclass(frozen=True, slots=True)
class Gate4Audit:
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
        raise ValueError("Gate-4 result must be one JSON object")
    return payload


def _seed_from_parts(*parts: object) -> int:
    digest = hashlib.sha256(":".join(str(part) for part in parts).encode("ascii")).digest()
    return int.from_bytes(digest[:8], "big")


def _runtime_seed(world_index: int) -> int:
    return _seed_from_parts("gate4-adaptive-activation-development-runtime", world_index, DEPTH)


def _bootstrap_ci(
    differences: tuple[int, ...], *, checkpoint: int, comparison: str
) -> tuple[float, float]:
    rng = random.Random(_seed_from_parts("gate4-adaptive-activation-bootstrap", checkpoint, comparison))
    estimates = sorted(
        sum(differences[rng.randrange(WORLD_COUNT)] for _ in range(WORLD_COUNT)) / WORLD_COUNT
        for _ in range(BOOTSTRAP_SAMPLES)
    )
    return (
        estimates[int(math.floor(0.025 * (BOOTSTRAP_SAMPLES - 1)))],
        estimates[int(math.ceil(0.975 * (BOOTSTRAP_SAMPLES - 1)))],
    )


def _float_equal(expected: float, observed: Any, *, tolerance: float = 1e-12) -> bool:
    try:
        return math.isclose(expected, float(observed), rel_tol=0.0, abs_tol=tolerance)
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
        raise ValueError("paired Gate-4 conditions use different world indices")
    a = treatment.get("covered_by_world")
    b = reference.get("covered_by_world")
    if not isinstance(a, list) or not isinstance(b, list) or len(a) != WORLD_COUNT or len(b) != WORLD_COUNT:
        raise ValueError("paired Gate-4 coverage vectors must each contain 256 worlds")
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
        "treatment_mode": treatment["mode"],
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


def _classify(lows: dict[str, float], highs: dict[str, float]) -> str:
    primary = [_metric_key(checkpoint, "adaptive_score_vs_static_generation") for checkpoint in CHECKPOINTS]
    routing = [_metric_key(checkpoint, "adaptive_score_vs_adaptive_hash") for checkpoint in CHECKPOINTS]

    # Frozen precedence: A4 -> A2 -> A1 -> A3 -> A0 -> mixed.
    if any(highs[key] < 0.0 for key in routing):
        return "G4_A4_LEARNED_ROUTING_HARMFUL"

    primary_significant = [lows[key] > 0.0 for key in primary]
    routing_significant = [lows[key] > 0.0 for key in routing]

    if all(primary_significant) and all(routing_significant):
        return "G4_A2_ROBUST_ADAPTIVE_ACTIVATION_BENEFIT"
    if all(routing_significant) and not all(primary_significant):
        return "G4_A1_ROUTING_SIGNAL_ONLY"
    if len(set(primary_significant)) > 1:
        return "G4_A3_CHECKPOINT_SENSITIVE_ADAPTIVE_EFFECT"
    if not any(primary_significant) and not all(routing_significant):
        return "G4_A0_NO_ADAPTIVE_ALLOCATION_BENEFIT"
    return "G4_MIXED_ADAPTIVE_PATTERN"


def _validate_vector_length(
    row: dict[str, Any], field: str, expected_length: int, errors: list[str], key: tuple[Any, ...]
) -> list[Any]:
    value = row.get(field)
    if not isinstance(value, list) or len(value) != expected_length:
        errors.append(f"condition {key} has invalid {field}")
        return []
    return value


def audit_gate4_adaptive_activation(path: Path) -> Gate4Audit:
    errors: list[str] = []
    try:
        payload = _load(path)
    except Exception as exc:  # noqa: BLE001
        return Gate4Audit(False, "INVALID_ARTIFACT", None, {}, {}, {}, (str(exc),))

    if payload.get("experiment_version") != EXPERIMENT_VERSION:
        errors.append("unexpected experiment version")
    if payload.get("scientific_status") != "DEVELOPMENT_ONLY_NO_GATE_VERDICT":
        errors.append("unexpected scientific status")
    if payload.get("scientific_decision") != "DEVELOPMENT_ONLY_NOT_ASSIGNED":
        errors.append("development artifact attempted to assign a Gate verdict")
    if payload.get("confirmation_opened") is not False:
        errors.append("Gate-4 confirmation must remain closed")
    if payload.get("training_performed") is not False:
        errors.append("Gate-4 v0 must perform no training")
    if payload.get("world_count") != WORLD_COUNT:
        errors.append("world count differs from frozen 256")
    if payload.get("evaluation_batch_size") != BATCH_SIZE:
        errors.append("evaluation batch size differs from frozen 64")
    if payload.get("bootstrap_samples") != BOOTSTRAP_SAMPLES:
        errors.append("bootstrap sample count differs from frozen 2000")
    if payload.get("depth") != DEPTH:
        errors.append("depth differs from frozen 8")
    if not _float_equal(HINT_RELIABILITY, payload.get("hint_reliability")):
        errors.append("hint reliability differs from frozen 0.70")
    if payload.get("reserve_capacity") != RESERVE_CAPACITY:
        errors.append("reserve capacity differs from frozen L256")
    if payload.get("scheduled_slots") != SCHEDULED_SLOTS:
        errors.append("scheduled slot budget differs from frozen 159")
    if payload.get("active_child_lanes") != 2 or payload.get("recurrent_updates_per_child") != 8:
        errors.append("active child-lane/update identity differs from frozen values")
    if payload.get("total_learned_updates_per_world") != TOTAL_UPDATES:
        errors.append("top-level learned-work total differs from 2544")

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
        errors.append("result must contain exactly nine conditions")
        conditions = []

    index: dict[tuple[int, str], dict[str, Any]] = {}
    expected_world_indices = list(range(WORLD_COUNT))
    expected_runtime_seeds = [_runtime_seed(world_index) for world_index in range(WORLD_COUNT)]

    for row in conditions:
        if not isinstance(row, dict):
            errors.append("condition is not an object")
            continue
        key = (row.get("checkpoint_index"), row.get("mode"))
        if key in index:
            errors.append(f"duplicate condition {key}")
            continue
        index[key] = row
        checkpoint, mode = key
        if checkpoint not in CHECKPOINTS or mode not in MODES:
            errors.append(f"unexpected condition {key}")
            continue
        if row.get("world_count") != WORLD_COUNT or row.get("world_indices") != expected_world_indices:
            errors.append(f"condition {key} does not use frozen world indices")
        if row.get("runtime_seeds") != expected_runtime_seeds:
            errors.append(f"condition {key} runtime seeds differ from frozen Gate-4 namespace")
        if row.get("reserve_capacity") != RESERVE_CAPACITY:
            errors.append(f"condition {key} does not keep L256 fixed")
        if row.get("total_learned_updates_per_world") != TOTAL_UPDATES:
            errors.append(f"condition {key} violates frozen learned-work total")
        if row.get("learned_parameter_count") != PARAMETER_COUNT:
            errors.append(f"condition {key} parameter count differs from 19649")
        if checkpoint in CHECKPOINTS and row.get("parameter_fingerprint") != CHECKPOINTS[checkpoint]["fingerprint"]:
            errors.append(f"condition {key} parameter fingerprint mismatch")

        covered = _validate_vector_length(row, "covered_by_world", WORLD_COUNT, errors, key)
        if covered:
            rate = sum(int(bool(value)) for value in covered) / WORLD_COUNT
            if not _float_equal(rate, row.get("coverage_rate")):
                errors.append(f"condition {key} coverage rate differs from raw vector")

        productive = _validate_vector_length(row, "productive_slots_by_world", WORLD_COUNT, errors, key)
        sink = _validate_vector_length(row, "sink_slots_by_world", WORLD_COUNT, errors, key)
        max_population = _validate_vector_length(
            row, "max_live_nonterminal_population_by_world", WORLD_COUNT, errors, key
        )
        mean_population = _validate_vector_length(
            row, "mean_live_nonterminal_population_by_world", WORLD_COUNT, errors, key
        )
        distinct_depths = _validate_vector_length(
            row, "distinct_parent_depths_activated_by_world", WORLD_COUNT, errors, key
        )
        terminal_counts = _validate_vector_length(
            row, "generated_terminal_count_by_world", WORLD_COUNT, errors, key
        )
        unique_terminal_counts = _validate_vector_length(
            row, "unique_generated_terminal_count_by_world", WORLD_COUNT, errors, key
        )

        if productive and sink:
            for world_index, (productive_value, sink_value) in enumerate(zip(productive, sink, strict=True)):
                if int(productive_value) + int(sink_value) != SCHEDULED_SLOTS:
                    errors.append(f"condition {key} world {world_index} violates 159-slot accounting")

        population_rows = _validate_vector_length(
            row, "live_nonterminal_population_by_slot_by_world", WORLD_COUNT, errors, key
        )
        depth_count_rows = _validate_vector_length(
            row, "productive_activations_by_parent_depth_by_world", WORLD_COUNT, errors, key
        )
        activated_rows = _validate_vector_length(
            row, "activated_parent_depth_by_slot_by_world", WORLD_COUNT, errors, key
        )
        terminal_slot_rows = _validate_vector_length(
            row, "terminal_generation_slot_indices_by_world", WORLD_COUNT, errors, key
        )

        for world_index in range(WORLD_COUNT):
            if world_index >= len(population_rows) or world_index >= len(depth_count_rows) or world_index >= len(activated_rows):
                break
            populations = population_rows[world_index]
            depth_counts = depth_count_rows[world_index]
            activated = activated_rows[world_index]
            terminal_slots = terminal_slot_rows[world_index] if world_index < len(terminal_slot_rows) else []
            if not isinstance(populations, list) or len(populations) != SCHEDULED_SLOTS:
                errors.append(f"condition {key} world {world_index} has invalid live-population trace")
                continue
            if any(not isinstance(value, int) or value < 0 or value > RESERVE_CAPACITY for value in populations):
                errors.append(f"condition {key} world {world_index} violates L256 population bound")
            observed_max = max(populations, default=0)
            observed_mean = sum(populations) / SCHEDULED_SLOTS
            if world_index < len(max_population) and int(max_population[world_index]) != observed_max:
                errors.append(f"condition {key} world {world_index} max-population summary mismatch")
            if world_index < len(mean_population) and not _float_equal(
                observed_mean, mean_population[world_index], tolerance=1e-10
            ):
                errors.append(f"condition {key} world {world_index} mean-population summary mismatch")

            if not isinstance(depth_counts, list) or len(depth_counts) != DEPTH or any(
                not isinstance(value, int) or value < 0 for value in depth_counts
            ):
                errors.append(f"condition {key} world {world_index} has invalid productive-by-depth vector")
                continue
            productive_value = int(productive[world_index]) if world_index < len(productive) else -1
            if sum(depth_counts) != productive_value:
                errors.append(f"condition {key} world {world_index} productive-by-depth sum mismatch")
            observed_distinct = sum(int(value > 0) for value in depth_counts)
            if world_index < len(distinct_depths) and int(distinct_depths[world_index]) != observed_distinct:
                errors.append(f"condition {key} world {world_index} distinct-depth summary mismatch")

            if not isinstance(activated, list) or len(activated) != SCHEDULED_SLOTS or any(
                not isinstance(value, int) or value < -1 or value >= DEPTH for value in activated
            ):
                errors.append(f"condition {key} world {world_index} has invalid activation-depth trace")
            else:
                sink_value = int(sink[world_index]) if world_index < len(sink) else -1
                if activated.count(-1) != sink_value:
                    errors.append(f"condition {key} world {world_index} sink-depth count mismatch")

            if not isinstance(terminal_slots, list) or any(
                not isinstance(value, int) or not 0 <= value < SCHEDULED_SLOTS for value in terminal_slots
            ):
                errors.append(f"condition {key} world {world_index} has invalid terminal-slot trace")
            else:
                terminal_count = int(terminal_counts[world_index]) if world_index < len(terminal_counts) else -1
                if len(terminal_slots) != terminal_count:
                    errors.append(f"condition {key} world {world_index} terminal-slot count mismatch")
            if world_index < len(terminal_counts) and world_index < len(unique_terminal_counts):
                if not 0 <= int(unique_terminal_counts[world_index]) <= int(terminal_counts[world_index]):
                    errors.append(f"condition {key} world {world_index} unique-terminal count is impossible")

            if mode == "static_generation":
                if productive_value != SCHEDULED_SLOTS or int(sink[world_index]) != 0:
                    errors.append(f"static condition {key} world {world_index} did not use all 159 productive slots")
                if depth_counts != STATIC_DEPTH_COUNTS:
                    errors.append(f"static condition {key} world {world_index} changed frozen depth schedule")
                if int(terminal_counts[world_index]) != STATIC_TERMINAL_COUNT:
                    errors.append(f"static condition {key} world {world_index} did not generate 64 terminals")
                if int(unique_terminal_counts[world_index]) != STATIC_TERMINAL_COUNT:
                    errors.append(f"static condition {key} world {world_index} generated duplicate terminals")

    expected_condition_keys = {(checkpoint, mode) for checkpoint in CHECKPOINTS for mode in MODES}
    if set(index) != expected_condition_keys:
        errors.append("condition matrix is incomplete")

    paired_rows = payload.get("paired_summaries")
    if not isinstance(paired_rows, list) or len(paired_rows) != EXPECTED_PAIRS:
        errors.append("result must contain exactly nine paired summaries")
        paired_rows = []
    observed_pairs: dict[tuple[str, int], dict[str, Any]] = {}
    for row in paired_rows:
        if not isinstance(row, dict):
            errors.append("paired summary is not an object")
            continue
        key = (str(row.get("comparison")), int(row.get("checkpoint_index", -1)))
        if key in observed_pairs:
            errors.append(f"duplicate paired summary {key}")
            continue
        observed_pairs[key] = row

    deltas: dict[str, float] = {}
    lows: dict[str, float] = {}
    highs: dict[str, float] = {}
    for checkpoint in CHECKPOINTS:
        for comparison, treatment_mode, reference_mode in PAIR_SPECS:
            treatment = index.get((checkpoint, treatment_mode))
            reference = index.get((checkpoint, reference_mode))
            if treatment is None or reference is None:
                continue
            try:
                recomputed = _recompute_pair(
                    comparison=comparison,
                    checkpoint=checkpoint,
                    treatment=treatment,
                    reference=reference,
                )
            except Exception as exc:  # noqa: BLE001
                errors.append(f"could not reconstruct pair c{checkpoint} {comparison}: {exc}")
                continue
            observed = observed_pairs.get((comparison, checkpoint))
            if observed is None:
                errors.append(f"missing paired summary c{checkpoint} {comparison}")
                continue
            for field in (
                "comparison",
                "checkpoint_index",
                "treatment_mode",
                "reference_mode",
                "world_count",
                "treatment_only",
                "reference_only",
                "both_covered",
                "neither_covered",
            ):
                if observed.get(field) != recomputed[field]:
                    errors.append(f"paired summary c{checkpoint} {comparison} differs in {field}")
            for field in ("coverage_delta", "bootstrap_ci_low", "bootstrap_ci_high"):
                if not _float_equal(recomputed[field], observed.get(field)):
                    errors.append(f"paired summary c{checkpoint} {comparison} differs in {field}")
            metric = _metric_key(checkpoint, comparison)
            deltas[metric] = float(recomputed["coverage_delta"])
            lows[metric] = float(recomputed["bootstrap_ci_low"])
            highs[metric] = float(recomputed["bootstrap_ci_high"])

    expected_pair_keys = {(comparison, checkpoint) for checkpoint in CHECKPOINTS for comparison, _, _ in PAIR_SPECS}
    if set(observed_pairs) != expected_pair_keys:
        errors.append("paired-summary matrix is incomplete")

    outcome = None if errors else _classify(lows, highs)
    return Gate4Audit(
        artifact_valid=not errors,
        scientific_status=(
            "DEVELOPMENT_ONLY_NO_GATE_VERDICT" if not errors else "INVALID_ARTIFACT"
        ),
        directional_outcome=outcome,
        primary_deltas=deltas,
        primary_ci_lows=lows,
        primary_ci_highs=highs,
        errors=tuple(errors),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    audit = audit_gate4_adaptive_activation(args.result)
    args.output.write_text(json.dumps(audit.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(audit.to_dict(), indent=2, sort_keys=True), flush=True)
    return 0 if audit.artifact_valid else 1


if __name__ == "__main__":
    raise SystemExit(main())
