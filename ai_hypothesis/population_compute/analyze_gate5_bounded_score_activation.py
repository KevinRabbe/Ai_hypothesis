"""Independent auditor for Gate-5 v0 bounded score-visibility development artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

EXPERIMENT_VERSION = "gate5-bounded-score-activation-v0"
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
MODES = (
    "global_score",
    "bounded_score_k4",
    "bounded_score_k8",
    "bounded_score_k16",
    "bounded_score_k32",
    "bounded_hash_k16",
)
BOUNDED_K = {
    "bounded_score_k4": 4,
    "bounded_score_k8": 8,
    "bounded_score_k16": 16,
    "bounded_score_k32": 32,
    "bounded_hash_k16": 16,
}
PAIR_SPECS = (
    ("bounded_score_k4_vs_global_score", "bounded_score_k4", "global_score"),
    ("bounded_score_k8_vs_global_score", "bounded_score_k8", "global_score"),
    ("bounded_score_k16_vs_global_score", "bounded_score_k16", "global_score"),
    ("bounded_score_k32_vs_global_score", "bounded_score_k32", "global_score"),
    ("bounded_score_k16_vs_bounded_hash_k16", "bounded_score_k16", "bounded_hash_k16"),
)
WORLD_COUNT = 256
BATCH_SIZE = 64
BOOTSTRAP_SAMPLES = 2_000
DEPTH = 8
HINT_RELIABILITY = 0.70
RESERVE_CAPACITY = 256
STAGE_A_SLOTS = 63
STAGE_B_SLOTS = 96
SCHEDULED_SLOTS = 159
TOTAL_UPDATES = 2_544
STAGE_A_FRONTIER = 64
NONINFERIORITY_MARGIN = 0.05
PARAMETER_COUNT = 19_649
EXPECTED_CONDITIONS = 18
EXPECTED_PAIRS = 15


@dataclass(frozen=True, slots=True)
class Gate5Audit:
    artifact_valid: bool
    scientific_status: str
    directional_outcome: str | None
    smallest_noninferior_k: int | None
    primary_deltas: dict[str, float]
    primary_ci_lows: dict[str, float]
    primary_ci_highs: dict[str, float]
    mean_stage_b_score_observations: dict[str, float]
    errors: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _load(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise ValueError("Gate-5 result must be one JSON object")
    return payload


def _seed_from_parts(*parts: object) -> int:
    digest = hashlib.sha256(":".join(str(part) for part in parts).encode("ascii")).digest()
    return int.from_bytes(digest[:8], "big")


def _runtime_seed(world_index: int) -> int:
    return _seed_from_parts("gate5-bounded-score-activation-development-runtime", world_index, DEPTH)


def _bootstrap_ci(
    differences: tuple[int, ...], *, checkpoint: int, comparison: str
) -> tuple[float, float]:
    rng = random.Random(
        _seed_from_parts("gate5-bounded-score-activation-bootstrap", checkpoint, comparison)
    )
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
        raise ValueError("paired Gate-5 conditions use different world indices")
    a = treatment.get("covered_by_world")
    b = reference.get("covered_by_world")
    if not isinstance(a, list) or not isinstance(b, list) or len(a) != WORLD_COUNT or len(b) != WORLD_COUNT:
        raise ValueError("paired Gate-5 coverage vectors must each contain 256 worlds")
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


def classify_gate5(lows: dict[str, float], highs: dict[str, float]) -> str:
    learned = [
        _metric_key(checkpoint, "bounded_score_k16_vs_bounded_hash_k16")
        for checkpoint in CHECKPOINTS
    ]
    global_gap = [
        _metric_key(checkpoint, "bounded_score_k16_vs_global_score")
        for checkpoint in CHECKPOINTS
    ]

    if any(highs[key] < 0.0 for key in learned):
        return "G5_B4_BOUNDED_LEARNED_ROUTING_HARMFUL"

    learned_positive = [lows[key] > 0.0 for key in learned]
    noninferior = [lows[key] > -NONINFERIORITY_MARGIN for key in global_gap]

    if all(learned_positive) and all(noninferior):
        return "G5_B2_ROBUST_BOUNDED_SCORE_ACTIVATION"
    if len(set(learned_positive)) > 1 or len(set(noninferior)) > 1:
        return "G5_B3_CHECKPOINT_SENSITIVE_BOUNDED_EFFECT"
    if all(learned_positive) and not any(noninferior):
        return "G5_B1_LEARNED_SIGNAL_WITH_GLOBAL_GAP"
    if not any(learned_positive):
        return "G5_B0_BOUNDED_LEARNED_ROUTING_NOT_ESTABLISHED"
    return "G5_MIXED_BOUNDED_SCORE_PATTERN"


def _smallest_noninferior_k(lows: dict[str, float]) -> int | None:
    for k in (4, 8, 16, 32):
        comparison = f"bounded_score_k{k}_vs_global_score"
        keys = [_metric_key(checkpoint, comparison) for checkpoint in CHECKPOINTS]
        if all(key in lows and lows[key] > -NONINFERIORITY_MARGIN for key in keys):
            return k
    return None


def _vector(
    row: dict[str, Any], field: str, expected_length: int, errors: list[str], key: tuple[Any, ...]
) -> list[Any]:
    value = row.get(field)
    if not isinstance(value, list) or len(value) != expected_length:
        errors.append(f"condition {key} has invalid {field}")
        return []
    return value


def audit_gate5_bounded_score_activation(path: Path) -> Gate5Audit:
    errors: list[str] = []
    try:
        payload = _load(path)
    except Exception as exc:  # noqa: BLE001
        return Gate5Audit(False, "INVALID_ARTIFACT", None, None, {}, {}, {}, {}, (str(exc),))

    if payload.get("experiment_version") != EXPERIMENT_VERSION:
        errors.append("unexpected experiment version")
    if payload.get("scientific_status") != "DEVELOPMENT_ONLY_NO_GATE_VERDICT":
        errors.append("unexpected scientific status")
    if payload.get("scientific_decision") != "DEVELOPMENT_ONLY_NOT_ASSIGNED":
        errors.append("development artifact attempted to assign a Gate verdict")
    if payload.get("confirmation_opened") is not False:
        errors.append("Gate-5 confirmation must remain closed")
    if payload.get("training_performed") is not False:
        errors.append("Gate-5 v0 must perform no training")
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
    if payload.get("stage_a_slots") != STAGE_A_SLOTS or payload.get("stage_b_slots") != STAGE_B_SLOTS:
        errors.append("Gate-5 63/96 slot split changed")
    if payload.get("scheduled_slots") != SCHEDULED_SLOTS:
        errors.append("scheduled slot budget differs from frozen 159")
    if payload.get("active_child_lanes") != 2 or payload.get("recurrent_updates_per_child") != 8:
        errors.append("active child-lane/update identity differs from frozen values")
    if payload.get("total_learned_updates_per_world") != TOTAL_UPDATES:
        errors.append("top-level learned-work total differs from 2544")
    if not _float_equal(NONINFERIORITY_MARGIN, payload.get("noninferiority_margin")):
        errors.append("non-inferiority margin differs from frozen 0.05")

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
        errors.append("result must contain exactly eighteen conditions")
        conditions = []

    index: dict[tuple[int, str], dict[str, Any]] = {}
    expected_world_indices = list(range(WORLD_COUNT))
    expected_runtime_seeds = [_runtime_seed(world_index) for world_index in range(WORLD_COUNT)]
    mean_score_observations: dict[str, float] = {}

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
            errors.append(f"condition {key} runtime seeds differ from frozen Gate-5 namespace")
        if row.get("reserve_capacity") != RESERVE_CAPACITY:
            errors.append(f"condition {key} does not keep L256 fixed")
        if row.get("total_learned_updates_per_world") != TOTAL_UPDATES:
            errors.append(f"condition {key} violates frozen learned-work total")
        if row.get("learned_parameter_count") != PARAMETER_COUNT:
            errors.append(f"condition {key} parameter count differs from 19649")
        if checkpoint in CHECKPOINTS and row.get("parameter_fingerprint") != CHECKPOINTS[checkpoint]["fingerprint"]:
            errors.append(f"condition {key} parameter fingerprint mismatch")

        covered = _vector(row, "covered_by_world", WORLD_COUNT, errors, key)
        if covered:
            rate = sum(int(bool(value)) for value in covered) / WORLD_COUNT
            if not _float_equal(rate, row.get("coverage_rate")):
                errors.append(f"condition {key} coverage rate differs from raw vector")

        productive = _vector(row, "productive_slots_by_world", WORLD_COUNT, errors, key)
        sink = _vector(row, "sink_slots_by_world", WORLD_COUNT, errors, key)
        frontier = _vector(row, "stage_a_frontier_width_by_world", WORLD_COUNT, errors, key)
        live_rows = _vector(row, "stage_b_live_population_by_slot_by_world", WORLD_COUNT, errors, key)
        depth_rows = _vector(row, "stage_b_activated_parent_depth_by_slot_by_world", WORLD_COUNT, errors, key)
        visible_rows = _vector(row, "stage_b_visible_candidate_count_by_slot_by_world", WORLD_COUNT, errors, key)
        score_obs_rows = _vector(row, "stage_b_score_observation_count_by_slot_by_world", WORLD_COUNT, errors, key)
        total_score_obs = _vector(row, "total_stage_b_score_observations_by_world", WORLD_COUNT, errors, key)
        max_score_obs = _vector(row, "max_stage_b_score_observations_by_world", WORLD_COUNT, errors, key)
        visible_rank_rows = _vector(row, "selected_visible_score_rank_by_slot_by_world", WORLD_COUNT, errors, key)
        global_rank_rows = _vector(row, "selected_global_score_rank_by_slot_by_world", WORLD_COUNT, errors, key)
        selected_path_rows = _vector(row, "selected_parent_paths_by_slot_by_world", WORLD_COUNT, errors, key)
        terminal_count = _vector(row, "generated_terminal_count_by_world", WORLD_COUNT, errors, key)
        unique_terminal_count = _vector(row, "unique_generated_terminal_count_by_world", WORLD_COUNT, errors, key)

        if total_score_obs:
            mean_score_observations[f"c{checkpoint}_{mode}"] = sum(int(v) for v in total_score_obs) / WORLD_COUNT

        for world_index in range(WORLD_COUNT):
            if world_index >= len(productive) or world_index >= len(sink):
                break
            if int(productive[world_index]) != SCHEDULED_SLOTS or int(sink[world_index]) != 0:
                errors.append(f"condition {key} world {world_index} violates frozen productive/sink accounting")
            if world_index < len(frontier) and int(frontier[world_index]) != STAGE_A_FRONTIER:
                errors.append(f"condition {key} world {world_index} did not create the 64-state Stage-A frontier")

            rows = (live_rows, depth_rows, visible_rows, score_obs_rows, visible_rank_rows, global_rank_rows, selected_path_rows)
            if any(world_index >= len(vector) for vector in rows):
                break
            live = live_rows[world_index]
            depths = depth_rows[world_index]
            visible = visible_rows[world_index]
            score_obs = score_obs_rows[world_index]
            visible_ranks = visible_rank_rows[world_index]
            global_ranks = global_rank_rows[world_index]
            paths = selected_path_rows[world_index]
            if any(not isinstance(v, list) or len(v) != STAGE_B_SLOTS for v in (live, depths, visible, score_obs, visible_ranks, global_ranks, paths)):
                errors.append(f"condition {key} world {world_index} has incomplete Stage-B telemetry")
                continue
            for slot in range(STAGE_B_SLOTS):
                n = int(live[slot])
                vis = int(visible[slot])
                obs = int(score_obs[slot])
                if not 1 <= n <= RESERVE_CAPACITY:
                    errors.append(f"condition {key} world {world_index} slot {slot} has impossible live population")
                    break
                if mode == "global_score":
                    if vis != n or obs != n:
                        errors.append(f"global condition {key} world {world_index} slot {slot} did not observe full reserve")
                        break
                else:
                    expected_vis = min(BOUNDED_K[str(mode)], n)
                    if vis != expected_vis:
                        errors.append(f"bounded condition {key} world {world_index} slot {slot} violated K visibility")
                        break
                    expected_obs = 0 if mode == "bounded_hash_k16" else expected_vis
                    if obs != expected_obs:
                        errors.append(f"condition {key} world {world_index} slot {slot} has wrong score-observation count")
                        break
                if not 1 <= int(visible_ranks[slot]) <= vis:
                    errors.append(f"condition {key} world {world_index} slot {slot} has invalid visible score rank")
                    break
                if not 1 <= int(global_ranks[slot]) <= n:
                    errors.append(f"condition {key} world {world_index} slot {slot} has invalid global score rank")
                    break
                path = paths[slot]
                if not isinstance(path, list) or len(path) not in (6, 7) or any(bit not in (0, 1) for bit in path):
                    errors.append(f"condition {key} world {world_index} slot {slot} has invalid selected parent path")
                    break

            if world_index < len(total_score_obs) and isinstance(score_obs, list):
                observed_total = sum(int(value) for value in score_obs)
                if int(total_score_obs[world_index]) != observed_total:
                    errors.append(f"condition {key} world {world_index} total score-observation summary mismatch")
                if world_index < len(max_score_obs) and int(max_score_obs[world_index]) != max((int(v) for v in score_obs), default=0):
                    errors.append(f"condition {key} world {world_index} max score-observation summary mismatch")
            if world_index < len(terminal_count) and world_index < len(unique_terminal_count):
                if not 0 <= int(unique_terminal_count[world_index]) <= int(terminal_count[world_index]):
                    errors.append(f"condition {key} world {world_index} has impossible terminal counts")

    expected_condition_keys = {(checkpoint, mode) for checkpoint in CHECKPOINTS for mode in MODES}
    if set(index) != expected_condition_keys:
        errors.append("condition matrix is incomplete")

    # K16 score/hash must receive exactly the same visible-subset mechanics.  Their live populations
    # may diverge after different selections, so exact candidate identities need not match; the K16
    # bound and shared sampling namespace are source/protocol invariants rather than post-hoc equality.

    paired_rows = payload.get("paired_summaries")
    if not isinstance(paired_rows, list) or len(paired_rows) != EXPECTED_PAIRS:
        errors.append("result must contain exactly fifteen paired summaries")
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

    outcome = None if errors else classify_gate5(lows, highs)
    smallest = None if errors else _smallest_noninferior_k(lows)
    return Gate5Audit(
        artifact_valid=not errors,
        scientific_status=("DEVELOPMENT_ONLY_NO_GATE_VERDICT" if not errors else "INVALID_ARTIFACT"),
        directional_outcome=outcome,
        smallest_noninferior_k=smallest,
        primary_deltas=deltas,
        primary_ci_lows=lows,
        primary_ci_highs=highs,
        mean_stage_b_score_observations=mean_score_observations,
        errors=tuple(errors),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    audit = audit_gate5_bounded_score_activation(args.result)
    args.output.write_text(json.dumps(audit.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(audit.to_dict(), indent=2, sort_keys=True), flush=True)
    return 0 if audit.artifact_valid else 1


if __name__ == "__main__":
    raise SystemExit(main())
