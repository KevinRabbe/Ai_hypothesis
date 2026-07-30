"""Independent auditor for the Gate-7 routing-bandwidth frontier confirmation artifact."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

EXPERIMENT_VERSION = "gate7-high-scale-routing-bandwidth-confirmation-v0"
SCIENTIFIC_STATUS = "FRESH_HIGH_SCALE_ROUTING_BANDWIDTH_CONFIRMATION_EVIDENCE"
PROTOCOL_HEAD = "b0f0cfca736186b9400f82a7539a54f888dc59e5"
SCREENING_RESULT_HEAD = "07b6397f2a9d4f71ed789d6c7011e12b4cbf90e0"
SCREENING_RESULT_SHA256 = "d76c8b0753a518b4c61b3ff42c1f3e85902e2e492342f23fa6706459ee13a9b5"
SCREENING_AUDIT_SHA256 = "7352621ef5c5199cba98070e2f2511674bd2f4aba8b20b48c0ec87436c5204d5"
CHECKPOINTS = {
    0: {
        "sha256": "be1b18e22a1f713b48de8934aca1c0302f1255342684158c6ae31a7c3618a719",
        "fingerprint": "0be5c26d1056da1bf12a53be5ba5e6d1cadb7815eb117e2a77db269391c1c5aa",
        "training_seed": 0,
    },
    1: {
        "sha256": "a9b712736f440168ccd86ee070b8d301be0eb834b9d17418d40c0175207cadfb",
        "fingerprint": "b9685382992cb2f94454b6faa3675d458f236e7b0c7d8399bea256bedcb02e46",
        "training_seed": 1,
    },
    2: {
        "sha256": "cd7869511d07679cb1ad430743cad031299a575413e8a853ef9116d545c7480a",
        "fingerprint": "1bca7012c7350c3b3fe8c9790a65c36eea8fcd8bef3e123034817ff78673a7bb",
        "training_seed": 2,
    },
}
CHECKPOINT_INDICES = (0, 1, 2)
TRAINING_GIT_HEAD = "07307650b2bbbfaa09b80e40caa4419ecdda2947"
TRANSITION_VERSION = "gate7-scale-neutral-scorer-transition-v0"
POPULATIONS = (4096, 8192)
ANCHOR_POPULATION = 4096
FRONTIER_POPULATION = 8192
K_LADDER = (16, 32, 64, 128, 256, 512)
ANCHOR_K = 512
WORLD_COUNT = 512
BATCH_SIZE = 64
BATCH_COUNT = 8
BOOTSTRAP_SAMPLES = 10_000
HINT_RELIABILITY = 0.70
NONINFERIORITY_MARGIN = 0.05
STAGE_B_SLOTS = 128
PARAMETER_COUNT = 19_649
GLOBAL_SCORE = "global_score"
GLOBAL_HASH = "global_hash"

FRONTIER_CONFIRMED = "G7_ROUTING_BANDWIDTH_FRONTIER_CONFIRMED"
FRONTIER_NOT_CONFIRMED = "G7_ROUTING_BANDWIDTH_FRONTIER_NOT_CONFIRMED"
ANCHOR_REFERENCE_NOT_REPLICATED = "G7_CONFIRMATION_ANCHOR_REFERENCE_NOT_REPLICATED"
ANCHOR_K_NOT_REPLICATED = "G7_CONFIRMATION_ANCHOR_K512_NOT_REPLICATED"
FRONTIER_REFERENCE_NOT_REPLICATED = "G7_CONFIRMATION_N8192_REFERENCE_NOT_REPLICATED"
VALID_OUTCOMES = {
    FRONTIER_CONFIRMED,
    FRONTIER_NOT_CONFIRMED,
    ANCHOR_REFERENCE_NOT_REPLICATED,
    ANCHOR_K_NOT_REPLICATED,
    FRONTIER_REFERENCE_NOT_REPLICATED,
}


@dataclass(frozen=True, slots=True)
class Gate7ConfirmationAudit:
    artifact_valid: bool
    scientific_status: str
    confirmation_outcome: str | None
    anchor_k512_passed: bool | None
    passing_k_at_n8192: tuple[int, ...]
    populations_audited: tuple[int, ...]
    errors: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _load(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise ValueError("Gate-7 confirmation result must be one JSON object")
    return payload


def _seed_from_parts(*parts: object) -> int:
    digest = hashlib.sha256(":".join(str(part) for part in parts).encode("ascii")).digest()
    return int.from_bytes(digest[:8], "big") & ((1 << 63) - 1)


def _runtime_seed(population: int, world_index: int) -> int:
    return _seed_from_parts(
        "gate7-high-scale-routing-bandwidth-confirmation-runtime-v0",
        population,
        world_index,
        population.bit_length(),
    )


def _float_equal(expected: float, observed: Any, *, tolerance: float = 1e-12) -> bool:
    try:
        return math.isclose(expected, float(observed), rel_tol=0.0, abs_tol=tolerance)
    except (TypeError, ValueError):
        return False


def _score_condition(k: int) -> str:
    return f"bounded_score_k{k}"


def _hash_condition(k: int) -> str:
    return f"bounded_hash_k{k}"


def _planned_k(population: int) -> tuple[int, ...]:
    if population == ANCHOR_POPULATION:
        return (ANCHOR_K,)
    if population == FRONTIER_POPULATION:
        return K_LADDER
    raise ValueError("unexpected confirmation population")


def _planned_conditions(population: int) -> tuple[str, ...]:
    conditions = [GLOBAL_SCORE, GLOBAL_HASH]
    for k in _planned_k(population):
        conditions.extend((_score_condition(k), _hash_condition(k)))
    return tuple(conditions)


def _expected_observations(population: int, condition: str) -> int:
    if condition == GLOBAL_SCORE:
        return STAGE_B_SLOTS * population - (STAGE_B_SLOTS - 1) * STAGE_B_SLOTS // 2
    if condition == GLOBAL_HASH or condition.startswith("bounded_hash_k"):
        return 0
    if condition.startswith("bounded_score_k"):
        return STAGE_B_SLOTS * int(condition.rsplit("k", 1)[1])
    raise ValueError("unknown confirmation condition")


def _bootstrap_quantiles(estimates: list[float]) -> tuple[float, float]:
    estimates.sort()
    return (
        estimates[int(math.floor(0.025 * (BOOTSTRAP_SAMPLES - 1)))],
        estimates[int(math.ceil(0.975 * (BOOTSTRAP_SAMPLES - 1)))],
    )


def _paired_bootstrap(
    differences: tuple[int, ...], *, population: int, checkpoint: int, comparison: str
) -> tuple[float, float]:
    if len(differences) != WORLD_COUNT:
        raise ValueError("confirmation paired bootstrap requires 512 values")
    rng = random.Random(
        _seed_from_parts(
            "gate7-high-scale-routing-bandwidth-confirmation-paired-bootstrap-v0",
            population,
            checkpoint,
            comparison,
        )
    )
    estimates = [
        sum(differences[rng.randrange(WORLD_COUNT)] for _ in range(WORLD_COUNT)) / WORLD_COUNT
        for _ in range(BOOTSTRAP_SAMPLES)
    ]
    return _bootstrap_quantiles(estimates)


def _stratified_bootstrap(
    differences_by_checkpoint: dict[int, tuple[int, ...]], *, population: int
) -> tuple[float, float]:
    if set(differences_by_checkpoint) != set(CHECKPOINT_INDICES):
        raise ValueError("confirmation stratified bootstrap requires all checkpoints")
    rng = random.Random(
        _seed_from_parts(
            "gate7-high-scale-routing-bandwidth-confirmation-stratified-bootstrap-v0",
            population,
            "global_score_vs_global_hash",
        )
    )
    estimates: list[float] = []
    for _ in range(BOOTSTRAP_SAMPLES):
        strata = []
        for checkpoint in CHECKPOINT_INDICES:
            vector = differences_by_checkpoint[checkpoint]
            strata.append(
                sum(vector[rng.randrange(WORLD_COUNT)] for _ in range(WORLD_COUNT)) / WORLD_COUNT
            )
        estimates.append(sum(strata) / len(strata))
    return _bootstrap_quantiles(estimates)


def _condition_k(condition: str) -> int | None:
    if condition in (GLOBAL_SCORE, GLOBAL_HASH):
        return None
    if condition.startswith("bounded_score_k") or condition.startswith("bounded_hash_k"):
        return int(condition.rsplit("k", 1)[1])
    raise ValueError("unknown condition")


def _validate_condition(
    row: dict[str, Any],
    *,
    population: int,
    expected_runtime_seeds: list[int],
    errors: list[str],
) -> tuple[int, str] | None:
    checkpoint = row.get("checkpoint_index")
    condition = row.get("condition")
    label = f"N{population}/C{checkpoint}/{condition}"
    if checkpoint not in CHECKPOINT_INDICES or not isinstance(condition, str):
        errors.append(f"invalid condition identity {label}")
        return None
    if condition not in _planned_conditions(population):
        errors.append(f"unexpected condition {label}")
        return None
    expected_k = _condition_k(condition)
    if row.get("k") != expected_k:
        errors.append(f"{label} K identity mismatch")
    if row.get("population") != population:
        errors.append(f"{label} population mismatch")
    if row.get("world_indices") != list(range(WORLD_COUNT)):
        errors.append(f"{label} world indices differ from 0..511")
    if row.get("runtime_seeds") != expected_runtime_seeds:
        errors.append(f"{label} runtime seeds differ from the frozen namespace")
    covered = row.get("covered_by_world")
    if not isinstance(covered, list) or len(covered) != WORLD_COUNT or any(type(value) is not bool for value in covered):
        errors.append(f"{label} must contain exactly 512 Boolean coverage values")
    else:
        expected_rate = sum(int(value) for value in covered) / WORLD_COUNT
        if not _float_equal(expected_rate, row.get("coverage_rate")):
            errors.append(f"{label} coverage rate does not match its vector")
    observations = row.get("score_observations_per_world")
    expected_observation = _expected_observations(population, condition)
    if (
        not isinstance(observations, list)
        or len(observations) != WORLD_COUNT
        or any(value != expected_observation for value in observations)
    ):
        errors.append(f"{label} score-observation accounting changed")
    if row.get("logical_stage_a_parent_slots") != population - 1:
        errors.append(f"{label} Stage-A work identity changed")
    if row.get("logical_stage_b_parent_slots") != STAGE_B_SLOTS:
        errors.append(f"{label} Stage-B work identity changed")
    if row.get("logical_learned_updates_per_world") != (population - 1 + STAGE_B_SLOTS) * 16:
        errors.append(f"{label} learned-update identity changed")
    if row.get("learned_parameter_count") != PARAMETER_COUNT:
        errors.append(f"{label} parameter count changed")
    if row.get("parameter_fingerprint") != CHECKPOINTS[checkpoint]["fingerprint"]:
        errors.append(f"{label} parameter fingerprint mismatch")
    if row.get("batch_count") != BATCH_COUNT:
        errors.append(f"{label} must aggregate exactly eight B64 batches")
    for telemetry in (
        "wall_seconds",
        "peak_allocated_bytes",
        "selected_frontier_index_checksum",
        "terminal_score_checksum",
    ):
        if telemetry not in row:
            errors.append(f"{label} missing telemetry {telemetry}")
    return checkpoint, condition


def _expected_pair(
    *,
    comparison: str,
    treatment: dict[str, Any],
    reference: dict[str, Any],
    population: int,
    checkpoint: int,
) -> tuple[float, float, float]:
    differences = tuple(
        int(left) - int(right)
        for left, right in zip(
            treatment["covered_by_world"],
            reference["covered_by_world"],
            strict=True,
        )
    )
    low, high = _paired_bootstrap(
        differences,
        population=population,
        checkpoint=checkpoint,
        comparison=comparison,
    )
    return sum(differences) / WORLD_COUNT, low, high


def _k_passes(k: int, lows: dict[str, float]) -> bool:
    return all(
        lows[f"c{checkpoint}_k{k}_score_vs_hash"] > 0.0
        and lows[f"c{checkpoint}_k{k}_score_vs_global"] > -NONINFERIORITY_MARGIN
        for checkpoint in CHECKPOINT_INDICES
    )


def _classify(
    *,
    anchor_reference_viable: bool,
    anchor_lows: dict[str, float],
    frontier_reference_viable: bool,
    frontier_lows_by_k: dict[int, dict[str, float]],
) -> tuple[str, bool, tuple[int, ...]]:
    anchor_pass = _k_passes(ANCHOR_K, anchor_lows)
    passing = tuple(k for k in K_LADDER if _k_passes(k, frontier_lows_by_k[k]))
    if not anchor_reference_viable:
        return ANCHOR_REFERENCE_NOT_REPLICATED, anchor_pass, passing
    if not anchor_pass:
        return ANCHOR_K_NOT_REPLICATED, anchor_pass, passing
    if not frontier_reference_viable:
        return FRONTIER_REFERENCE_NOT_REPLICATED, anchor_pass, passing
    if passing:
        return FRONTIER_NOT_CONFIRMED, anchor_pass, passing
    return FRONTIER_CONFIRMED, anchor_pass, passing


def audit_gate7_high_scale_routing_bandwidth_confirmation(
    path: Path,
) -> Gate7ConfirmationAudit:
    errors: list[str] = []
    try:
        payload = _load(path)
    except Exception as exc:  # noqa: BLE001
        return Gate7ConfirmationAudit(
            False,
            "INVALID_ARTIFACT",
            None,
            None,
            (),
            (),
            (str(exc),),
        )

    if payload.get("experiment_version") != EXPERIMENT_VERSION:
        errors.append("unexpected confirmation version")
    if payload.get("scientific_status") != SCIENTIFIC_STATUS:
        errors.append("unexpected scientific status")
    if payload.get("execution_admitted") is not True:
        errors.append("confirmation execution was not admitted")
    if payload.get("confirmation_opened") is not True:
        errors.append("confirmation artifact must record confirmation_opened=true")
    if payload.get("second_confirmation_opened") is not False:
        errors.append("second confirmation must remain closed")
    if payload.get("training_performed") is not False:
        errors.append("confirmation must perform no training")
    if payload.get("checkpoint_selection_performed") is not False:
        errors.append("confirmation must perform no checkpoint selection")
    if payload.get("confirmation_protocol_head") != PROTOCOL_HEAD:
        errors.append("confirmation protocol head mismatch")
    if payload.get("screening_result_head") != SCREENING_RESULT_HEAD:
        errors.append("screening result head mismatch")
    if payload.get("screening_result_sha256") != SCREENING_RESULT_SHA256:
        errors.append("screening result SHA mismatch")
    if payload.get("screening_audit_sha256") != SCREENING_AUDIT_SHA256:
        errors.append("screening audit SHA mismatch")
    if payload.get("world_count") != WORLD_COUNT:
        errors.append("confirmation world count changed")
    if payload.get("evaluation_batch_size") != BATCH_SIZE:
        errors.append("confirmation physical batch changed")
    if payload.get("bootstrap_samples") != BOOTSTRAP_SAMPLES:
        errors.append("confirmation bootstrap count changed")
    if payload.get("stage_b_parent_slots") != STAGE_B_SLOTS:
        errors.append("confirmation Stage-B slots changed")
    if not _float_equal(HINT_RELIABILITY, payload.get("hint_reliability")):
        errors.append("confirmation hint reliability changed")
    if not _float_equal(NONINFERIORITY_MARGIN, payload.get("noninferiority_margin")):
        errors.append("confirmation non-inferiority margin changed")
    if payload.get("populations") != list(POPULATIONS):
        errors.append("confirmation populations changed")
    if payload.get("k_ladder") != list(K_LADDER):
        errors.append("confirmation K ladder changed")
    for flag in ("compiler_enabled", "cuda_graphs_enabled", "mixed_precision_enabled"):
        if payload.get(flag) is not False:
            errors.append(f"{flag} must remain false")

    checkpoint_rows = payload.get("transition_checkpoints")
    if not isinstance(checkpoint_rows, list) or len(checkpoint_rows) != 3:
        errors.append("artifact must bind exactly three transition checkpoints")
        checkpoint_rows = []
    seen_checkpoints: set[int] = set()
    for row in checkpoint_rows:
        if not isinstance(row, dict) or row.get("checkpoint_index") not in CHECKPOINTS:
            errors.append("invalid transition checkpoint identity")
            continue
        checkpoint = int(row["checkpoint_index"])
        if checkpoint in seen_checkpoints:
            errors.append(f"duplicate checkpoint {checkpoint}")
            continue
        seen_checkpoints.add(checkpoint)
        expected = CHECKPOINTS[checkpoint]
        if str(row.get("checkpoint_sha256", "")).lower() != expected["sha256"]:
            errors.append(f"checkpoint {checkpoint} SHA mismatch")
        if row.get("parameter_fingerprint") != expected["fingerprint"]:
            errors.append(f"checkpoint {checkpoint} fingerprint mismatch")
        if row.get("training_seed") != expected["training_seed"]:
            errors.append(f"checkpoint {checkpoint} training seed mismatch")
        if row.get("learned_parameter_count") != PARAMETER_COUNT:
            errors.append(f"checkpoint {checkpoint} parameter count mismatch")
        if row.get("transition_version") != TRANSITION_VERSION:
            errors.append(f"checkpoint {checkpoint} transition version mismatch")
        if row.get("training_git_head") != TRAINING_GIT_HEAD:
            errors.append(f"checkpoint {checkpoint} training head mismatch")
    if seen_checkpoints != set(CHECKPOINT_INDICES):
        errors.append("checkpoint identity set is incomplete")

    tiers = payload.get("tiers")
    if not isinstance(tiers, list) or [row.get("population") for row in tiers if isinstance(row, dict)] != list(POPULATIONS):
        errors.append("artifact must contain exactly ordered N4096/N8192 tiers")
        tiers = []

    reference_viable_by_population: dict[int, bool] = {}
    lows_by_population: dict[int, dict[int, dict[str, float]]] = {}
    populations_audited: list[int] = []

    for tier in tiers:
        if not isinstance(tier, dict):
            errors.append("tier is not an object")
            continue
        population = tier.get("population")
        if population not in POPULATIONS:
            errors.append(f"unexpected tier population {population}")
            continue
        populations_audited.append(population)
        expected_runtime_seeds = [_runtime_seed(population, index) for index in range(WORLD_COUNT)]
        if tier.get("world_indices") != list(range(WORLD_COUNT)):
            errors.append(f"N{population} tier world indices changed")
        if tier.get("runtime_seeds") != expected_runtime_seeds:
            errors.append(f"N{population} tier runtime seeds changed")
        if tier.get("world_count") != WORLD_COUNT:
            errors.append(f"N{population} tier world count changed")
        if tier.get("evaluation_batch_size") != BATCH_SIZE:
            errors.append(f"N{population} tier physical batch changed")
        if tier.get("physical_batch_count") != BATCH_COUNT:
            errors.append(f"N{population} tier batch count changed")
        if tier.get("conditions_planned") != list(_planned_conditions(population)):
            errors.append(f"N{population} planned condition matrix changed")
        if tier.get("k_values_planned") != list(_planned_k(population)):
            errors.append(f"N{population} planned K matrix changed")
        if tier.get("logical_stage_a_parent_slots") != population - 1:
            errors.append(f"N{population} tier Stage-A work changed")
        if tier.get("logical_stage_b_parent_slots") != STAGE_B_SLOTS:
            errors.append(f"N{population} tier Stage-B work changed")
        if tier.get("logical_learned_updates_per_world") != (population - 1 + STAGE_B_SLOTS) * 16:
            errors.append(f"N{population} tier learned work changed")

        frontier_builds = tier.get("frontier_builds")
        expected_frontier_builds = len(CHECKPOINT_INDICES) * BATCH_COUNT
        if not isinstance(frontier_builds, list) or len(frontier_builds) != expected_frontier_builds:
            errors.append(f"N{population} must contain exactly {expected_frontier_builds} frontier builds")

        conditions = tier.get("conditions")
        expected_condition_count = len(CHECKPOINT_INDICES) * len(_planned_conditions(population))
        if not isinstance(conditions, list) or len(conditions) != expected_condition_count:
            errors.append(f"N{population} condition count changed")
            conditions = []
        condition_index: dict[tuple[int, str], dict[str, Any]] = {}
        for row in conditions:
            if not isinstance(row, dict):
                errors.append(f"N{population} condition is not an object")
                continue
            identity = _validate_condition(
                row,
                population=population,
                expected_runtime_seeds=expected_runtime_seeds,
                errors=errors,
            )
            if identity is None:
                continue
            if identity in condition_index:
                errors.append(f"duplicate N{population} condition {identity}")
            condition_index[identity] = row
        expected_keys = {
            (checkpoint, condition)
            for checkpoint in CHECKPOINT_INDICES
            for condition in _planned_conditions(population)
        }
        if set(condition_index) != expected_keys:
            errors.append(f"N{population} condition identity set is incomplete")

        pair_rows = tier.get("paired_summaries")
        expected_pair_count = len(CHECKPOINT_INDICES) * (1 + 2 * len(_planned_k(population)))
        if not isinstance(pair_rows, list) or len(pair_rows) != expected_pair_count:
            errors.append(f"N{population} paired-summary count changed")
            pair_rows = []
        pair_index: dict[str, dict[str, Any]] = {}
        for row in pair_rows:
            if not isinstance(row, dict) or not isinstance(row.get("comparison"), str):
                errors.append(f"N{population} invalid paired summary")
                continue
            comparison = row["comparison"]
            if comparison in pair_index:
                errors.append(f"N{population} duplicate paired summary {comparison}")
            pair_index[comparison] = row

        differences_by_checkpoint: dict[int, tuple[int, ...]] = {}
        points: dict[int, float] = {}
        for checkpoint in CHECKPOINT_INDICES:
            score = condition_index.get((checkpoint, GLOBAL_SCORE))
            hash_control = condition_index.get((checkpoint, GLOBAL_HASH))
            if score is None or hash_control is None:
                continue
            comparison = f"c{checkpoint}_global_score_vs_global_hash"
            pair = pair_index.get(comparison)
            if pair is None:
                errors.append(f"missing N{population} pair {comparison}")
                continue
            expected_delta, expected_low, expected_high = _expected_pair(
                comparison=comparison,
                treatment=score,
                reference=hash_control,
                population=population,
                checkpoint=checkpoint,
            )
            for name, expected in (
                ("coverage_delta", expected_delta),
                ("bootstrap_ci_low", expected_low),
                ("bootstrap_ci_high", expected_high),
            ):
                if not _float_equal(expected, pair.get(name)):
                    errors.append(f"N{population} pair {comparison} {name} mismatch")
            vector = tuple(
                int(left) - int(right)
                for left, right in zip(
                    score["covered_by_world"],
                    hash_control["covered_by_world"],
                    strict=True,
                )
            )
            differences_by_checkpoint[checkpoint] = vector
            points[checkpoint] = sum(vector) / WORLD_COUNT

        stratified = tier.get("reference_stratified_summary")
        if not isinstance(stratified, dict) or set(differences_by_checkpoint) != set(CHECKPOINT_INDICES):
            errors.append(f"N{population} stratified reference is incomplete")
            stratified = {}
        else:
            pooled_low, pooled_high = _stratified_bootstrap(
                differences_by_checkpoint,
                population=population,
            )
            expected_pooled = sum(points.values()) / len(points)
            observed_points = stratified.get("checkpoint_point_deltas")
            expected_points_json = {str(key): value for key, value in points.items()}
            if observed_points not in (points, expected_points_json):
                errors.append(f"N{population} stratified checkpoint deltas mismatch")
            if not _float_equal(expected_pooled, stratified.get("pooled_delta")):
                errors.append(f"N{population} stratified pooled delta mismatch")
            if not _float_equal(pooled_low, stratified.get("bootstrap_ci_low")):
                errors.append(f"N{population} stratified CI low mismatch")
            if not _float_equal(pooled_high, stratified.get("bootstrap_ci_high")):
                errors.append(f"N{population} stratified CI high mismatch")
            viable = all(points[index] > 0.0 for index in CHECKPOINT_INDICES) and pooled_low > 0.0
            reference_viable_by_population[population] = viable
            if tier.get("reference_viable") is not viable:
                errors.append(f"N{population} reference viability mismatch")

        k_lows: dict[int, dict[str, float]] = {}
        for k in _planned_k(population):
            lows: dict[str, float] = {}
            for checkpoint in CHECKPOINT_INDICES:
                score = condition_index.get((checkpoint, _score_condition(k)))
                hash_control = condition_index.get((checkpoint, _hash_condition(k)))
                global_score = condition_index.get((checkpoint, GLOBAL_SCORE))
                if score is None or hash_control is None or global_score is None:
                    continue
                for suffix, reference in (("score_vs_hash", hash_control), ("score_vs_global", global_score)):
                    comparison = f"c{checkpoint}_k{k}_{suffix}"
                    pair = pair_index.get(comparison)
                    if pair is None:
                        errors.append(f"missing N{population} pair {comparison}")
                        continue
                    expected_delta, expected_low, expected_high = _expected_pair(
                        comparison=comparison,
                        treatment=score,
                        reference=reference,
                        population=population,
                        checkpoint=checkpoint,
                    )
                    for name, expected in (
                        ("coverage_delta", expected_delta),
                        ("bootstrap_ci_low", expected_low),
                        ("bootstrap_ci_high", expected_high),
                    ):
                        if not _float_equal(expected, pair.get(name)):
                            errors.append(f"N{population} pair {comparison} {name} mismatch")
                    lows[comparison] = expected_low
            k_lows[k] = lows
        lows_by_population[population] = k_lows
        expected_lows_json = {str(k): values for k, values in k_lows.items()}
        if tier.get("primary_ci_lows_by_k") != expected_lows_json:
            errors.append(f"N{population} primary CI-low map mismatch")
        expected_passing = [k for k in _planned_k(population) if _k_passes(k, k_lows[k])]
        if tier.get("passing_k") != expected_passing:
            errors.append(f"N{population} passing-K set mismatch")

    anchor_reference = reference_viable_by_population.get(ANCHOR_POPULATION)
    frontier_reference = reference_viable_by_population.get(FRONTIER_POPULATION)
    anchor_lows = lows_by_population.get(ANCHOR_POPULATION, {}).get(ANCHOR_K)
    frontier_lows = lows_by_population.get(FRONTIER_POPULATION)
    expected_outcome: str | None = None
    expected_anchor_pass: bool | None = None
    expected_passing: tuple[int, ...] = ()
    if (
        anchor_reference is not None
        and frontier_reference is not None
        and anchor_lows is not None
        and frontier_lows is not None
        and tuple(frontier_lows) == K_LADDER
    ):
        expected_outcome, expected_anchor_pass, expected_passing = _classify(
            anchor_reference_viable=anchor_reference,
            anchor_lows=anchor_lows,
            frontier_reference_viable=frontier_reference,
            frontier_lows_by_k=frontier_lows,
        )
        classification = payload.get("confirmation_classification")
        if not isinstance(classification, dict):
            errors.append("missing confirmation classification")
        else:
            if classification.get("outcome") != expected_outcome:
                errors.append("confirmation classification outcome mismatch")
            if classification.get("anchor_k512_passed") is not expected_anchor_pass:
                errors.append("confirmation anchor pass mismatch")
            if classification.get("passing_k_at_n8192") != list(expected_passing):
                errors.append("confirmation passing-K vector mismatch")
            expected_smallest = expected_passing[0] if expected_passing else None
            if classification.get("smallest_passing_k_at_n8192") != expected_smallest:
                errors.append("confirmation smallest passing K mismatch")
        if payload.get("confirmation_outcome") != expected_outcome:
            errors.append("top-level confirmation outcome mismatch")
    else:
        errors.append("could not reconstruct complete confirmation classifier inputs")

    observed_outcome = payload.get("confirmation_outcome")
    if observed_outcome not in VALID_OUTCOMES:
        errors.append("unknown confirmation outcome")

    return Gate7ConfirmationAudit(
        artifact_valid=not errors,
        scientific_status=SCIENTIFIC_STATUS if not errors else "INVALID_ARTIFACT",
        confirmation_outcome=expected_outcome if not errors else observed_outcome,
        anchor_k512_passed=expected_anchor_pass,
        passing_k_at_n8192=expected_passing,
        populations_audited=tuple(populations_audited),
        errors=tuple(errors),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    audit = audit_gate7_high_scale_routing_bandwidth_confirmation(args.result)
    args.output.write_text(
        json.dumps(audit.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(audit.to_dict(), indent=2, sort_keys=True), flush=True)
    return 0 if audit.artifact_valid else 1


if __name__ == "__main__":
    raise SystemExit(main())
