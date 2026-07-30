"""Independent auditor for Gate-7 high-scale routing-bandwidth screening artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

EXPERIMENT_VERSION = "gate7-high-scale-routing-bandwidth-screening-v0"
SCIENTIFIC_STATUS = "FRESH_HIGH_SCALE_ROUTING_BANDWIDTH_SCREENING_EVIDENCE"
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
TRAINING_GIT_HEAD = "07307650b2bbbfaa09b80e40caa4419ecdda2947"
TRANSITION_VERSION = "gate7-scale-neutral-scorer-transition-v0"
ENGINEERING_HEAD = "5305475ea1e295c84fadbce3533f13489b10d60d"
ENGINEERING_SUMMARY_SHA256 = "e40823e3e2787151f2a63607aa3d396f18e03428b715b8864af4f549631e2953"
ENGINEERING_MANIFEST_SHA256 = "8393f9b4f11aa90aa333c3443669306675d1e9cc746e1f1dc3aa5acd1523afe4"
POPULATIONS = (1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072)
K_LADDER = (16, 32, 64, 128, 256, 512)
CHECKPOINT_INDICES = (0, 1, 2)
WORLD_COUNT = 64
BATCH_SIZE = 64
BOOTSTRAP_SAMPLES = 2_000
HINT_RELIABILITY = 0.70
NONINFERIORITY_MARGIN = 0.05
STAGE_B_SLOTS = 128
PARAMETER_COUNT = 19_649
GLOBAL_SCORE = "global_score"
GLOBAL_HASH = "global_hash"
CONTINUE = "G7_CONTINUE_TO_NEXT_POPULATION"
REFERENCE_FRONTIER = "G7_REFERENCE_FRONTIER_REACHED"
ROUTING_FRONTIER = "G7_ROUTING_BANDWIDTH_FRONTIER_REACHED"
RESOURCE_FRONTIER = "G7_RESOURCE_FRONTIER_REACHED"
CAMPAIGN_CEILING = "G7_CAMPAIGN_CEILING_REACHED"


@dataclass(frozen=True, slots=True)
class Gate7HighScaleAudit:
    artifact_valid: bool
    scientific_status: str
    campaign_outcome: str | None
    populations_completed: tuple[int, ...]
    k_required_by_population: dict[int, int | None]
    errors: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _load(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise ValueError("Gate-7 high-scale result must be one JSON object")
    return payload


def _seed_from_parts(*parts: object) -> int:
    digest = hashlib.sha256(":".join(str(part) for part in parts).encode("ascii")).digest()
    return int.from_bytes(digest[:8], "big") & ((1 << 63) - 1)


def _world_depth(population: int) -> int:
    if population <= 0 or population & (population - 1):
        raise ValueError("population must remain a positive power of two")
    return population.bit_length()


def _runtime_seed(population: int, world_index: int) -> int:
    return _seed_from_parts(
        "gate7-high-scale-routing-bandwidth-runtime-v0",
        population,
        world_index,
        _world_depth(population),
    )


def _float_equal(expected: float, observed: Any, *, tolerance: float = 1e-12) -> bool:
    try:
        return math.isclose(expected, float(observed), rel_tol=0.0, abs_tol=tolerance)
    except (TypeError, ValueError):
        return False


def _quantiles(estimates: list[float]) -> tuple[float, float]:
    estimates.sort()
    return (
        estimates[int(math.floor(0.025 * (BOOTSTRAP_SAMPLES - 1)))],
        estimates[int(math.ceil(0.975 * (BOOTSTRAP_SAMPLES - 1)))],
    )


def _paired_bootstrap(
    differences: tuple[int, ...], *, population: int, checkpoint: int, comparison: str
) -> tuple[float, float]:
    rng = random.Random(
        _seed_from_parts(
            "gate7-high-scale-routing-bandwidth-paired-bootstrap-v0",
            population,
            checkpoint,
            comparison,
        )
    )
    estimates = [
        sum(differences[rng.randrange(WORLD_COUNT)] for _ in range(WORLD_COUNT)) / WORLD_COUNT
        for _ in range(BOOTSTRAP_SAMPLES)
    ]
    return _quantiles(estimates)


def _stratified_bootstrap(
    differences: dict[int, tuple[int, ...]], *, population: int
) -> tuple[float, float]:
    rng = random.Random(
        _seed_from_parts(
            "gate7-high-scale-routing-bandwidth-stratified-bootstrap-v0",
            population,
            "global_score_vs_global_hash",
        )
    )
    estimates: list[float] = []
    for _ in range(BOOTSTRAP_SAMPLES):
        means = []
        for checkpoint in CHECKPOINT_INDICES:
            vector = differences[checkpoint]
            means.append(
                sum(vector[rng.randrange(WORLD_COUNT)] for _ in range(WORLD_COUNT)) / WORLD_COUNT
            )
        estimates.append(sum(means) / len(means))
    return _quantiles(estimates)


def _condition_name_score(k: int) -> str:
    return f"bounded_score_k{k}"


def _condition_name_hash(k: int) -> str:
    return f"bounded_hash_k{k}"


def _expected_observations(population: int, condition: str) -> int:
    if condition == GLOBAL_SCORE:
        return STAGE_B_SLOTS * population - (STAGE_B_SLOTS - 1) * STAGE_B_SLOTS // 2
    if condition == GLOBAL_HASH or condition.startswith("bounded_hash_k"):
        return 0
    if condition.startswith("bounded_score_k"):
        return STAGE_B_SLOTS * int(condition.rsplit("k", 1)[1])
    raise ValueError("unknown Gate-7 condition")


def _coverage_vector(row: dict[str, Any], errors: list[str], label: str) -> tuple[bool, ...]:
    values = row.get("covered_by_world")
    if not isinstance(values, list) or len(values) != WORLD_COUNT or any(type(x) is not bool for x in values):
        errors.append(f"{label} must contain exactly 64 Boolean coverage values")
        return tuple()
    vector = tuple(values)
    expected_rate = sum(int(value) for value in vector) / WORLD_COUNT
    if not _float_equal(expected_rate, row.get("coverage_rate")):
        errors.append(f"{label} coverage rate does not match its vector")
    return vector


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
    valid_conditions = {GLOBAL_SCORE, GLOBAL_HASH}
    valid_conditions.update(_condition_name_score(k) for k in K_LADDER)
    valid_conditions.update(_condition_name_hash(k) for k in K_LADDER)
    if condition not in valid_conditions:
        errors.append(f"unexpected condition {label}")
        return None
    if row.get("population") != population:
        errors.append(f"{label} population mismatch")
    if row.get("world_indices") != list(range(WORLD_COUNT)):
        errors.append(f"{label} world indices differ from 0..63")
    if row.get("runtime_seeds") != expected_runtime_seeds:
        errors.append(f"{label} runtime seeds differ from the frozen namespace")
    vector = _coverage_vector(row, errors, label)
    observations = row.get("score_observations_per_world")
    expected_observation = _expected_observations(population, condition)
    if (
        not isinstance(observations, list)
        or len(observations) != WORLD_COUNT
        or any(value != expected_observation for value in observations)
    ):
        errors.append(f"{label} neural-score observation accounting changed")
    expected_stage_a = population - 1
    expected_updates = (population - 1 + STAGE_B_SLOTS) * 16
    if row.get("logical_stage_a_parent_slots") != expected_stage_a:
        errors.append(f"{label} Stage-A work identity changed")
    if row.get("logical_stage_b_parent_slots") != STAGE_B_SLOTS:
        errors.append(f"{label} Stage-B work identity changed")
    if row.get("logical_learned_updates_per_world") != expected_updates:
        errors.append(f"{label} learned-update identity changed")
    if row.get("learned_parameter_count") != PARAMETER_COUNT:
        errors.append(f"{label} parameter count changed")
    expected_fingerprint = CHECKPOINTS[checkpoint]["fingerprint"]
    if row.get("parameter_fingerprint") != expected_fingerprint:
        errors.append(f"{label} parameter fingerprint mismatch")
    if not vector:
        return None
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


def _validate_pair_row(
    row: dict[str, Any],
    *,
    comparison: str,
    treatment: dict[str, Any],
    reference: dict[str, Any],
    population: int,
    checkpoint: int,
    errors: list[str],
) -> float:
    label = f"N{population}/{comparison}"
    expected_delta, expected_low, expected_high = _expected_pair(
        comparison=comparison,
        treatment=treatment,
        reference=reference,
        population=population,
        checkpoint=checkpoint,
    )
    if row.get("comparison") != comparison:
        errors.append(f"{label} comparison name mismatch")
    if row.get("checkpoint_index") != checkpoint or row.get("population") != population:
        errors.append(f"{label} pair identity mismatch")
    if row.get("treatment_condition") != treatment.get("condition"):
        errors.append(f"{label} treatment condition mismatch")
    if row.get("reference_condition") != reference.get("condition"):
        errors.append(f"{label} reference condition mismatch")
    for field, expected in (
        ("coverage_delta", expected_delta),
        ("bootstrap_ci_low", expected_low),
        ("bootstrap_ci_high", expected_high),
    ):
        if not _float_equal(expected, row.get(field)):
            errors.append(f"{label} {field} mismatch")
    return expected_low


def _classify_k(test_lows: dict[int, dict[str, float]]) -> int | None:
    for position, k in enumerate(test_lows):
        lows = test_lows[k]
        passed = all(
            lows[f"c{checkpoint}_k{k}_score_vs_hash"] > 0.0
            and lows[f"c{checkpoint}_k{k}_score_vs_global"] > -NONINFERIORITY_MARGIN
            for checkpoint in CHECKPOINT_INDICES
        )
        if passed:
            if position != len(test_lows) - 1:
                raise ValueError("larger K exposed after first all-checkpoint pass")
            return k
    return None


def audit_gate7_high_scale_routing_bandwidth(path: Path) -> Gate7HighScaleAudit:
    errors: list[str] = []
    try:
        payload = _load(path)
    except Exception as exc:  # noqa: BLE001
        return Gate7HighScaleAudit(False, "INVALID_ARTIFACT", None, tuple(), {}, (str(exc),))

    if payload.get("experiment_version") != EXPERIMENT_VERSION:
        errors.append("unexpected experiment version")
    if payload.get("scientific_status") != SCIENTIFIC_STATUS:
        errors.append("unexpected scientific status")
    for field, expected in (
        ("execution_admitted", True),
        ("high_scale_gate7_screening_opened", True),
        ("confirmation_opened", False),
        ("training_performed", False),
        ("checkpoint_selection_performed", False),
        ("compiler_enabled", False),
        ("cuda_graphs_enabled", False),
        ("mixed_precision_enabled", False),
    ):
        if payload.get(field) is not expected:
            errors.append(f"{field} differs from the frozen execution contract")
    if payload.get("engineering_prerequisite_head") != ENGINEERING_HEAD:
        errors.append("engineering prerequisite head mismatch")
    if payload.get("engineering_summary_sha256") != ENGINEERING_SUMMARY_SHA256:
        errors.append("engineering summary hash mismatch")
    if payload.get("engineering_manifest_sha256") != ENGINEERING_MANIFEST_SHA256:
        errors.append("engineering manifest hash mismatch")
    if payload.get("world_count_per_checkpoint_tier") != WORLD_COUNT:
        errors.append("world count differs from 64")
    if payload.get("evaluation_batch_size") != BATCH_SIZE:
        errors.append("evaluation batch differs from 64")
    if payload.get("bootstrap_samples") != BOOTSTRAP_SAMPLES:
        errors.append("bootstrap samples differ from 2000")
    if not _float_equal(HINT_RELIABILITY, payload.get("hint_reliability")):
        errors.append("hint reliability differs from 0.70")
    if not _float_equal(NONINFERIORITY_MARGIN, payload.get("noninferiority_margin")):
        errors.append("non-inferiority margin differs from 0.05")
    if payload.get("populations") != list(POPULATIONS):
        errors.append("population ladder changed")
    if payload.get("k_ladder") != list(K_LADDER):
        errors.append("K ladder changed")
    if payload.get("stage_b_parent_slots") != STAGE_B_SLOTS:
        errors.append("Stage-B slot count changed")

    checkpoint_rows = payload.get("transition_checkpoints")
    if not isinstance(checkpoint_rows, list) or len(checkpoint_rows) != 3:
        errors.append("artifact must identify exactly three transition checkpoints")
        checkpoint_rows = []
    seen_checkpoints: set[int] = set()
    for row in checkpoint_rows:
        if not isinstance(row, dict) or row.get("checkpoint_index") not in CHECKPOINTS:
            errors.append("invalid transition checkpoint identity")
            continue
        checkpoint = int(row["checkpoint_index"])
        if checkpoint in seen_checkpoints:
            errors.append(f"duplicate checkpoint identity {checkpoint}")
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
    if not isinstance(tiers, list):
        errors.append("tiers must be a list")
        tiers = []
    populations_completed: list[int] = []
    k_required_by_population: dict[int, int | None] = {}
    expected_campaign = None

    for tier_position, tier in enumerate(tiers):
        if not isinstance(tier, dict):
            errors.append("tier is not an object")
            continue
        if tier_position >= len(POPULATIONS):
            errors.append("artifact contains too many population tiers")
            break
        population = POPULATIONS[tier_position]
        if tier.get("population") != population:
            errors.append(f"tier {tier_position} population is not the next frozen population")
            continue
        populations_completed.append(population)
        expected_runtime_seeds = [_runtime_seed(population, index) for index in range(WORLD_COUNT)]
        if tier.get("world_indices") != list(range(WORLD_COUNT)):
            errors.append(f"N{population} world indices differ from 0..63")
        if tier.get("runtime_seeds") != expected_runtime_seeds:
            errors.append(f"N{population} runtime seeds differ from the frozen namespace")
        expected_depth = _world_depth(population)
        if tier.get("frontier_depth") != expected_depth - 1 or tier.get("world_depth") != expected_depth:
            errors.append(f"N{population} depth geometry changed")
        if tier.get("logical_stage_a_parent_slots") != population - 1:
            errors.append(f"N{population} Stage-A slots changed")
        if tier.get("logical_stage_b_parent_slots") != STAGE_B_SLOTS:
            errors.append(f"N{population} Stage-B slots changed")
        if tier.get("logical_learned_updates_per_world") != (population - 1 + STAGE_B_SLOTS) * 16:
            errors.append(f"N{population} learned-work identity changed")

        condition_rows = tier.get("conditions")
        if not isinstance(condition_rows, list):
            errors.append(f"N{population} conditions must be a list")
            condition_rows = []
        condition_index: dict[tuple[int, str], dict[str, Any]] = {}
        for row in condition_rows:
            if not isinstance(row, dict):
                errors.append(f"N{population} condition is not an object")
                continue
            key = _validate_condition(
                row,
                population=population,
                expected_runtime_seeds=expected_runtime_seeds,
                errors=errors,
            )
            if key is None:
                continue
            if key in condition_index:
                errors.append(f"N{population} duplicate condition {key}")
            condition_index[key] = row

        tier_outcome = tier.get("tier_outcome")
        if tier_outcome == RESOURCE_FRONTIER:
            expected_campaign = RESOURCE_FRONTIER
            k_required_by_population[population] = None
            if tier_position != len(tiers) - 1:
                errors.append("tiers continue after a resource frontier")
            continue

        pair_rows = tier.get("paired_summaries")
        if not isinstance(pair_rows, list):
            errors.append(f"N{population} paired summaries must be a list")
            pair_rows = []
        pair_index = {
            row.get("comparison"): row
            for row in pair_rows
            if isinstance(row, dict) and isinstance(row.get("comparison"), str)
        }
        if len(pair_index) != len(pair_rows):
            errors.append(f"N{population} duplicate or invalid paired summary")

        global_differences: dict[int, tuple[int, ...]] = {}
        checkpoint_points: dict[int, float] = {}
        for checkpoint in CHECKPOINT_INDICES:
            score = condition_index.get((checkpoint, GLOBAL_SCORE))
            control = condition_index.get((checkpoint, GLOBAL_HASH))
            if score is None or control is None:
                errors.append(f"N{population} missing global pair for checkpoint {checkpoint}")
                continue
            comparison = f"c{checkpoint}_global_score_vs_global_hash"
            pair = pair_index.get(comparison)
            if pair is None:
                errors.append(f"N{population} missing {comparison}")
                continue
            _validate_pair_row(
                pair,
                comparison=comparison,
                treatment=score,
                reference=control,
                population=population,
                checkpoint=checkpoint,
                errors=errors,
            )
            vector = tuple(
                int(left) - int(right)
                for left, right in zip(score["covered_by_world"], control["covered_by_world"], strict=True)
            )
            global_differences[checkpoint] = vector
            checkpoint_points[checkpoint] = sum(vector) / WORLD_COUNT

        stratified = tier.get("reference_stratified_summary")
        viable = False
        if len(global_differences) == 3 and isinstance(stratified, dict):
            pooled_low, pooled_high = _stratified_bootstrap(global_differences, population=population)
            pooled_delta = sum(checkpoint_points.values()) / 3
            expected_points = {str(key): value for key, value in checkpoint_points.items()}
            observed_points = stratified.get("checkpoint_point_deltas")
            normalized_points = (
                {str(key): value for key, value in observed_points.items()}
                if isinstance(observed_points, dict)
                else {}
            )
            if normalized_points.keys() != expected_points.keys() or any(
                not _float_equal(expected_points[key], normalized_points.get(key))
                for key in expected_points
            ):
                errors.append(f"N{population} stratified checkpoint deltas mismatch")
            for field, expected in (
                ("pooled_delta", pooled_delta),
                ("bootstrap_ci_low", pooled_low),
                ("bootstrap_ci_high", pooled_high),
            ):
                if not _float_equal(expected, stratified.get(field)):
                    errors.append(f"N{population} stratified {field} mismatch")
            viable = all(value > 0.0 for value in checkpoint_points.values()) and pooled_low > 0.0
        else:
            errors.append(f"N{population} stratified global reference summary missing")
        if tier.get("reference_viable") is not viable:
            errors.append(f"N{population} reference viability mismatch")

        tested_k_raw = tier.get("tested_k")
        if not isinstance(tested_k_raw, list) or any(type(k) is not int for k in tested_k_raw):
            errors.append(f"N{population} tested_k must be an integer list")
            tested_k: tuple[int, ...] = tuple()
        else:
            tested_k = tuple(tested_k_raw)
        if tested_k != K_LADDER[: len(tested_k)]:
            errors.append(f"N{population} K exposure is not a contiguous ascending prefix")

        lows_by_k: dict[int, dict[str, float]] = {}
        for k in tested_k:
            k_lows: dict[str, float] = {}
            for checkpoint in CHECKPOINT_INDICES:
                score = condition_index.get((checkpoint, _condition_name_score(k)))
                control = condition_index.get((checkpoint, _condition_name_hash(k)))
                global_score = condition_index.get((checkpoint, GLOBAL_SCORE))
                if score is None or control is None or global_score is None:
                    errors.append(f"N{population}/K{k} missing checkpoint {checkpoint} conditions")
                    continue
                for suffix, reference in (("score_vs_hash", control), ("score_vs_global", global_score)):
                    comparison = f"c{checkpoint}_k{k}_{suffix}"
                    pair = pair_index.get(comparison)
                    if pair is None:
                        errors.append(f"N{population} missing {comparison}")
                        continue
                    k_lows[comparison] = _validate_pair_row(
                        pair,
                        comparison=comparison,
                        treatment=score,
                        reference=reference,
                        population=population,
                        checkpoint=checkpoint,
                        errors=errors,
                    )
            lows_by_k[k] = k_lows

        passing_k: int | None = None
        if viable:
            try:
                passing_k = _classify_k(lows_by_k)
            except ValueError as exc:
                errors.append(f"N{population} {exc}")
        elif tested_k:
            errors.append(f"N{population} exposed K after reference failure")

        expected_tier_outcome: str
        if not viable:
            expected_tier_outcome = REFERENCE_FRONTIER
        elif passing_k is not None:
            expected_tier_outcome = f"G7_K_REQUIRED_{passing_k}"
        elif tested_k == K_LADDER:
            expected_tier_outcome = ROUTING_FRONTIER
        else:
            expected_tier_outcome = "G7_SCREENING_INCOMPLETE"
        if tier_outcome != expected_tier_outcome:
            errors.append(f"N{population} tier outcome mismatch: {tier_outcome} != {expected_tier_outcome}")

        observed_k_required = tier.get("k_required")
        if observed_k_required != passing_k:
            errors.append(f"N{population} k_required mismatch")
        expected_ratio = None if passing_k is None else passing_k / population
        if expected_ratio is None:
            if tier.get("k_required_over_n") is not None:
                errors.append(f"N{population} K/N must be null")
        elif not _float_equal(expected_ratio, tier.get("k_required_over_n")):
            errors.append(f"N{population} K/N mismatch")
        k_required_by_population[population] = passing_k

        unexposed = tier.get("unexposed_k")
        if not isinstance(unexposed, list):
            errors.append(f"N{population} unexposed_k must be a list")
            unexposed = []
        unexposed_map = {
            row.get("k"): row.get("status")
            for row in unexposed
            if isinstance(row, dict)
        }
        expected_unexposed = set(K_LADDER) - set(tested_k)
        if set(unexposed_map) != expected_unexposed:
            errors.append(f"N{population} unexposed K set mismatch")
        if passing_k is not None and any(
            unexposed_map.get(k) != "NOT_RUN_BY_FIRST_PASS_RULE"
            for k in expected_unexposed
        ):
            errors.append(f"N{population} larger K values are not explicitly first-pass suppressed")

        if expected_tier_outcome.startswith("G7_K_REQUIRED_"):
            expected_campaign = CAMPAIGN_CEILING if population == POPULATIONS[-1] else CONTINUE
        else:
            expected_campaign = expected_tier_outcome
        if expected_campaign != CONTINUE and tier_position != len(tiers) - 1:
            errors.append(f"tiers continue after terminal N{population} outcome")

    observed_campaign = payload.get("campaign_outcome")
    if expected_campaign is None:
        errors.append("campaign contains no completed tier")
    elif observed_campaign != expected_campaign:
        errors.append(f"campaign outcome mismatch: {observed_campaign} != {expected_campaign}")
    if expected_campaign == CONTINUE:
        errors.append("artifact stops while protocol requires the next population")

    return Gate7HighScaleAudit(
        artifact_valid=not errors,
        scientific_status=SCIENTIFIC_STATUS if not errors else "INVALID_ARTIFACT",
        campaign_outcome=observed_campaign if isinstance(observed_campaign, str) else None,
        populations_completed=tuple(populations_completed),
        k_required_by_population=k_required_by_population,
        errors=tuple(errors),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    audit = audit_gate7_high_scale_routing_bandwidth(args.result)
    payload = audit.to_dict()
    rendered = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0 if audit.artifact_valid else 1


if __name__ == "__main__":
    raise SystemExit(main())
