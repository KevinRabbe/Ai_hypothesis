"""Independent tier reconstruction for Gate-7 continuation artifacts."""

from __future__ import annotations

from typing import Any

from . import gate7_high_scale_routing_bandwidth_continuation_audit_spec as spec


def validate_condition(
    row: dict[str, Any],
    *,
    population: int,
    expected_runtime_seeds: list[int],
    errors: list[str],
) -> tuple[int, str] | None:
    checkpoint = row.get("checkpoint_index")
    condition = row.get("condition")
    label = f"N{population}/C{checkpoint}/{condition}"
    if checkpoint not in spec.CHECKPOINT_INDICES or not isinstance(condition, str):
        errors.append(f"invalid condition identity {label}")
        return None
    if condition not in spec.planned_conditions():
        errors.append(f"unexpected condition {label}")
        return None
    try:
        expected_k = spec.condition_k(condition)
    except Exception as exc:  # noqa: BLE001
        errors.append(f"{label} invalid condition K: {exc}")
        return None
    if row.get("k") != expected_k:
        errors.append(f"{label} K identity mismatch")
    if row.get("population") != population:
        errors.append(f"{label} population mismatch")
    if row.get("world_indices") != list(range(spec.WORLD_COUNT)):
        errors.append(f"{label} world indices differ from 0..511")
    if row.get("runtime_seeds") != expected_runtime_seeds:
        errors.append(f"{label} runtime seeds differ from the frozen namespace")
    covered = row.get("covered_by_world")
    if (
        not isinstance(covered, list)
        or len(covered) != spec.WORLD_COUNT
        or any(type(value) is not bool for value in covered)
    ):
        errors.append(f"{label} must contain exactly 512 Boolean coverage values")
    else:
        expected_rate = sum(int(value) for value in covered) / spec.WORLD_COUNT
        if not spec.float_equal(expected_rate, row.get("coverage_rate")):
            errors.append(f"{label} coverage rate does not match its vector")
    observations = row.get("score_observations_per_world")
    expected_observation = spec.expected_observations(population, condition)
    if (
        not isinstance(observations, list)
        or len(observations) != spec.WORLD_COUNT
        or any(value != expected_observation for value in observations)
    ):
        errors.append(f"{label} score-observation accounting changed")
    if row.get("logical_stage_a_parent_slots") != population - 1:
        errors.append(f"{label} Stage-A work identity changed")
    if row.get("logical_stage_b_parent_slots") != spec.STAGE_B_SLOTS:
        errors.append(f"{label} Stage-B work identity changed")
    if row.get("logical_learned_updates_per_world") != (
        population - 1 + spec.STAGE_B_SLOTS
    ) * 16:
        errors.append(f"{label} learned-update identity changed")
    if row.get("learned_parameter_count") != spec.PARAMETER_COUNT:
        errors.append(f"{label} parameter count changed")
    if row.get("parameter_fingerprint") != spec.CHECKPOINTS[checkpoint]["fingerprint"]:
        errors.append(f"{label} parameter fingerprint mismatch")
    if row.get("batch_count") != spec.BATCH_COUNT:
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


def expected_pair(
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
    low, high = spec.paired_bootstrap(
        differences,
        population=population,
        checkpoint=checkpoint,
        comparison=comparison,
    )
    return sum(differences) / spec.WORLD_COUNT, low, high


def validate_pair(
    *,
    pair: dict[str, Any] | None,
    comparison: str,
    treatment: dict[str, Any],
    reference: dict[str, Any],
    population: int,
    checkpoint: int,
    expected_k: int | None,
    errors: list[str],
) -> float | None:
    if pair is None:
        errors.append(f"missing N{population} pair {comparison}")
        return None
    if pair.get("checkpoint_index") != checkpoint:
        errors.append(f"{comparison} checkpoint mismatch")
    if pair.get("population") != population:
        errors.append(f"{comparison} population mismatch")
    if pair.get("k") != expected_k:
        errors.append(f"{comparison} K mismatch")
    if pair.get("treatment_condition") != treatment.get("condition"):
        errors.append(f"{comparison} treatment identity mismatch")
    if pair.get("reference_condition") != reference.get("condition"):
        errors.append(f"{comparison} reference identity mismatch")
    try:
        expected_delta, expected_low, expected_high = expected_pair(
            comparison=comparison,
            treatment=treatment,
            reference=reference,
            population=population,
            checkpoint=checkpoint,
        )
    except Exception as exc:  # noqa: BLE001
        errors.append(f"{comparison} bootstrap reconstruction failed: {exc}")
        return None
    for name, expected in (
        ("coverage_delta", expected_delta),
        ("bootstrap_ci_low", expected_low),
        ("bootstrap_ci_high", expected_high),
    ):
        if not spec.float_equal(expected, pair.get(name)):
            errors.append(f"{comparison} {name} mismatch")
    return expected_low


def validate_frontier_builds(
    rows: Any,
    *,
    population: int,
    errors: list[str],
) -> None:
    expected_count = len(spec.CHECKPOINT_INDICES) * spec.BATCH_COUNT
    if not isinstance(rows, list) or len(rows) != expected_count:
        errors.append(f"N{population} must contain exactly {expected_count} frontier builds")
        return
    expected = {
        (checkpoint, batch_index, (batch_index - 1) * spec.BATCH_SIZE)
        for checkpoint in spec.CHECKPOINT_INDICES
        for batch_index in range(1, spec.BATCH_COUNT + 1)
    }
    observed: set[tuple[int, int, int]] = set()
    for row in rows:
        if not isinstance(row, dict):
            errors.append(f"N{population} frontier build is not an object")
            continue
        identity = (
            row.get("checkpoint_index"),
            row.get("batch_index"),
            row.get("batch_start"),
        )
        if identity in observed:
            errors.append(f"N{population} duplicate frontier build {identity}")
        observed.add(identity)
        for telemetry in ("wall_seconds", "peak_allocated_bytes", "frontier_storage_bytes"):
            if telemetry not in row:
                errors.append(f"N{population} frontier build missing {telemetry}")
    if observed != expected:
        errors.append(f"N{population} frontier-build identity set changed")


def validate_tier(tier: dict[str, Any], errors: list[str]) -> dict[str, Any] | None:
    population = tier.get("population")
    if population not in spec.POPULATIONS:
        errors.append(f"unexpected tier population {population}")
        return None
    expected_runtime_seeds = [
        spec.runtime_seed(population, index) for index in range(spec.WORLD_COUNT)
    ]
    if tier.get("world_indices") != list(range(spec.WORLD_COUNT)):
        errors.append(f"N{population} tier world indices changed")
    if tier.get("runtime_seeds") != expected_runtime_seeds:
        errors.append(f"N{population} tier runtime seeds changed")
    if tier.get("world_count") != spec.WORLD_COUNT:
        errors.append(f"N{population} tier world count changed")
    if tier.get("evaluation_batch_size") != spec.BATCH_SIZE:
        errors.append(f"N{population} tier physical batch changed")
    if tier.get("physical_batch_count") != spec.BATCH_COUNT:
        errors.append(f"N{population} tier batch count changed")
    if tier.get("conditions_planned") != list(spec.planned_conditions()):
        errors.append(f"N{population} planned condition matrix changed")
    if tier.get("k_values_planned") != list(spec.K_LADDER):
        errors.append(f"N{population} planned K matrix changed")
    if tier.get("logical_stage_a_parent_slots") != population - 1:
        errors.append(f"N{population} tier Stage-A work changed")
    if tier.get("logical_stage_b_parent_slots") != spec.STAGE_B_SLOTS:
        errors.append(f"N{population} tier Stage-B work changed")
    if tier.get("logical_learned_updates_per_world") != (
        population - 1 + spec.STAGE_B_SLOTS
    ) * 16:
        errors.append(f"N{population} tier learned work changed")

    validate_frontier_builds(tier.get("frontier_builds"), population=population, errors=errors)

    conditions = tier.get("conditions")
    expected_condition_count = len(spec.CHECKPOINT_INDICES) * len(spec.planned_conditions())
    if not isinstance(conditions, list) or len(conditions) != expected_condition_count:
        errors.append(f"N{population} condition count changed")
        conditions = []
    condition_index: dict[tuple[int, str], dict[str, Any]] = {}
    for row in conditions:
        if not isinstance(row, dict):
            errors.append(f"N{population} condition is not an object")
            continue
        identity = validate_condition(
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
        for checkpoint in spec.CHECKPOINT_INDICES
        for condition in spec.planned_conditions()
    }
    if set(condition_index) != expected_keys:
        errors.append(f"N{population} condition identity set is incomplete")

    pair_rows = tier.get("paired_summaries")
    expected_pair_count = len(spec.CHECKPOINT_INDICES) * (1 + 2 * len(spec.K_LADDER))
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
    lows_by_k: dict[int, dict[str, float]] = {k: {} for k in spec.K_LADDER}
    for checkpoint in spec.CHECKPOINT_INDICES:
        score = condition_index.get((checkpoint, spec.GLOBAL_SCORE))
        hash_control = condition_index.get((checkpoint, spec.GLOBAL_HASH))
        if score is None or hash_control is None:
            continue
        comparison = f"c{checkpoint}_global_score_vs_global_hash"
        validate_pair(
            pair=pair_index.get(comparison),
            comparison=comparison,
            treatment=score,
            reference=hash_control,
            population=population,
            checkpoint=checkpoint,
            expected_k=None,
            errors=errors,
        )
        differences = tuple(
            int(left) - int(right)
            for left, right in zip(
                score["covered_by_world"],
                hash_control["covered_by_world"],
                strict=True,
            )
        )
        differences_by_checkpoint[checkpoint] = differences
        points[checkpoint] = sum(differences) / spec.WORLD_COUNT

        for k in spec.K_LADDER:
            bounded_score = condition_index.get((checkpoint, spec.score_condition(k)))
            bounded_hash = condition_index.get((checkpoint, spec.hash_condition(k)))
            if bounded_score is None or bounded_hash is None:
                continue
            learned_comparison = f"c{checkpoint}_k{k}_score_vs_hash"
            global_comparison = f"c{checkpoint}_k{k}_score_vs_global"
            learned_low = validate_pair(
                pair=pair_index.get(learned_comparison),
                comparison=learned_comparison,
                treatment=bounded_score,
                reference=bounded_hash,
                population=population,
                checkpoint=checkpoint,
                expected_k=k,
                errors=errors,
            )
            global_low = validate_pair(
                pair=pair_index.get(global_comparison),
                comparison=global_comparison,
                treatment=bounded_score,
                reference=score,
                population=population,
                checkpoint=checkpoint,
                expected_k=k,
                errors=errors,
            )
            if learned_low is not None:
                lows_by_k[k][learned_comparison] = learned_low
            if global_low is not None:
                lows_by_k[k][global_comparison] = global_low

    stratified = tier.get("reference_stratified_summary")
    expected_reference_viable = False
    if set(differences_by_checkpoint) == set(spec.CHECKPOINT_INDICES):
        try:
            pooled_low, pooled_high = spec.stratified_bootstrap(
                differences_by_checkpoint,
                population=population,
            )
            pooled_delta = sum(points.values()) / len(points)
            expected_reference_viable = spec.reference_is_viable(
                point_deltas=points,
                pooled_ci_low=pooled_low,
            )
            if not isinstance(stratified, dict):
                errors.append(f"N{population} missing stratified reference summary")
            else:
                if stratified.get("comparison") != "global_score_vs_global_hash_stratified":
                    errors.append(f"N{population} stratified comparison mismatch")
                if stratified.get("population") != population:
                    errors.append(f"N{population} stratified population mismatch")
                observed_points = stratified.get("checkpoint_point_deltas")
                normalized_points = (
                    {int(key): float(value) for key, value in observed_points.items()}
                    if isinstance(observed_points, dict)
                    else {}
                )
                if normalized_points != points:
                    errors.append(f"N{population} stratified point deltas mismatch")
                for name, expected in (
                    ("pooled_delta", pooled_delta),
                    ("bootstrap_ci_low", pooled_low),
                    ("bootstrap_ci_high", pooled_high),
                ):
                    if not spec.float_equal(expected, stratified.get(name)):
                        errors.append(f"N{population} stratified {name} mismatch")
        except Exception as exc:  # noqa: BLE001
            errors.append(f"N{population} stratified reconstruction failed: {exc}")

    if tier.get("reference_viable") is not expected_reference_viable:
        errors.append(f"N{population} reference viability mismatch")

    if any(len(lows_by_k[k]) != 2 * len(spec.CHECKPOINT_INDICES) for k in spec.K_LADDER):
        errors.append(f"N{population} primary CI set is incomplete")
        return {
            "population": population,
            "reference_viable": expected_reference_viable,
            "lows_by_k": lows_by_k,
            "outcome": None,
            "passing_k": (),
            "smallest_passing_k": None,
        }

    expected_outcome, expected_passing, expected_smallest = spec.classify_tier(
        reference_viable=expected_reference_viable,
        lows_by_k=lows_by_k,
    )
    expected_ratio = expected_smallest / population if expected_smallest is not None else None

    stored_lows = tier.get("primary_ci_lows_by_k")
    normalized_lows = (
        {int(key): value for key, value in stored_lows.items()}
        if isinstance(stored_lows, dict)
        else {}
    )
    if normalized_lows != lows_by_k:
        errors.append(f"N{population} stored primary CI lows mismatch")
    if tier.get("passing_k") != list(expected_passing):
        errors.append(f"N{population} passing-K set mismatch")
    if tier.get("smallest_passing_k") != expected_smallest:
        errors.append(f"N{population} smallest passing K mismatch")
    if not (
        tier.get("smallest_passing_k_over_n") is None
        if expected_ratio is None
        else spec.float_equal(expected_ratio, tier.get("smallest_passing_k_over_n"))
    ):
        errors.append(f"N{population} K/N ratio mismatch")
    if tier.get("tier_outcome") != expected_outcome:
        errors.append(f"N{population} tier outcome mismatch")

    classification = tier.get("classification")
    if not isinstance(classification, dict):
        errors.append(f"N{population} missing classification object")
    else:
        if classification.get("population") != population:
            errors.append(f"N{population} classification population mismatch")
        if classification.get("outcome") != expected_outcome:
            errors.append(f"N{population} classification outcome mismatch")
        if classification.get("reference_viable") is not expected_reference_viable:
            errors.append(f"N{population} classification reference mismatch")
        if classification.get("passing_k") != list(expected_passing):
            errors.append(f"N{population} classification passing-K mismatch")
        if classification.get("smallest_passing_k") != expected_smallest:
            errors.append(f"N{population} classification smallest K mismatch")
        if not (
            classification.get("smallest_passing_k_over_n") is None
            if expected_ratio is None
            else spec.float_equal(expected_ratio, classification.get("smallest_passing_k_over_n"))
        ):
            errors.append(f"N{population} classification K/N mismatch")

    return {
        "population": population,
        "reference_viable": expected_reference_viable,
        "lows_by_k": lows_by_k,
        "outcome": expected_outcome,
        "passing_k": expected_passing,
        "smallest_passing_k": expected_smallest,
    }
