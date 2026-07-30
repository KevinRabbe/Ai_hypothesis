"""Paired statistics and immutable provenance for Gate-7 continuation."""

from __future__ import annotations

import math
import random
from dataclasses import asdict, dataclass
from typing import Any

from .gate7_high_scale_routing_bandwidth_continuation_conditions import (
    Gate7ContinuationCondition,
)
from .gate7_high_scale_routing_bandwidth_continuation_protocol import (
    GATE7_CONTINUATION_ACTIVE_CHILD_LANES,
    GATE7_CONTINUATION_BOOTSTRAP_SAMPLES,
    GATE7_CONTINUATION_CHECKPOINT_INDICES,
    GATE7_CONTINUATION_CONFIRMATION_AUDIT_SHA256,
    GATE7_CONTINUATION_CONFIRMATION_EXECUTION_HEAD,
    GATE7_CONTINUATION_CONFIRMATION_MANIFEST_SHA256,
    GATE7_CONTINUATION_CONFIRMATION_OUTCOME,
    GATE7_CONTINUATION_CONFIRMATION_RESULT_HEAD,
    GATE7_CONTINUATION_CONFIRMATION_RESULT_SHA256,
    GATE7_CONTINUATION_CONFIRMED_N8192_K_REQUIRED,
    GATE7_CONTINUATION_CONFIRMED_N8192_PASSING_K,
    GATE7_CONTINUATION_EVALUATION_BATCH_SIZE,
    GATE7_CONTINUATION_RECURRENT_UPDATES_PER_CHILD,
    GATE7_CONTINUATION_STAGE_B_PARENT_SLOTS,
    GATE7_CONTINUATION_VERSION,
    GATE7_CONTINUATION_WORLD_COUNT,
)
from .gate7_high_scale_routing_bandwidth_continuation_worlds import (
    continuation_seed_from_parts,
)

GATE7_CONTINUATION_PROTOCOL_HEAD = "4f05f8b1f9a33aed712edbf28691b927d2e220d3"


@dataclass(frozen=True, slots=True)
class Gate7ContinuationPairedSummary:
    comparison: str
    checkpoint_index: int
    population: int
    k: int | None
    treatment_condition: str
    reference_condition: str
    coverage_delta: float
    bootstrap_ci_low: float
    bootstrap_ci_high: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class Gate7ContinuationStratifiedSummary:
    comparison: str
    population: int
    checkpoint_point_deltas: dict[int, float]
    pooled_delta: float
    bootstrap_ci_low: float
    bootstrap_ci_high: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _bootstrap_quantiles(estimates: list[float]) -> tuple[float, float]:
    estimates.sort()
    return (
        estimates[int(math.floor(0.025 * (GATE7_CONTINUATION_BOOTSTRAP_SAMPLES - 1)))],
        estimates[int(math.ceil(0.975 * (GATE7_CONTINUATION_BOOTSTRAP_SAMPLES - 1)))],
    )


def paired_gate7_continuation_summary(
    *,
    comparison: str,
    treatment: Gate7ContinuationCondition,
    reference: Gate7ContinuationCondition,
) -> Gate7ContinuationPairedSummary:
    if treatment.checkpoint_index != reference.checkpoint_index:
        raise ValueError("continuation pair checkpoint mismatch")
    if treatment.population != reference.population:
        raise ValueError("continuation pair population mismatch")
    if treatment.world_indices != reference.world_indices or treatment.runtime_seeds != reference.runtime_seeds:
        raise ValueError("continuation pair does not share the exact same worlds")
    differences = tuple(
        int(left) - int(right)
        for left, right in zip(treatment.covered_by_world, reference.covered_by_world, strict=True)
    )
    rng = random.Random(
        continuation_seed_from_parts(
            "gate7-high-scale-routing-bandwidth-continuation-paired-bootstrap-v0",
            treatment.population,
            treatment.checkpoint_index,
            comparison,
        )
    )
    count = GATE7_CONTINUATION_WORLD_COUNT
    estimates = [
        sum(differences[rng.randrange(count)] for _ in range(count)) / count
        for _ in range(GATE7_CONTINUATION_BOOTSTRAP_SAMPLES)
    ]
    low, high = _bootstrap_quantiles(estimates)
    return Gate7ContinuationPairedSummary(
        comparison=comparison,
        checkpoint_index=treatment.checkpoint_index,
        population=treatment.population,
        k=treatment.k,
        treatment_condition=treatment.condition,
        reference_condition=reference.condition,
        coverage_delta=sum(differences) / count,
        bootstrap_ci_low=low,
        bootstrap_ci_high=high,
    )


def stratified_gate7_continuation_global_summary(
    *,
    population: int,
    treatment_by_checkpoint: dict[int, Gate7ContinuationCondition],
    reference_by_checkpoint: dict[int, Gate7ContinuationCondition],
) -> Gate7ContinuationStratifiedSummary:
    if set(treatment_by_checkpoint) != set(GATE7_CONTINUATION_CHECKPOINT_INDICES):
        raise ValueError("continuation stratified treatment requires all three checkpoints")
    if set(reference_by_checkpoint) != set(GATE7_CONTINUATION_CHECKPOINT_INDICES):
        raise ValueError("continuation stratified reference requires all three checkpoints")
    differences: dict[int, tuple[int, ...]] = {}
    points: dict[int, float] = {}
    for checkpoint in GATE7_CONTINUATION_CHECKPOINT_INDICES:
        treatment = treatment_by_checkpoint[checkpoint]
        reference = reference_by_checkpoint[checkpoint]
        if treatment.population != population or reference.population != population:
            raise ValueError("continuation stratified population mismatch")
        if treatment.world_indices != reference.world_indices or treatment.runtime_seeds != reference.runtime_seeds:
            raise ValueError("continuation stratified pair does not share exact worlds")
        vector = tuple(
            int(left) - int(right)
            for left, right in zip(treatment.covered_by_world, reference.covered_by_world, strict=True)
        )
        differences[checkpoint] = vector
        points[checkpoint] = sum(vector) / GATE7_CONTINUATION_WORLD_COUNT

    rng = random.Random(
        continuation_seed_from_parts(
            "gate7-high-scale-routing-bandwidth-continuation-stratified-bootstrap-v0",
            population,
            "global_score_vs_global_hash",
        )
    )
    count = GATE7_CONTINUATION_WORLD_COUNT
    estimates: list[float] = []
    for _ in range(GATE7_CONTINUATION_BOOTSTRAP_SAMPLES):
        stratum_means = []
        for checkpoint in GATE7_CONTINUATION_CHECKPOINT_INDICES:
            vector = differences[checkpoint]
            stratum_means.append(
                sum(vector[rng.randrange(count)] for _ in range(count)) / count
            )
        estimates.append(sum(stratum_means) / len(stratum_means))
    low, high = _bootstrap_quantiles(estimates)
    return Gate7ContinuationStratifiedSummary(
        comparison="global_score_vs_global_hash_stratified",
        population=population,
        checkpoint_point_deltas=points,
        pooled_delta=sum(points.values()) / len(points),
        bootstrap_ci_low=low,
        bootstrap_ci_high=high,
    )


def continuation_provenance() -> dict[str, Any]:
    return {
        "continuation_version": GATE7_CONTINUATION_VERSION,
        "continuation_protocol_head": GATE7_CONTINUATION_PROTOCOL_HEAD,
        "confirmation_execution_head": GATE7_CONTINUATION_CONFIRMATION_EXECUTION_HEAD,
        "confirmation_result_head": GATE7_CONTINUATION_CONFIRMATION_RESULT_HEAD,
        "confirmation_result_sha256": GATE7_CONTINUATION_CONFIRMATION_RESULT_SHA256,
        "confirmation_audit_sha256": GATE7_CONTINUATION_CONFIRMATION_AUDIT_SHA256,
        "confirmation_manifest_sha256": GATE7_CONTINUATION_CONFIRMATION_MANIFEST_SHA256,
        "confirmation_outcome": GATE7_CONTINUATION_CONFIRMATION_OUTCOME,
        "confirmed_n8192_passing_k": list(GATE7_CONTINUATION_CONFIRMED_N8192_PASSING_K),
        "confirmed_n8192_k_required": GATE7_CONTINUATION_CONFIRMED_N8192_K_REQUIRED,
        "world_count": GATE7_CONTINUATION_WORLD_COUNT,
        "evaluation_batch_size": GATE7_CONTINUATION_EVALUATION_BATCH_SIZE,
        "bootstrap_samples": GATE7_CONTINUATION_BOOTSTRAP_SAMPLES,
        "stage_b_parent_slots": GATE7_CONTINUATION_STAGE_B_PARENT_SLOTS,
        "active_child_lanes": GATE7_CONTINUATION_ACTIVE_CHILD_LANES,
        "recurrent_updates_per_child": GATE7_CONTINUATION_RECURRENT_UPDATES_PER_CHILD,
    }
