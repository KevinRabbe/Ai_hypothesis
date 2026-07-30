"""Execute one fixed population tier of the Gate-7 continuation campaign."""

from __future__ import annotations

import gc
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

from .gate7_high_scale_routing_bandwidth_continuation import (
    Gate7ContinuationBatchCondition,
    Gate7ContinuationCondition,
    aggregate_gate7_continuation_condition,
    build_gate7_continuation_frontier,
    continuation_world_batch,
    evaluate_gate7_continuation_batch_condition,
    load_verified_gate7_continuation_checkpoint,
    paired_gate7_continuation_summary,
    stratified_gate7_continuation_global_summary,
)
from .gate7_high_scale_routing_bandwidth_continuation_protocol import (
    GATE7_CONTINUATION_CHECKPOINT_INDICES,
    GATE7_CONTINUATION_EVALUATION_BATCH_SIZE,
    GATE7_CONTINUATION_GLOBAL_HASH,
    GATE7_CONTINUATION_GLOBAL_SCORE,
    GATE7_CONTINUATION_K_LADDER,
    GATE7_CONTINUATION_WORLD_COUNT,
    bounded_hash_condition,
    bounded_score_condition,
    build_continuation_tier_plan,
    classify_continuation_tier,
    reference_is_viable,
)


def _release_cuda() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def execute_gate7_continuation_tier(
    *,
    population: int,
    transition_checkpoint_paths: tuple[Path, Path, Path],
) -> dict[str, Any]:
    plan = build_continuation_tier_plan(population)
    print(
        f"\nN={population} fixed continuation matrix: "
        f"{len(plan.conditions)} conditions/checkpoint...",
        flush=True,
    )
    tier: dict[str, Any] = {
        "population": population,
        "world_indices": list(range(GATE7_CONTINUATION_WORLD_COUNT)),
        "runtime_seeds": [],
        "world_count": GATE7_CONTINUATION_WORLD_COUNT,
        "evaluation_batch_size": GATE7_CONTINUATION_EVALUATION_BATCH_SIZE,
        "physical_batch_count": GATE7_CONTINUATION_WORLD_COUNT
        // GATE7_CONTINUATION_EVALUATION_BATCH_SIZE,
        "conditions_planned": list(plan.conditions),
        "k_values_planned": list(plan.k_values),
        "logical_stage_a_parent_slots": plan.stage_a_parent_slots,
        "logical_stage_b_parent_slots": plan.stage_b_parent_slots,
        "logical_learned_updates_per_world": plan.logical_learned_updates_per_world,
        "frontier_builds": [],
        "conditions": [],
        "paired_summaries": [],
        "reference_stratified_summary": None,
        "reference_viable": None,
        "primary_ci_lows_by_k": {},
        "passing_k": [],
        "smallest_passing_k": None,
        "smallest_passing_k_over_n": None,
        "tier_outcome": "RUNNING",
    }

    batch_rows: dict[tuple[int, str], list[Gate7ContinuationBatchCondition]] = {
        (checkpoint, condition): []
        for checkpoint in GATE7_CONTINUATION_CHECKPOINT_INDICES
        for condition in plan.conditions
    }

    for checkpoint, path in zip(
        GATE7_CONTINUATION_CHECKPOINT_INDICES,
        transition_checkpoint_paths,
        strict=True,
    ):
        model, _ = load_verified_gate7_continuation_checkpoint(
            checkpoint_index=checkpoint,
            checkpoint_path=path,
            device="cuda",
        )
        print(f"  C{checkpoint}: eight B64 frontier batches", flush=True)
        for batch_index, batch_start in enumerate(
            range(0, GATE7_CONTINUATION_WORLD_COUNT, GATE7_CONTINUATION_EVALUATION_BATCH_SIZE),
            start=1,
        ):
            worlds = continuation_world_batch(
                population=population,
                batch_start=batch_start,
            )
            if checkpoint == GATE7_CONTINUATION_CHECKPOINT_INDICES[0]:
                tier["runtime_seeds"].extend(world.runtime_seed for world in worlds)
            frontier, frontier_metrics = build_gate7_continuation_frontier(
                model,
                worlds=worlds,
                device="cuda",
            )
            tier["frontier_builds"].append(
                {
                    "checkpoint_index": checkpoint,
                    "batch_index": batch_index,
                    "batch_start": batch_start,
                    **frontier_metrics,
                }
            )
            for condition in plan.conditions:
                row = evaluate_gate7_continuation_batch_condition(
                    model,
                    frontier,
                    checkpoint_index=checkpoint,
                    worlds=worlds,
                    condition=condition,
                )
                batch_rows[(checkpoint, condition)].append(row)
            del frontier, worlds
            _release_cuda()
            print(
                f"    batch {batch_index}/8 complete "
                f"({batch_start:03d}..{batch_start + 63:03d})",
                flush=True,
            )
        del model
        _release_cuda()

    condition_index: dict[tuple[int, str], Gate7ContinuationCondition] = {}
    for checkpoint in GATE7_CONTINUATION_CHECKPOINT_INDICES:
        for condition in plan.conditions:
            row = aggregate_gate7_continuation_condition(
                tuple(batch_rows[(checkpoint, condition)])
            )
            condition_index[(checkpoint, condition)] = row
            tier["conditions"].append(row.to_dict())

    global_score_by_checkpoint = {
        checkpoint: condition_index[(checkpoint, GATE7_CONTINUATION_GLOBAL_SCORE)]
        for checkpoint in GATE7_CONTINUATION_CHECKPOINT_INDICES
    }
    global_hash_by_checkpoint = {
        checkpoint: condition_index[(checkpoint, GATE7_CONTINUATION_GLOBAL_HASH)]
        for checkpoint in GATE7_CONTINUATION_CHECKPOINT_INDICES
    }
    for checkpoint in GATE7_CONTINUATION_CHECKPOINT_INDICES:
        pair = paired_gate7_continuation_summary(
            comparison=f"c{checkpoint}_global_score_vs_global_hash",
            treatment=global_score_by_checkpoint[checkpoint],
            reference=global_hash_by_checkpoint[checkpoint],
        )
        tier["paired_summaries"].append(pair.to_dict())
        print(
            f"  C{checkpoint} global={global_score_by_checkpoint[checkpoint].coverage_rate:.4f} "
            f"hash={global_hash_by_checkpoint[checkpoint].coverage_rate:.4f} "
            f"delta={pair.coverage_delta:+.4f} "
            f"CI=[{pair.bootstrap_ci_low:+.4f},{pair.bootstrap_ci_high:+.4f}]",
            flush=True,
        )

    stratified = stratified_gate7_continuation_global_summary(
        population=population,
        treatment_by_checkpoint=global_score_by_checkpoint,
        reference_by_checkpoint=global_hash_by_checkpoint,
    )
    tier["reference_stratified_summary"] = stratified.to_dict()
    viable = reference_is_viable(
        checkpoint_point_deltas=stratified.checkpoint_point_deltas,
        pooled_ci_low=stratified.bootstrap_ci_low,
    )
    tier["reference_viable"] = viable
    print(
        f"  pooled global-reference delta={stratified.pooled_delta:+.4f} "
        f"CI=[{stratified.bootstrap_ci_low:+.4f},{stratified.bootstrap_ci_high:+.4f}] "
        f"viable={viable}",
        flush=True,
    )

    lows_by_k: dict[int, dict[str, float]] = {}
    for k in GATE7_CONTINUATION_K_LADDER:
        k_lows: dict[str, float] = {}
        for checkpoint in GATE7_CONTINUATION_CHECKPOINT_INDICES:
            score = condition_index[(checkpoint, bounded_score_condition(k))]
            hash_control = condition_index[(checkpoint, bounded_hash_condition(k))]
            learned_pair = paired_gate7_continuation_summary(
                comparison=f"c{checkpoint}_k{k}_score_vs_hash",
                treatment=score,
                reference=hash_control,
            )
            global_pair = paired_gate7_continuation_summary(
                comparison=f"c{checkpoint}_k{k}_score_vs_global",
                treatment=score,
                reference=global_score_by_checkpoint[checkpoint],
            )
            tier["paired_summaries"].extend(
                (learned_pair.to_dict(), global_pair.to_dict())
            )
            k_lows[learned_pair.comparison] = learned_pair.bootstrap_ci_low
            k_lows[global_pair.comparison] = global_pair.bootstrap_ci_low
            print(
                f"  K{k} C{checkpoint} score={score.coverage_rate:.4f} "
                f"hash={hash_control.coverage_rate:.4f} "
                f"learned_CI_low={learned_pair.bootstrap_ci_low:+.4f} "
                f"global_CI_low={global_pair.bootstrap_ci_low:+.4f}",
                flush=True,
            )
        lows_by_k[k] = k_lows

    tier["primary_ci_lows_by_k"] = {
        str(k): lows for k, lows in lows_by_k.items()
    }
    classification = classify_continuation_tier(
        population=population,
        reference_viable=viable,
        primary_ci_lows_by_k=lows_by_k,
    )
    tier["classification"] = asdict(classification)
    tier["passing_k"] = list(classification.passing_k)
    tier["smallest_passing_k"] = classification.smallest_passing_k
    tier["smallest_passing_k_over_n"] = classification.smallest_passing_k_over_n
    tier["tier_outcome"] = classification.outcome
    print(
        f"  tier outcome: {classification.outcome} | "
        f"passing K={list(classification.passing_k)} | "
        f"K_required={classification.smallest_passing_k}",
        flush=True,
    )
    return tier
