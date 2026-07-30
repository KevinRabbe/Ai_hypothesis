"""Run the one admitted Gate-7 routing-bandwidth frontier confirmation matrix."""

from __future__ import annotations

import argparse
import gc
import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

from .gate7_high_scale_routing_bandwidth_confirmation import (
    GATE7_CONFIRMATION_EXECUTION_ADMITTED,
    GATE7_CONFIRMATION_SCIENTIFIC_STATUS,
    Gate7ConfirmationBatchCondition,
    Gate7ConfirmationCondition,
    aggregate_gate7_confirmation_condition,
    build_gate7_confirmation_frontier,
    confirmation_provenance,
    confirmation_world_batch,
    evaluate_gate7_confirmation_batch_condition,
    load_verified_gate7_confirmation_checkpoint,
    paired_gate7_confirmation_summary,
    stratified_gate7_confirmation_global_summary,
)
from .gate7_high_scale_routing_bandwidth_confirmation_protocol import (
    GATE7_CONFIRMATION_ANCHOR_K,
    GATE7_CONFIRMATION_ANCHOR_POPULATION,
    GATE7_CONFIRMATION_BOOTSTRAP_SAMPLES,
    GATE7_CONFIRMATION_CHECKPOINT_INDICES,
    GATE7_CONFIRMATION_EVALUATION_BATCH_SIZE,
    GATE7_CONFIRMATION_FRONTIER_POPULATION,
    GATE7_CONFIRMATION_GLOBAL_HASH,
    GATE7_CONFIRMATION_GLOBAL_SCORE,
    GATE7_CONFIRMATION_HINT_RELIABILITY,
    GATE7_CONFIRMATION_K_LADDER,
    GATE7_CONFIRMATION_NONINFERIORITY_MARGIN,
    GATE7_CONFIRMATION_POPULATIONS,
    GATE7_CONFIRMATION_STAGE_B_PARENT_SLOTS,
    GATE7_CONFIRMATION_VERSION,
    GATE7_CONFIRMATION_WORLD_COUNT,
    bounded_hash_condition,
    bounded_score_condition,
    build_confirmation_tier_plan,
    classify_confirmation,
    k_passes_all_checkpoints,
    reference_is_viable,
)
from .gate7_scale_neutral_transition_bridge import sha256_file


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _release_cuda() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _checkpoint_paths(args: argparse.Namespace) -> tuple[Path, Path, Path]:
    return (
        args.transition_checkpoint0,
        args.transition_checkpoint1,
        args.transition_checkpoint2,
    )


def run_gate7_high_scale_routing_bandwidth_confirmation(
    *,
    output_root: Path,
    transition_checkpoint_paths: tuple[Path, Path, Path],
) -> int:
    if output_root.exists():
        raise FileExistsError(f"Gate-7 confirmation output already exists: {output_root}")
    if not torch.cuda.is_available():
        raise RuntimeError("admitted Gate-7 confirmation requires CUDA")
    if not GATE7_CONFIRMATION_EXECUTION_ADMITTED:
        raise RuntimeError("Gate-7 confirmation execution is not admitted")

    output_root = output_root.resolve()
    output_root.mkdir(parents=True)
    result_path = output_root / "gate7-high-scale-routing-bandwidth-confirmation.json"
    runtime_path = output_root / "runtime.json"
    started = time.monotonic()

    checkpoint_identities = []
    for checkpoint, path in zip(
        GATE7_CONFIRMATION_CHECKPOINT_INDICES,
        transition_checkpoint_paths,
        strict=True,
    ):
        model, identity = load_verified_gate7_confirmation_checkpoint(
            checkpoint_index=checkpoint,
            checkpoint_path=path,
            device="cpu",
        )
        checkpoint_identities.append(identity)
        del model

    result: dict[str, Any] = {
        "experiment_version": GATE7_CONFIRMATION_VERSION,
        "scientific_status": GATE7_CONFIRMATION_SCIENTIFIC_STATUS,
        "execution_admitted": True,
        "confirmation_opened": True,
        "second_confirmation_opened": False,
        "training_performed": False,
        "checkpoint_selection_performed": False,
        **confirmation_provenance(),
        "hint_reliability": GATE7_CONFIRMATION_HINT_RELIABILITY,
        "noninferiority_margin": GATE7_CONFIRMATION_NONINFERIORITY_MARGIN,
        "populations": list(GATE7_CONFIRMATION_POPULATIONS),
        "k_ladder": list(GATE7_CONFIRMATION_K_LADDER),
        "compiler_enabled": False,
        "cuda_graphs_enabled": False,
        "mixed_precision_enabled": False,
        "transition_checkpoints": [identity.to_dict() for identity in checkpoint_identities],
        "tiers": [],
        "confirmation_classification": None,
        "confirmation_outcome": "RUNNING",
    }
    _write_json(result_path, result)

    print("Gate-7 routing-bandwidth frontier confirmation — FRESH CONFIRMATION EVIDENCE", flush=True)
    print(f"CUDA device: {torch.cuda.get_device_name(0)}", flush=True)
    print("Training/checkpoint selection: NONE", flush=True)
    print("Worlds: untouched confirmation namespace 0..511", flush=True)
    print(f"Populations: {GATE7_CONFIRMATION_POPULATIONS}", flush=True)
    print("N4096 matrix: global/hash + K512/hash", flush=True)
    print(f"N8192 matrix: global/hash + complete K ladder {GATE7_CONFIRMATION_K_LADDER}", flush=True)
    print("Physical world batch: 64 x 8", flush=True)
    print("Adaptive exposure: NONE", flush=True)
    print("Second confirmation: CLOSED", flush=True)

    tiers: list[dict[str, Any]] = []
    anchor_reference_viable: bool | None = None
    anchor_k512_lows: dict[str, float] | None = None
    frontier_reference_viable: bool | None = None
    frontier_lows_by_k: dict[int, dict[str, float]] | None = None

    for population in GATE7_CONFIRMATION_POPULATIONS:
        plan = build_confirmation_tier_plan(population)
        print(
            f"\nN={population} fixed confirmation matrix: {len(plan.conditions)} conditions/checkpoint...",
            flush=True,
        )
        tier: dict[str, Any] = {
            "population": population,
            "world_indices": list(range(GATE7_CONFIRMATION_WORLD_COUNT)),
            "runtime_seeds": [],
            "world_count": GATE7_CONFIRMATION_WORLD_COUNT,
            "evaluation_batch_size": GATE7_CONFIRMATION_EVALUATION_BATCH_SIZE,
            "physical_batch_count": GATE7_CONFIRMATION_WORLD_COUNT
            // GATE7_CONFIRMATION_EVALUATION_BATCH_SIZE,
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
        }

        batch_rows: dict[tuple[int, str], list[Gate7ConfirmationBatchCondition]] = {
            (checkpoint, condition): []
            for checkpoint in GATE7_CONFIRMATION_CHECKPOINT_INDICES
            for condition in plan.conditions
        }

        for checkpoint, path in zip(
            GATE7_CONFIRMATION_CHECKPOINT_INDICES,
            transition_checkpoint_paths,
            strict=True,
        ):
            model, _ = load_verified_gate7_confirmation_checkpoint(
                checkpoint_index=checkpoint,
                checkpoint_path=path,
                device="cuda",
            )
            print(f"  C{checkpoint}: eight B64 frontier batches", flush=True)
            for batch_index, batch_start in enumerate(
                range(0, GATE7_CONFIRMATION_WORLD_COUNT, GATE7_CONFIRMATION_EVALUATION_BATCH_SIZE),
                start=1,
            ):
                worlds = confirmation_world_batch(
                    population=population,
                    batch_start=batch_start,
                )
                if checkpoint == GATE7_CONFIRMATION_CHECKPOINT_INDICES[0]:
                    tier["runtime_seeds"].extend(world.runtime_seed for world in worlds)
                frontier, frontier_metrics = build_gate7_confirmation_frontier(
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
                    row = evaluate_gate7_confirmation_batch_condition(
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
                    f"    batch {batch_index}/8 complete ({batch_start:03d}..{batch_start + 63:03d})",
                    flush=True,
                )
            del model
            _release_cuda()

        condition_index: dict[tuple[int, str], Gate7ConfirmationCondition] = {}
        for checkpoint in GATE7_CONFIRMATION_CHECKPOINT_INDICES:
            for condition in plan.conditions:
                row = aggregate_gate7_confirmation_condition(
                    tuple(batch_rows[(checkpoint, condition)])
                )
                condition_index[(checkpoint, condition)] = row
                tier["conditions"].append(row.to_dict())

        global_score_by_checkpoint = {
            checkpoint: condition_index[(checkpoint, GATE7_CONFIRMATION_GLOBAL_SCORE)]
            for checkpoint in GATE7_CONFIRMATION_CHECKPOINT_INDICES
        }
        global_hash_by_checkpoint = {
            checkpoint: condition_index[(checkpoint, GATE7_CONFIRMATION_GLOBAL_HASH)]
            for checkpoint in GATE7_CONFIRMATION_CHECKPOINT_INDICES
        }
        for checkpoint in GATE7_CONFIRMATION_CHECKPOINT_INDICES:
            pair = paired_gate7_confirmation_summary(
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

        stratified = stratified_gate7_confirmation_global_summary(
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
        for k in plan.k_values:
            k_lows: dict[str, float] = {}
            for checkpoint in GATE7_CONFIRMATION_CHECKPOINT_INDICES:
                score = condition_index[(checkpoint, bounded_score_condition(k))]
                hash_control = condition_index[(checkpoint, bounded_hash_condition(k))]
                learned_pair = paired_gate7_confirmation_summary(
                    comparison=f"c{checkpoint}_k{k}_score_vs_hash",
                    treatment=score,
                    reference=hash_control,
                )
                global_pair = paired_gate7_confirmation_summary(
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
        passing = [
            k
            for k in plan.k_values
            if k_passes_all_checkpoints(k=k, primary_ci_lows=lows_by_k[k])
        ]
        tier["passing_k"] = passing
        tiers.append(tier)
        result["tiers"] = tiers
        _write_json(result_path, result)

        if population == GATE7_CONFIRMATION_ANCHOR_POPULATION:
            anchor_reference_viable = viable
            anchor_k512_lows = lows_by_k[GATE7_CONFIRMATION_ANCHOR_K]
        elif population == GATE7_CONFIRMATION_FRONTIER_POPULATION:
            frontier_reference_viable = viable
            frontier_lows_by_k = lows_by_k

    if anchor_reference_viable is None or anchor_k512_lows is None:
        raise RuntimeError("confirmation anchor tier did not complete")
    if frontier_reference_viable is None or frontier_lows_by_k is None:
        raise RuntimeError("confirmation frontier tier did not complete")

    classification = classify_confirmation(
        anchor_reference_viable=anchor_reference_viable,
        anchor_k512_primary_ci_lows=anchor_k512_lows,
        frontier_reference_viable=frontier_reference_viable,
        frontier_primary_ci_lows_by_k=frontier_lows_by_k,
    )
    result["confirmation_classification"] = asdict(classification)
    result["confirmation_outcome"] = classification.outcome
    result["tiers"] = tiers
    _write_json(result_path, result)

    runtime = {
        "scientific_status": GATE7_CONFIRMATION_SCIENTIFIC_STATUS,
        "confirmation_outcome": classification.outcome,
        "confirmation_opened": True,
        "second_confirmation_opened": False,
        "training_performed": False,
        "checkpoint_selection_performed": False,
        "torch_version": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cuda_device_name": torch.cuda.get_device_name(0),
        "wall_seconds": time.monotonic() - started,
        "result_sha256": sha256_file(result_path),
        "transition_checkpoint_paths": [str(path.resolve()) for path in transition_checkpoint_paths],
    }
    _write_json(runtime_path, runtime)
    print(
        json.dumps(
            {
                "status": "GATE7_ROUTING_BANDWIDTH_CONFIRMATION_COMPLETE",
                "confirmation_outcome": classification.outcome,
                "passing_k_at_n8192": list(classification.passing_k_at_n8192),
                "second_confirmation_opened": False,
                "result": str(result_path),
                "result_sha256": runtime["result_sha256"],
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--transition-checkpoint0", type=Path, required=True)
    parser.add_argument("--transition-checkpoint1", type=Path, required=True)
    parser.add_argument("--transition-checkpoint2", type=Path, required=True)
    args = parser.parse_args()
    return run_gate7_high_scale_routing_bandwidth_confirmation(
        output_root=args.output_root,
        transition_checkpoint_paths=_checkpoint_paths(args),
    )


if __name__ == "__main__":
    raise SystemExit(main())
