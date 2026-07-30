"""Run the one admitted Gate-7 high-scale routing-bandwidth screening campaign."""

from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path
from typing import Any

import torch

from .gate7_high_scale_routing_bandwidth import (
    GATE7_HIGH_SCALE_ENGINEERING_MANIFEST_SHA256,
    GATE7_HIGH_SCALE_ENGINEERING_RESULT_HEAD,
    GATE7_HIGH_SCALE_ENGINEERING_SUMMARY_SHA256,
    GATE7_HIGH_SCALE_EXECUTION_ADMITTED,
    GATE7_HIGH_SCALE_SCIENTIFIC_STATUS,
    Gate7HighScaleCondition,
    Gate7HighScalePairedSummary,
    build_gate7_high_scale_scientific_frontier,
    evaluate_gate7_high_scale_condition,
    generate_gate7_high_scale_world,
    load_verified_gate7_high_scale_checkpoint,
    paired_gate7_high_scale_summary,
    stratified_gate7_high_scale_global_summary,
)
from .gate7_high_scale_routing_bandwidth_protocol import (
    GATE7_HIGH_SCALE_BOOTSTRAP_SAMPLES,
    GATE7_HIGH_SCALE_CAMPAIGN_CEILING_REACHED,
    GATE7_HIGH_SCALE_CHECKPOINT_INDICES,
    GATE7_HIGH_SCALE_CONTINUE,
    GATE7_HIGH_SCALE_EVALUATION_BATCH_SIZE,
    GATE7_HIGH_SCALE_GLOBAL_HASH,
    GATE7_HIGH_SCALE_GLOBAL_SCORE,
    GATE7_HIGH_SCALE_HINT_RELIABILITY,
    GATE7_HIGH_SCALE_K_LADDER,
    GATE7_HIGH_SCALE_NONINFERIORITY_MARGIN,
    GATE7_HIGH_SCALE_POPULATIONS,
    GATE7_HIGH_SCALE_RESOURCE_FRONTIER_REACHED,
    GATE7_HIGH_SCALE_STAGE_B_PARENT_SLOTS,
    GATE7_HIGH_SCALE_VERSION,
    GATE7_HIGH_SCALE_WORLD_COUNT,
    bounded_hash_condition,
    bounded_score_condition,
    build_gate7_high_scale_tier_plan,
    campaign_action_after_tier,
    classify_completed_tier,
    reference_is_viable,
)
from .gate7_scale_neutral_transition_bridge import sha256_file


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _condition_dict(row: Gate7HighScaleCondition) -> dict[str, Any]:
    return row.to_dict()


def _pair_dict(row: Gate7HighScalePairedSummary) -> dict[str, Any]:
    return row.to_dict()


def _bar(done: int, total: int, width: int = 28) -> str:
    filled = min(width, max(0, round(width * done / total)))
    return "[" + "#" * filled + "-" * (width - filled) + "]"


def _checkpoint_paths(args: argparse.Namespace) -> tuple[Path, Path, Path]:
    return (
        args.transition_checkpoint0,
        args.transition_checkpoint1,
        args.transition_checkpoint2,
    )


def _release(*objects: object) -> None:
    del objects
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def run_gate7_high_scale_routing_bandwidth(
    *,
    output_root: Path,
    transition_checkpoint_paths: tuple[Path, Path, Path],
) -> int:
    if output_root.exists():
        raise FileExistsError(f"Gate-7 high-scale output already exists: {output_root}")
    if not torch.cuda.is_available():
        raise RuntimeError("admitted Gate-7 high-scale screening requires CUDA")
    if not GATE7_HIGH_SCALE_EXECUTION_ADMITTED:
        raise RuntimeError("Gate-7 high-scale scientific execution is not admitted")

    output_root = output_root.resolve()
    output_root.mkdir(parents=True)
    result_path = output_root / "gate7-high-scale-routing-bandwidth.json"
    runtime_path = output_root / "runtime.json"
    started = time.monotonic()

    checkpoint_identities = []
    for checkpoint, path in zip(
        GATE7_HIGH_SCALE_CHECKPOINT_INDICES,
        transition_checkpoint_paths,
        strict=True,
    ):
        model, identity = load_verified_gate7_high_scale_checkpoint(
            checkpoint_index=checkpoint,
            checkpoint_path=path,
            device="cpu",
        )
        checkpoint_identities.append(identity)
        del model

    result: dict[str, Any] = {
        "experiment_version": GATE7_HIGH_SCALE_VERSION,
        "scientific_status": GATE7_HIGH_SCALE_SCIENTIFIC_STATUS,
        "execution_admitted": True,
        "high_scale_gate7_screening_opened": True,
        "confirmation_opened": False,
        "training_performed": False,
        "checkpoint_selection_performed": False,
        "engineering_prerequisite_head": GATE7_HIGH_SCALE_ENGINEERING_RESULT_HEAD,
        "engineering_summary_sha256": GATE7_HIGH_SCALE_ENGINEERING_SUMMARY_SHA256,
        "engineering_manifest_sha256": GATE7_HIGH_SCALE_ENGINEERING_MANIFEST_SHA256,
        "world_count_per_checkpoint_tier": GATE7_HIGH_SCALE_WORLD_COUNT,
        "evaluation_batch_size": GATE7_HIGH_SCALE_EVALUATION_BATCH_SIZE,
        "bootstrap_samples": GATE7_HIGH_SCALE_BOOTSTRAP_SAMPLES,
        "hint_reliability": GATE7_HIGH_SCALE_HINT_RELIABILITY,
        "noninferiority_margin": GATE7_HIGH_SCALE_NONINFERIORITY_MARGIN,
        "populations": list(GATE7_HIGH_SCALE_POPULATIONS),
        "k_ladder": list(GATE7_HIGH_SCALE_K_LADDER),
        "stage_b_parent_slots": GATE7_HIGH_SCALE_STAGE_B_PARENT_SLOTS,
        "compiler_enabled": False,
        "cuda_graphs_enabled": False,
        "mixed_precision_enabled": False,
        "transition_checkpoints": [identity.to_dict() for identity in checkpoint_identities],
        "tiers": [],
        "campaign_outcome": "RUNNING",
    }
    _write_json(result_path, result)

    print("Gate-7 high-scale routing-bandwidth screening — FRESH SCIENTIFIC EVIDENCE", flush=True)
    print(f"CUDA device: {torch.cuda.get_device_name(0)}", flush=True)
    print("Training/checkpoint selection: NONE", flush=True)
    print("Worlds: fresh population-specific high-scale namespace", flush=True)
    print(f"Populations: {GATE7_HIGH_SCALE_POPULATIONS}", flush=True)
    print(f"K ladder: {GATE7_HIGH_SCALE_K_LADDER}", flush=True)
    print("Physical world batch: 64", flush=True)
    print("Confirmation: CLOSED", flush=True)

    tiers: list[dict[str, Any]] = []
    campaign_outcome = "RUNNING"
    resource_error: str | None = None

    for tier_index, population in enumerate(GATE7_HIGH_SCALE_POPULATIONS, start=1):
        plan = build_gate7_high_scale_tier_plan(population)
        worlds = tuple(
            generate_gate7_high_scale_world(population=population, world_index=world_index)
            for world_index in range(GATE7_HIGH_SCALE_WORLD_COUNT)
        )
        tier: dict[str, Any] = {
            "population": population,
            "frontier_depth": plan.frontier_depth,
            "world_depth": plan.world_depth,
            "world_indices": [world.world_index for world in worlds],
            "runtime_seeds": [world.runtime_seed for world in worlds],
            "logical_stage_a_parent_slots": plan.stage_a_parent_slots,
            "logical_stage_b_parent_slots": plan.stage_b_parent_slots,
            "logical_learned_updates_per_world": plan.total_logical_learned_updates,
            "conditions": [],
            "paired_summaries": [],
            "frontier_builds": [],
            "tested_k": [],
            "unexposed_k": [],
            "reference_viable": None,
            "reference_stratified_summary": None,
            "k_required": None,
            "k_required_over_n": None,
            "tier_outcome": "RUNNING",
        }
        print(
            f"\nN={population} ({tier_index}/{len(GATE7_HIGH_SCALE_POPULATIONS)}) "
            "opening global reference pair...",
            flush=True,
        )

        condition_index: dict[tuple[int, str], Gate7HighScaleCondition] = {}
        pair_rows: list[Gate7HighScalePairedSummary] = []
        global_score_by_checkpoint: dict[int, Gate7HighScaleCondition] = {}
        global_hash_by_checkpoint: dict[int, Gate7HighScaleCondition] = {}

        try:
            for checkpoint, path in zip(
                GATE7_HIGH_SCALE_CHECKPOINT_INDICES,
                transition_checkpoint_paths,
                strict=True,
            ):
                model, _ = load_verified_gate7_high_scale_checkpoint(
                    checkpoint_index=checkpoint,
                    checkpoint_path=path,
                    device="cuda",
                )
                frontier, frontier_metrics = build_gate7_high_scale_scientific_frontier(
                    model,
                    worlds=worlds,
                    device="cuda",
                )
                tier["frontier_builds"].append(
                    {
                        "checkpoint_index": checkpoint,
                        "phase": "global_reference",
                        **frontier_metrics,
                    }
                )
                score = evaluate_gate7_high_scale_condition(
                    model,
                    frontier,
                    checkpoint_index=checkpoint,
                    worlds=worlds,
                    condition=GATE7_HIGH_SCALE_GLOBAL_SCORE,
                )
                hash_control = evaluate_gate7_high_scale_condition(
                    model,
                    frontier,
                    checkpoint_index=checkpoint,
                    worlds=worlds,
                    condition=GATE7_HIGH_SCALE_GLOBAL_HASH,
                )
                global_score_by_checkpoint[checkpoint] = score
                global_hash_by_checkpoint[checkpoint] = hash_control
                condition_index[(checkpoint, score.condition)] = score
                condition_index[(checkpoint, hash_control.condition)] = hash_control
                tier["conditions"].extend((_condition_dict(score), _condition_dict(hash_control)))
                pair = paired_gate7_high_scale_summary(
                    comparison=f"c{checkpoint}_global_score_vs_global_hash",
                    treatment=score,
                    reference=hash_control,
                )
                pair_rows.append(pair)
                tier["paired_summaries"].append(_pair_dict(pair))
                print(
                    f"  C{checkpoint} global={score.coverage_rate:.4f} "
                    f"hash={hash_control.coverage_rate:.4f} "
                    f"delta={pair.coverage_delta:+.4f} "
                    f"CI=[{pair.bootstrap_ci_low:+.4f},{pair.bootstrap_ci_high:+.4f}]",
                    flush=True,
                )
                del frontier, model
                _release()

            stratified = stratified_gate7_high_scale_global_summary(
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

            primary_ci_lows_by_k: dict[int, dict[str, float]] = {}
            if not viable:
                tier_outcome = classify_completed_tier(
                    population=population,
                    reference_viable=False,
                    primary_ci_lows_by_k={},
                )
            else:
                tier_outcome = "G7_SCREENING_INCOMPLETE"
                for k_position, k in enumerate(GATE7_HIGH_SCALE_K_LADDER, start=1):
                    print(
                        f"  K{k} ({k_position}/{len(GATE7_HIGH_SCALE_K_LADDER)}) "
                        "opening matched score/hash pair...",
                        flush=True,
                    )
                    k_lows: dict[str, float] = {}
                    for checkpoint, path in zip(
                        GATE7_HIGH_SCALE_CHECKPOINT_INDICES,
                        transition_checkpoint_paths,
                        strict=True,
                    ):
                        model, _ = load_verified_gate7_high_scale_checkpoint(
                            checkpoint_index=checkpoint,
                            checkpoint_path=path,
                            device="cuda",
                        )
                        frontier, frontier_metrics = build_gate7_high_scale_scientific_frontier(
                            model,
                            worlds=worlds,
                            device="cuda",
                        )
                        tier["frontier_builds"].append(
                            {
                                "checkpoint_index": checkpoint,
                                "phase": f"k{k}",
                                **frontier_metrics,
                            }
                        )
                        score_condition_name = bounded_score_condition(k)
                        hash_condition_name = bounded_hash_condition(k)
                        score_k = evaluate_gate7_high_scale_condition(
                            model,
                            frontier,
                            checkpoint_index=checkpoint,
                            worlds=worlds,
                            condition=score_condition_name,
                        )
                        hash_k = evaluate_gate7_high_scale_condition(
                            model,
                            frontier,
                            checkpoint_index=checkpoint,
                            worlds=worlds,
                            condition=hash_condition_name,
                        )
                        condition_index[(checkpoint, score_condition_name)] = score_k
                        condition_index[(checkpoint, hash_condition_name)] = hash_k
                        tier["conditions"].extend((_condition_dict(score_k), _condition_dict(hash_k)))
                        learned_pair = paired_gate7_high_scale_summary(
                            comparison=f"c{checkpoint}_k{k}_score_vs_hash",
                            treatment=score_k,
                            reference=hash_k,
                        )
                        global_pair = paired_gate7_high_scale_summary(
                            comparison=f"c{checkpoint}_k{k}_score_vs_global",
                            treatment=score_k,
                            reference=global_score_by_checkpoint[checkpoint],
                        )
                        pair_rows.extend((learned_pair, global_pair))
                        tier["paired_summaries"].extend(
                            (_pair_dict(learned_pair), _pair_dict(global_pair))
                        )
                        k_lows[learned_pair.comparison] = learned_pair.bootstrap_ci_low
                        k_lows[global_pair.comparison] = global_pair.bootstrap_ci_low
                        print(
                            f"    C{checkpoint} score={score_k.coverage_rate:.4f} "
                            f"hash={hash_k.coverage_rate:.4f} "
                            f"learned_CI_low={learned_pair.bootstrap_ci_low:+.4f} "
                            f"global_CI_low={global_pair.bootstrap_ci_low:+.4f}",
                            flush=True,
                        )
                        del frontier, model
                        _release()

                    tier["tested_k"].append(k)
                    primary_ci_lows_by_k[k] = k_lows
                    tier_outcome = classify_completed_tier(
                        population=population,
                        reference_viable=True,
                        primary_ci_lows_by_k=primary_ci_lows_by_k,
                    )
                    if tier_outcome.startswith("G7_K_REQUIRED_"):
                        break

            tier["tier_outcome"] = tier_outcome
            if tier_outcome.startswith("G7_K_REQUIRED_"):
                k_required = int(tier_outcome.rsplit("_", 1)[1])
                tier["k_required"] = k_required
                tier["k_required_over_n"] = k_required / population
            tested_set = set(tier["tested_k"])
            first_pass = tier["k_required"]
            tier["unexposed_k"] = [
                {
                    "k": k,
                    "status": "NOT_RUN_BY_FIRST_PASS_RULE"
                    if first_pass is not None and k > first_pass
                    else "NOT_RUN_AFTER_TIER_STOP",
                }
                for k in GATE7_HIGH_SCALE_K_LADDER
                if k not in tested_set
            ]
            campaign_outcome = campaign_action_after_tier(
                population=population,
                tier_outcome=tier_outcome,
            )
        except torch.OutOfMemoryError as exc:
            resource_error = str(exc)
            tier["tier_outcome"] = GATE7_HIGH_SCALE_RESOURCE_FRONTIER_REACHED
            tier["resource_error"] = resource_error
            tier["unexposed_k"] = [
                {"k": k, "status": "NOT_RUN_AFTER_RESOURCE_STOP"}
                for k in GATE7_HIGH_SCALE_K_LADDER
                if k not in set(tier["tested_k"])
            ]
            campaign_outcome = GATE7_HIGH_SCALE_RESOURCE_FRONTIER_REACHED
            _release()

        tiers.append(tier)
        result["tiers"] = tiers
        result["campaign_outcome"] = campaign_outcome
        _write_json(result_path, result)
        print(
            f"  tier outcome: {tier['tier_outcome']} | campaign action: {campaign_outcome}",
            flush=True,
        )
        if campaign_outcome != GATE7_HIGH_SCALE_CONTINUE:
            break

    if campaign_outcome == "RUNNING":
        campaign_outcome = GATE7_HIGH_SCALE_CAMPAIGN_CEILING_REACHED
    result["campaign_outcome"] = campaign_outcome
    result["resource_error"] = resource_error
    result["tiers"] = tiers
    _write_json(result_path, result)

    runtime = {
        "scientific_status": GATE7_HIGH_SCALE_SCIENTIFIC_STATUS,
        "campaign_outcome": campaign_outcome,
        "high_scale_gate7_screening_opened": True,
        "confirmation_opened": False,
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
                "status": "GATE7_HIGH_SCALE_ROUTING_BANDWIDTH_SCREENING_COMPLETE",
                "campaign_outcome": campaign_outcome,
                "confirmation_opened": False,
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
    return run_gate7_high_scale_routing_bandwidth(
        output_root=args.output_root,
        transition_checkpoint_paths=_checkpoint_paths(args),
    )


if __name__ == "__main__":
    raise SystemExit(main())
