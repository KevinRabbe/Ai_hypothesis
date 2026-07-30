"""Run the one admitted fresh low-scale Gate-7 scale-neutral transition bridge.

The runner binds exact transition and original checkpoint identities, executes only depth-10 N128/N256
bridge worlds, and leaves the high-scale Gate-7 population/K ladder closed.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import torch

from .gate6_fixed_k_population_scaling import Gate6SchedulerMode
from .gate7_scale_neutral_transition_bridge import (
    GATE7_TRANSITION_BRIDGE_HIGH_SCALE_OPENED,
    Gate7OriginalCheckpointIdentity,
    load_verified_gate7_transition_checkpoint,
    sha256_file,
)
from .gate7_scale_neutral_transition_bridge_prep import (
    GATE7_TRANSITION_BRIDGE_BATCH_SIZE,
    GATE7_TRANSITION_BRIDGE_BOOTSTRAP_SAMPLES,
    GATE7_TRANSITION_BRIDGE_CHECKPOINT_INDICES,
    GATE7_TRANSITION_BRIDGE_DEPTH,
    GATE7_TRANSITION_BRIDGE_HINT_RELIABILITY,
    GATE7_TRANSITION_BRIDGE_NONINFERIORITY_MARGIN,
    GATE7_TRANSITION_BRIDGE_POPULATIONS,
    GATE7_TRANSITION_BRIDGE_VERSION,
    GATE7_TRANSITION_BRIDGE_WORLD_COUNT,
    Gate7TransitionBridgeCondition,
    Gate7TransitionBridgePairedSummary,
    build_gate7_transition_bridge_pair,
    classify_gate7_scale_neutral_transition_bridge,
    evaluate_gate7_transition_bridge_condition,
)
from .run_gate3_v2_frontier import load_verified_checkpoint


def _condition_dict(row: Gate7TransitionBridgeCondition) -> dict[str, Any]:
    return {
        "checkpoint_index": row.checkpoint_index,
        "checkpoint_family": row.checkpoint_family,
        "population_size": row.population_size,
        "mode": row.mode.value,
        "world_indices": list(row.world_indices),
        "runtime_seeds": list(row.runtime_seeds),
        "covered_by_world": list(row.covered_by_world),
        "coverage_rate": row.coverage_rate,
        "total_learned_updates_per_world": row.total_learned_updates_per_world,
        "stage_a_parent_slots": row.stage_a_parent_slots,
        "stage_b_parent_slots": row.stage_b_parent_slots,
        "learned_parameter_count": row.learned_parameter_count,
        "parameter_fingerprint": row.parameter_fingerprint,
    }


def _summary_dict(row: Gate7TransitionBridgePairedSummary) -> dict[str, Any]:
    return {
        "comparison": row.comparison,
        "checkpoint_index": row.checkpoint_index,
        "population_size": row.population_size,
        "treatment_family": row.treatment_family,
        "reference_family": row.reference_family,
        "treatment_mode": row.treatment_mode.value,
        "reference_mode": row.reference_mode.value,
        "coverage_delta": row.coverage_delta,
        "bootstrap_ci_low": row.bootstrap_ci_low,
        "bootstrap_ci_high": row.bootstrap_ci_high,
    }


def _original_identity(identity: object) -> Gate7OriginalCheckpointIdentity:
    return Gate7OriginalCheckpointIdentity(
        checkpoint_index=int(getattr(identity, "checkpoint_index")),
        checkpoint_sha256=str(getattr(identity, "checkpoint_sha256")),
        parameter_fingerprint=str(getattr(identity, "parameter_fingerprint")),
        learned_parameter_count=int(getattr(identity, "learned_parameter_count")),
    )


def _bar(done: int, total: int, width: int = 30) -> str:
    filled = min(width, max(0, round(width * done / total)))
    return "[" + "#" * filled + "-" * (width - filled) + "]"


def run_gate7_scale_neutral_transition_bridge(
    *,
    output_root: Path,
    transition_checkpoint_paths: tuple[Path, Path, Path],
    original_checkpoint_paths: tuple[Path, Path, Path],
) -> int:
    if output_root.exists():
        raise FileExistsError(f"Gate-7 transition bridge output already exists: {output_root}")
    if not torch.cuda.is_available():
        raise RuntimeError("admitted Gate-7 transition bridge requires CUDA")
    if GATE7_TRANSITION_BRIDGE_HIGH_SCALE_OPENED:
        raise RuntimeError("high-scale Gate-7 must remain closed during the transition bridge")

    output_root = output_root.resolve()
    output_root.mkdir(parents=True)
    started = time.monotonic()
    conditions: list[Gate7TransitionBridgeCondition] = []
    summaries: list[Gate7TransitionBridgePairedSummary] = []
    transition_identities = []
    original_identities = []
    total_cells = len(GATE7_TRANSITION_BRIDGE_CHECKPOINT_INDICES) * 7
    completed = 0

    print("Gate-7 scale-neutral transition bridge — FRESH LOW-SCALE EVIDENCE", flush=True)
    print(f"CUDA device: {torch.cuda.get_device_name(0)}", flush=True)
    print("Training: NONE — exact bound transition/original checkpoints only", flush=True)
    print("Worlds: fresh bridge-only depth-10 namespace", flush=True)
    print("Populations: N128 / N256", flush=True)
    print("Modes: K16 learned / matched K16 hash / global", flush=True)
    print("Gate-7 high-scale population/K ladder: CLOSED", flush=True)

    for checkpoint_index, transition_path, original_path in zip(
        GATE7_TRANSITION_BRIDGE_CHECKPOINT_INDICES,
        transition_checkpoint_paths,
        original_checkpoint_paths,
        strict=True,
    ):
        transition_model, transition_identity = load_verified_gate7_transition_checkpoint(
            checkpoint_index=checkpoint_index,
            checkpoint_path=transition_path,
            device="cuda",
        )
        original_model, gate3_identity = load_verified_checkpoint(
            checkpoint_index=checkpoint_index,
            checkpoint_path=original_path,
            device="cuda",
        )
        original_identity = _original_identity(gate3_identity)
        transition_identities.append(transition_identity)
        original_identities.append(original_identity)
        print(
            f"T{checkpoint_index}: SHA256={transition_identity.checkpoint_sha256} "
            f"fingerprint={transition_identity.parameter_fingerprint}",
            flush=True,
        )
        print(
            f"O{checkpoint_index}: SHA256={original_identity.checkpoint_sha256} "
            f"fingerprint={original_identity.parameter_fingerprint}",
            flush=True,
        )

        cell: dict[tuple[str, int, Gate6SchedulerMode], Gate7TransitionBridgeCondition] = {}
        matrix = (
            ("transition", 128, Gate6SchedulerMode.GLOBAL_SCORE, transition_model),
            ("transition", 128, Gate6SchedulerMode.BOUNDED_SCORE_K16, transition_model),
            ("transition", 128, Gate6SchedulerMode.BOUNDED_HASH_K16, transition_model),
            ("transition", 256, Gate6SchedulerMode.GLOBAL_SCORE, transition_model),
            ("transition", 256, Gate6SchedulerMode.BOUNDED_SCORE_K16, transition_model),
            ("transition", 256, Gate6SchedulerMode.BOUNDED_HASH_K16, transition_model),
            ("original", 256, Gate6SchedulerMode.GLOBAL_SCORE, original_model),
        )
        with torch.no_grad():
            for family, population, mode, model in matrix:
                condition = evaluate_gate7_transition_bridge_condition(
                    model,
                    checkpoint_index=checkpoint_index,
                    checkpoint_family=family,
                    population_size=population,
                    mode=mode,
                    device="cuda",
                )
                conditions.append(condition)
                cell[(family, population, mode)] = condition
                completed += 1
                elapsed = (time.monotonic() - started) / 60.0
                print(
                    f"EVAL {_bar(completed, total_cells)} {100.0 * completed / total_cells:6.2f}% "
                    f"{completed:2d}/{total_cells} C{checkpoint_index} {family:<10s} "
                    f"N={population:<3d} {mode.value:<20s} coverage={condition.coverage_rate:.4f} "
                    f"elapsed={elapsed:.1f}m",
                    flush=True,
                )

        pair_specs = (
            (
                f"t{checkpoint_index}_n128_k16_vs_hash",
                cell[("transition", 128, Gate6SchedulerMode.BOUNDED_SCORE_K16)],
                cell[("transition", 128, Gate6SchedulerMode.BOUNDED_HASH_K16)],
            ),
            (
                f"t{checkpoint_index}_n256_k16_vs_hash",
                cell[("transition", 256, Gate6SchedulerMode.BOUNDED_SCORE_K16)],
                cell[("transition", 256, Gate6SchedulerMode.BOUNDED_HASH_K16)],
            ),
            (
                f"t{checkpoint_index}_n128_k16_vs_global",
                cell[("transition", 128, Gate6SchedulerMode.BOUNDED_SCORE_K16)],
                cell[("transition", 128, Gate6SchedulerMode.GLOBAL_SCORE)],
            ),
            (
                f"t{checkpoint_index}_n256_transition_global_vs_original_global",
                cell[("transition", 256, Gate6SchedulerMode.GLOBAL_SCORE)],
                cell[("original", 256, Gate6SchedulerMode.GLOBAL_SCORE)],
            ),
            (
                f"t{checkpoint_index}_n256_k16_vs_global",
                cell[("transition", 256, Gate6SchedulerMode.BOUNDED_SCORE_K16)],
                cell[("transition", 256, Gate6SchedulerMode.GLOBAL_SCORE)],
            ),
        )
        summaries.extend(
            build_gate7_transition_bridge_pair(
                comparison=comparison,
                treatment=treatment,
                reference=reference,
            )
            for comparison, treatment, reference in pair_specs
        )
        del transition_model
        del original_model

    primary_lows = {
        row.comparison: row.bootstrap_ci_low
        for row in summaries
        if not row.comparison.endswith("_n256_k16_vs_global")
    }
    outcome = classify_gate7_scale_neutral_transition_bridge(primary_lows)
    result = {
        "experiment_version": GATE7_TRANSITION_BRIDGE_VERSION,
        "scientific_status": "FRESH_LOW_SCALE_TRANSITION_BRIDGE_EVIDENCE",
        "transition_outcome": outcome,
        "training_performed": False,
        "checkpoint_selection_performed": False,
        "high_scale_gate7_opened": False,
        "world_count": GATE7_TRANSITION_BRIDGE_WORLD_COUNT,
        "evaluation_batch_size": GATE7_TRANSITION_BRIDGE_BATCH_SIZE,
        "bootstrap_samples": GATE7_TRANSITION_BRIDGE_BOOTSTRAP_SAMPLES,
        "depth": GATE7_TRANSITION_BRIDGE_DEPTH,
        "hint_reliability": GATE7_TRANSITION_BRIDGE_HINT_RELIABILITY,
        "populations": list(GATE7_TRANSITION_BRIDGE_POPULATIONS),
        "noninferiority_margin": GATE7_TRANSITION_BRIDGE_NONINFERIORITY_MARGIN,
        "transition_checkpoints": [identity.to_dict() for identity in transition_identities],
        "original_checkpoints": [identity.to_dict() for identity in original_identities],
        "conditions": [_condition_dict(row) for row in conditions],
        "paired_summaries": [_summary_dict(row) for row in summaries],
        "primary_ci_lows": primary_lows,
    }
    result_path = output_root / "gate7-scale-neutral-transition-bridge.json"
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    runtime = {
        "scientific_status": result["scientific_status"],
        "transition_outcome": outcome,
        "high_scale_gate7_opened": False,
        "training_performed": False,
        "torch_version": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cuda_device_name": torch.cuda.get_device_name(0),
        "wall_seconds": time.monotonic() - started,
        "result_sha256": sha256_file(result_path),
        "transition_checkpoint_paths": [str(path.resolve()) for path in transition_checkpoint_paths],
        "original_checkpoint_paths": [str(path.resolve()) for path in original_checkpoint_paths],
    }
    runtime_path = output_root / "runtime.json"
    runtime_path.write_text(json.dumps(runtime, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "status": "GATE7_SCALE_NEUTRAL_TRANSITION_BRIDGE_COMPLETE",
                "transition_outcome": outcome,
                "high_scale_gate7_opened": False,
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
    parser.add_argument("--original-checkpoint0", type=Path, required=True)
    parser.add_argument("--original-checkpoint1", type=Path, required=True)
    parser.add_argument("--original-checkpoint2", type=Path, required=True)
    args = parser.parse_args()
    return run_gate7_scale_neutral_transition_bridge(
        output_root=args.output_root,
        transition_checkpoint_paths=(
            args.transition_checkpoint0,
            args.transition_checkpoint1,
            args.transition_checkpoint2,
        ),
        original_checkpoint_paths=(
            args.original_checkpoint0,
            args.original_checkpoint1,
            args.original_checkpoint2,
        ),
    )


if __name__ == "__main__":
    raise SystemExit(main())
