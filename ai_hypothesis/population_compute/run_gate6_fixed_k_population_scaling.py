"""Run the frozen Gate-6 v0 fixed-K population-scaling development matrix."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import torch

from .gate6_fixed_k_population_scaling import (
    GATE6_BOOTSTRAP_SAMPLES,
    GATE6_CHECKPOINT_INDICES,
    GATE6_CONDITIONS,
    GATE6_DEPTH,
    GATE6_DESCRIPTIVE_K,
    GATE6_EVAL_BATCH_SIZE,
    GATE6_EXPERIMENT_VERSION,
    GATE6_FRONTIER_DEPTH,
    GATE6_HINT_RELIABILITY,
    GATE6_NONINFERIORITY_MARGIN,
    GATE6_POPULATION_LADDER,
    GATE6_PRIMARY_K,
    GATE6_SCHEDULED_PARENT_SLOTS,
    GATE6_STAGE_A_PARENT_SLOTS,
    GATE6_STAGE_B_PARENT_SLOTS,
    GATE6_TOTAL_LEARNED_UPDATES,
    GATE6_WORLD_COUNT,
    Gate6CheckpointIdentity,
    Gate6DevelopmentResult,
    build_gate6_paired_summaries,
    evaluate_gate6_condition,
)
from .run_gate3_v2_frontier import load_verified_checkpoint


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _bar(done: int, total: int, width: int = 30) -> str:
    filled = min(width, max(0, round(width * done / total)))
    return "[" + "#" * filled + "-" * (width - filled) + "]"


def _identity_from_gate3(identity: object) -> Gate6CheckpointIdentity:
    return Gate6CheckpointIdentity(
        checkpoint_index=int(getattr(identity, "checkpoint_index")),
        checkpoint_sha256=str(getattr(identity, "checkpoint_sha256")),
        parameter_fingerprint=str(getattr(identity, "parameter_fingerprint")),
        learned_parameter_count=int(getattr(identity, "learned_parameter_count")),
    )


def run_gate6_fixed_k_population_scaling(
    *, output_root: Path, checkpoint_paths: tuple[Path, Path, Path]
) -> int:
    if output_root.exists():
        raise FileExistsError(f"Gate-6 output already exists: {output_root}")
    if not torch.cuda.is_available():
        raise RuntimeError("admitted Gate-6 development runner requires CUDA")

    output_root = output_root.resolve()
    output_root.mkdir(parents=True)
    started = time.monotonic()
    conditions = []
    identities: list[Gate6CheckpointIdentity] = []
    total_cells = (
        len(GATE6_CHECKPOINT_INDICES)
        * len(GATE6_POPULATION_LADDER)
        * len(GATE6_CONDITIONS)
    )
    completed = 0

    print("Gate-6 v0 fixed-K population scaling DEVELOPMENT ONLY", flush=True)
    print(f"CUDA device: {torch.cuda.get_device_name(0)}", flush=True)
    print("Training: NONE — reusing three frozen Gate-3 v1 checkpoints", flush=True)
    print(f"Population ladder: {GATE6_POPULATION_LADDER}", flush=True)
    print(
        f"Work: {GATE6_STAGE_A_PARENT_SLOTS} common frontier slots + "
        f"{GATE6_STAGE_B_PARENT_SLOTS} routing slots = "
        f"{GATE6_TOTAL_LEARNED_UPDATES} learned recurrent updates/world",
        flush=True,
    )
    print(f"Primary bounded visibility: K{GATE6_PRIMARY_K}; NI margin: {GATE6_NONINFERIORITY_MARGIN:.2f}", flush=True)
    print(f"Descriptive bounded visibility: K{GATE6_DESCRIPTIVE_K}", flush=True)
    print("Confirmation remains CLOSED.", flush=True)

    for checkpoint_index, checkpoint_path in zip(
        GATE6_CHECKPOINT_INDICES, checkpoint_paths, strict=True
    ):
        model, gate3_identity = load_verified_checkpoint(
            checkpoint_index=checkpoint_index,
            checkpoint_path=checkpoint_path,
            device="cuda",
        )
        identity = _identity_from_gate3(gate3_identity)
        identities.append(identity)
        print(
            f"CHECKPOINT {checkpoint_index}: SHA256={identity.checkpoint_sha256} "
            f"fingerprint={identity.parameter_fingerprint}",
            flush=True,
        )

        for population_size in GATE6_POPULATION_LADDER:
            for mode in GATE6_CONDITIONS:
                condition = evaluate_gate6_condition(
                    model,
                    checkpoint_index=checkpoint_index,
                    population_size=population_size,
                    mode=mode,
                    device="cuda",
                )
                conditions.append(condition)
                completed += 1
                elapsed = (time.monotonic() - started) / 60.0
                mean_score_obs = (
                    sum(condition.total_stage_b_score_observations_by_world) / condition.world_count
                )
                mean_terminals = (
                    sum(condition.generated_terminal_count_by_world) / condition.world_count
                )
                mean_live = sum(
                    sum(row) / len(row)
                    for row in condition.stage_b_live_population_by_slot_by_world
                ) / condition.world_count
                mean_pruned = sum(
                    sum(row) for row in condition.overflow_pruned_count_by_slot_by_world
                ) / condition.world_count
                print(
                    f"EVAL  {_bar(completed, total_cells)} {100.0 * completed / total_cells:6.2f}% "
                    f"{completed:2d}/{total_cells} C{checkpoint_index} N={population_size:<3d} "
                    f"{mode.value:<20s} coverage={condition.coverage_rate:.4f} "
                    f"mean-live={mean_live:.1f} score-obs={mean_score_obs:.1f} "
                    f"pruned={mean_pruned:.1f} terminals={mean_terminals:.1f} "
                    f"elapsed={elapsed:.1f}m",
                    flush=True,
                )
        del model

    paired = build_gate6_paired_summaries(conditions)
    result = Gate6DevelopmentResult(
        experiment_version=GATE6_EXPERIMENT_VERSION,
        scientific_status="DEVELOPMENT_ONLY_NO_GATE_VERDICT",
        confirmation_opened=False,
        training_performed=False,
        checkpoints=tuple(identities),
        world_count=GATE6_WORLD_COUNT,
        evaluation_batch_size=GATE6_EVAL_BATCH_SIZE,
        bootstrap_samples=GATE6_BOOTSTRAP_SAMPLES,
        depth=GATE6_DEPTH,
        frontier_depth=GATE6_FRONTIER_DEPTH,
        hint_reliability=GATE6_HINT_RELIABILITY,
        population_ladder=GATE6_POPULATION_LADDER,
        stage_a_parent_slots=GATE6_STAGE_A_PARENT_SLOTS,
        stage_b_parent_slots=GATE6_STAGE_B_PARENT_SLOTS,
        scheduled_parent_slots=GATE6_SCHEDULED_PARENT_SLOTS,
        total_learned_updates_per_world=GATE6_TOTAL_LEARNED_UPDATES,
        primary_k=GATE6_PRIMARY_K,
        descriptive_k=GATE6_DESCRIPTIVE_K,
        noninferiority_margin=GATE6_NONINFERIORITY_MARGIN,
        conditions=tuple(conditions),
        paired_summaries=paired,
    )
    result_path = output_root / "gate6-fixed-k-population-scaling-development.json"
    result_path.write_text(
        json.dumps(result.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    runtime = {
        "scientific_status": "DEVELOPMENT_ONLY_NO_GATE_VERDICT",
        "confirmation_opened": False,
        "training_performed": False,
        "torch_version": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cuda_device_name": torch.cuda.get_device_name(0),
        "wall_seconds": time.monotonic() - started,
        "result_sha256": _sha256(result_path),
        "checkpoint_paths": [str(path.resolve()) for path in checkpoint_paths],
        "checkpoint_sha256": [identity.checkpoint_sha256 for identity in identities],
    }
    runtime_path = output_root / "runtime.json"
    runtime_path.write_text(json.dumps(runtime, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(
        json.dumps(
            {
                "status": "GATE6_FIXED_K_POPULATION_SCALING_DEVELOPMENT_COMPLETE",
                "scientific_decision": "DEVELOPMENT_ONLY_NOT_ASSIGNED",
                "confirmation_opened": False,
                "training_performed": False,
                "result": str(result_path),
                "result_sha256": runtime["result_sha256"],
                "checkpoints": [
                    {
                        "checkpoint_index": identity.checkpoint_index,
                        "checkpoint_sha256": identity.checkpoint_sha256,
                        "parameter_fingerprint": identity.parameter_fingerprint,
                        "learned_parameter_count": identity.learned_parameter_count,
                    }
                    for identity in identities
                ],
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
    parser.add_argument("--checkpoint0", type=Path, required=True)
    parser.add_argument("--checkpoint1", type=Path, required=True)
    parser.add_argument("--checkpoint2", type=Path, required=True)
    args = parser.parse_args()
    return run_gate6_fixed_k_population_scaling(
        output_root=args.output_root,
        checkpoint_paths=(args.checkpoint0, args.checkpoint1, args.checkpoint2),
    )


if __name__ == "__main__":
    raise SystemExit(main())
