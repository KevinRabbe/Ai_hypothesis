"""Run the frozen Gate-3 v3 generation-pressure development matrix.

No training occurs. The runner loads and verifies the exact three frozen Gate-3 v1 checkpoints,
then evaluates the preregistered 15-cell generation-synchronous population matrix on CUDA.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import torch

from .gate3_v2_frontier import Gate3V2CheckpointIdentity
from .run_gate3_v2_frontier import load_verified_checkpoint
from .gate3_v3_generation_pressure import (
    GATE3_V3_BOOTSTRAP_SAMPLES,
    GATE3_V3_CHECKPOINT_INDICES,
    GATE3_V3_CONDITIONS,
    GATE3_V3_DEPTH,
    GATE3_V3_EVAL_BATCH_SIZE,
    GATE3_V3_EXPERIMENT_VERSION,
    GATE3_V3_HINT_RELIABILITY,
    GATE3_V3_SCHEDULED_SLOTS,
    GATE3_V3_TOTAL_LEARNED_UPDATES,
    GATE3_V3_WORLD_COUNT,
    Gate3V3CheckpointIdentity,
    Gate3V3DevelopmentResult,
    build_gate3_v3_paired_summaries,
    evaluate_gate3_v3_condition,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _bar(done: int, total: int, width: int = 30) -> str:
    filled = min(width, max(0, round(width * done / total)))
    return "[" + "#" * filled + "-" * (width - filled) + "]"


def _identity_from_v2(identity: Gate3V2CheckpointIdentity) -> Gate3V3CheckpointIdentity:
    return Gate3V3CheckpointIdentity(
        checkpoint_index=identity.checkpoint_index,
        checkpoint_sha256=identity.checkpoint_sha256,
        parameter_fingerprint=identity.parameter_fingerprint,
        learned_parameter_count=identity.learned_parameter_count,
    )


def run_gate3_v3_generation_pressure(
    *, output_root: Path, checkpoint_paths: tuple[Path, Path, Path]
) -> int:
    if output_root.exists():
        raise FileExistsError(f"Gate-3 v3 output already exists: {output_root}")
    if not torch.cuda.is_available():
        raise RuntimeError("admitted Gate-3 v3 runner requires CUDA")

    output_root = output_root.resolve()
    output_root.mkdir(parents=True)
    started = time.monotonic()
    conditions = []
    identities: list[Gate3V3CheckpointIdentity] = []
    total_cells = len(GATE3_V3_CHECKPOINT_INDICES) * len(GATE3_V3_CONDITIONS)
    completed = 0

    print("Gate-3 v3 generation-pressure DEVELOPMENT ONLY", flush=True)
    print(f"CUDA device: {torch.cuda.get_device_name(0)}", flush=True)
    print("Training: NONE — reusing three frozen Gate-3 v1 checkpoints", flush=True)
    print("Scheduler: generation-synchronous; frozen depth-7 frontier pressure", flush=True)
    print("Confirmation remains CLOSED.", flush=True)

    for checkpoint_index, checkpoint_path in zip(
        GATE3_V3_CHECKPOINT_INDICES, checkpoint_paths, strict=True
    ):
        model, v2_identity = load_verified_checkpoint(
            checkpoint_index=checkpoint_index,
            checkpoint_path=checkpoint_path,
            device="cuda",
        )
        identity = _identity_from_v2(v2_identity)
        identities.append(identity)
        print(
            f"CHECKPOINT {checkpoint_index}: SHA256={identity.checkpoint_sha256} "
            f"fingerprint={identity.parameter_fingerprint}",
            flush=True,
        )

        for reserve_capacity, mode in GATE3_V3_CONDITIONS:
            condition = evaluate_gate3_v3_condition(
                model,
                checkpoint_index=checkpoint_index,
                reserve_capacity=reserve_capacity,
                mode=mode,
                device="cuda",
            )
            conditions.append(condition)
            completed += 1
            elapsed = (time.monotonic() - started) / 60.0
            mean_productive = sum(condition.productive_slots_by_world) / condition.world_count
            mean_sink = sum(condition.sink_slots_by_world) / condition.world_count
            mean_depth7 = sum(condition.depth7_retained_width_by_world) / condition.world_count
            print(
                f"EVAL  {_bar(completed, total_cells)} {100.0 * completed / total_cells:6.2f}% "
                f"{completed:2d}/{total_cells} C{checkpoint_index} "
                f"L={reserve_capacity:<3d} {mode.value:<23s} "
                f"coverage={condition.coverage_rate:.4f} depth7={mean_depth7:.1f} "
                f"productive={mean_productive:.1f} sink={mean_sink:.1f} elapsed={elapsed:.1f}m",
                flush=True,
            )
        del model

    paired = build_gate3_v3_paired_summaries(conditions)
    result = Gate3V3DevelopmentResult(
        experiment_version=GATE3_V3_EXPERIMENT_VERSION,
        scientific_status="DEVELOPMENT_ONLY_NO_GATE_VERDICT",
        confirmation_opened=False,
        training_performed=False,
        checkpoints=tuple(identities),
        world_count=GATE3_V3_WORLD_COUNT,
        evaluation_batch_size=GATE3_V3_EVAL_BATCH_SIZE,
        bootstrap_samples=GATE3_V3_BOOTSTRAP_SAMPLES,
        depth=GATE3_V3_DEPTH,
        hint_reliability=GATE3_V3_HINT_RELIABILITY,
        scheduled_slots=GATE3_V3_SCHEDULED_SLOTS,
        total_learned_updates_per_world=GATE3_V3_TOTAL_LEARNED_UPDATES,
        conditions=tuple(conditions),
        paired_summaries=paired,
    )
    result_path = output_root / "gate3-v3-generation-pressure-development.json"
    result_path.write_text(json.dumps(result.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")

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
                "status": "GATE3_V3_GENERATION_PRESSURE_DEVELOPMENT_COMPLETE",
                "scientific_decision": "DEVELOPMENT_ONLY_NOT_ASSIGNED",
                "confirmation_opened": False,
                "training_performed": False,
                "result": str(result_path),
                "result_sha256": runtime["result_sha256"],
                "checkpoints": [asdict_identity(identity) for identity in identities],
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


def asdict_identity(identity: Gate3V3CheckpointIdentity) -> dict[str, object]:
    return {
        "checkpoint_index": identity.checkpoint_index,
        "checkpoint_sha256": identity.checkpoint_sha256,
        "parameter_fingerprint": identity.parameter_fingerprint,
        "learned_parameter_count": identity.learned_parameter_count,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--checkpoint0", type=Path, required=True)
    parser.add_argument("--checkpoint1", type=Path, required=True)
    parser.add_argument("--checkpoint2", type=Path, required=True)
    args = parser.parse_args()
    return run_gate3_v3_generation_pressure(
        output_root=args.output_root,
        checkpoint_paths=(args.checkpoint0, args.checkpoint1, args.checkpoint2),
    )


if __name__ == "__main__":
    raise SystemExit(main())
