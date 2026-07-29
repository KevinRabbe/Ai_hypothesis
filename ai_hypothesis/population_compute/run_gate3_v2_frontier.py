"""Run the frozen Gate-3 v2 ambiguity-frontier development matrix.

This runner performs no training.  It only loads the three exact frozen Gate-3 v1 checkpoints,
verifies their identities, and evaluates the preregistered 36-cell frontier matrix on CUDA.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import torch

from .gate3_v1_model import Gate3V1Scorer
from .gate3_v2_frontier import (
    GATE3_V2_BOOTSTRAP_SAMPLES,
    GATE3_V2_CHECKPOINT_INDICES,
    GATE3_V2_CONDITIONS,
    GATE3_V2_DEPTH,
    GATE3_V2_EVAL_BATCH_SIZE,
    GATE3_V2_EXPERIMENT_VERSION,
    GATE3_V2_SEARCH_ROUNDS,
    GATE3_V2_TOTAL_LEARNED_UPDATES,
    GATE3_V2_WORLD_COUNT,
    Gate3V2AmbiguityTier,
    Gate3V2CheckpointIdentity,
    Gate3V2DevelopmentResult,
    build_gate3_v2_paired_summaries,
    evaluate_gate3_v2_condition,
)

CHECKPOINT_EXPECTED = {
    0: {
        "sha256": "e63748a1182472d03c540f9123c3442ce44e130519e2176681034568826db590",
        "fingerprint": "e4f9990f08e85393a155637cfa50b5831d779770447ec3716fc9e67946992afc",
        "training_seed": 0,
    },
    1: {
        "sha256": "8ce384627f5543fa4fe78498f9990f8214cf2a4afed0c9b734e86761ad13c989",
        "fingerprint": "2a57984f755ee1404fd828c2df36a5efff8f13a541b7d7c93a891014fbf4897c",
        "training_seed": 1,
    },
    2: {
        "sha256": "103826bbd9451b965eced5134d1674cb8e893f5d3b378cb828312ffdb6fc9a37",
        "fingerprint": "8afaf956f200f41ea914eafdd1b5f151dd303cae1552165c74e12bb8c945af02",
        "training_seed": 2,
    },
}
PARAMETER_COUNT = 19_649


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _bar(done: int, total: int, width: int = 30) -> str:
    filled = min(width, max(0, round(width * done / total)))
    return "[" + "#" * filled + "-" * (width - filled) + "]"


def load_verified_checkpoint(
    *, checkpoint_index: int, checkpoint_path: Path, device: torch.device | str
) -> tuple[Gate3V1Scorer, Gate3V2CheckpointIdentity]:
    if checkpoint_index not in CHECKPOINT_EXPECTED:
        raise ValueError("checkpoint index must be 0, 1 or 2")
    path = checkpoint_path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Gate-3 v2 checkpoint does not exist: {path}")
    expected = CHECKPOINT_EXPECTED[checkpoint_index]
    observed_sha = _sha256(path)
    if observed_sha.lower() != expected["sha256"]:
        raise RuntimeError(
            f"checkpoint {checkpoint_index} SHA256 mismatch: {observed_sha} != {expected['sha256']}"
        )

    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise RuntimeError(f"checkpoint {checkpoint_index} payload must be one mapping")
    if payload.get("training_seed") != expected["training_seed"]:
        raise RuntimeError(f"checkpoint {checkpoint_index} training-seed identity mismatch")
    if payload.get("learned_parameter_count") != PARAMETER_COUNT:
        raise RuntimeError(f"checkpoint {checkpoint_index} parameter count mismatch")
    if payload.get("parameter_fingerprint") != expected["fingerprint"]:
        raise RuntimeError(f"checkpoint {checkpoint_index} stored fingerprint mismatch")
    state_dict = payload.get("state_dict")
    if not isinstance(state_dict, dict):
        raise RuntimeError(f"checkpoint {checkpoint_index} state_dict is missing")

    model = Gate3V1Scorer()
    model.load_state_dict(state_dict, strict=True)
    if model.trainable_parameter_count() != PARAMETER_COUNT:
        raise RuntimeError(f"checkpoint {checkpoint_index} instantiated parameter count mismatch")
    fingerprint = model.parameter_fingerprint()
    if fingerprint != expected["fingerprint"]:
        raise RuntimeError(f"checkpoint {checkpoint_index} reconstructed fingerprint mismatch")
    model.eval()
    model.to(device)

    return model, Gate3V2CheckpointIdentity(
        checkpoint_index=checkpoint_index,
        checkpoint_sha256=observed_sha.lower(),
        parameter_fingerprint=fingerprint,
        learned_parameter_count=model.trainable_parameter_count(),
    )


def run_gate3_v2_frontier(
    *,
    output_root: Path,
    checkpoint_paths: tuple[Path, Path, Path],
) -> int:
    if output_root.exists():
        raise FileExistsError(f"Gate-3 v2 output already exists: {output_root}")
    if not torch.cuda.is_available():
        raise RuntimeError("admitted Gate-3 v2 frontier runner requires CUDA")

    output_root = output_root.resolve()
    output_root.mkdir(parents=True)
    started = time.monotonic()
    conditions = []
    identities: list[Gate3V2CheckpointIdentity] = []
    total_cells = len(GATE3_V2_CHECKPOINT_INDICES) * len(Gate3V2AmbiguityTier) * len(GATE3_V2_CONDITIONS)
    completed = 0

    print("Gate-3 v2 ambiguity-frontier DEVELOPMENT ONLY", flush=True)
    print(f"CUDA device: {torch.cuda.get_device_name(0)}", flush=True)
    print("Training: NONE — reusing three frozen Gate-3 v1 checkpoints", flush=True)
    print("Confirmation remains CLOSED.", flush=True)

    for checkpoint_index, checkpoint_path in zip(
        GATE3_V2_CHECKPOINT_INDICES, checkpoint_paths, strict=True
    ):
        model, identity = load_verified_checkpoint(
            checkpoint_index=checkpoint_index,
            checkpoint_path=checkpoint_path,
            device="cuda",
        )
        identities.append(identity)
        print(
            f"CHECKPOINT {checkpoint_index}: SHA256={identity.checkpoint_sha256} "
            f"fingerprint={identity.parameter_fingerprint}",
            flush=True,
        )

        for tier in Gate3V2AmbiguityTier:
            for reserve_capacity, mode in GATE3_V2_CONDITIONS:
                condition = evaluate_gate3_v2_condition(
                    model,
                    checkpoint_index=checkpoint_index,
                    tier=tier,
                    reserve_capacity=reserve_capacity,
                    mode=mode,
                    device="cuda",
                )
                conditions.append(condition)
                completed += 1
                reached = sum(int(value) for value in condition.reached_capacity_by_world) / condition.world_count
                elapsed = (time.monotonic() - started) / 60.0
                print(
                    f"EVAL  {_bar(completed, total_cells)} {100.0 * completed / total_cells:6.2f}% "
                    f"{completed:2d}/{total_cells} C{checkpoint_index} {tier.value} "
                    f"L={reserve_capacity:<3d} {mode.value:<23s} "
                    f"coverage={condition.coverage_rate:.4f} reached-cap={reached:.3f} "
                    f"elapsed={elapsed:.1f}m",
                    flush=True,
                )

        # Only one frozen checkpoint is needed on the GPU at a time.
        del model

    paired = build_gate3_v2_paired_summaries(conditions)
    result = Gate3V2DevelopmentResult(
        experiment_version=GATE3_V2_EXPERIMENT_VERSION,
        scientific_status="DEVELOPMENT_ONLY_NO_GATE_VERDICT",
        confirmation_opened=False,
        checkpoints=tuple(identities),
        world_count_per_tier=GATE3_V2_WORLD_COUNT,
        evaluation_batch_size=GATE3_V2_EVAL_BATCH_SIZE,
        bootstrap_samples=GATE3_V2_BOOTSTRAP_SAMPLES,
        depth=GATE3_V2_DEPTH,
        search_rounds=GATE3_V2_SEARCH_ROUNDS,
        total_learned_updates_per_world=GATE3_V2_TOTAL_LEARNED_UPDATES,
        conditions=tuple(conditions),
        paired_summaries=paired,
    )
    result_path = output_root / "gate3-v2-frontier-development.json"
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
                "status": "GATE3_V2_FRONTIER_DEVELOPMENT_COMPLETE",
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


def asdict_identity(identity: Gate3V2CheckpointIdentity) -> dict[str, object]:
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
    return run_gate3_v2_frontier(
        output_root=args.output_root,
        checkpoint_paths=(args.checkpoint0, args.checkpoint1, args.checkpoint2),
    )


if __name__ == "__main__":
    raise SystemExit(main())
