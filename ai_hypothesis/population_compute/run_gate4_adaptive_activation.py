"""Run the frozen Gate-4 v0 adaptive-activation development matrix.

No training occurs.  The runner verifies and reuses the exact three Gate-3 v1 checkpoints and
evaluates the preregistered nine-cell scheduler matrix on CUDA.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import torch

from .gate4_adaptive_activation import (
    GATE4_BOOTSTRAP_SAMPLES,
    GATE4_CHECKPOINT_INDICES,
    GATE4_CONDITIONS,
    GATE4_DEPTH,
    GATE4_EVAL_BATCH_SIZE,
    GATE4_EXPERIMENT_VERSION,
    GATE4_HINT_RELIABILITY,
    GATE4_RESERVE_CAPACITY,
    GATE4_SCHEDULED_SLOTS,
    GATE4_TOTAL_LEARNED_UPDATES,
    GATE4_WORLD_COUNT,
    Gate4CheckpointIdentity,
    Gate4DevelopmentResult,
    Gate4SchedulerMode,
    build_gate4_paired_summaries,
    evaluate_gate4_condition,
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


def _identity_from_gate3(identity: object) -> Gate4CheckpointIdentity:
    return Gate4CheckpointIdentity(
        checkpoint_index=int(getattr(identity, "checkpoint_index")),
        checkpoint_sha256=str(getattr(identity, "checkpoint_sha256")),
        parameter_fingerprint=str(getattr(identity, "parameter_fingerprint")),
        learned_parameter_count=int(getattr(identity, "learned_parameter_count")),
    )


def run_gate4_adaptive_activation(
    *, output_root: Path, checkpoint_paths: tuple[Path, Path, Path]
) -> int:
    if output_root.exists():
        raise FileExistsError(f"Gate-4 output already exists: {output_root}")
    if not torch.cuda.is_available():
        raise RuntimeError("admitted Gate-4 development runner requires CUDA")

    output_root = output_root.resolve()
    output_root.mkdir(parents=True)
    started = time.monotonic()
    conditions = []
    identities: list[Gate4CheckpointIdentity] = []
    total_cells = len(GATE4_CHECKPOINT_INDICES) * len(GATE4_CONDITIONS)
    completed = 0

    print("Gate-4 v0 adaptive-activation DEVELOPMENT ONLY", flush=True)
    print(f"CUDA device: {torch.cuda.get_device_name(0)}", flush=True)
    print("Training: NONE — reusing three frozen Gate-3 v1 checkpoints", flush=True)
    print(f"Reserve capacity: L{GATE4_RESERVE_CAPACITY} in every condition", flush=True)
    print(
        f"Work: {GATE4_SCHEDULED_SLOTS} scheduled parent slots = "
        f"{GATE4_TOTAL_LEARNED_UPDATES} learned recurrent updates/world",
        flush=True,
    )
    print("Confirmation remains CLOSED.", flush=True)

    for checkpoint_index, checkpoint_path in zip(
        GATE4_CHECKPOINT_INDICES, checkpoint_paths, strict=True
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

        for mode in GATE4_CONDITIONS:
            condition = evaluate_gate4_condition(
                model,
                checkpoint_index=checkpoint_index,
                mode=mode,
                device="cuda",
            )
            conditions.append(condition)
            completed += 1
            elapsed = (time.monotonic() - started) / 60.0
            mean_productive = sum(condition.productive_slots_by_world) / condition.world_count
            mean_sink = sum(condition.sink_slots_by_world) / condition.world_count
            mean_live = sum(condition.mean_live_nonterminal_population_by_world) / condition.world_count
            mean_terminals = sum(condition.generated_terminal_count_by_world) / condition.world_count
            print(
                f"EVAL  {_bar(completed, total_cells)} {100.0 * completed / total_cells:6.2f}% "
                f"{completed:2d}/{total_cells} C{checkpoint_index} {mode.value:<18s} "
                f"coverage={condition.coverage_rate:.4f} productive={mean_productive:.1f} "
                f"sink={mean_sink:.1f} mean-live={mean_live:.1f} terminals={mean_terminals:.1f} "
                f"elapsed={elapsed:.1f}m",
                flush=True,
            )
        del model

    paired = build_gate4_paired_summaries(conditions)
    result = Gate4DevelopmentResult(
        experiment_version=GATE4_EXPERIMENT_VERSION,
        scientific_status="DEVELOPMENT_ONLY_NO_GATE_VERDICT",
        confirmation_opened=False,
        training_performed=False,
        checkpoints=tuple(identities),
        world_count=GATE4_WORLD_COUNT,
        evaluation_batch_size=GATE4_EVAL_BATCH_SIZE,
        bootstrap_samples=GATE4_BOOTSTRAP_SAMPLES,
        depth=GATE4_DEPTH,
        hint_reliability=GATE4_HINT_RELIABILITY,
        reserve_capacity=GATE4_RESERVE_CAPACITY,
        scheduled_slots=GATE4_SCHEDULED_SLOTS,
        total_learned_updates_per_world=GATE4_TOTAL_LEARNED_UPDATES,
        conditions=tuple(conditions),
        paired_summaries=paired,
    )
    result_path = output_root / "gate4-adaptive-activation-development.json"
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
                "status": "GATE4_ADAPTIVE_ACTIVATION_DEVELOPMENT_COMPLETE",
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
    return run_gate4_adaptive_activation(
        output_root=args.output_root,
        checkpoint_paths=(args.checkpoint0, args.checkpoint1, args.checkpoint2),
    )


if __name__ == "__main__":
    raise SystemExit(main())
