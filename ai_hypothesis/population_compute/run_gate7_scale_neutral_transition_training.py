"""Train the three frozen Gate-7 scale-neutral transition checkpoints only.

No bridge or high-scale Gate-7 capability evaluation is reachable from this CLI.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from dataclasses import asdict
from pathlib import Path

import torch

from .gate7_scale_neutral_model_prep import (
    GATE7_SCALE_NEUTRAL_PARAMETER_COUNT,
    Gate7ScaleNeutralModelConfig,
    Gate7ScaleNeutralScorer,
)
from .gate7_scale_neutral_transition_training import (
    GATE7_SCALE_NEUTRAL_GRADIENT_CLIP_NORM,
    GATE7_SCALE_NEUTRAL_LEARNING_RATE,
    GATE7_SCALE_NEUTRAL_TRAINING_BATCH_SIZE,
    GATE7_SCALE_NEUTRAL_TRAINING_DEPTHS,
    GATE7_SCALE_NEUTRAL_TRAINING_SEEDS,
    GATE7_SCALE_NEUTRAL_TRAINING_STEPS,
    GATE7_SCALE_NEUTRAL_TRANSITION_VERSION,
    GATE7_SCALE_NEUTRAL_WEIGHT_DECAY,
    Gate7ScaleNeutralTrainingSummary,
    train_gate7_scale_neutral_checkpoint,
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


def _checkpoint_payload(
    model: Gate7ScaleNeutralScorer,
    *,
    summary: Gate7ScaleNeutralTrainingSummary,
) -> dict[str, object]:
    return {
        "transition_version": GATE7_SCALE_NEUTRAL_TRANSITION_VERSION,
        "scientific_status": "GATE7_SCALE_NEUTRAL_TRANSITION_CHECKPOINT_UNBRIDGED",
        "gate7_high_scale_opened": False,
        "bridge_opened": False,
        "training_seed": summary.training_seed,
        "learned_parameter_count": summary.learned_parameter_count,
        "parameter_fingerprint": summary.parameter_fingerprint,
        "training_config": {
            "steps": GATE7_SCALE_NEUTRAL_TRAINING_STEPS,
            "batch_size": GATE7_SCALE_NEUTRAL_TRAINING_BATCH_SIZE,
            "depth_schedule": list(GATE7_SCALE_NEUTRAL_TRAINING_DEPTHS),
            "learning_rate": GATE7_SCALE_NEUTRAL_LEARNING_RATE,
            "weight_decay": GATE7_SCALE_NEUTRAL_WEIGHT_DECAY,
            "gradient_clip_norm": GATE7_SCALE_NEUTRAL_GRADIENT_CLIP_NORM,
            "model": asdict(Gate7ScaleNeutralModelConfig()),
        },
        "state_dict": model.state_dict(),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.output_root.exists():
        raise FileExistsError(f"Gate-7 transition training output already exists: {args.output_root}")
    if not torch.cuda.is_available():
        raise RuntimeError("Gate-7 scale-neutral transition training requires CUDA")

    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True)
    print("Gate-7 scale-neutral scorer transition training", flush=True)
    print(f"CUDA device: {torch.cuda.get_device_name(0)}", flush=True)
    print(f"Checkpoints: {GATE7_SCALE_NEUTRAL_TRAINING_SEEDS}", flush=True)
    print(f"Depth schedule: {GATE7_SCALE_NEUTRAL_TRAINING_DEPTHS}", flush=True)
    print(f"Steps/checkpoint: {GATE7_SCALE_NEUTRAL_TRAINING_STEPS}", flush=True)
    print(f"Learned parameters: {GATE7_SCALE_NEUTRAL_PARAMETER_COUNT:,}", flush=True)
    print("Bridge: CLOSED", flush=True)
    print("Gate-7 high-scale capability worlds: CLOSED", flush=True)

    results: list[dict[str, object]] = []
    overall_started = time.monotonic()
    for seed_index, training_seed in enumerate(GATE7_SCALE_NEUTRAL_TRAINING_SEEDS, start=1):
        seed_started = time.monotonic()
        last_reported = 0

        def progress(done: int, total: int, depth: int, loss: float) -> None:
            nonlocal last_reported
            if done != total and done - last_reported < 20:
                return
            last_reported = done
            elapsed = (time.monotonic() - seed_started) / 60.0
            print(
                f"T{training_seed} {_bar(done, total)} {100.0 * done / total:6.2f}% "
                f"{done:4d}/{total} D{depth:02d} loss={loss:.6f} elapsed={elapsed:.1f}m",
                flush=True,
            )

        model, summary = train_gate7_scale_neutral_checkpoint(
            training_seed=training_seed,
            device="cuda",
            progress=progress,
        )
        if summary.learned_parameter_count != GATE7_SCALE_NEUTRAL_PARAMETER_COUNT:
            raise RuntimeError("trained checkpoint parameter count differs from frozen transition")

        checkpoint_path = output_root / f"gate7-scale-neutral-transition-seed-{training_seed}.pt"
        torch.save(_checkpoint_payload(model, summary=summary), checkpoint_path)
        checkpoint_sha = _sha256(checkpoint_path)
        row = {
            "training_seed": training_seed,
            "checkpoint_file": checkpoint_path.name,
            "checkpoint_sha256": checkpoint_sha,
            "training_summary": summary.to_dict(),
        }
        results.append(row)
        print(
            f"CHECKPOINT {seed_index}/3 T{training_seed} "
            f"loss={summary.final_loss:.6f} mean50={summary.mean_last_50_loss:.6f} "
            f"sha256={checkpoint_sha} fingerprint={summary.parameter_fingerprint}",
            flush=True,
        )

    payload = {
        "transition_version": GATE7_SCALE_NEUTRAL_TRANSITION_VERSION,
        "status": "GATE7_SCALE_NEUTRAL_TRANSITION_TRAINING_COMPLETE_UNBRIDGED",
        "scientific_evidence": False,
        "bridge_opened": False,
        "gate7_high_scale_opened": False,
        "torch_version": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cuda_device_name": torch.cuda.get_device_name(0),
        "training_seeds": list(GATE7_SCALE_NEUTRAL_TRAINING_SEEDS),
        "depth_schedule": list(GATE7_SCALE_NEUTRAL_TRAINING_DEPTHS),
        "steps_per_checkpoint": GATE7_SCALE_NEUTRAL_TRAINING_STEPS,
        "batch_size": GATE7_SCALE_NEUTRAL_TRAINING_BATCH_SIZE,
        "learned_parameter_count": GATE7_SCALE_NEUTRAL_PARAMETER_COUNT,
        "checkpoints": results,
        "elapsed_minutes": (time.monotonic() - overall_started) / 60.0,
    }
    summary_path = output_root / "training-summary.json"
    summary_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(f"Training summary: {summary_path}", flush=True)
    print("No bridge or Gate-7 capability evaluation was executed.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
