"""CLI for one frozen Gate-2 confirmation training seed with live progress."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import torch

from .gate2_confirmation import (
    GATE2_CONFIRMATION_TRAINING_SEEDS,
    run_gate2_confirmation_seed,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--training-seed", type=int, required=True, choices=GATE2_CONFIRMATION_TRAINING_SEEDS)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def _bar(done: int, total: int, width: int = 30) -> str:
    filled = min(width, max(0, round(width * done / total)))
    return "[" + "#" * filled + "-" * (width - filled) + "]"


def main() -> int:
    args = _parse_args()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available to PyTorch")

    output_root = args.output_root.resolve()
    if output_root.exists():
        raise FileExistsError("Gate-2 confirmation output root already exists; use a new root")
    output_root.mkdir(parents=True)

    result_path = output_root / "gate2-confirmation.json"
    checkpoint_path = output_root / "gate2-confirmation-checkpoint.pt"
    runtime_path = output_root / "runtime.json"
    start = time.monotonic()
    last_training_print = 0

    def training_progress(step: int, total: int, entity_count: int, width: int, loss: float) -> None:
        nonlocal last_training_print
        if step != total and step - last_training_print < 10:
            return
        last_training_print = step
        percent = 100.0 * step / total
        elapsed = time.monotonic() - start
        print(
            f"TRAIN {_bar(step, total)} {percent:6.2f}%  {step:4d}/{total}  "
            f"C={entity_count:<3d} W={width:<3d} loss={loss:.6f}  elapsed={elapsed/60:.1f}m",
            flush=True,
        )

    evaluation_start: list[float] = []

    def evaluation_progress(done: int, total: int, entity_count: int, width: int, mode: object) -> None:
        if not evaluation_start:
            evaluation_start.append(time.monotonic())
        percent = 100.0 * done / total
        elapsed = time.monotonic() - evaluation_start[0]
        mode_value = getattr(mode, "value", str(mode))
        print(
            f"EVAL  {_bar(done, total)} {percent:6.2f}%  {done:2d}/{total}  "
            f"C={entity_count:<3d} W={width:<3d} {mode_value:<20s} elapsed={elapsed/60:.1f}m",
            flush=True,
        )

    model, result = run_gate2_confirmation_seed(
        training_seed=args.training_seed,
        device=device,
        training_progress=training_progress,
        evaluation_progress=evaluation_progress,
    )

    torch.save(
        {
            "experiment_version": result.experiment_version,
            "evaluation_split": result.evaluation_split,
            "confirmation_opened": result.confirmation_opened,
            "training_seed": args.training_seed,
            "model_config": {
                "state_width": result.training_config.model.state_width,
                "query_width": result.training_config.model.query_width,
            },
            "learned_parameter_count": model.trainable_parameter_count(),
            "parameter_fingerprint": model.parameter_fingerprint(),
            "state_dict": model.state_dict(),
        },
        checkpoint_path,
    )
    checkpoint_sha256 = _sha256_file(checkpoint_path)

    payload = result.to_dict()
    payload["checkpoint_file"] = checkpoint_path.name
    payload["checkpoint_sha256"] = checkpoint_sha256
    result_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    runtime = {
        "torch_version": torch.__version__,
        "device": str(device),
        "cuda_available": torch.cuda.is_available(),
        "cuda_runtime": torch.version.cuda,
        "cuda_device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "checkpoint_sha256": checkpoint_sha256,
        "scientific_status": "FROZEN_GATE2_CONFIRMATION_SEED",
        "seed_passed": result.seed_passed,
        "wall_seconds": time.monotonic() - start,
    }
    runtime_path.write_text(json.dumps(runtime, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(
        json.dumps(
            {
                "status": "CONFIRMATION_SEED_COMPLETE",
                "training_seed": args.training_seed,
                "seed_passed": result.seed_passed,
                "width1_identity_passed": result.width1_identity_passed,
                "learned_parameter_count": model.trainable_parameter_count(),
                "parameter_fingerprint": model.parameter_fingerprint(),
                "result": str(result_path),
                "checkpoint": str(checkpoint_path),
                "checkpoint_sha256": checkpoint_sha256,
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
