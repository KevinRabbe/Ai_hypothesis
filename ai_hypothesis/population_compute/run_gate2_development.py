"""CLI for one development-only Gate-2 training/evaluation run."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

from .gate2_development import Gate2TrainingConfig, run_gate2_development
from .gate2_persistent_model import Gate2PersistentModelConfig


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--training-seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=1_000)
    parser.add_argument("--training-batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--gradient-clip-norm", type=float, default=1.0)
    parser.add_argument("--state-width", type=int, default=64)
    parser.add_argument("--query-width", type=int, default=24)
    parser.add_argument("--evaluation-world-count", type=int, default=256)
    parser.add_argument("--evaluation-batch-size", type=int, default=64)
    parser.add_argument("--bootstrap-samples", type=int, default=2_000)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available to PyTorch")

    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    result_path = output_root / "gate2-development.json"
    checkpoint_path = output_root / "gate2-development-checkpoint.pt"
    runtime_path = output_root / "runtime.json"
    if result_path.exists() or checkpoint_path.exists():
        raise FileExistsError("Gate-2 development output already exists; use a new output root")

    config = Gate2TrainingConfig(
        steps=args.steps,
        batch_size=args.training_batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        gradient_clip_norm=args.gradient_clip_norm,
        model=Gate2PersistentModelConfig(
            state_width=args.state_width,
            query_width=args.query_width,
        ),
    )
    model, result = run_gate2_development(
        training_seed=args.training_seed,
        training_config=config,
        evaluation_world_count=args.evaluation_world_count,
        evaluation_batch_size=args.evaluation_batch_size,
        bootstrap_samples=args.bootstrap_samples,
        device=device,
    )

    torch.save(
        {
            "experiment_version": result.experiment_version,
            "evaluation_split": result.evaluation_split,
            "confirmation_opened": result.confirmation_opened,
            "training_seed": args.training_seed,
            "model_config": {
                "state_width": config.model.state_width,
                "query_width": config.model.query_width,
            },
            "learned_parameter_count": model.trainable_parameter_count(),
            "parameter_fingerprint": model.parameter_fingerprint(),
            "state_dict": model.state_dict(),
        },
        checkpoint_path,
    )
    checkpoint_sha256 = _sha256_file(checkpoint_path)

    result_payload = result.to_dict()
    result_payload["checkpoint_file"] = checkpoint_path.name
    result_payload["checkpoint_sha256"] = checkpoint_sha256
    result_path.write_text(
        json.dumps(result_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    runtime = {
        "torch_version": torch.__version__,
        "device": str(device),
        "cuda_available": torch.cuda.is_available(),
        "cuda_runtime": torch.version.cuda,
        "cuda_device_name": (
            torch.cuda.get_device_name(device) if device.type == "cuda" else None
        ),
        "checkpoint_sha256": checkpoint_sha256,
        "scientific_status": "DEVELOPMENT_ONLY_NOT_CONFIRMATION",
    }
    runtime_path.write_text(json.dumps(runtime, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(
        json.dumps(
            {
                "status": "DEVELOPMENT_ONLY_COMPLETE",
                "training_seed": args.training_seed,
                "learned_parameter_count": model.trainable_parameter_count(),
                "parameter_fingerprint": model.parameter_fingerprint(),
                "initial_loss": result.training.initial_loss,
                "final_loss": result.training.final_loss,
                "result": str(result_path),
                "checkpoint": str(checkpoint_path),
                "checkpoint_sha256": checkpoint_sha256,
            },
            indent=2,
            sort_keys=True,
        )
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
