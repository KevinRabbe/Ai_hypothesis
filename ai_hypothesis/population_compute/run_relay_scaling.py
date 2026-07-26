"""CLI for the first train-once fixed-parameter collective-relay development curve."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .relay_experiment import (
    RelayTrainingConfig,
    evaluate_relay_development,
    load_relay_checkpoint,
    train_relay_checkpoint,
)
from .relay_model import RelayPopulationConfig


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Train one shared-weight relay model, reload the exact checkpoint, and "
            "evaluate the frozen 1/4/16/64/256 development curve."
        )
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--training-seed", type=int, default=0)
    parser.add_argument("--benchmark-seed", type=int, default=0)
    parser.add_argument("--train-steps", type=int, default=2000)
    parser.add_argument("--train-batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--gradient-clip-norm", type=float, default=1.0)
    parser.add_argument("--state-width", type=int, default=64)
    parser.add_argument("--message-width", type=int, default=24)
    parser.add_argument("--development-world-count", type=int, default=1000)
    parser.add_argument("--eval-batch-size", type=int, default=64)
    parser.add_argument(
        "--checkpoint",
        default="results/population_compute_scaling_v0/relay_seed_0.pt",
    )
    parser.add_argument(
        "--output",
        default="results/population_compute_scaling_v0/development_seed_0.json",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    model_config = RelayPopulationConfig(
        state_width=args.state_width,
        message_width=args.message_width,
    )
    training_config = RelayTrainingConfig(
        training_seed=args.training_seed,
        steps=args.train_steps,
        batch_size=args.train_batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        gradient_clip_norm=args.gradient_clip_norm,
    )
    model_config.validate()
    training_config.validate()

    checkpoint = Path(args.checkpoint)
    summary = train_relay_checkpoint(
        checkpoint,
        model_config=model_config,
        training_config=training_config,
        device=args.device,
    )
    model, loaded_summary = load_relay_checkpoint(checkpoint, device=args.device)
    if loaded_summary != summary:
        raise RuntimeError("reloaded relay training summary differs from saved checkpoint")

    result = evaluate_relay_development(
        model,
        loaded_summary,
        benchmark_seed=args.benchmark_seed,
        world_count_per_difficulty=args.development_world_count,
        batch_size=args.eval_batch_size,
        device=args.device,
    )
    payload = result.to_dict()
    payload["provenance"] = {
        "checkpoint": str(checkpoint),
        "device": args.device,
        "execution_mode": "eager",
        "confirmation_opened": False,
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
