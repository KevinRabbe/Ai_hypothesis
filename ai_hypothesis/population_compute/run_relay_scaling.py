"""CLI for the first trained fixed-parameter population-compute development curve."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .relay_experiment import (
    RelayTrainingConfig,
    run_development_experiment,
    save_relay_checkpoint,
    train_relay_model,
    write_relay_result,
)
from .relay_model import RelayPopulationConfig


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Train one shared collective-relay checkpoint and evaluate fixed-parameter "
            "population curves."
        )
    )
    parser.add_argument("--training-seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--steps", type=int, default=2_000)
    parser.add_argument("--train-batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--gradient-clip-norm", type=float, default=1.0)
    parser.add_argument("--state-width", type=int, default=64)
    parser.add_argument("--message-width", type=int, default=24)
    parser.add_argument("--evaluation-world-count", type=int, default=1_000)
    parser.add_argument("--evaluation-batch-size", type=int, default=64)
    parser.add_argument(
        "--evaluation-split",
        choices=("development", "confirmation"),
        default="development",
    )
    parser.add_argument("--allow-confirmation", action="store_true")
    parser.add_argument(
        "--output-dir",
        default="results/population_compute_scaling_v0",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.evaluation_split == "confirmation" and not args.allow_confirmation:
        raise SystemExit(
            "Refusing to open frozen confirmation worlds without --allow-confirmation"
        )

    training_config = RelayTrainingConfig(
        steps=args.steps,
        batch_size=args.train_batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        gradient_clip_norm=args.gradient_clip_norm,
        model=RelayPopulationConfig(
            state_width=args.state_width,
            message_width=args.message_width,
        ),
    )
    training_config.validate()

    # Keep the checkpoint and result as separate artifacts: the result must always be
    # traceable back to the exact shared parameter fingerprint it evaluated.
    model, training = train_relay_model(
        training_seed=args.training_seed,
        config=training_config,
        device=args.device,
    )
    from .relay_experiment import evaluate_relay_split, assess_relay_results, RelayDevelopmentResult, RELAY_EXPERIMENT_VERSION

    evaluations = evaluate_relay_split(
        model,
        training_seed=args.training_seed,
        split=args.evaluation_split,
        world_count=args.evaluation_world_count,
        batch_size=args.evaluation_batch_size,
        device=args.device,
        allow_confirmation=args.allow_confirmation,
    )
    result = RelayDevelopmentResult(
        experiment_version=RELAY_EXPERIMENT_VERSION,
        evaluation_split=args.evaluation_split,
        training=training,
        training_config=training_config,
        evaluation_world_count=args.evaluation_world_count,
        evaluation_batch_size=args.evaluation_batch_size,
        evaluations=evaluations,
        assessments=assess_relay_results(evaluations),
    )

    output_dir = Path(args.output_dir) / f"seed_{args.training_seed}"
    checkpoint = save_relay_checkpoint(
        model,
        training,
        training_config,
        output_dir / "model.pt",
    )
    result_path = write_relay_result(
        result,
        output_dir / f"{args.evaluation_split}.json",
    )

    summary = {
        "checkpoint": str(checkpoint),
        "result": str(result_path),
        "parameter_fingerprint": training.parameter_fingerprint,
        "learned_parameter_count": training.learned_parameter_count,
        "training_final_loss": training.final_loss,
        "evaluation_split": args.evaluation_split,
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
