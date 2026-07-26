"""CLI for the first trained fixed-parameter population-compute development curve."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .relay_experiment import (
    RELAY_EXPERIMENT_VERSION,
    RelayDevelopmentResult,
    RelayTrainingConfig,
    assess_relay_results,
    evaluate_relay_split,
    load_relay_checkpoint,
    save_relay_checkpoint,
    train_relay_model,
)
from .relay_model import RelayPopulationConfig


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Train one shared collective-relay checkpoint, reload the exact persisted "
            "weights, and evaluate the frozen development population curve."
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
        "--output-dir",
        default="results/population_compute_scaling_v0",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
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

    model, training = train_relay_model(
        training_seed=args.training_seed,
        config=training_config,
        device=args.device,
    )
    output_dir = Path(args.output_dir) / f"seed_{args.training_seed}"
    checkpoint = save_relay_checkpoint(
        model,
        training,
        training_config,
        output_dir / "model.pt",
    )

    # Development evidence is produced from a fresh model loaded from the artifact that
    # will later be used for re-evaluation/confirmation. In-memory training state is not
    # accepted as the scientific checkpoint boundary.
    loaded_model, checkpoint_payload = load_relay_checkpoint(
        checkpoint,
        device=args.device,
    )
    if loaded_model.parameter_fingerprint() != training.parameter_fingerprint:
        raise RuntimeError("reloaded development checkpoint changed parameter fingerprint")
    if loaded_model.trainable_parameter_count() != training.learned_parameter_count:
        raise RuntimeError("reloaded development checkpoint changed parameter count")
    saved_summary = checkpoint_payload.get("training_summary")
    if not isinstance(saved_summary, dict):
        raise RuntimeError("saved development checkpoint lost training summary")
    if saved_summary.get("parameter_fingerprint") != training.parameter_fingerprint:
        raise RuntimeError("saved training summary fingerprint differs from trained model")

    evaluations = evaluate_relay_split(
        loaded_model,
        training_seed=args.training_seed,
        split="development",
        world_count=args.evaluation_world_count,
        batch_size=args.evaluation_batch_size,
        device=args.device,
    )
    result = RelayDevelopmentResult(
        experiment_version=RELAY_EXPERIMENT_VERSION,
        evaluation_split="development",
        training=training,
        training_config=training_config,
        evaluation_world_count=args.evaluation_world_count,
        evaluation_batch_size=args.evaluation_batch_size,
        evaluations=evaluations,
        assessments=assess_relay_results(evaluations),
    )
    payload = result.to_dict()
    payload["provenance"] = {
        "checkpoint": str(checkpoint),
        "device": args.device,
        "execution_mode": "eager",
        "confirmation_opened": False,
        "evaluated_parameter_fingerprint": loaded_model.parameter_fingerprint(),
    }
    result_path = output_dir / "development.json"
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    summary = {
        "checkpoint": str(checkpoint),
        "result": str(result_path),
        "parameter_fingerprint": training.parameter_fingerprint,
        "learned_parameter_count": training.learned_parameter_count,
        "training_final_loss": training.final_loss,
        "evaluation_split": "development",
        "confirmation_opened": False,
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
