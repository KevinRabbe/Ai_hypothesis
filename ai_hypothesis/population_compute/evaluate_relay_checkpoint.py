"""Evaluate an already-trained collective-relay checkpoint without retraining it."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .relay_experiment import (
    RELAY_EXPERIMENT_VERSION,
    RelayDevelopmentResult,
    RelayTrainingConfig,
    RelayTrainingSummary,
    assess_relay_results,
    evaluate_relay_split,
    load_relay_checkpoint,
    write_relay_result,
)
from .relay_model import RelayPopulationConfig


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate one exact saved relay checkpoint on development or explicitly "
            "unlocked confirmation worlds."
        )
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--evaluation-split",
        choices=("development", "confirmation"),
        default="development",
    )
    parser.add_argument("--allow-confirmation", action="store_true")
    parser.add_argument("--evaluation-world-count", type=int, default=1_000)
    parser.add_argument("--evaluation-batch-size", type=int, default=64)
    parser.add_argument("--output", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.evaluation_split == "confirmation" and not args.allow_confirmation:
        raise SystemExit(
            "Refusing to open frozen confirmation worlds without --allow-confirmation"
        )

    model, payload = load_relay_checkpoint(args.checkpoint, device=args.device)
    training, training_config = _training_metadata(payload)
    if model.parameter_fingerprint() != training.parameter_fingerprint:
        raise ValueError("checkpoint model does not match saved training fingerprint")
    if model.trainable_parameter_count() != training.learned_parameter_count:
        raise ValueError("checkpoint model does not match saved training parameter count")

    evaluations = evaluate_relay_split(
        model,
        training_seed=training.training_seed,
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
    output = write_relay_result(result, Path(args.output))
    print(
        json.dumps(
            {
                "checkpoint": str(Path(args.checkpoint)),
                "result": str(output),
                "parameter_fingerprint": training.parameter_fingerprint,
                "learned_parameter_count": training.learned_parameter_count,
                "evaluation_split": args.evaluation_split,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def _training_metadata(
    payload: dict[str, object],
) -> tuple[RelayTrainingSummary, RelayTrainingConfig]:
    summary_data = payload.get("training_summary")
    config_data = payload.get("training_config")
    if not isinstance(summary_data, dict):
        raise ValueError("relay checkpoint is missing training_summary")
    if not isinstance(config_data, dict):
        raise ValueError("relay checkpoint is missing training_config")
    model_data = config_data.get("model")
    if not isinstance(model_data, dict):
        raise ValueError("relay checkpoint is missing training model config")

    summary = RelayTrainingSummary(**summary_data)
    config = RelayTrainingConfig(
        steps=int(config_data["steps"]),
        batch_size=int(config_data["batch_size"]),
        learning_rate=float(config_data["learning_rate"]),
        weight_decay=float(config_data["weight_decay"]),
        gradient_clip_norm=float(config_data["gradient_clip_norm"]),
        model=RelayPopulationConfig(**model_data),
    )
    config.validate()
    return summary, config


if __name__ == "__main__":
    raise SystemExit(main())
