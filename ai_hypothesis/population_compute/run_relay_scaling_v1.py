"""CLI for canonical repaired fixed-parameter collective-relay protocol v1."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .relay_experiment_v1 import (
    RELAY_EXPERIMENT_V1,
    RelayDevelopmentResultV1,
    RelayTrainingConfigV1,
    assess_relay_results_v1,
    evaluate_relay_split_v1,
    load_relay_checkpoint_v1,
    save_relay_checkpoint_v1,
    train_relay_model_v1,
)
from .relay_model import RelayPopulationConfig
from .relay_protocol_v1 import RELAY_PROTOCOL_VERSION
from .collective_relay import COLLECTIVE_RELAY_VERSION


FROZEN_CONFIRMATION_WORLD_COUNT = 1_000
FROZEN_CONFIRMATION_BATCH_SIZE = 64
FROZEN_CONFIRMATION_CONFIG = RelayTrainingConfigV1()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Train one canonical repaired relay-v1 checkpoint, reload the exact persisted "
            "weights, and evaluate the fixed population ladder."
        )
    )
    parser.add_argument("--training-seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--steps", type=int, default=2_000)
    parser.add_argument("--train-batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--gradient-clip-norm", type=float, default=1.0)
    parser.add_argument("--gate-supervision-weight", type=float, default=1.0)
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
        default="results/population_compute_scaling_v1",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.evaluation_split == "confirmation" and not args.allow_confirmation:
        raise SystemExit(
            "Refusing to open frozen confirmation without --allow-confirmation"
        )

    training_config = RelayTrainingConfigV1(
        steps=args.steps,
        batch_size=args.train_batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        gradient_clip_norm=args.gradient_clip_norm,
        gate_supervision_weight=args.gate_supervision_weight,
        model=RelayPopulationConfig(
            state_width=args.state_width,
            message_width=args.message_width,
        ),
    )
    training_config.validate()

    if args.evaluation_split == "confirmation":
        _require_frozen_confirmation_configuration(
            training_config,
            evaluation_world_count=args.evaluation_world_count,
            evaluation_batch_size=args.evaluation_batch_size,
        )

    model, training = train_relay_model_v1(
        training_seed=args.training_seed,
        config=training_config,
        device=args.device,
    )
    output_dir = Path(args.output_dir) / f"seed_{args.training_seed}"
    checkpoint = save_relay_checkpoint_v1(
        model,
        training,
        training_config,
        output_dir / "model-v1.pt",
    )

    loaded_model, checkpoint_payload = load_relay_checkpoint_v1(
        checkpoint,
        device=args.device,
    )
    if loaded_model.parameter_fingerprint() != training.parameter_fingerprint:
        raise RuntimeError("reloaded relay-v1 checkpoint changed parameter fingerprint")
    if loaded_model.trainable_parameter_count() != training.learned_parameter_count:
        raise RuntimeError("reloaded relay-v1 checkpoint changed parameter count")
    saved_summary = checkpoint_payload.get("training_summary")
    if not isinstance(saved_summary, dict):
        raise RuntimeError("saved relay-v1 checkpoint lost training summary")
    if saved_summary.get("parameter_fingerprint") != training.parameter_fingerprint:
        raise RuntimeError("saved relay-v1 training summary fingerprint differs")

    evaluations = evaluate_relay_split_v1(
        loaded_model,
        training_seed=args.training_seed,
        split=args.evaluation_split,
        world_count=args.evaluation_world_count,
        batch_size=args.evaluation_batch_size,
        device=args.device,
        allow_confirmation=args.allow_confirmation,
    )
    result = RelayDevelopmentResultV1(
        experiment_version=RELAY_EXPERIMENT_V1,
        protocol_version=RELAY_PROTOCOL_VERSION,
        benchmark_version=COLLECTIVE_RELAY_VERSION,
        evaluation_split=args.evaluation_split,
        confirmation_opened=args.evaluation_split == "confirmation",
        training=training,
        training_config=training_config,
        evaluation_world_count=args.evaluation_world_count,
        evaluation_batch_size=args.evaluation_batch_size,
        evaluations=evaluations,
        assessments=assess_relay_results_v1(evaluations),
    )
    payload = result.to_dict()
    payload["provenance"] = {
        "checkpoint": str(checkpoint),
        "device": args.device,
        "execution_mode": "eager",
        "evaluated_parameter_fingerprint": loaded_model.parameter_fingerprint(),
        "serial_schedule_result": "serial_schedule_equivalence_result_v0.md",
    }
    result_name = f"{args.evaluation_split}.json"
    result_path = output_dir / result_name
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    print(
        json.dumps(
            {
                "checkpoint": str(checkpoint),
                "result": str(result_path),
                "experiment_version": RELAY_EXPERIMENT_V1,
                "protocol_version": RELAY_PROTOCOL_VERSION,
                "benchmark_version": COLLECTIVE_RELAY_VERSION,
                "parameter_fingerprint": training.parameter_fingerprint,
                "learned_parameter_count": training.learned_parameter_count,
                "evaluation_split": args.evaluation_split,
                "confirmation_opened": args.evaluation_split == "confirmation",
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def _require_frozen_confirmation_configuration(
    training_config: RelayTrainingConfigV1,
    *,
    evaluation_world_count: int,
    evaluation_batch_size: int,
) -> None:
    if training_config != FROZEN_CONFIRMATION_CONFIG:
        raise SystemExit(
            "Confirmation requires the frozen canonical relay-v1 training configuration"
        )
    if evaluation_world_count != FROZEN_CONFIRMATION_WORLD_COUNT:
        raise SystemExit(
            "Confirmation requires exactly 1000 frozen worlds per relay difficulty"
        )
    if evaluation_batch_size != FROZEN_CONFIRMATION_BATCH_SIZE:
        raise SystemExit("Confirmation requires the frozen evaluation batch size 64")


if __name__ == "__main__":
    raise SystemExit(main())
