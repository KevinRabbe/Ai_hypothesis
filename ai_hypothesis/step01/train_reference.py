"""Command-line entry point for the Step 1 10M reference experiment."""

from __future__ import annotations

import argparse
from dataclasses import replace

from .model import UnitConfig
from .training import TrainConfig, load_experiment_config, run_training


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train and evaluate the Step 1 reference neural unit."
    )
    parser.add_argument(
        "--config",
        default="configs/step01/reference_10m.json",
        help="Path to the experiment JSON configuration.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Optional device override, for example cuda, cuda:0, or cpu.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Optional result/checkpoint directory override.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Optional training-step override.",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run a very small end-to-end training check instead of the real 10M experiment.",
    )
    return parser.parse_args()


def _smoke_configs(train_config: TrainConfig) -> tuple[UnitConfig, TrainConfig]:
    unit_config = UnitConfig(
        d_model=32,
        block_count=2,
        attention_heads=4,
        feed_forward_width=64,
        dropout=0.0,
    )
    train_config = replace(
        train_config,
        experiment_name="step01_smoke_test",
        train_count=256,
        validation_count=80,
        test_count=80,
        batch_size=16,
        max_training_steps=4,
        eval_interval=2,
        early_stopping_patience=2,
        output_dir="results/step01/smoke_test",
    )
    return unit_config, train_config


def main() -> None:
    args = _parse_args()
    unit_config, train_config = load_experiment_config(args.config)

    if args.smoke_test:
        unit_config, train_config = _smoke_configs(train_config)

    if args.device is not None:
        train_config = replace(train_config, device=args.device)
    if args.output_dir is not None:
        train_config = replace(train_config, output_dir=args.output_dir)
    if args.max_steps is not None:
        train_config = replace(train_config, max_training_steps=args.max_steps)

    run_training(unit_config=unit_config, train_config=train_config, verbose=True)


if __name__ == "__main__":
    main()
