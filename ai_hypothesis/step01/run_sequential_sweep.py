"""Run the Step 1 mid-size checkpoint sweep strictly one experiment at a time."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

DEFAULT_CONFIGS: tuple[str, ...] = (
    "configs/step01/checkpoint_25k_extended_15k.json",
    "configs/step01/checkpoint_50k_extended_15k.json",
    "configs/step01/checkpoint_75k_extended_15k.json",
)

QUALITY_THRESHOLDS: tuple[float, ...] = (0.90, 0.92, 0.93, 0.9382)
DEFAULT_SUMMARY_PATH = "results/step01/sweep_25k_50k_75k/summary.json"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run Step 1 checkpoint experiments sequentially. The next experiment "
            "starts only after the previous process exits successfully."
        )
    )
    parser.add_argument(
        "--configs",
        nargs="+",
        default=list(DEFAULT_CONFIGS),
        help="Experiment config paths in the exact order they should run.",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="Device passed to each training process, for example cuda or cpu.",
    )
    parser.add_argument(
        "--summary-path",
        default=DEFAULT_SUMMARY_PATH,
        help="Combined JSON summary written after all experiments complete.",
    )
    parser.add_argument(
        "--rerun-completed",
        action="store_true",
        help="Run experiments again even when their result.json already exists.",
    )
    return parser.parse_args()


def _load_output_dir(config_path: Path) -> Path:
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    output_dir = payload.get("training", {}).get("output_dir")
    if not isinstance(output_dir, str) or not output_dir:
        raise ValueError(f"config {config_path} has no valid training.output_dir")
    return Path(output_dir)


def _first_step_at_or_above(
    validation_history: list[dict[str, Any]], threshold: float
) -> int | None:
    for record in validation_history:
        score = float(record["validation"]["macro_task_accuracy"])
        if score >= threshold:
            return int(record["step"])
    return None


def _summarize_result(result_path: Path) -> dict[str, Any]:
    result = json.loads(result_path.read_text(encoding="utf-8"))
    history = result.get("validation_history", [])
    threshold_steps = {
        f"{threshold:.4f}": _first_step_at_or_above(history, threshold)
        for threshold in QUALITY_THRESHOLDS
    }
    return {
        "experiment_name": result["experiment_name"],
        "parameter_count": result["parameter_count"],
        "best_step": result["best_step"],
        "best_validation_score": result["best_validation_score"],
        "test_accuracy": result["test"]["accuracy"],
        "training_duration_seconds": result["training_duration_seconds"],
        "first_step_at_or_above": threshold_steps,
        "result_path": str(result_path),
    }


def main() -> None:
    args = _parse_args()
    configs = [Path(path) for path in args.configs]
    summaries: list[dict[str, Any]] = []

    for index, config_path in enumerate(configs, start=1):
        if not config_path.is_file():
            raise FileNotFoundError(f"experiment config not found: {config_path}")

        output_dir = _load_output_dir(config_path)
        result_path = output_dir / "result.json"

        if result_path.exists() and not args.rerun_completed:
            print(
                json.dumps(
                    {
                        "event": "sweep_skip_completed",
                        "index": index,
                        "total": len(configs),
                        "config": str(config_path),
                        "result_path": str(result_path),
                    }
                )
            )
            summaries.append(_summarize_result(result_path))
            continue

        command = [
            sys.executable,
            "-m",
            "ai_hypothesis.step01.train_reference",
            "--config",
            str(config_path),
            "--device",
            args.device,
        ]
        print(
            json.dumps(
                {
                    "event": "sweep_experiment_start",
                    "index": index,
                    "total": len(configs),
                    "config": str(config_path),
                    "command": command,
                }
            )
        )

        completed = subprocess.run(command, check=False)
        if completed.returncode != 0:
            print(
                json.dumps(
                    {
                        "event": "sweep_stopped_on_failure",
                        "index": index,
                        "config": str(config_path),
                        "return_code": completed.returncode,
                    }
                )
            )
            raise SystemExit(completed.returncode)

        if not result_path.is_file():
            raise RuntimeError(
                f"training process succeeded but result file was not created: {result_path}"
            )

        summary = _summarize_result(result_path)
        summaries.append(summary)
        print(json.dumps({"event": "sweep_experiment_complete", **summary}))

    summary_path = Path(args.summary_path)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    combined = {
        "execution_mode": "strictly_sequential",
        "device": args.device,
        "quality_thresholds": list(QUALITY_THRESHOLDS),
        "target_100k_best_validation": 0.9382,
        "experiments": summaries,
    }
    summary_path.write_text(json.dumps(combined, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "event": "sweep_complete",
                "experiment_count": len(summaries),
                "summary_path": str(summary_path),
            }
        )
    )


if __name__ == "__main__":
    main()
