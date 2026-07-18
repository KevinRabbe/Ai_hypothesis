"""Run the Step 1 candidate-size confirmation sweep strictly sequentially."""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Any

EXPERIMENTS: tuple[tuple[str, str, str], ...] = (
    (
        "25k",
        "configs/step01/checkpoint_25k_extended_15k.json",
        "results/step01/checkpoint_25k_extended_15k",
    ),
    (
        "50k",
        "configs/step01/checkpoint_50k_extended_15k.json",
        "results/step01/checkpoint_50k_extended_15k",
    ),
    (
        "75k",
        "configs/step01/checkpoint_75k_extended_15k.json",
        "results/step01/checkpoint_75k_extended_15k",
    ),
    (
        "100k",
        "configs/step01/checkpoint_100k_extended_15k.json",
        "results/step01/checkpoint_100k_extended_15k",
    ),
)

DEFAULT_SEEDS: tuple[int, ...] = (1, 2, 3, 4, 5)
QUALITY_THRESHOLDS: tuple[float, ...] = (0.90, 0.92, 0.93, 0.9382)
DEFAULT_SUMMARY_PATH = "results/step01/confirmation_25k_50k_75k_100k/summary.json"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run 25K, 50K, 75K, and 100K Step 1 confirmation experiments across "
            "multiple seeds, strictly one training process at a time."
        )
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="Device passed to every training process, for example cuda or cpu.",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=list(DEFAULT_SEEDS),
        help="Random seeds to run. Default: 1 2 3 4 5.",
    )
    parser.add_argument(
        "--summary-path",
        default=DEFAULT_SUMMARY_PATH,
        help="Combined JSON summary written after all requested runs complete.",
    )
    parser.add_argument(
        "--rerun-completed",
        action="store_true",
        help="Run an experiment again even when its result.json already exists.",
    )
    return parser.parse_args()


def _first_step_at_or_above(
    validation_history: list[dict[str, Any]], threshold: float
) -> int | None:
    for record in validation_history:
        score = float(record["validation"]["macro_task_accuracy"])
        if score >= threshold:
            return int(record["step"])
    return None


def _summarize_result(
    *, size_label: str, result_path: Path
) -> dict[str, Any]:
    result = json.loads(result_path.read_text(encoding="utf-8"))
    history = result.get("validation_history", [])
    threshold_steps = {
        f"{threshold:.4f}": _first_step_at_or_above(history, threshold)
        for threshold in QUALITY_THRESHOLDS
    }
    return {
        "size_label": size_label,
        "experiment_name": result["experiment_name"],
        "seed": int(result["train_config"]["seed"]),
        "parameter_count": int(result["parameter_count"]),
        "best_step": int(result["best_step"]),
        "best_validation_score": float(result["best_validation_score"]),
        "test_accuracy": float(result["test"]["accuracy"]),
        "training_duration_seconds": float(result["training_duration_seconds"]),
        "first_step_at_or_above": threshold_steps,
        "result_path": str(result_path),
    }


def _stats(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {
            "count": 0,
            "mean": None,
            "sample_stdev": None,
            "min": None,
            "max": None,
        }
    return {
        "count": len(values),
        "mean": statistics.mean(values),
        "sample_stdev": statistics.stdev(values) if len(values) >= 2 else None,
        "min": min(values),
        "max": max(values),
    }


def _aggregate_runs(runs: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for run in runs:
        grouped.setdefault(str(run["size_label"]), []).append(run)

    aggregates: dict[str, Any] = {}
    for size_label, size_runs in grouped.items():
        size_runs.sort(key=lambda item: int(item["seed"]))
        threshold_stats: dict[str, Any] = {}
        for threshold in QUALITY_THRESHOLDS:
            key = f"{threshold:.4f}"
            reached_steps = [
                float(run["first_step_at_or_above"][key])
                for run in size_runs
                if run["first_step_at_or_above"][key] is not None
            ]
            threshold_stats[key] = {
                "reached_count": len(reached_steps),
                "requested_seed_count": len(size_runs),
                "step_stats_for_reached_runs": _stats(reached_steps),
            }

        aggregates[size_label] = {
            "parameter_count": size_runs[0]["parameter_count"],
            "seeds": [run["seed"] for run in size_runs],
            "best_validation_score": _stats(
                [float(run["best_validation_score"]) for run in size_runs]
            ),
            "test_accuracy": _stats(
                [float(run["test_accuracy"]) for run in size_runs]
            ),
            "best_step": _stats([float(run["best_step"]) for run in size_runs]),
            "training_duration_seconds": _stats(
                [float(run["training_duration_seconds"]) for run in size_runs]
            ),
            "first_step_at_or_above": threshold_stats,
        }

    return aggregates


def main() -> None:
    args = _parse_args()
    seeds = list(dict.fromkeys(args.seeds))
    if not seeds:
        raise ValueError("at least one seed is required")

    total_runs = len(seeds) * len(EXPERIMENTS)
    run_index = 0
    summaries: list[dict[str, Any]] = []

    for seed in seeds:
        for size_label, config_string, output_root_string in EXPERIMENTS:
            run_index += 1
            config_path = Path(config_string)
            if not config_path.is_file():
                raise FileNotFoundError(f"experiment config not found: {config_path}")

            output_dir = Path(output_root_string) / f"seed_{seed}"
            result_path = output_dir / "result.json"

            if result_path.exists() and not args.rerun_completed:
                summary = _summarize_result(
                    size_label=size_label,
                    result_path=result_path,
                )
                summaries.append(summary)
                print(
                    json.dumps(
                        {
                            "event": "confirmation_skip_completed",
                            "index": run_index,
                            "total": total_runs,
                            "size_label": size_label,
                            "seed": seed,
                            "result_path": str(result_path),
                        }
                    )
                )
                continue

            command = [
                sys.executable,
                "-m",
                "ai_hypothesis.step01.train_reference",
                "--config",
                str(config_path),
                "--device",
                args.device,
                "--seed",
                str(seed),
                "--output-dir",
                str(output_dir),
            ]
            print(
                json.dumps(
                    {
                        "event": "confirmation_experiment_start",
                        "index": run_index,
                        "total": total_runs,
                        "size_label": size_label,
                        "seed": seed,
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
                            "event": "confirmation_stopped_on_failure",
                            "index": run_index,
                            "size_label": size_label,
                            "seed": seed,
                            "return_code": completed.returncode,
                        }
                    )
                )
                raise SystemExit(completed.returncode)

            if not result_path.is_file():
                raise RuntimeError(
                    "training process succeeded but result file was not created: "
                    f"{result_path}"
                )

            summary = _summarize_result(
                size_label=size_label,
                result_path=result_path,
            )
            summaries.append(summary)
            print(json.dumps({"event": "confirmation_experiment_complete", **summary}))

    summary_path = Path(args.summary_path)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    combined = {
        "execution_mode": "strictly_sequential",
        "device": args.device,
        "requested_seeds": seeds,
        "quality_thresholds": list(QUALITY_THRESHOLDS),
        "target_100k_original_best_validation": 0.9382,
        "runs": summaries,
        "aggregates_by_size": _aggregate_runs(summaries),
    }
    summary_path.write_text(json.dumps(combined, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "event": "confirmation_complete",
                "run_count": len(summaries),
                "summary_path": str(summary_path),
            }
        )
    )


if __name__ == "__main__":
    main()
