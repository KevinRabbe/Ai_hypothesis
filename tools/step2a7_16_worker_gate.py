"""Step 2A.7 staged 16-worker population-scaling gate.

Validation/development only. Does not access the frozen test split for population
evaluation and does not train or reference workers beyond seed 16.
"""

from __future__ import annotations

import json
import statistics
import subprocess
import time
import warnings
from dataclasses import asdict
from itertools import combinations
from pathlib import Path
from typing import Any

import torch

from ai_hypothesis.step01.model import NON_UNCERTAIN_LABELS, Step01Output, decode_predictions
from ai_hypothesis.step01.schema import BENCHMARK_VERSION, TaskFamily
from ai_hypothesis.step01.torch_data import make_loader
from ai_hypothesis.step02.evidence import AggregationConfig, aggregate_evidence, build_evidence_matrix
from ai_hypothesis.step02.evaluation import (
    _majority_prediction_indices,
    _mean_probability_predictions,
    _worker_prediction_indices,
)
from ai_hypothesis.step02.metrics import conditional_rate, is_true_minority_opportunity
from ai_hypothesis.step02.population import HomogeneousWorkerBank, PopulationOutput

ROOT = Path("results/step02/step2a7_16_worker_scaling_gate")
CACHE_DIR = ROOT / "cache"
COUNT = 20_000
BATCH_SIZE = 256
BENCHMARK_COUNT = 4096
DATASET_SPLIT = "validation"
DATASET_SEED = 1
DEVICE = "cuda"
WIDTHS = (1, 4, 16)
LABELS = (*NON_UNCERTAIN_LABELS, "UNCERTAIN")


def dump(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def git_revision() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()


def checkpoint_paths(width: int = 16) -> list[Path]:
    root = Path("results/step01/checkpoint_50k_extended_15k")
    return [root / f"seed_{seed}" / "best.pt" for seed in range(1, width + 1)]


def result_paths() -> list[Path]:
    root = Path("results/step01/checkpoint_50k_extended_15k")
    return [root / f"seed_{seed}" / "result.json" for seed in range(1, 17)]


def first_step_at_or_above(history: list[dict[str, Any]], threshold: float) -> int | None:
    for record in history:
        if float(record["validation"]["macro_task_accuracy"]) >= threshold:
            return int(record["step"])
    return None


def verify_workers() -> dict[str, Any]:
    rows = []
    for seed, result_path in enumerate(result_paths(), start=1):
        checkpoint_path = result_path.parent / "best.pt"
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"missing checkpoint for seed {seed}: {checkpoint_path}")
        if not result_path.is_file():
            raise FileNotFoundError(f"missing result for seed {seed}: {result_path}")
        result = json.loads(result_path.read_text(encoding="utf-8"))
        if int(result["parameter_count"]) != 50268:
            raise RuntimeError(f"seed {seed} parameter count mismatch")
        if result["benchmark_version"] != BENCHMARK_VERSION:
            raise RuntimeError(f"seed {seed} benchmark mismatch")
        if result["architecture_version"] != "step01-unit-v0":
            raise RuntimeError(f"seed {seed} architecture mismatch")
        rows.append(
            {
                "seed": seed,
                "checkpoint_path": str(checkpoint_path),
                "result_path": str(result_path),
                "parameter_count": int(result["parameter_count"]),
                "benchmark_version": result["benchmark_version"],
                "architecture_version": result["architecture_version"],
                "best_validation_score": float(result["best_validation_score"]),
                "best_step": int(result["best_step"]),
                "training_duration_seconds": float(result["training_duration_seconds"]),
                "first_step_at_or_above": {
                    f"{threshold:.4f}": first_step_at_or_above(result["validation_history"], threshold)
                    for threshold in (0.90, 0.92, 0.93, 0.94, 0.945)
                },
                "test_accuracy_from_existing_step1_protocol": float(result["test"]["accuracy"]),
            }
        )
    validations = [row["best_validation_score"] for row in rows]
    durations = [row["training_duration_seconds"] for row in rows]
    return {
        "rows": rows,
        "validation_stats": {
            "count": len(validations),
            "mean": statistics.mean(validations),
            "sample_stdev": statistics.stdev(validations),
            "min": min(validations),
            "max": max(validations),
        },
        "training_duration_seconds_stats": {
            "mean": statistics.mean(durations),
            "sample_stdev": statistics.stdev(durations),
            "min": min(durations),
            "max": max(durations),
        },
    }


def task_members(values: list[str]) -> tuple[TaskFamily, ...]:
    lookup = {member.value: member for member in TaskFamily}
    return tuple(lookup[value] for value in values)


def collect_outputs(width: int, backend: str, count: int) -> tuple[dict[str, Any], dict[str, Any]]:
    bank = HomogeneousWorkerBank.from_checkpoints(
        checkpoint_paths(width),
        device=DEVICE,
        execution_backend=backend,
    )
    loader = make_loader(
        split=DATASET_SPLIT,
        count=count,
        batch_size=BATCH_SIZE,
        shuffle=False,
        seed=DATASET_SEED,
        num_workers=0,
    )
    label_chunks = []
    uncertainty_chunks = []
    labels: list[str] = []
    tasks: list[str] = []
    difficulties: list[str] = []
    seeds: list[int] = []
    if DEVICE == "cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    start = time.perf_counter()
    captured_warnings: list[str] = []
    with warnings.catch_warnings(record=True) as records:
        warnings.simplefilter("always")
        for batch in loader:
            with torch.inference_mode():
                output = bank(batch["features"], batch["mask"])
            label_chunks.append(output.label_logits.detach().cpu())
            uncertainty_chunks.append(output.uncertainty_logits.detach().cpu())
            for sample in batch["samples"]:
                labels.append(sample.label)
                tasks.append(sample.task.value)
                difficulties.append(sample.difficulty.value)
                seeds.append(sample.seed)
        captured_warnings = [str(record.message) for record in records]
    if DEVICE == "cuda":
        torch.cuda.synchronize()
        peak_bytes = int(torch.cuda.max_memory_allocated())
    else:
        peak_bytes = None
    wall = time.perf_counter() - start
    cache = {
        "phase": "step2a7_validation_population_cache",
        "split": DATASET_SPLIT,
        "count": count,
        "seed": DATASET_SEED,
        "batch_size": BATCH_SIZE,
        "backend": backend,
        "device": DEVICE,
        "checkpoint_paths": [str(path) for path in checkpoint_paths(width)],
        "label_logits": torch.cat(label_chunks, dim=1),
        "uncertainty_logits": torch.cat(uncertainty_chunks, dim=1),
        "labels": labels,
        "tasks": tasks,
        "difficulties": difficulties,
        "sample_seeds": seeds,
    }
    runtime = {
        "width": width,
        "backend": backend,
        "count": count,
        "wall_time_seconds": wall,
        "samples_per_second": count / wall,
        "worker_evaluations_per_second": count * width / wall,
        "peak_gpu_memory_bytes": peak_bytes,
        "fallback_warning_count": sum(
            "BatchedFallback" in warning or "batching rule" in warning
            for warning in captured_warnings
        ),
        "warnings": sorted(set(captured_warnings)),
    }
    return cache, runtime


def benchmark_w16() -> dict[str, Any]:
    rows = []
    for backend in ("loop", "vmap"):
        _, runtime = collect_outputs(16, backend, BENCHMARK_COUNT)
        rows.append(runtime)
    chosen = max(rows, key=lambda row: row["samples_per_second"])["backend"]
    return {"rows": rows, "chosen_backend": chosen}


def accuracy(predictions: list[str], labels: list[str]) -> float:
    return sum(pred == truth for pred, truth in zip(predictions, labels, strict=True)) / len(labels)


def evaluate_cache(cache: dict[str, Any], width: int) -> dict[str, Any]:
    output = PopulationOutput(
        cache["label_logits"][:width],
        cache["uncertainty_logits"][:width],
    )
    labels = cache["labels"]
    tasks = task_members(cache["tasks"])
    evidence = build_evidence_matrix(output, tasks, AggregationConfig())
    summary, decision = aggregate_evidence(evidence, AggregationConfig())
    worker_idx = _worker_prediction_indices(output, uncertainty_threshold=0.5)
    majority_idx = _majority_prediction_indices(worker_idx)
    majority = [LABELS[int(idx)] for idx in majority_idx]
    mean_logit_output = Step01Output(output.label_logits.mean(dim=0), output.uncertainty_logits.mean(dim=0))
    mean_logit = list(decode_predictions(mean_logit_output, uncertainty_threshold=0.5))
    mean_prob = list(_mean_probability_predictions(output, uncertainty_threshold=0.5))
    reducer = list(decision.predictions)

    worker_correct = torch.zeros((width, len(labels)), dtype=torch.bool)
    for worker in range(width):
        worker_correct[worker] = torch.tensor(
            [LABELS[int(worker_idx[worker, sample])] == labels[sample] for sample in range(len(labels))],
            dtype=torch.bool,
        )
    correct_counts = worker_correct.sum(dim=0)
    oracle_flags = correct_counts > 0
    majority_correct_flags = torch.tensor([majority[i] == labels[i] for i in range(len(labels))], dtype=torch.bool)
    reducer_correct_flags = torch.tensor([reducer[i] == labels[i] for i in range(len(labels))], dtype=torch.bool)

    true_minority_opp = []
    for sample_index in range(len(labels)):
        true_minority_opp.append(
            is_true_minority_opportunity(
                correct_worker_count=int(correct_counts[sample_index]),
                population_width=width,
                majority_is_correct=bool(majority_correct_flags[sample_index]),
            )
        )
    true_minority_opp_t = torch.tensor(true_minority_opp, dtype=torch.bool)
    true_minority_rescues = true_minority_opp_t & reducer_correct_flags
    true_minority_suppressions = true_minority_opp_t & ~reducer_correct_flags
    majority_harm = majority_correct_flags & ~reducer_correct_flags

    true_minority_rescue_rate = conditional_rate(int(true_minority_rescues.sum()), int(true_minority_opp_t.sum()))
    true_minority_suppression_rate = conditional_rate(int(true_minority_suppressions.sum()), int(true_minority_opp_t.sum()))
    return {
        "width": width,
        "count": len(labels),
        "majority_accuracy": accuracy(majority, labels),
        "mean_logit_accuracy": accuracy(mean_logit, labels),
        "mean_probability_accuracy": accuracy(mean_prob, labels),
        "reducer_v0_accuracy": accuracy(reducer, labels),
        "oracle_any_correct_coverage": float(oracle_flags.float().mean()),
        "all_wrong_rate": float((~oracle_flags).float().mean()),
        "evidence_utilization_gap": float(oracle_flags.float().mean()) - accuracy(reducer, labels),
        "mean_disagreement_entropy": float(summary.disagreement_entropy.mean()),
        "mean_population_uncertainty": float(summary.mean_uncertainty.mean()),
        "mean_invalid_label_mass": float(summary.mean_invalid_label_mass.mean()),
        "true_minority_opportunity": {
            "numerator": int(true_minority_opp_t.sum()),
            "denominator": len(labels),
            "rate": float(true_minority_opp_t.float().mean()),
        },
        "true_minority_rescue": asdict(true_minority_rescue_rate),
        "true_minority_suppression": asdict(true_minority_suppression_rate),
        "majority_harm": {
            "numerator": int(majority_harm.sum()),
            "denominator": len(labels),
            "rate": float(majority_harm.float().mean()),
        },
        "correct_worker_count_histogram": {
            str(i): int((correct_counts == i).sum()) for i in range(width + 1)
        },
    }


def diversity_analysis(cache: dict[str, Any]) -> dict[str, Any]:
    width = 16
    output = PopulationOutput(cache["label_logits"], cache["uncertainty_logits"])
    labels = cache["labels"]
    worker_idx = _worker_prediction_indices(output, uncertainty_threshold=0.5)
    worker_correct = torch.zeros((width, len(labels)), dtype=torch.bool)
    for worker in range(width):
        worker_correct[worker] = torch.tensor(
            [LABELS[int(worker_idx[worker, sample])] == labels[sample] for sample in range(len(labels))],
            dtype=torch.bool,
        )
    pair_agreements = []
    pair_error_correlations = []
    pair_joint_wrong = []
    for a, b in combinations(range(width), 2):
        pair_agreements.append(float((worker_idx[a] == worker_idx[b]).float().mean()))
        err_a = (~worker_correct[a]).float()
        err_b = (~worker_correct[b]).float()
        pair_joint_wrong.append(float((err_a.bool() & err_b.bool()).float().mean()))
        if float(err_a.std(unbiased=False)) == 0.0 or float(err_b.std(unbiased=False)) == 0.0:
            corr = None
        else:
            corr = float(torch.corrcoef(torch.stack([err_a, err_b]))[0, 1])
        pair_error_correlations.append(corr)

    correct_counts = worker_correct.sum(dim=0)
    unique_correct = []
    for worker in range(width):
        unique_correct.append(
            {
                "worker_index": worker,
                "seed": worker + 1,
                "unique_correct_count": int((worker_correct[worker] & (correct_counts == 1)).sum()),
            }
        )
    discovered_w1 = worker_correct[:1].any(dim=0)
    discovered_w4 = worker_correct[:4].any(dim=0)
    discovered_w16 = worker_correct[:16].any(dim=0)
    return {
        "pairwise_prediction_agreement": {
            "mean": statistics.mean(pair_agreements),
            "min": min(pair_agreements),
            "max": max(pair_agreements),
        },
        "pairwise_error_correlation": {
            "mean": statistics.mean([value for value in pair_error_correlations if value is not None]),
            "min": min(value for value in pair_error_correlations if value is not None),
            "max": max(value for value in pair_error_correlations if value is not None),
            "undefined_count": sum(value is None for value in pair_error_correlations),
        },
        "pairwise_joint_wrong_rate": {
            "mean": statistics.mean(pair_joint_wrong),
            "min": min(pair_joint_wrong),
            "max": max(pair_joint_wrong),
        },
        "unique_correct_contribution": unique_correct,
        "correct_worker_count_histogram": {
            str(i): int((correct_counts == i).sum()) for i in range(width + 1)
        },
        "new_discoveries_1_to_4": int((~discovered_w1 & discovered_w4).sum()),
        "new_discoveries_4_to_16": int((~discovered_w4 & discovered_w16).sum()),
    }


def deltas(rows: dict[int, dict[str, Any]]) -> dict[str, Any]:
    practical_keys = ("majority_accuracy", "mean_logit_accuracy", "mean_probability_accuracy", "reducer_v0_accuracy")
    return {
        "oracle_gain": {
            "w1_to_w4": rows[4]["oracle_any_correct_coverage"] - rows[1]["oracle_any_correct_coverage"],
            "w4_to_w16": rows[16]["oracle_any_correct_coverage"] - rows[4]["oracle_any_correct_coverage"],
        },
        "all_wrong_reduction": {
            "w1_to_w4": rows[1]["all_wrong_rate"] - rows[4]["all_wrong_rate"],
            "w4_to_w16": rows[4]["all_wrong_rate"] - rows[16]["all_wrong_rate"],
        },
        "practical_aggregator_gain": {
            key: {
                "w1_to_w4": rows[4][key] - rows[1][key],
                "w4_to_w16": rows[16][key] - rows[4][key],
            }
            for key in practical_keys
        },
        "utilization_gap_change": {
            "w1_to_w4": rows[4]["evidence_utilization_gap"] - rows[1]["evidence_utilization_gap"],
            "w4_to_w16": rows[16]["evidence_utilization_gap"] - rows[4]["evidence_utilization_gap"],
        },
    }


def verdict_from(rows: dict[int, dict[str, Any]], delta_rows: dict[str, Any]) -> str:
    oracle_w4_to_w16 = delta_rows["oracle_gain"]["w4_to_w16"]
    best_practical_w4_to_w16 = max(
        item["w4_to_w16"] for item in delta_rows["practical_aggregator_gain"].values()
    )
    if oracle_w4_to_w16 >= 0.005 and best_practical_w4_to_w16 >= 0.002:
        return "PRACTICAL_SCALING"
    if oracle_w4_to_w16 >= 0.005:
        return "CONTINUING_DISCOVERY"
    if oracle_w4_to_w16 < 0.002:
        return "DISCOVERY_SATURATION"
    return "MIXED_INCONCLUSIVE"


def main() -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    verification = verify_workers()
    benchmark = benchmark_w16()
    chosen_backend = benchmark["chosen_backend"]

    runtime_scaling = []
    caches = {}
    for width in WIDTHS:
        cache_path = CACHE_DIR / f"w{width}_validation_{chosen_backend}.pt"
        cache, runtime = collect_outputs(width, chosen_backend, COUNT)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(cache, cache_path)
        runtime["cache_path"] = str(cache_path)
        runtime_scaling.append(runtime)
        caches[width] = cache

    population_rows = {width: evaluate_cache(caches[width], width) for width in WIDTHS}
    diversity = diversity_analysis(caches[16])
    delta_rows = deltas(population_rows)
    verdict = verdict_from(population_rows, delta_rows)
    manifest = {
        "phase": "step2a7_16_worker_scaling_gate",
        "frozen_test_accessed_for_population": False,
        "workers_trained": list(range(6, 17)),
        "workers_beyond_seed_16_trained": False,
        "population_widths": list(WIDTHS),
        "dataset_contract": {
            "split": DATASET_SPLIT,
            "count": COUNT,
            "seed": DATASET_SEED,
            "batch_size": BATCH_SIZE,
        },
        "benchmark_version": BENCHMARK_VERSION,
        "git_revision": git_revision(),
        "aggregation_config": asdict(AggregationConfig()),
        "checkpoint_paths": [str(path) for path in checkpoint_paths(16)],
    }
    summary = {
        **manifest,
        "worker_verification": verification,
        "w16_backend_benchmark": benchmark,
        "chosen_backend": chosen_backend,
        "runtime_scaling": runtime_scaling,
        "population_results": {str(width): population_rows[width] for width in WIDTHS},
        "scaling_deltas": delta_rows,
        "diversity_analysis": diversity,
        "verdict": verdict,
        "recommendation": (
            "proceed_to_w64"
            if verdict in {"CONTINUING_DISCOVERY", "PRACTICAL_SCALING"}
            else "stop_and_reconsider_worker_diversity_or_evidence_design"
        ),
    }
    dump(ROOT / "manifest.json", manifest)
    dump(ROOT / "summary.json", summary)
    dump(ROOT / "worker_verification.json", verification)
    dump(ROOT / "population_results.json", {str(width): population_rows[width] for width in WIDTHS})
    dump(ROOT / "diversity_analysis.json", diversity)
    print(json.dumps({"event": "step2a7_complete", "verdict": verdict, "summary": str(ROOT / "summary.json")}, sort_keys=True))


if __name__ == "__main__":
    main()
