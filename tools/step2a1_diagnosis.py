"""Step 2A.1 validation-only aggregation diagnosis.

This script does not touch the test split. It builds a small W=5 validation cache
of worker outputs and writes diagnostic summaries for aggregation analysis.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from statistics import mean
from typing import Any

import torch

from ai_hypothesis.step01.model import LABEL_TO_INDEX, NON_UNCERTAIN_LABELS, Step01Output, decode_predictions
from ai_hypothesis.step01.torch_data import make_loader
from ai_hypothesis.step02.evidence import AggregationConfig, aggregate_evidence, build_evidence_matrix
from ai_hypothesis.step02.evaluation import _majority_prediction_indices, _mean_probability_predictions, _worker_prediction_indices
from ai_hypothesis.step02.population import HomogeneousWorkerBank, PopulationOutput

LABELS_WITH_UNCERTAIN = (*NON_UNCERTAIN_LABELS, "UNCERTAIN")
ROOT = Path("results/step02/step2a1_aggregation_diagnosis")
CACHE_PATH = ROOT / "cache" / "w5_validation_logits.pt"
COUNT = 20_000
BATCH_SIZE = 256
SEED = 1
DEVICE = "cuda"
BACKEND = "vmap"


def checkpoint_paths() -> list[Path]:
    base = Path("results/step01/checkpoint_50k_extended_15k")
    return [base / f"seed_{seed}" / "best.pt" for seed in range(1, 6)]


def dump_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def metric_definitions() -> dict[str, dict[str, Any]]:
    return {
        "oracle_any_correct_coverage": {
            "numerator": "samples where at least one worker decoded prediction equals the ground-truth label",
            "denominator": "all evaluated samples",
            "conditional": False,
            "zero_denominator": "evaluation raises if total samples is zero",
            "w1_meaning": "meaningful; equals single decoded worker accuracy",
        },
        "all_wrong_rate": {
            "numerator": "samples where no worker decoded prediction equals the ground-truth label",
            "denominator": "all evaluated samples",
            "conditional": False,
            "zero_denominator": "evaluation raises if total samples is zero",
            "w1_meaning": "meaningful; equals 1 - single decoded worker accuracy",
        },
        "evidence_utilization_gap": {
            "numerator": "oracle_any_correct_count - evidence_reducer_correct_count",
            "denominator": "all evaluated samples",
            "conditional": False,
            "zero_denominator": "evaluation raises if total samples is zero",
            "w1_meaning": "mathematically defined but not a population utilization measure",
            "note": "Can differ from utilization_failure_rate when reducer is correct on samples where no worker decoded the truth.",
        },
        "utilization_failure_rate": {
            "numerator": "samples where at least one worker decoded truth and evidence reducer prediction is wrong",
            "denominator": "all evaluated samples",
            "conditional": False,
            "zero_denominator": "not in production evaluate_population; diagnostic-only here",
            "w1_meaning": "defined, but describes disagreement between single decoded worker and reducer, not population utilization",
        },
        "minority_rescue_opportunity_rate": {
            "numerator": "samples where majority vote is wrong and at least one worker decoded truth",
            "denominator": "all evaluated samples",
            "conditional": False,
            "zero_denominator": "evaluation raises if total samples is zero",
            "w1_meaning": "not meaningful as a minority metric; with W=1 it should be zero",
        },
        "minority_rescue_rate": {
            "numerator": "minority_rescues: rescue opportunities where evidence reducer is correct",
            "denominator": "minority_rescue_opportunities",
            "conditional": True,
            "zero_denominator": "currently reported as 0.0",
            "w1_meaning": "not applicable when no rescue opportunities exist; current 0.0 is a sentinel, not a real rate",
        },
        "minority_suppression_rate": {
            "numerator": "samples with strong_correct_minority true where evidence reducer is not correct",
            "denominator": "strong_correct_minority_cases",
            "conditional": True,
            "zero_denominator": "currently reported as 0.0",
            "w1_meaning": "not meaningful; current strong_correct_minority is actually strong-correct-evidence while majority != truth, not a true minority condition",
            "bug_or_issue": "misleading name/denominator for W=1 and non-minority cases",
        },
        "majority_harm_rate": {
            "numerator": "samples where majority vote is correct and evidence reducer is wrong",
            "denominator": "all evaluated samples",
            "conditional": False,
            "zero_denominator": "evaluation raises if total samples is zero",
            "w1_meaning": "defined, but at W=1 describes reducer changing a correct single-worker decoded answer to wrong",
        },
        "reducer_rescue_rate_total": {
            "numerator": "samples where majority vote is wrong, at least one worker decoded truth, and evidence reducer is correct",
            "denominator": "all evaluated samples",
            "conditional": False,
            "zero_denominator": "diagnostic-only here",
            "w1_meaning": "not meaningful as population rescue; should be zero",
        },
    }


def build_cache() -> dict[str, Any]:
    paths = checkpoint_paths()
    bank = HomogeneousWorkerBank.from_checkpoints(paths, device=DEVICE, execution_backend=BACKEND)
    loader = make_loader(split="validation", count=COUNT, batch_size=BATCH_SIZE, shuffle=False, seed=SEED, num_workers=0)
    label_chunks = []
    uncertainty_chunks = []
    labels: list[str] = []
    tasks: list[str] = []
    difficulties: list[str] = []
    sample_seeds: list[int] = []
    for batch in loader:
        with torch.inference_mode():
            output = bank(batch["features"], batch["mask"])
        label_chunks.append(output.label_logits.detach().cpu())
        uncertainty_chunks.append(output.uncertainty_logits.detach().cpu())
        for sample in batch["samples"]:
            labels.append(sample.label)
            tasks.append(sample.task.value)
            difficulties.append(sample.difficulty.value)
            sample_seeds.append(sample.seed)
    payload = {
        "runtime_version": "step02-population-runtime-v0",
        "evidence_contract_version": "step02-evidence-v0",
        "phase": "step2a1_validation_only_diagnostic_cache",
        "split": "validation",
        "count": COUNT,
        "batch_size": BATCH_SIZE,
        "seed": SEED,
        "device": DEVICE,
        "backend": BACKEND,
        "checkpoints": [str(path) for path in paths],
        "label_logits": torch.cat(label_chunks, dim=1),
        "uncertainty_logits": torch.cat(uncertainty_chunks, dim=1),
        "labels": labels,
        "tasks": tasks,
        "difficulties": difficulties,
        "sample_seeds": sample_seeds,
    }
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, CACHE_PATH)
    return payload


def load_or_build_cache() -> dict[str, Any]:
    if CACHE_PATH.exists():
        return torch.load(CACHE_PATH, map_location="cpu", weights_only=False)
    return build_cache()


def summarize(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"count": 0, "mean": None, "min": None, "p10": None, "median": None, "p90": None, "max": None}
    ordered = sorted(values)
    def q(frac: float) -> float:
        idx = min(len(ordered) - 1, max(0, round(frac * (len(ordered) - 1))))
        return ordered[idx]
    return {
        "count": len(values),
        "mean": mean(values),
        "min": ordered[0],
        "p10": q(0.10),
        "median": q(0.50),
        "p90": q(0.90),
        "max": ordered[-1],
    }


def analyze(cache: dict[str, Any]) -> None:
    config = AggregationConfig()
    label_logits = cache["label_logits"]
    uncertainty_logits = cache["uncertainty_logits"]
    output = PopulationOutput(label_logits=label_logits, uncertainty_logits=uncertainty_logits)
    task_lookup = {member.value: member for member in __import__("ai_hypothesis.step01.schema", fromlist=["TaskFamily"]).TaskFamily}
    tasks = tuple(task_lookup[value] for value in cache["tasks"])
    evidence = build_evidence_matrix(output, tasks, config)
    summary, decision = aggregate_evidence(evidence, config)
    worker_idx = _worker_prediction_indices(output, uncertainty_threshold=0.5)
    majority_idx = _majority_prediction_indices(worker_idx)
    majority_predictions = [LABELS_WITH_UNCERTAIN[int(idx)] for idx in majority_idx]
    mean_logit_output = Step01Output(label_logits=label_logits.mean(dim=0), uncertainty_logits=uncertainty_logits.mean(dim=0))
    mean_logit_predictions = list(decode_predictions(mean_logit_output, uncertainty_threshold=0.5))
    mean_probability_predictions = list(_mean_probability_predictions(output, uncertainty_threshold=0.5))

    counts = {
        "A_all_workers_wrong": 0,
        "B_any_worker_correct_and_reducer_correct": 0,
        "C_any_worker_correct_and_reducer_wrong": 0,
        "D_majority_wrong_and_reducer_correct": 0,
        "E_majority_wrong_reducer_wrong_any_worker_correct": 0,
        "F_majority_correct_reducer_wrong": 0,
        "G_mean_logit_correct_reducer_wrong": 0,
        "H_reducer_correct_mean_logit_wrong": 0,
    }
    overlap = {
        "all_samples_both_correct": 0,
        "all_samples_mean_logit_only": 0,
        "all_samples_reducer_only": 0,
        "all_samples_both_wrong": 0,
        "all_samples_both_wrong_oracle_available": 0,
        "utilization_failures_mean_logit_correct": 0,
        "utilization_failures_mean_logit_wrong": 0,
        "utilization_failures_majority_correct": 0,
        "utilization_failures_majority_wrong": 0,
    }
    totals = {
        "total": len(cache["labels"]),
        "oracle_any_correct": 0,
        "reducer_correct": 0,
        "majority_correct": 0,
        "mean_logit_correct": 0,
        "mean_probability_correct": 0,
        "utilization_failures": 0,
        "reducer_correct_without_decoded_worker_correct": 0,
        "strong_correct_minority_current_definition": 0,
        "minority_suppressions_current_definition": 0,
        "true_minority_correct_cases": 0,
        "true_minority_suppressions": 0,
        "minority_rescue_opportunities": 0,
        "minority_rescues": 0,
        "majority_harm": 0,
    }
    groups: dict[str, dict[str, list[float]]] = {
        "utilization_failure": {},
        "reducer_rescue": {},
        "all_oracle_available": {},
        "reducer_success_with_oracle": {},
    }
    failure_cases = []

    def add(group: str, key: str, value: float | None) -> None:
        if value is None:
            return
        groups[group].setdefault(key, []).append(float(value))

    for i, truth in enumerate(cache["labels"]):
        worker_predictions = [LABELS_WITH_UNCERTAIN[int(worker_idx[w, i])] for w in range(label_logits.shape[0])]
        worker_correct_flags = [prediction == truth for prediction in worker_predictions]
        correct_count = sum(worker_correct_flags)
        any_correct = correct_count > 0
        reducer_prediction = decision.predictions[i]
        reducer_correct = reducer_prediction == truth
        majority_prediction = majority_predictions[i]
        majority_correct = majority_prediction == truth
        mean_logit_correct = mean_logit_predictions[i] == truth
        mean_probability_correct = mean_probability_predictions[i] == truth
        totals["oracle_any_correct"] += int(any_correct)
        totals["reducer_correct"] += int(reducer_correct)
        totals["majority_correct"] += int(majority_correct)
        totals["mean_logit_correct"] += int(mean_logit_correct)
        totals["mean_probability_correct"] += int(mean_probability_correct)
        if not any_correct:
            counts["A_all_workers_wrong"] += 1
        if any_correct and reducer_correct:
            counts["B_any_worker_correct_and_reducer_correct"] += 1
        if any_correct and not reducer_correct:
            counts["C_any_worker_correct_and_reducer_wrong"] += 1
            totals["utilization_failures"] += 1
        if not majority_correct and reducer_correct:
            counts["D_majority_wrong_and_reducer_correct"] += 1
        if not majority_correct and not reducer_correct and any_correct:
            counts["E_majority_wrong_reducer_wrong_any_worker_correct"] += 1
        if majority_correct and not reducer_correct:
            counts["F_majority_correct_reducer_wrong"] += 1
            totals["majority_harm"] += 1
        if mean_logit_correct and not reducer_correct:
            counts["G_mean_logit_correct_reducer_wrong"] += 1
        if reducer_correct and not mean_logit_correct:
            counts["H_reducer_correct_mean_logit_wrong"] += 1
        if reducer_correct and not any_correct:
            totals["reducer_correct_without_decoded_worker_correct"] += 1

        if mean_logit_correct and reducer_correct:
            overlap["all_samples_both_correct"] += 1
        elif mean_logit_correct and not reducer_correct:
            overlap["all_samples_mean_logit_only"] += 1
        elif reducer_correct and not mean_logit_correct:
            overlap["all_samples_reducer_only"] += 1
        else:
            overlap["all_samples_both_wrong"] += 1
            overlap["all_samples_both_wrong_oracle_available"] += int(any_correct)
        if any_correct and not reducer_correct:
            overlap["utilization_failures_mean_logit_correct"] += int(mean_logit_correct)
            overlap["utilization_failures_mean_logit_wrong"] += int(not mean_logit_correct)
            overlap["utilization_failures_majority_correct"] += int(majority_correct)
            overlap["utilization_failures_majority_wrong"] += int(not majority_correct)

        rescue_opportunity = (not majority_correct) and any_correct
        totals["minority_rescue_opportunities"] += int(rescue_opportunity)
        totals["minority_rescues"] += int(rescue_opportunity and reducer_correct)

        truth_index = LABEL_TO_INDEX.get(truth)
        correct_max = correct_sum = correct_mean = correct_rank = protected_truth = unresolved = None
        true_evidence = None
        incorrect_max = incorrect_sum = None
        if truth_index is not None:
            true_evidence = evidence.evidence_scores[:, i, truth_index]
            correct_max = float(true_evidence.max())
            correct_sum = float(true_evidence.sum())
            correct_mean = float(summary.mean_evidence_per_label[i, truth_index])
            valid_mask = evidence.valid_label_mask[i]
            other_mask = valid_mask.clone()
            other_mask[truth_index] = False
            other_scores_worker = evidence.evidence_scores[:, i, :][:, other_mask]
            other_scores_sum = summary.sum_evidence_per_label[i, other_mask]
            other_scores_mean = summary.mean_evidence_per_label[i, other_mask]
            incorrect_max = float(other_scores_worker.max()) if other_scores_worker.numel() else None
            incorrect_sum = float(other_scores_sum.max()) if other_scores_sum.numel() else None
            ranked = torch.argsort(summary.mean_evidence_per_label[i].masked_fill(~valid_mask, float("-inf")), descending=True)
            correct_rank = int((ranked == truth_index).nonzero()[0].item()) + 1
            protected_truth = bool(summary.protected_label_mask[i, truth_index])
            unresolved = bool(decision.unresolved_contradiction[i])
            strong_correct_evidence = bool((true_evidence >= config.strong_evidence_threshold).any())
            current_strong_correct_minority = strong_correct_evidence and majority_prediction != truth
            totals["strong_correct_minority_current_definition"] += int(current_strong_correct_minority)
            totals["minority_suppressions_current_definition"] += int(current_strong_correct_minority and not reducer_correct)
            true_minority = any_correct and correct_count < (label_logits.shape[0] / 2) and majority_prediction != truth
            totals["true_minority_correct_cases"] += int(true_minority)
            totals["true_minority_suppressions"] += int(true_minority and not reducer_correct)

        common_metrics = {
            "correct_worker_count": correct_count,
            "correct_worker_fraction": correct_count / label_logits.shape[0],
            "strongest_correct_label_evidence": correct_max,
            "strongest_incorrect_label_evidence": incorrect_max,
            "cumulative_correct_label_evidence": correct_sum,
            "cumulative_best_incorrect_label_evidence": incorrect_sum,
            "mean_correct_label_evidence": correct_mean,
            "evidence_margin_sum_correct_minus_best_incorrect": None if correct_sum is None or incorrect_sum is None else correct_sum - incorrect_sum,
            "evidence_margin_max_correct_minus_best_incorrect": None if correct_max is None or incorrect_max is None else correct_max - incorrect_max,
            "population_mean_uncertainty": float(summary.mean_uncertainty[i]),
            "mean_invalid_label_mass": float(summary.mean_invalid_label_mass[i]),
            "disagreement_entropy": float(summary.disagreement_entropy[i]),
            "truth_rank_by_mean_evidence": correct_rank,
            "truth_protected": protected_truth,
            "unresolved_contradiction": unresolved,
            "truth_is_decoded_minority": any_correct and correct_count < (label_logits.shape[0] / 2),
        }
        if any_correct:
            for key, value in common_metrics.items():
                add("all_oracle_available", key, value if not isinstance(value, bool) else int(value))
        if any_correct and reducer_correct:
            for key, value in common_metrics.items():
                add("reducer_success_with_oracle", key, value if not isinstance(value, bool) else int(value))
        if any_correct and not reducer_correct:
            for key, value in common_metrics.items():
                add("utilization_failure", key, value if not isinstance(value, bool) else int(value))
            if len(failure_cases) < 200:
                failure_cases.append({
                    "sample_index": i,
                    "task": cache["tasks"][i],
                    "difficulty": cache["difficulties"][i],
                    "sample_seed": cache["sample_seeds"][i],
                    "truth": truth,
                    "worker_predictions": worker_predictions,
                    "majority_prediction": majority_prediction,
                    "mean_logit_prediction": mean_logit_predictions[i],
                    "mean_probability_prediction": mean_probability_predictions[i],
                    "reducer_prediction": reducer_prediction,
                    "uncertainty_reasons": list(decision.uncertainty_reasons[i]),
                    "metrics": common_metrics,
                })
        if rescue_opportunity and reducer_correct:
            for key, value in common_metrics.items():
                add("reducer_rescue", key, value if not isinstance(value, bool) else int(value))

    rate_counts = {key: {"count": value, "percent_all_samples": value / totals["total"]} for key, value in counts.items()}
    derived = {
        "oracle_any_correct_coverage": totals["oracle_any_correct"] / totals["total"],
        "evidence_reducer_accuracy": totals["reducer_correct"] / totals["total"],
        "evidence_utilization_gap": (totals["oracle_any_correct"] - totals["reducer_correct"]) / totals["total"],
        "utilization_failure_rate": totals["utilization_failures"] / totals["total"],
        "reducer_correct_without_decoded_worker_correct_rate": totals["reducer_correct_without_decoded_worker_correct"] / totals["total"],
        "gap_reconciliation": "utilization_failure_rate - reducer_correct_without_decoded_worker_correct_rate = evidence_utilization_gap",
        "minority_suppression_rate_current_definition": None if totals["strong_correct_minority_current_definition"] == 0 else totals["minority_suppressions_current_definition"] / totals["strong_correct_minority_current_definition"],
        "true_minority_suppression_rate": None if totals["true_minority_correct_cases"] == 0 else totals["true_minority_suppressions"] / totals["true_minority_correct_cases"],
        "minority_rescue_rate": None if totals["minority_rescue_opportunities"] == 0 else totals["minority_rescues"] / totals["minority_rescue_opportunities"],
    }
    distributions = {group: {key: summarize(values) for key, values in metrics.items()} for group, metrics in groups.items()}
    report = {
        "phase": "step2a1_validation_only_aggregation_diagnosis",
        "split": "validation",
        "count": totals["total"],
        "checkpoint_paths": cache["checkpoints"],
        "aggregation_config": asdict(config),
        "metric_definitions": metric_definitions(),
        "totals": totals,
        "failure_category_counts": rate_counts,
        "derived_rates": derived,
        "aggregator_overlap": overlap,
        "diagnostic_distributions": distributions,
        "by_task_failure_counts": {},
        "artifact_paths": {
            "cache": str(CACHE_PATH),
            "summary": str(ROOT / "analysis_summary.json"),
            "failure_cases_sample": str(ROOT / "utilization_failure_cases_sample.json"),
            "metric_definitions": str(ROOT / "metric_definitions.json"),
        },
    }
    by_task: dict[str, dict[str, int]] = {}
    for i, task in enumerate(cache["tasks"]):
        truth = cache["labels"][i]
        worker_predictions = [LABELS_WITH_UNCERTAIN[int(worker_idx[w, i])] for w in range(label_logits.shape[0])]
        any_correct = any(pred == truth for pred in worker_predictions)
        reducer_correct = decision.predictions[i] == truth
        row = by_task.setdefault(task, {"count": 0, "oracle_available": 0, "utilization_failure": 0, "reducer_correct": 0})
        row["count"] += 1
        row["oracle_available"] += int(any_correct)
        row["utilization_failure"] += int(any_correct and not reducer_correct)
        row["reducer_correct"] += int(reducer_correct)
    report["by_task_failure_counts"] = by_task
    dump_json(ROOT / "metric_definitions.json", metric_definitions())
    dump_json(ROOT / "analysis_summary.json", report)
    dump_json(ROOT / "utilization_failure_cases_sample.json", failure_cases)
    print(json.dumps({"event": "step2a1_diagnosis_complete", "output": str(ROOT / "analysis_summary.json")}, sort_keys=True))


def main() -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    dump_json(ROOT / "metric_definitions.json", metric_definitions())
    cache = load_or_build_cache()
    analyze(cache)


if __name__ == "__main__":
    main()
