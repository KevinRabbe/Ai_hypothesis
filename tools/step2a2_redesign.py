"""Step 2A.2 validation/dev-only evidence utilization redesign."""

from __future__ import annotations

import json
from dataclasses import asdict, replace
from pathlib import Path
from statistics import mean
from typing import Any, Callable

import torch

from ai_hypothesis.step01.generator import generate_sample
from ai_hypothesis.step01.model import LABEL_TO_INDEX, NON_UNCERTAIN_LABELS, Step01Output, decode_predictions
from ai_hypothesis.step01.schema import BENCHMARK_VERSION, DIFFICULTIES, TASKS
from ai_hypothesis.step01.torch_data import collate_samples
from ai_hypothesis.step02.evidence import AggregationConfig, EvidenceBatch, aggregate_evidence, build_evidence_matrix
from ai_hypothesis.step02.evaluation import _majority_prediction_indices, _mean_probability_predictions, _worker_prediction_indices
from ai_hypothesis.step02.population import HomogeneousWorkerBank, PopulationOutput

ROOT = Path("results/step02/step2a2_evidence_utilization_redesign")
DEV_CACHE = Path("results/step02/step2a1_aggregation_diagnosis/cache/w5_validation_logits.pt")
CONFIRM_CACHE = ROOT / "cache" / "w5_dev_confirmation_logits.pt"
COUNT = 20_000
BATCH_SIZE = 256
DEV_CONFIRM_BASE_SEED = 3_000_000_000
DEVICE = "cuda"
BACKEND = "vmap"
LABELS_WITH_UNCERTAIN = (*NON_UNCERTAIN_LABELS, "UNCERTAIN")


def checkpoint_paths() -> list[Path]:
    base = Path("results/step01/checkpoint_50k_extended_15k")
    return [base / f"seed_{seed}" / "best.pt" for seed in range(1, 6)]


def dump(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def load_dev_cache() -> dict[str, Any]:
    return torch.load(DEV_CACHE, map_location="cpu", weights_only=False)


def dev_confirmation_samples(count: int):
    pair_count = len(TASKS) * len(DIFFICULTIES)
    for index in range(count):
        task = TASKS[index % len(TASKS)]
        difficulty = DIFFICULTIES[(index // len(TASKS)) % len(DIFFICULTIES)]
        cycle = index // pair_count
        seed = DEV_CONFIRM_BASE_SEED + cycle * pair_count + index % pair_count
        yield generate_sample(task, difficulty, seed)


def build_confirmation_cache() -> dict[str, Any]:
    paths = checkpoint_paths()
    bank = HomogeneousWorkerBank.from_checkpoints(paths, device=DEVICE, execution_backend=BACKEND)
    label_chunks = []
    uncertainty_chunks = []
    labels: list[str] = []
    tasks: list[str] = []
    difficulties: list[str] = []
    seeds: list[int] = []
    batch: list[Any] = []
    for sample in dev_confirmation_samples(COUNT):
        batch.append(sample)
        if len(batch) == BATCH_SIZE:
            collated = collate_samples(batch)
            with torch.inference_mode():
                output = bank(collated["features"], collated["mask"])
            label_chunks.append(output.label_logits.detach().cpu())
            uncertainty_chunks.append(output.uncertainty_logits.detach().cpu())
            for item in batch:
                labels.append(item.label)
                tasks.append(item.task.value)
                difficulties.append(item.difficulty.value)
                seeds.append(item.seed)
            batch = []
    if batch:
        collated = collate_samples(batch)
        with torch.inference_mode():
            output = bank(collated["features"], collated["mask"])
        label_chunks.append(output.label_logits.detach().cpu())
        uncertainty_chunks.append(output.uncertainty_logits.detach().cpu())
        for item in batch:
            labels.append(item.label)
            tasks.append(item.task.value)
            difficulties.append(item.difficulty.value)
            seeds.append(item.seed)
    payload = {
        "runtime_version": "step02-population-runtime-v0",
        "evidence_contract_version": "step02-evidence-v0",
        "phase": "step2a2_dev_confirmation_only",
        "split": "development_confirmation_not_test",
        "benchmark_version": BENCHMARK_VERSION,
        "generation_contract": "balanced TASKS x DIFFICULTIES stream using generate_sample with explicit non-test base seed",
        "base_seed": DEV_CONFIRM_BASE_SEED,
        "count": COUNT,
        "batch_size": BATCH_SIZE,
        "device": DEVICE,
        "backend": BACKEND,
        "checkpoints": [str(path) for path in paths],
        "label_logits": torch.cat(label_chunks, dim=1),
        "uncertainty_logits": torch.cat(uncertainty_chunks, dim=1),
        "labels": labels,
        "tasks": tasks,
        "difficulties": difficulties,
        "sample_seeds": seeds,
    }
    CONFIRM_CACHE.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, CONFIRM_CACHE)
    return payload


def load_or_build_confirmation_cache() -> dict[str, Any]:
    if CONFIRM_CACHE.exists():
        return torch.load(CONFIRM_CACHE, map_location="cpu", weights_only=False)
    return build_confirmation_cache()


def task_members(values: list[str]):
    lookup = {member.value: member for member in TASKS}
    return tuple(lookup[value] for value in values)


def base_objects(cache: dict[str, Any], config: AggregationConfig = AggregationConfig()):
    output = PopulationOutput(cache["label_logits"], cache["uncertainty_logits"])
    tasks = task_members(cache["tasks"])
    evidence = build_evidence_matrix(output, tasks, config)
    summary, decision = aggregate_evidence(evidence, config)
    worker_idx = _worker_prediction_indices(output, uncertainty_threshold=0.5)
    majority_idx = _majority_prediction_indices(worker_idx)
    majority = [LABELS_WITH_UNCERTAIN[int(idx)] for idx in majority_idx]
    mean_logit_output = Step01Output(output.label_logits.mean(dim=0), output.uncertainty_logits.mean(dim=0))
    mean_logit = list(decode_predictions(mean_logit_output, uncertainty_threshold=0.5))
    mean_prob = list(_mean_probability_predictions(output, uncertainty_threshold=0.5))
    return output, evidence, summary, decision, worker_idx, majority, mean_logit, mean_prob


def v0_predictions(cache: dict[str, Any]) -> list[str]:
    return list(base_objects(cache)[3].predictions)


def majority_predictions(cache: dict[str, Any]) -> list[str]:
    return base_objects(cache)[5]


def mean_logit_predictions(cache: dict[str, Any]) -> list[str]:
    return base_objects(cache)[6]


def mean_probability_predictions(cache: dict[str, Any]) -> list[str]:
    return base_objects(cache)[7]


def hypothesis_a_predictions(cache: dict[str, Any]) -> list[str]:
    """Cohesive minority challenger v1a."""
    output, evidence, summary, decision, worker_idx, _, _, _ = base_objects(cache)
    preds = list(decision.predictions)
    worker_top = evidence.top_valid_label_indices
    for i in range(len(preds)):
        if preds[i] == "UNCERTAIN":
            continue
        valid = evidence.valid_label_mask[i]
        mean_scores = summary.mean_evidence_per_label[i].masked_fill(~valid, float("-inf"))
        primary = int(mean_scores.argmax())
        best_label = None
        best_score = None
        for label_index in torch.where(valid)[0].tolist():
            if label_index == primary:
                continue
            supporters = (worker_top[:, i] == label_index) & (evidence.evidence_scores[:, i, label_index] >= 2.0)
            support_count = int(supporters.sum())
            if support_count < 2:
                continue
            supporter_scores = evidence.evidence_scores[:, i, label_index][supporters]
            supporter_unc = evidence.uncertainty_probability[:, i][supporters]
            supporter_invalid = evidence.invalid_label_mass[:, i][supporters]
            challenger_sum = float(supporter_scores.sum())
            challenger_mean = float(supporter_scores.mean())
            primary_supporters = (worker_top[:, i] == primary)
            primary_sum = float(evidence.evidence_scores[:, i, primary][primary_supporters].sum()) if bool(primary_supporters.any()) else float(summary.sum_evidence_per_label[i, primary])
            cohesive = (
                challenger_mean >= 5.5
                and challenger_sum >= 11.0
                and float(supporter_unc.mean()) <= 0.62
                and float(supporter_invalid.mean()) <= 0.002
                and challenger_sum >= primary_sum - 4.0
            )
            if cohesive and (best_score is None or challenger_sum > best_score):
                best_score = challenger_sum
                best_label = label_index
        if best_label is not None:
            preds[i] = NON_UNCERTAIN_LABELS[best_label]
    return preds


def worker_reliability(cache: dict[str, Any]) -> dict[str, Any]:
    output, _, _, _, worker_idx, _, _, _ = base_objects(cache)
    labels = cache["labels"]
    tasks = cache["tasks"]
    task_names = sorted(set(tasks))
    per_task = {task: [] for task in task_names}
    global_weights = []
    for w in range(output.label_logits.shape[0]):
        correct_global = 0
        for i, truth in enumerate(labels):
            correct_global += int(LABELS_WITH_UNCERTAIN[int(worker_idx[w, i])] == truth)
        global_weights.append((correct_global + 5.0) / (len(labels) + 10.0))
        for task in task_names:
            indices = [i for i, value in enumerate(tasks) if value == task]
            correct = sum(int(LABELS_WITH_UNCERTAIN[int(worker_idx[w, i])] == labels[i]) for i in indices)
            per_task[task].append((correct + 2.0) / (len(indices) + 4.0))
    return {"global": global_weights, "task_conditioned": per_task}


def weighted_predictions(cache: dict[str, Any], reliability: dict[str, Any]) -> list[str]:
    config = AggregationConfig()
    output = PopulationOutput(cache["label_logits"], cache["uncertainty_logits"])
    evidence = build_evidence_matrix(output, task_members(cache["tasks"]), config)
    weights = torch.ones((output.label_logits.shape[0], output.label_logits.shape[1]), dtype=evidence.evidence_scores.dtype)
    for i, task in enumerate(cache["tasks"]):
        vals = reliability["task_conditioned"][task]
        centered = [max(0.5, min(1.5, value / mean(vals))) for value in vals]
        weights[:, i] = torch.tensor(centered, dtype=weights.dtype)
    weighted = EvidenceBatch(
        label_probabilities_all=evidence.label_probabilities_all,
        valid_label_probabilities=evidence.valid_label_probabilities,
        valid_label_mask=evidence.valid_label_mask,
        invalid_label_mass=evidence.invalid_label_mass,
        uncertainty_probability=evidence.uncertainty_probability,
        reliability=evidence.reliability,
        evidence_scores=evidence.evidence_scores * weights.unsqueeze(-1),
        top_valid_label_indices=evidence.top_valid_label_indices,
        top_margin=evidence.top_margin,
    )
    _, decision = aggregate_evidence(weighted, config)
    return list(decision.predictions)


def hypothesis_c_predictions(cache: dict[str, Any]) -> list[str]:
    """Contradiction resolution v1c: resolve only clean primary evidence conflicts."""
    output, evidence, summary, decision, _, _, _, _ = base_objects(cache)
    preds = list(decision.predictions)
    for i, reasons in enumerate(decision.uncertainty_reasons):
        if "protected_minority_contradiction" not in reasons:
            continue
        other_reasons = [r for r in reasons if r != "protected_minority_contradiction"]
        if other_reasons:
            continue
        valid = evidence.valid_label_mask[i]
        mean_scores = summary.mean_evidence_per_label[i].masked_fill(~valid, float("-inf"))
        top = torch.topk(mean_scores, k=2)
        primary = int(top.indices[0])
        primary_margin = float(top.values[0] - top.values[1])
        support = int((evidence.top_valid_label_indices[:, i] == primary).sum())
        primary_sum = float(summary.sum_evidence_per_label[i, primary])
        competitor_sum = float(summary.sum_evidence_per_label[i].masked_fill(~valid, float("-inf")).topk(2).values[1])
        if primary_margin >= 0.05 and support >= 3 and primary_sum >= competitor_sum + 2.0 and float(summary.mean_uncertainty[i]) <= 0.62:
            preds[i] = NON_UNCERTAIN_LABELS[primary]
    return preds


def combined_predictions(cache: dict[str, Any], reliability: dict[str, Any]) -> list[str]:
    # Apply reliability weighting first, then the conservative contradiction resolver.
    # This intentionally does not include the challenger override unless A wins alone.
    config = AggregationConfig(protected_conflict_mean_gap=0.0, min_primary_margin=0.05)
    output = PopulationOutput(cache["label_logits"], cache["uncertainty_logits"])
    evidence = build_evidence_matrix(output, task_members(cache["tasks"]), config)
    weights = torch.ones((output.label_logits.shape[0], output.label_logits.shape[1]), dtype=evidence.evidence_scores.dtype)
    for i, task in enumerate(cache["tasks"]):
        vals = reliability["task_conditioned"][task]
        centered = [max(0.5, min(1.5, value / mean(vals))) for value in vals]
        weights[:, i] = torch.tensor(centered, dtype=weights.dtype)
    weighted = EvidenceBatch(
        label_probabilities_all=evidence.label_probabilities_all,
        valid_label_probabilities=evidence.valid_label_probabilities,
        valid_label_mask=evidence.valid_label_mask,
        invalid_label_mass=evidence.invalid_label_mass,
        uncertainty_probability=evidence.uncertainty_probability,
        reliability=evidence.reliability,
        evidence_scores=evidence.evidence_scores * weights.unsqueeze(-1),
        top_valid_label_indices=evidence.top_valid_label_indices,
        top_margin=evidence.top_margin,
    )
    _, decision = aggregate_evidence(weighted, config)
    return list(decision.predictions)


def evaluate_predictions(cache: dict[str, Any], predictions: list[str], name: str, baseline: list[str] | None = None, mean_logit: list[str] | None = None) -> dict[str, Any]:
    output, evidence, summary, decision, worker_idx, majority, ml, mp = base_objects(cache)
    labels = cache["labels"]
    total = len(labels)
    correct = [predictions[i] == labels[i] for i in range(total)]
    worker_correct_counts = []
    any_correct_flags = []
    majority_correct_flags = []
    true_minority_cases = 0
    true_minority_rescues = 0
    true_minority_suppressions = 0
    strong_cases = 0
    strong_suppressions = 0
    majority_harm = 0
    utilization_failures = 0
    all_wrong = 0
    uncertain_predictions = 0
    uncertain_correct = 0
    uncertain_wrong_with_oracle = 0
    useful_label_evidence_suppressed_by_uncertain = 0
    for i, truth in enumerate(labels):
        worker_preds = [LABELS_WITH_UNCERTAIN[int(worker_idx[w, i])] for w in range(output.label_logits.shape[0])]
        wc = sum(pred == truth for pred in worker_preds)
        worker_correct_counts.append(wc)
        any_correct = wc > 0
        any_correct_flags.append(any_correct)
        maj_correct = majority[i] == truth
        majority_correct_flags.append(maj_correct)
        pred_correct = predictions[i] == truth
        all_wrong += int(not any_correct)
        utilization_failures += int(any_correct and not pred_correct)
        majority_harm += int(maj_correct and not pred_correct)
        if any_correct and wc < output.label_logits.shape[0] / 2 and not maj_correct:
            true_minority_cases += 1
            true_minority_rescues += int(pred_correct)
            true_minority_suppressions += int(not pred_correct)
        truth_idx = LABEL_TO_INDEX.get(truth)
        if truth_idx is not None:
            strong = bool((evidence.evidence_scores[:, i, truth_idx] >= AggregationConfig().strong_evidence_threshold).any() and not maj_correct)
            strong_cases += int(strong)
            strong_suppressions += int(strong and not pred_correct)
        if predictions[i] == "UNCERTAIN":
            uncertain_predictions += 1
            uncertain_correct += int(truth == "UNCERTAIN")
            uncertain_wrong_with_oracle += int(truth != "UNCERTAIN" and any_correct)
            useful_label_evidence_suppressed_by_uncertain += int(truth_idx is not None and any_correct)
    result = {
        "name": name,
        "count": total,
        "accuracy": sum(correct) / total,
        "oracle_any_correct_coverage": sum(any_correct_flags) / total,
        "all_wrong_rate": all_wrong / total,
        "utilization_gap": (sum(any_correct_flags) - sum(correct)) / total,
        "utilization_failure_rate": utilization_failures / total,
        "true_minority_metrics": {
            "numerator_rescues": true_minority_rescues,
            "numerator_suppressions": true_minority_suppressions,
            "denominator_true_minority_opportunities": true_minority_cases,
            "rescue_rate": None if true_minority_cases == 0 else true_minority_rescues / true_minority_cases,
            "suppression_rate": None if true_minority_cases == 0 else true_minority_suppressions / true_minority_cases,
        },
        "legacy_strong_correct_evidence_suppression": {
            "numerator": strong_suppressions,
            "denominator": strong_cases,
            "rate": None if strong_cases == 0 else strong_suppressions / strong_cases,
        },
        "majority_harm_rate": majority_harm / total,
        "mean_disagreement_entropy": float(summary.disagreement_entropy.mean()),
        "mean_population_uncertainty": float(summary.mean_uncertainty.mean()),
        "mean_invalid_label_mass": float(summary.mean_invalid_label_mass.mean()),
        "uncertain_behavior": {
            "uncertain_prediction_count": uncertain_predictions,
            "uncertain_prediction_rate": uncertain_predictions / total,
            "uncertain_correct_count": uncertain_correct,
            "uncertain_wrong_with_oracle_count": uncertain_wrong_with_oracle,
            "useful_label_evidence_suppressed_by_uncertain_count": useful_label_evidence_suppressed_by_uncertain,
        },
    }
    if baseline is not None:
        result["overlap_vs_baseline"] = overlap(labels, predictions, baseline, any_correct_flags)
    if mean_logit is not None:
        result["overlap_vs_mean_logit"] = overlap(labels, predictions, mean_logit, any_correct_flags)
    return result


def overlap(labels: list[str], a: list[str], b: list[str], oracle: list[bool]) -> dict[str, int]:
    out = {"a_correct_b_wrong": 0, "b_correct_a_wrong": 0, "both_correct": 0, "both_wrong": 0, "both_wrong_oracle_available": 0}
    for i, truth in enumerate(labels):
        ac = a[i] == truth
        bc = b[i] == truth
        if ac and bc:
            out["both_correct"] += 1
        elif ac and not bc:
            out["a_correct_b_wrong"] += 1
        elif bc and not ac:
            out["b_correct_a_wrong"] += 1
        else:
            out["both_wrong"] += 1
            out["both_wrong_oracle_available"] += int(oracle[i])
    return out


def run_set(cache: dict[str, Any], reliability: dict[str, Any], selected: str | None = None) -> dict[str, Any]:
    preds = {
        "reducer_v0": v0_predictions(cache),
        "majority": majority_predictions(cache),
        "mean_logit": mean_logit_predictions(cache),
        "mean_probability": mean_probability_predictions(cache),
        "hypothesis_a_cohesive_minority_challenger": hypothesis_a_predictions(cache),
        "hypothesis_b_task_reliability_weighted": weighted_predictions(cache, reliability),
        "hypothesis_c_contradiction_resolution": hypothesis_c_predictions(cache),
        "combined_b_c_candidate": combined_predictions(cache, reliability),
    }
    baseline = preds["reducer_v0"]
    ml = preds["mean_logit"]
    results = [evaluate_predictions(cache, value, key, baseline=baseline, mean_logit=ml) for key, value in preds.items()]
    if selected:
        results = [r for r in results if r["name"] in {"reducer_v0", "majority", "mean_logit", "mean_probability", selected}]
    return {"results": results, "predictions": preds}


def select_candidate(dev_results: list[dict[str, Any]]) -> str | None:
    base = next(r for r in dev_results if r["name"] == "reducer_v0")
    controls = {"reducer_v0", "majority", "mean_logit", "mean_probability"}
    candidates = [r for r in dev_results if r["name"] not in controls]
    candidates.sort(key=lambda r: (r["accuracy"], -r["majority_harm_rate"], -r["utilization_gap"]), reverse=True)
    best = candidates[0]
    if best["accuracy"] > base["accuracy"] and best["utilization_gap"] < base["utilization_gap"]:
        return best["name"]
    return None


def main() -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    dev = load_dev_cache()
    confirm = load_or_build_confirmation_cache()
    reliability = worker_reliability(dev)
    dev_run = run_set(dev, reliability)
    selected = select_candidate(dev_run["results"])
    confirm_run = run_set(confirm, reliability, selected=selected) if selected else run_set(confirm, reliability)
    manifest = {
        "phase": "step2a2_evidence_utilization_redesign_validation_dev_only",
        "test_split_accessed": False,
        "historical_artifacts_modified": False,
        "benchmark_version": BENCHMARK_VERSION,
        "development_set": {
            "source": str(DEV_CACHE),
            "interpretation": "existing 20K validation set now treated as aggregation-development set",
        },
        "dev_confirmation_set": {
            "source": str(CONFIRM_CACHE),
            "split_name": "development_confirmation_not_test",
            "base_seed": DEV_CONFIRM_BASE_SEED,
            "count": COUNT,
            "distribution": "balanced TASKS x DIFFICULTIES, same generate_sample contract",
            "test_overlap": "none by seed range; test base is 2_000_000_000, confirmation base is 3_000_000_000",
        },
        "reliability_from_development": reliability,
        "candidate_selected_from_development": selected,
        "candidate_status": "TEMPORARY_DEV_SELECTED_NOT_FROZEN" if selected else "NO_CANDIDATE_SELECTED",
        "hypotheses": {
            "A": "cohesive minority challenger based on supporter count, supporter evidence, uncertainty, invalid mass, and primary margin",
            "B": "task-conditioned worker reliability weighting calibrated from development data",
            "C": "conservative protected-contradiction resolution when primary evidence is clean",
            "combined": "B plus relaxed v0 conflict/margin configuration; evaluated only as a candidate check",
        },
    }
    summary = {
        "manifest": manifest,
        "development_results": dev_run["results"],
        "dev_confirmation_results": confirm_run["results"],
    }
    dump(ROOT / "manifest.json", manifest)
    dump(ROOT / "development_results.json", dev_run["results"])
    dump(ROOT / "dev_confirmation_results.json", confirm_run["results"])
    dump(ROOT / "summary.json", summary)
    print(json.dumps({"event": "step2a2_complete", "candidate": selected, "summary": str(ROOT / "summary.json")}, sort_keys=True))


if __name__ == "__main__":
    main()
