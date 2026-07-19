"""Step 2A.3 evidence-contract sufficiency study.

Diagnostic only: no frozen test access, no worker training, no new confirmation set.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

import torch
import torch.nn.functional as F

from ai_hypothesis.step01.model import LABEL_TO_INDEX, NON_UNCERTAIN_LABELS, Step01Output, decode_predictions
from ai_hypothesis.step01.schema import VALID_LABELS, TaskFamily
from ai_hypothesis.step02.evidence import AggregationConfig, aggregate_evidence, build_evidence_matrix
from ai_hypothesis.step02.evaluation import _majority_prediction_indices, _mean_probability_predictions, _worker_prediction_indices
from ai_hypothesis.step02.population import PopulationOutput
from tools.step2a2_redesign import combined_predictions, worker_reliability

ROOT = Path("results/step02/step2a3_evidence_contract_sufficiency")
DEV_CACHE = Path("results/step02/step2a1_aggregation_diagnosis/cache/w5_validation_logits.pt")
CONFIRM_CACHE = Path("results/step02/step2a2_evidence_utilization_redesign/cache/w5_dev_confirmation_logits.pt")
LABELS = (*NON_UNCERTAIN_LABELS, "UNCERTAIN")
LABEL_INDEX_WITH_UNCERTAIN = {label: i for i, label in enumerate(LABELS)}
PRACTICAL = ("reducer_v0", "majority", "mean_logit", "mean_probability", "combined_b_c")


def dump(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def load_cache(path: Path) -> dict[str, Any]:
    return torch.load(path, map_location="cpu", weights_only=False)


def task_members(values: list[str]) -> tuple[TaskFamily, ...]:
    lookup = {member.value: member for member in TaskFamily}
    return tuple(lookup[value] for value in values)


def objects(cache: dict[str, Any], reliability: dict[str, Any]) -> dict[str, Any]:
    output = PopulationOutput(cache["label_logits"], cache["uncertainty_logits"])
    tasks = task_members(cache["tasks"])
    evidence = build_evidence_matrix(output, tasks, AggregationConfig())
    summary, decision = aggregate_evidence(evidence, AggregationConfig())
    worker_idx = _worker_prediction_indices(output, uncertainty_threshold=0.5)
    majority_idx = _majority_prediction_indices(worker_idx)
    majority = [LABELS[int(idx)] for idx in majority_idx]
    mean_logit_output = Step01Output(output.label_logits.mean(dim=0), output.uncertainty_logits.mean(dim=0))
    mean_logit = list(decode_predictions(mean_logit_output, uncertainty_threshold=0.5))
    mean_probability = list(_mean_probability_predictions(output, uncertainty_threshold=0.5))
    return {
        "output": output,
        "tasks": tasks,
        "evidence": evidence,
        "summary": summary,
        "decision": decision,
        "worker_idx": worker_idx,
        "predictions": {
            "reducer_v0": list(decision.predictions),
            "majority": majority,
            "mean_logit": mean_logit,
            "mean_probability": mean_probability,
            "combined_b_c": combined_predictions(cache, reliability),
        },
    }


def bools(labels: list[str], predictions: list[str]) -> list[bool]:
    return [p == y for p, y in zip(predictions, labels)]


def oracle_flags(cache: dict[str, Any], obj: dict[str, Any]) -> list[bool]:
    flags = []
    worker_idx = obj["worker_idx"]
    for i, truth in enumerate(cache["labels"]):
        flags.append(any(LABELS[int(worker_idx[w, i])] == truth for w in range(worker_idx.shape[0])))
    return flags


def overlap(labels: list[str], a: list[str], b: list[str], oracle: list[bool]) -> dict[str, int]:
    out = {"a_correct_b_wrong": 0, "b_correct_a_wrong": 0, "both_correct": 0, "both_wrong": 0, "both_wrong_oracle_available": 0}
    for i, truth in enumerate(labels):
        ac = a[i] == truth
        bc = b[i] == truth
        if ac and bc:
            out["both_correct"] += 1
        elif ac:
            out["a_correct_b_wrong"] += 1
        elif bc:
            out["b_correct_a_wrong"] += 1
        else:
            out["both_wrong"] += 1
            out["both_wrong_oracle_available"] += int(oracle[i])
    return out


def bin_key(value: float, bins: tuple[float, ...]) -> str:
    last = 0.0
    for b in bins:
        if value <= b:
            return f"{last:.2f}-{b:.2f}"
        last = b
    return f">{bins[-1]:.2f}"


def practical_analysis(cache: dict[str, Any], obj: dict[str, Any]) -> dict[str, Any]:
    labels = cache["labels"]
    preds = obj["predictions"]
    oracle = oracle_flags(cache, obj)
    summary = obj["summary"]
    worker_idx = obj["worker_idx"]
    output = {}
    for name in PRACTICAL:
        correct = bools(labels, preds[name])
        output[name] = {
            "accuracy": sum(correct) / len(labels),
            "utilization_gap": (sum(oracle) - sum(correct)) / len(labels),
            "uncertain_rate": sum(p == "UNCERTAIN" for p in preds[name]) / len(labels),
        }
    output["overlaps"] = {
        "mean_prob_vs_mean_logit": overlap(labels, preds["mean_probability"], preds["mean_logit"], oracle),
        "mean_prob_vs_reducer_v0": overlap(labels, preds["mean_probability"], preds["reducer_v0"], oracle),
        "reducer_v0_vs_majority": overlap(labels, preds["reducer_v0"], preds["majority"], oracle),
        "combined_vs_reducer_v0": overlap(labels, preds["combined_b_c"], preds["reducer_v0"], oracle),
    }
    all_wrong_practical = 0
    for i, truth in enumerate(labels):
        if oracle[i] and all(preds[name][i] != truth for name in PRACTICAL):
            all_wrong_practical += 1
    output["all_practical_aggregators_wrong_despite_oracle"] = all_wrong_practical

    breakdown: dict[str, dict[str, Any]] = {}
    for dim_name, values in {
        "task": cache["tasks"],
        "difficulty": cache["difficulties"],
        "disagreement_entropy": [bin_key(float(v), (0.0, 0.1, 0.4, 0.7, 1.0)) for v in summary.disagreement_entropy],
        "mean_uncertainty": [bin_key(float(v), (0.1, 0.3, 0.5, 0.7, 0.9)) for v in summary.mean_uncertainty],
        "invalid_label_mass": [bin_key(float(v), (0.00001, 0.0001, 0.001, 0.01)) for v in summary.mean_invalid_label_mass],
        "distinct_worker_predictions": [str(len(set(LABELS[int(worker_idx[w, i])] for w in range(worker_idx.shape[0])))) for i in range(len(labels))],
    }.items():
        rows: dict[str, dict[str, int]] = {}
        for i, value in enumerate(values):
            row = rows.setdefault(str(value), {"count": 0, **{name: 0 for name in PRACTICAL}})
            row["count"] += 1
            for name in PRACTICAL:
                row[name] += int(preds[name][i] == labels[i])
        breakdown[dim_name] = {k: {"count": v["count"], **{name: v[name] / v["count"] for name in PRACTICAL}} for k, v in sorted(rows.items())}
    output["breakdown"] = breakdown
    return output


def rank_of(values: torch.Tensor, truth_index: int, valid_mask: torch.Tensor) -> int:
    masked = values.masked_fill(~valid_mask, float("-inf"))
    ranked = torch.argsort(masked, descending=True)
    return int((ranked == truth_index).nonzero()[0].item()) + 1


def rank_bucket(rank: int) -> str:
    return str(rank) if rank <= 3 else ">3"


def rank_analysis(cache: dict[str, Any], obj: dict[str, Any]) -> dict[str, Any]:
    output = obj["output"]
    evidence = obj["evidence"]
    summary = obj["summary"]
    preds = obj["predictions"]
    oracle = oracle_flags(cache, obj)
    worker_idx = obj["worker_idx"]
    measures = {
        "summed_logits": output.label_logits.sum(dim=0),
        "mean_logits": output.label_logits.mean(dim=0),
        "summed_probabilities": torch.softmax(output.label_logits, dim=-1).sum(dim=0),
        "mean_probabilities": torch.softmax(output.label_logits, dim=-1).mean(dim=0),
        "current_evidence_support": summary.sum_evidence_per_label,
        "strongest_worker_evidence": summary.max_evidence_per_label,
        "topk_evidence_sum": summary.top_k_evidence_per_label.sum(dim=-1),
    }
    groups = {
        "oracle_available": lambda i, truth: oracle[i],
        "reducer_v0_success_with_oracle": lambda i, truth: oracle[i] and preds["reducer_v0"][i] == truth,
        "reducer_v0_utilization_failure": lambda i, truth: oracle[i] and preds["reducer_v0"][i] != truth,
        "true_minority_cases": lambda i, truth: true_minority(i, truth, cache, obj),
        "true_minority_suppressions": lambda i, truth: true_minority(i, truth, cache, obj) and preds["reducer_v0"][i] != truth,
    }
    result = {group: {measure: {"1": 0, "2": 0, "3": 0, ">3": 0, "excluded_uncertain_truth": 0} for measure in measures} for group in groups}
    for i, truth in enumerate(cache["labels"]):
        truth_index = LABEL_TO_INDEX.get(truth)
        for group, predicate in groups.items():
            if not predicate(i, truth):
                continue
            for measure, values in measures.items():
                if truth_index is None:
                    result[group][measure]["excluded_uncertain_truth"] += 1
                else:
                    rank = rank_of(values[i], truth_index, evidence.valid_label_mask[i])
                    result[group][measure][rank_bucket(rank)] += 1
    return result


def true_minority(i: int, truth: str, cache: dict[str, Any], obj: dict[str, Any]) -> bool:
    worker_idx = obj["worker_idx"]
    majority = obj["predictions"]["majority"]
    correct_count = sum(LABELS[int(worker_idx[w, i])] == truth for w in range(worker_idx.shape[0]))
    return correct_count > 0 and correct_count < (worker_idx.shape[0] / 2) and majority[i] != truth


def uncertain_analysis(cache: dict[str, Any], obj: dict[str, Any]) -> dict[str, Any]:
    labels = cache["labels"]
    preds = obj["predictions"]["reducer_v0"]
    reasons = obj["decision"].uncertainty_reasons
    oracle = oracle_flags(cache, obj)
    counts = {"correct_uncertain": 0, "false_uncertain_unnecessary_abstention": 0, "confident_incorrect_label_when_uncertain_truth": 0, "correct_label_evidence_suppressed_by_uncertainty": 0, "uncertainty_prevents_false_label_decision": 0}
    util = {"total_utilization_failures": 0, "wrong_non_uncertain_label": 0, "uncertain_or_abstention": 0, "high_mean_uncertainty": 0, "unresolved_contradiction": 0}
    for i, truth in enumerate(labels):
        pred = preds[i]
        if truth == "UNCERTAIN" and pred == "UNCERTAIN":
            counts["correct_uncertain"] += 1
            if obj["predictions"]["majority"][i] != "UNCERTAIN" or obj["predictions"]["mean_logit"][i] != "UNCERTAIN":
                counts["uncertainty_prevents_false_label_decision"] += 1
        if truth != "UNCERTAIN" and pred == "UNCERTAIN":
            counts["false_uncertain_unnecessary_abstention"] += 1
            if oracle[i] and "high_mean_uncertainty" in reasons[i]:
                counts["correct_label_evidence_suppressed_by_uncertainty"] += 1
        if truth == "UNCERTAIN" and pred != "UNCERTAIN":
            counts["confident_incorrect_label_when_uncertain_truth"] += 1
        if oracle[i] and pred != truth:
            util["total_utilization_failures"] += 1
            util["uncertain_or_abstention"] += int(pred == "UNCERTAIN")
            util["wrong_non_uncertain_label"] += int(pred != "UNCERTAIN")
            util["high_mean_uncertainty"] += int("high_mean_uncertainty" in reasons[i])
            util["unresolved_contradiction"] += int("protected_minority_contradiction" in reasons[i])
    return {"counts": counts, "utilization_failure_attribution": util}


def stats(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "mean": None, "std": None, "min": None, "max": None}
    return {"count": len(values), "mean": mean(values), "std": pstdev(values) if len(values) > 1 else 0.0, "min": min(values), "max": max(values)}


def effect(a: list[float], b: list[float]) -> float | None:
    if not a or not b:
        return None
    pooled = math.sqrt((pstdev(a) ** 2 + pstdev(b) ** 2) / 2) if len(a) > 1 and len(b) > 1 else 0.0
    return None if pooled == 0 else (mean(a) - mean(b)) / pooled


def feature_separability(cache: dict[str, Any], obj: dict[str, Any]) -> dict[str, Any]:
    evidence = obj["evidence"]
    summary = obj["summary"]
    worker_idx = obj["worker_idx"]
    preds = obj["predictions"]
    useful: dict[str, list[float]] = {}
    noisy: dict[str, list[float]] = {}
    def add(bucket: dict[str, list[float]], key: str, value: float) -> None:
        bucket.setdefault(key, []).append(float(value))
    for i, truth in enumerate(cache["labels"]):
        worker_labels = [LABELS[int(worker_idx[w, i])] for w in range(worker_idx.shape[0])]
        majority = preds["majority"][i]
        for candidate in set(worker_labels):
            if candidate == majority or candidate == "UNCERTAIN" or candidate not in LABEL_INDEX_WITH_UNCERTAIN:
                continue
            label_index = LABEL_TO_INDEX.get(candidate)
            if label_index is None:
                continue
            supporters = torch.tensor([label == candidate for label in worker_labels], dtype=torch.bool)
            support_count = int(supporters.sum())
            if support_count == 0:
                continue
            vals = {
                "supporter_count": support_count,
                "supporter_fraction": support_count / worker_idx.shape[0],
                "cumulative_evidence": float(evidence.evidence_scores[:, i, label_index][supporters].sum()),
                "mean_supporter_evidence": float(evidence.evidence_scores[:, i, label_index][supporters].mean()),
                "max_supporter_evidence": float(evidence.evidence_scores[:, i, label_index][supporters].max()),
                "supporter_evidence_std": float(evidence.evidence_scores[:, i, label_index][supporters].std(unbiased=False)),
                "mean_supporter_uncertainty": float(evidence.uncertainty_probability[:, i][supporters].mean()),
                "min_supporter_uncertainty": float(evidence.uncertainty_probability[:, i][supporters].min()),
                "mean_supporter_invalid_mass": float(evidence.invalid_label_mass[:, i][supporters].mean()),
                "disagreement_entropy": float(summary.disagreement_entropy[i]),
                "margin_against_primary_sum": float(summary.sum_evidence_per_label[i, label_index] - summary.sum_evidence_per_label[i, LABEL_TO_INDEX.get(majority, label_index)]),
            }
            bucket = useful if candidate == truth else noisy
            for key, value in vals.items():
                add(bucket, key, value)
    keys = sorted(set(useful) | set(noisy))
    return {key: {"useful_minority": stats(useful.get(key, [])), "noisy_minority": stats(noisy.get(key, [])), "cohens_d_useful_minus_noisy": effect(useful.get(key, []), noisy.get(key, []))} for key in keys}


def feature_matrix(cache: dict[str, Any], obj: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor, list[str]]:
    output = obj["output"]
    evidence = obj["evidence"]
    summary = obj["summary"]
    probs = torch.softmax(output.label_logits, dim=-1)
    parts = [
        output.label_logits.mean(dim=0),
        output.label_logits.sum(dim=0),
        probs.mean(dim=0),
        probs.sum(dim=0),
        summary.mean_evidence_per_label,
        summary.sum_evidence_per_label,
        summary.max_evidence_per_label,
        summary.support_count_per_label.to(torch.float32),
    ]
    global_features = torch.stack([
        summary.mean_uncertainty,
        summary.max_uncertainty,
        summary.mean_invalid_label_mass,
        summary.max_invalid_label_mass,
        summary.disagreement_entropy,
    ], dim=1)
    x = torch.cat([*parts, global_features], dim=1).to(torch.float32)
    y = torch.tensor([LABEL_INDEX_WITH_UNCERTAIN[label] for label in cache["labels"]], dtype=torch.long)
    names = [f"label_feature_{i}" for i in range(x.shape[1])]
    return x, y, names


def train_probe(dev_cache: dict[str, Any], dev_obj: dict[str, Any], confirm_cache: dict[str, Any], confirm_obj: dict[str, Any]) -> dict[str, Any]:
    torch.manual_seed(123)
    x, y, _ = feature_matrix(dev_cache, dev_obj)
    cx, cy, _ = feature_matrix(confirm_cache, confirm_obj)
    n = x.shape[0]
    train_n = int(n * 0.8)
    mean_x = x[:train_n].mean(dim=0, keepdim=True)
    std_x = x[:train_n].std(dim=0, keepdim=True).clamp_min(1e-6)
    tx = (x[:train_n] - mean_x) / std_x
    ty = y[:train_n]
    vx = (x[train_n:] - mean_x) / std_x
    vy = y[train_n:]
    cxs = (cx - mean_x) / std_x
    model = torch.nn.Linear(x.shape[1], len(LABELS))
    opt = torch.optim.Adam(model.parameters(), lr=0.05, weight_decay=1e-3)
    for _ in range(250):
        opt.zero_grad()
        loss = F.cross_entropy(model(tx), ty)
        loss.backward()
        opt.step()
    with torch.no_grad():
        train_pred = model(tx).argmax(dim=1)
        dev_cal_pred = model(vx).argmax(dim=1)
        confirm_pred = model(cxs).argmax(dim=1)
    dev_oracle = oracle_flags(dev_cache, dev_obj)
    confirm_oracle = oracle_flags(confirm_cache, confirm_obj)
    def eval_pred(pred: torch.Tensor, labels: list[str], oracle: list[bool]) -> dict[str, Any]:
        pred_labels = [LABELS[int(i)] for i in pred]
        correct = [p == t for p, t in zip(pred_labels, labels)]
        return {"accuracy": sum(correct) / len(correct), "utilization_gap": (sum(oracle) - sum(correct)) / len(correct), "uncertain_rate": sum(p == "UNCERTAIN" for p in pred_labels) / len(pred_labels)}
    return {
        "model": "single linear softmax classifier over current evidence-contract aggregate features",
        "feature_count": x.shape[1],
        "train_count": train_n,
        "internal_calibration_count": n - train_n,
        "development_train": eval_pred(train_pred, dev_cache["labels"][:train_n], dev_oracle[:train_n]),
        "development_internal_calibration": eval_pred(dev_cal_pred, dev_cache["labels"][train_n:], dev_oracle[train_n:]),
        "consumed_confirmation_1": eval_pred(confirm_pred, confirm_cache["labels"], confirm_oracle),
    }


def diversity(cache: dict[str, Any], obj: dict[str, Any]) -> dict[str, Any]:
    worker_idx = obj["worker_idx"]
    labels = cache["labels"]
    n_workers = worker_idx.shape[0]
    wrong = torch.zeros((n_workers, len(labels)), dtype=torch.float32)
    pred_labels = [[LABELS[int(worker_idx[w, i])] for i in range(len(labels))] for w in range(n_workers)]
    for w in range(n_workers):
        for i, truth in enumerate(labels):
            wrong[w, i] = float(pred_labels[w][i] != truth)
    pairs = {}
    for a in range(n_workers):
        for b in range(a + 1, n_workers):
            agree = sum(pred_labels[a][i] == pred_labels[b][i] for i in range(len(labels))) / len(labels)
            joint_wrong = float((wrong[a] * wrong[b]).mean())
            av = wrong[a] - wrong[a].mean()
            bv = wrong[b] - wrong[b].mean()
            denom = float(torch.sqrt((av * av).mean() * (bv * bv).mean()))
            corr = None if denom == 0 else float((av * bv).mean() / denom)
            pairs[f"worker_{a+1}_worker_{b+1}"] = {"prediction_agreement": agree, "joint_wrong_rate": joint_wrong, "error_correlation": corr}
    unique = {}
    correct_count_dist: dict[str, int] = {}
    for i, truth in enumerate(labels):
        flags = [pred_labels[w][i] == truth for w in range(n_workers)]
        correct_count_dist[str(sum(flags))] = correct_count_dist.get(str(sum(flags)), 0) + 1
        if sum(flags) == 1:
            w = flags.index(True)
            unique[f"worker_{w+1}"] = unique.get(f"worker_{w+1}", 0) + 1
    return {"pairwise": pairs, "unique_correct_contribution": unique, "correct_worker_count_distribution": correct_count_dist}


def run_dataset(name: str, cache: dict[str, Any], reliability: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    obj = objects(cache, reliability)
    return {
        "practical_aggregators": practical_analysis(cache, obj),
        "true_label_rank_distributions": rank_analysis(cache, obj),
        "uncertain_abstention": uncertain_analysis(cache, obj),
        "feature_separability": feature_separability(cache, obj),
        "worker_diversity": diversity(cache, obj),
    }, obj


def main() -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    dev = load_cache(DEV_CACHE)
    confirm = load_cache(CONFIRM_CACHE)
    reliability = worker_reliability(dev)
    dev_report, dev_obj = run_dataset("aggregation_development", dev, reliability)
    confirm_report, confirm_obj = run_dataset("dev_confirmation_1_consumed", confirm, reliability)
    probe = train_probe(dev, dev_obj, confirm, confirm_obj)
    summary = {
        "phase": "step2a3_evidence_contract_sufficiency_diagnostic_only",
        "frozen_test_accessed": False,
        "workers_trained": False,
        "dev_confirmation_2_generated": False,
        "git_revision_recorded_at_start": "4848450dc2f93acda93fbed34a134ef4fec0db05",
        "datasets": {
            "aggregation_development": str(DEV_CACHE),
            "dev_confirmation_1_consumed": str(CONFIRM_CACHE),
        },
        "aggregation_development": dev_report,
        "dev_confirmation_1_consumed": confirm_report,
        "diagnostic_probe": probe,
    }
    dump(ROOT / "summary.json", summary)
    dump(ROOT / "aggregation_development_analysis.json", dev_report)
    dump(ROOT / "dev_confirmation_1_consumed_analysis.json", confirm_report)
    dump(ROOT / "diagnostic_probe.json", probe)
    print(json.dumps({"event": "step2a3_complete", "summary": str(ROOT / "summary.json")}, sort_keys=True))


if __name__ == "__main__":
    main()

