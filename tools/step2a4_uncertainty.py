"""Step 2A.4 uncertainty and abstention isolation.

Diagnostic only: no frozen test access, no worker training, no new confirmation set.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

import torch

from ai_hypothesis.step01.model import LABEL_TO_INDEX, NON_UNCERTAIN_LABELS, Step01Output, decode_predictions
from ai_hypothesis.step01.schema import TaskFamily
from ai_hypothesis.step02.evidence import AggregationConfig, aggregate_evidence, build_evidence_matrix
from ai_hypothesis.step02.evaluation import _majority_prediction_indices, _mean_probability_predictions, _worker_prediction_indices
from ai_hypothesis.step02.population import PopulationOutput

ROOT = Path("results/step02/step2a4_uncertainty_abstention_isolation")
DEV_CACHE = Path("results/step02/step2a1_aggregation_diagnosis/cache/w5_validation_logits.pt")
CONFIRM_CACHE = Path("results/step02/step2a2_evidence_utilization_redesign/cache/w5_dev_confirmation_logits.pt")
LABELS = (*NON_UNCERTAIN_LABELS, "UNCERTAIN")


def dump(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def load_cache(path: Path) -> dict[str, Any]:
    return torch.load(path, map_location="cpu", weights_only=False)


def task_members(values: list[str]) -> tuple[TaskFamily, ...]:
    lookup = {member.value: member for member in TaskFamily}
    return tuple(lookup[value] for value in values)


def objects(cache: dict[str, Any]) -> dict[str, Any]:
    output = PopulationOutput(cache["label_logits"], cache["uncertainty_logits"])
    evidence = build_evidence_matrix(output, task_members(cache["tasks"]), AggregationConfig())
    summary, decision = aggregate_evidence(evidence, AggregationConfig())
    worker_idx = _worker_prediction_indices(output, uncertainty_threshold=0.5)
    majority_idx = _majority_prediction_indices(worker_idx)
    majority = [LABELS[int(idx)] for idx in majority_idx]
    mean_logit_output = Step01Output(output.label_logits.mean(dim=0), output.uncertainty_logits.mean(dim=0))
    mean_logit = list(decode_predictions(mean_logit_output, uncertainty_threshold=0.5))
    mean_logit_label = [NON_UNCERTAIN_LABELS[int(i)] for i in output.label_logits.mean(dim=0).argmax(dim=-1)]
    mean_prob_tensor = torch.softmax(output.label_logits, dim=-1).mean(dim=0)
    mean_prob_label = [NON_UNCERTAIN_LABELS[int(i)] for i in mean_prob_tensor.argmax(dim=-1)]
    mean_prob = list(_mean_probability_predictions(output, uncertainty_threshold=0.5))
    reducer_label = [NON_UNCERTAIN_LABELS[int(i)] for i in decision.primary_label_indices]
    return {
        "output": output,
        "evidence": evidence,
        "summary": summary,
        "decision": decision,
        "worker_idx": worker_idx,
        "majority": majority,
        "mean_logit": mean_logit,
        "mean_logit_label": mean_logit_label,
        "mean_probability": mean_prob,
        "mean_probability_label": mean_prob_label,
        "reducer_v0": list(decision.predictions),
        "reducer_label": reducer_label,
    }


def oracle_flags(cache: dict[str, Any], obj: dict[str, Any]) -> list[bool]:
    flags = []
    worker_idx = obj["worker_idx"]
    for i, truth in enumerate(cache["labels"]):
        flags.append(any(LABELS[int(worker_idx[w, i])] == truth for w in range(worker_idx.shape[0])))
    return flags


def evaluate_predictions(cache: dict[str, Any], obj: dict[str, Any], predictions: list[str], name: str) -> dict[str, Any]:
    labels = cache["labels"]
    oracle = oracle_flags(cache, obj)
    total = len(labels)
    correct = sum(p == y for p, y in zip(predictions, labels))
    pred_unc = [p == "UNCERTAIN" for p in predictions]
    truth_unc = [y == "UNCERTAIN" for y in labels]
    tp_unc = sum(p and y for p, y in zip(pred_unc, truth_unc))
    fp_unc = sum(p and not y for p, y in zip(pred_unc, truth_unc))
    fn_unc = sum((not p) and y for p, y in zip(pred_unc, truth_unc))
    non_unc_indices = [i for i, y in enumerate(labels) if y != "UNCERTAIN"]
    non_unc_correct = sum(predictions[i] == labels[i] for i in non_unc_indices)
    majority_harm = sum(obj["majority"][i] == labels[i] and predictions[i] != labels[i] for i in range(total))
    return {
        "name": name,
        "accuracy": correct / total,
        "non_uncertain_label_accuracy": non_unc_correct / len(non_unc_indices),
        "uncertain_precision": None if sum(pred_unc) == 0 else tp_unc / sum(pred_unc),
        "uncertain_recall": tp_unc / sum(truth_unc),
        "false_abstention_count": fp_unc,
        "false_abstention_rate": fp_unc / total,
        "missed_uncertain_count": fn_unc,
        "missed_uncertain_rate_among_uncertain_truth": fn_unc / sum(truth_unc),
        "utilization_gap": (sum(oracle) - correct) / total,
        "majority_harm_rate": majority_harm / total,
        "uncertain_prediction_rate": sum(pred_unc) / total,
    }


def audit_pipeline() -> dict[str, Any]:
    return {
        "worker_uncertainty_probability": "sigmoid(worker uncertainty logit)",
        "worker_uncertain_decode_threshold": 0.5,
        "worker_uncertain_decode": "worker decoded prediction becomes UNCERTAIN when sigmoid(uncertainty_logit) >= 0.5; otherwise argmax over 11 label logits",
        "calibration_exists": False,
        "population_uncertainty": "mean_uncertainty is arithmetic mean of worker uncertainty probabilities; max_uncertainty and quantiles are diagnostics",
        "evidence_reliability": "worker evidence is multiplied by (1 - uncertainty_probability) * (1 - invalid_label_mass)",
        "max_mean_uncertainty": "final reducer appends high_mean_uncertainty when mean_uncertainty > 0.5",
        "other_abstention_paths": ["low_primary_margin", "high_invalid_label_mass", "protected_minority_contradiction"],
        "contradiction_can_emit_uncertain": True,
        "invalid_label_mass_can_emit_uncertain": True,
        "multiple_reasons_possible": True,
        "decision_tree": [
            "Build valid-label evidence from worker logits and uncertainty-weighted reliability.",
            "Select primary candidate as valid label with highest mean evidence.",
            "Compute primary margin versus runner-up mean evidence.",
            "Mark protected competitor labels when max evidence >= strong_evidence_threshold.",
            "If primary margin < min_primary_margin, add low_primary_margin.",
            "If mean_uncertainty > max_mean_uncertainty, add high_mean_uncertainty.",
            "If mean_invalid_label_mass > max_mean_invalid_mass, add high_invalid_label_mass.",
            "If protected competitor remains within protected_conflict_mean_gap, add protected_minority_contradiction.",
            "If any reason exists, output UNCERTAIN; otherwise output primary label.",
        ],
    }


def abstention_decomposition(cache: dict[str, Any], obj: dict[str, Any]) -> dict[str, Any]:
    labels = cache["labels"]
    reasons = [tuple(r) for r in obj["decision"].uncertainty_reasons]
    preds = obj["reducer_v0"]
    exclusive = Counter()
    overlapping = Counter()
    false_exclusive = Counter()
    false_overlapping = Counter()
    useful_exclusive = Counter()
    useful_overlapping = Counter()
    false_cases = useful_cases = neutral_cases = 0
    for i, pred in enumerate(preds):
        if pred != "UNCERTAIN":
            continue
        combo = "+".join(reasons[i]) if reasons[i] else "other"
        exclusive[combo] += 1
        for reason in reasons[i] or ("other",):
            overlapping[reason] += 1
        cf = obj["reducer_label"][i]
        truth = labels[i]
        if truth == "UNCERTAIN":
            neutral_cases += 1
        elif cf == truth:
            false_cases += 1
            false_exclusive[combo] += 1
            for reason in reasons[i] or ("other",):
                false_overlapping[reason] += 1
        else:
            useful_cases += 1
            useful_exclusive[combo] += 1
            for reason in reasons[i] or ("other",):
                useful_overlapping[reason] += 1
    return {
        "total_uncertain_outputs": sum(exclusive.values()),
        "exclusive_reason_counts": dict(exclusive),
        "overlapping_reason_counts": dict(overlapping),
        "false_abstention_counterfactual_label_correct": false_cases,
        "useful_abstention_counterfactual_label_wrong": useful_cases,
        "neutral_truth_uncertain": neutral_cases,
        "false_abstention_exclusive_reasons": dict(false_exclusive),
        "false_abstention_overlapping_reasons": dict(false_overlapping),
        "useful_abstention_exclusive_reasons": dict(useful_exclusive),
        "useful_abstention_overlapping_reasons": dict(useful_overlapping),
    }


def counterfactuals(cache: dict[str, Any], obj: dict[str, Any]) -> dict[str, Any]:
    labels = cache["labels"]
    systems = {
        "reducer_v0": (obj["reducer_v0"], obj["reducer_label"]),
        "mean_logit": (obj["mean_logit"], obj["mean_logit_label"]),
        "mean_probability": (obj["mean_probability"], obj["mean_probability_label"]),
    }
    out = {}
    for name, (actual, cf_label) in systems.items():
        false = useful = neutral = total_unc = cf_correct = 0
        for p, cf, truth in zip(actual, cf_label, labels):
            cf_correct += int(cf == truth)
            if p != "UNCERTAIN":
                continue
            total_unc += 1
            if truth == "UNCERTAIN":
                neutral += 1
            elif cf == truth:
                false += 1
            else:
                useful += 1
        out[name] = {
            "uncertain_outputs": total_unc,
            "counterfactual_label_accuracy_if_abstention_disabled": cf_correct / len(labels),
            "false_abstention_label_would_be_correct": false,
            "useful_abstention_label_would_be_wrong": useful,
            "neutral_truth_uncertain": neutral,
        }
    return out


def calibration(cache: dict[str, Any], obj: dict[str, Any]) -> dict[str, Any]:
    output = obj["output"]
    worker_idx = obj["worker_idx"]
    labels = cache["labels"]
    unc = torch.sigmoid(output.uncertainty_logits)
    rows = []
    for w in range(unc.shape[0]):
        for i, truth in enumerate(labels):
            pred = LABELS[int(worker_idx[w, i])]
            event_need_uncertain = truth == "UNCERTAIN" or pred != truth
            rows.append({
                "worker": w + 1,
                "task": cache["tasks"][i],
                "difficulty": cache["difficulties"][i],
                "score": float(unc[w, i]),
                "need_uncertain": event_need_uncertain,
                "truth_uncertain": truth == "UNCERTAIN",
                "prediction_correct": pred == truth,
            })
    def summarize(vals: list[float]) -> dict[str, float | int | None]:
        if not vals:
            return {"count": 0, "mean": None, "min": None, "max": None}
        return {"count": len(vals), "mean": mean(vals), "min": min(vals), "max": max(vals)}
    by = {}
    for key in ["worker", "task", "difficulty"]:
        groups = defaultdict(list)
        for row in rows:
            groups[str(row[key])].append(row)
        by[key] = {
            g: {
                "uncertainty_when_prediction_correct": summarize([r["score"] for r in rs if r["prediction_correct"]]),
                "uncertainty_when_prediction_wrong": summarize([r["score"] for r in rs if not r["prediction_correct"]]),
                "uncertainty_when_truth_uncertain": summarize([r["score"] for r in rs if r["truth_uncertain"]]),
                "uncertain_precision_at_0_5": precision_recall(rs)[0],
                "uncertain_recall_at_0_5": precision_recall(rs)[1],
                "ece_10_bins_need_uncertain": ece(rs),
            }
            for g, rs in sorted(groups.items())
        }
    thresholds = []
    for threshold in [i / 20 for i in range(1, 20)]:
        tp = fp = fn = tn = 0
        for row in rows:
            pred_unc = row["score"] >= threshold
            need = row["need_uncertain"]
            tp += int(pred_unc and need)
            fp += int(pred_unc and not need)
            fn += int((not pred_unc) and need)
            tn += int((not pred_unc) and not need)
        thresholds.append({"threshold": threshold, "precision": None if tp + fp == 0 else tp / (tp + fp), "recall": None if tp + fn == 0 else tp / (tp + fn), "false_positive_rate": fp / (fp + tn)})
    return {
        "overall": {
            "correct_prediction_scores": summarize([r["score"] for r in rows if r["prediction_correct"]]),
            "wrong_prediction_scores": summarize([r["score"] for r in rows if not r["prediction_correct"]]),
            "truth_uncertain_scores": summarize([r["score"] for r in rows if r["truth_uncertain"]]),
            "ece_10_bins_need_uncertain": ece(rows),
        },
        "by_worker_task_difficulty": by,
        "threshold_curve": thresholds,
    }


def precision_recall(rows: list[dict[str, Any]]) -> tuple[float | None, float | None]:
    tp = fp = fn = 0
    for row in rows:
        pred = row["score"] >= 0.5
        need = row["need_uncertain"]
        tp += int(pred and need)
        fp += int(pred and not need)
        fn += int((not pred) and need)
    return (None if tp + fp == 0 else tp / (tp + fp), None if tp + fn == 0 else tp / (tp + fn))


def ece(rows: list[dict[str, Any]], bins: int = 10) -> float:
    total = len(rows)
    acc = 0.0
    for b in range(bins):
        lo = b / bins
        hi = (b + 1) / bins
        bucket = [r for r in rows if lo <= r["score"] < hi or (b == bins - 1 and r["score"] == 1.0)]
        if not bucket:
            continue
        conf = mean([r["score"] for r in bucket])
        freq = mean([float(r["need_uncertain"]) for r in bucket])
        acc += (len(bucket) / total) * abs(conf - freq)
    return acc


def interaction_analysis(cache: dict[str, Any], obj: dict[str, Any]) -> dict[str, Any]:
    labels = cache["labels"]
    false_indices = [i for i, (p, y, cf) in enumerate(zip(obj["reducer_v0"], labels, obj["reducer_label"])) if p == "UNCERTAIN" and y != "UNCERTAIN" and cf == y]
    evidence = obj["evidence"]
    summary = obj["summary"]
    worker_idx = obj["worker_idx"]
    out = {"false_abstention_count": len(false_indices), "truth_rank_1": 0, "multiple_workers_support_truth": 0, "high_mean_uncertainty_despite_truth_rank_1": 0, "strong_truth_evidence": 0, "high_disagreement": 0}
    for i in false_indices:
        truth_idx = LABEL_TO_INDEX[labels[i]]
        valid = evidence.valid_label_mask[i]
        rank = int((torch.argsort(summary.mean_evidence_per_label[i].masked_fill(~valid, float("-inf")), descending=True) == truth_idx).nonzero()[0].item()) + 1
        truth_supporters = sum(LABELS[int(worker_idx[w, i])] == labels[i] for w in range(worker_idx.shape[0]))
        out["truth_rank_1"] += int(rank == 1)
        out["multiple_workers_support_truth"] += int(truth_supporters >= 2)
        out["high_mean_uncertainty_despite_truth_rank_1"] += int(rank == 1 and float(summary.mean_uncertainty[i]) > 0.5)
        out["strong_truth_evidence"] += int(float(summary.max_evidence_per_label[i, truth_idx]) >= 2.0)
        out["high_disagreement"] += int(float(summary.disagreement_entropy[i]) >= 0.4)
    return out


def policies(cache: dict[str, Any], obj: dict[str, Any]) -> dict[str, list[str]]:
    labels = cache["labels"]
    summary = obj["summary"]
    base_label = obj["reducer_label"]
    # Thresholds selected from development audit: current 0.5 is aggressive; test small shifts only.
    out = {
        "current_reducer_v0": obj["reducer_v0"],
        "reducer_label_no_abstention": base_label,
        "global_uncertainty_threshold_0_6": ["UNCERTAIN" if float(summary.mean_uncertainty[i]) > 0.6 else base_label[i] for i in range(len(labels))],
        "high_uncertainty_and_weak_margin": ["UNCERTAIN" if float(summary.mean_uncertainty[i]) > 0.5 and float(obj["decision"].primary_margin[i]) < 0.5 else base_label[i] for i in range(len(labels))],
        "high_uncertainty_and_disagreement": ["UNCERTAIN" if float(summary.mean_uncertainty[i]) > 0.5 and float(summary.disagreement_entropy[i]) > 0.1 else base_label[i] for i in range(len(labels))],
    }
    return out


def recoverable_gap(cache: dict[str, Any], obj: dict[str, Any]) -> dict[str, Any]:
    labels = cache["labels"]
    oracle = oracle_flags(cache, obj)
    current = obj["reducer_v0"]
    cf = obj["reducer_label"]
    util = [i for i in range(len(labels)) if oracle[i] and current[i] != labels[i]]
    false_abstain_recoverable = [i for i in util if current[i] == "UNCERTAIN" and labels[i] != "UNCERTAIN" and cf[i] == labels[i]]
    label_failures = [i for i in util if current[i] != "UNCERTAIN"]
    abstain_but_label_wrong = [i for i in util if current[i] == "UNCERTAIN" and cf[i] != labels[i]]
    return {
        "utilization_failures": len(util),
        "theoretically_recoverable_false_abstentions": len(false_abstain_recoverable),
        "wrong_non_uncertain_label_selection_failures": len(label_failures),
        "abstentions_where_counterfactual_label_still_wrong_or_truth_uncertain": len(abstain_but_label_wrong),
        "max_accuracy_gain_from_perfect_abstention_fix_only": len(false_abstain_recoverable) / len(labels),
    }


def analyze_dataset(cache: dict[str, Any]) -> dict[str, Any]:
    obj = objects(cache)
    pol = policies(cache, obj)
    return {
        "abstention_decomposition": abstention_decomposition(cache, obj),
        "counterfactual_label_selection": counterfactuals(cache, obj),
        "uncertainty_calibration": calibration(cache, obj),
        "interactions": interaction_analysis(cache, obj),
        "policy_results": [evaluate_predictions(cache, obj, preds, name) for name, preds in pol.items()],
        "recoverable_gap": recoverable_gap(cache, obj),
    }


def main() -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    dev = load_cache(DEV_CACHE)
    confirm = load_cache(CONFIRM_CACHE)
    summary = {
        "phase": "step2a4_uncertainty_abstention_isolation_diagnostic_only",
        "frozen_test_accessed": False,
        "workers_trained": False,
        "dev_confirmation_2_generated": False,
        "current_uncertainty_decision_tree": audit_pipeline(),
        "aggregation_development": analyze_dataset(dev),
        "dev_confirmation_1_consumed": analyze_dataset(confirm),
    }
    dump(ROOT / "summary.json", summary)
    dump(ROOT / "uncertainty_decision_tree.json", summary["current_uncertainty_decision_tree"])
    dump(ROOT / "aggregation_development_uncertainty_analysis.json", summary["aggregation_development"])
    dump(ROOT / "dev_confirmation_1_uncertainty_analysis.json", summary["dev_confirmation_1_consumed"])
    print(json.dumps({"event": "step2a4_complete", "summary": str(ROOT / "summary.json")}, sort_keys=True))


if __name__ == "__main__":
    main()
