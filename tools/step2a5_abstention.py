"""Step 2A.5 calibrated abstention layer protocol and development."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from ai_hypothesis.step01.model import NON_UNCERTAIN_LABELS, Step01Output, decode_predictions
from ai_hypothesis.step02.abstention import (
    AbstentionConfig,
    apply_abstention,
    standardize_features,
    validate_inference_feature_names,
)
from ai_hypothesis.step02.evidence import AggregationConfig, aggregate_evidence, build_evidence_matrix
from ai_hypothesis.step02.evaluation import _mean_probability_predictions, _worker_prediction_indices
from ai_hypothesis.step02.population import PopulationOutput
from tools.step2a4_uncertainty import load_cache, task_members, DEV_CACHE, CONFIRM_CACHE

ROOT = Path("results/step02/step2a5_calibrated_abstention")
LABEL_CANDIDATES = ("reducer_v0_pre_abstention", "mean_logit", "mean_probability")


def dump(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def base_objects(cache: dict[str, Any]) -> dict[str, Any]:
    output = PopulationOutput(cache["label_logits"], cache["uncertainty_logits"])
    evidence = build_evidence_matrix(output, task_members(cache["tasks"]), AggregationConfig())
    summary, decision = aggregate_evidence(evidence, AggregationConfig())
    mean_logits = output.label_logits.mean(dim=0)
    probs = torch.softmax(output.label_logits, dim=-1)
    mean_probs = probs.mean(dim=0)
    mean_logit_output = Step01Output(mean_logits, output.uncertainty_logits.mean(dim=0))
    current_mean_logit = list(decode_predictions(mean_logit_output, uncertainty_threshold=0.5))
    current_mean_prob = list(_mean_probability_predictions(output, uncertainty_threshold=0.5))
    return {
        "output": output,
        "evidence": evidence,
        "summary": summary,
        "decision": decision,
        "mean_logits": mean_logits,
        "mean_probs": mean_probs,
        "candidates": {
            "reducer_v0_pre_abstention": [NON_UNCERTAIN_LABELS[int(i)] for i in decision.primary_label_indices],
            "mean_logit": [NON_UNCERTAIN_LABELS[int(i)] for i in mean_logits.argmax(dim=-1)],
            "mean_probability": [NON_UNCERTAIN_LABELS[int(i)] for i in mean_probs.argmax(dim=-1)],
        },
        "current_predictions": {
            "reducer_v0": list(decision.predictions),
            "mean_logit": current_mean_logit,
            "mean_probability": current_mean_prob,
        },
    }


def feature_matrix(cache: dict[str, Any], obj: dict[str, Any], source: str) -> tuple[torch.Tensor, tuple[str, ...]]:
    output = obj["output"]
    evidence = obj["evidence"]
    summary = obj["summary"]
    candidates = obj["candidates"][source]
    probs = torch.softmax(output.label_logits, dim=-1)
    worker_unc = torch.sigmoid(output.uncertainty_logits)
    worker_decoded_unc = (worker_unc >= 0.5).to(torch.float32)
    mean_logits = obj["mean_logits"]
    mean_probs = obj["mean_probs"]
    rows = []
    for i, candidate in enumerate(candidates):
        cand_idx = NON_UNCERTAIN_LABELS.index(candidate)
        worker_label_argmax = output.label_logits[:, i, :].argmax(dim=-1)
        supporters = worker_label_argmax == cand_idx
        support_count = int(supporters.sum())
        cand_evidence = evidence.evidence_scores[:, i, cand_idx]
        supporter_evidence = cand_evidence[supporters] if support_count else cand_evidence[:0]
        valid = evidence.valid_label_mask[i]
        mean_evidence_valid = summary.mean_evidence_per_label[i].masked_fill(~valid, float("-inf"))
        top_ev = torch.topk(mean_evidence_valid, k=2).values
        top_logit = torch.topk(mean_logits[i].masked_fill(~valid, float("-inf")), k=2).values
        top_prob = torch.topk(mean_probs[i].masked_fill(~valid, float("-inf")), k=2).values
        distinct_predictions = len(set(int(v) for v in worker_label_argmax.tolist()))
        protected_competitor = bool(summary.protected_label_mask[i].any())
        row = [
            float(summary.mean_uncertainty[i]),
            float(summary.max_uncertainty[i]),
            float(worker_unc[:, i].min()),
            float(worker_unc[:, i].var(unbiased=False)),
            float(worker_decoded_unc[:, i].mean()),
            support_count / output.label_logits.shape[0],
            float(support_count),
            float(summary.mean_evidence_per_label[i, cand_idx]),
            float(summary.sum_evidence_per_label[i, cand_idx]),
            float(summary.max_evidence_per_label[i, cand_idx]),
            float(supporter_evidence.mean()) if support_count else 0.0,
            float(supporter_evidence.var(unbiased=False)) if support_count else 0.0,
            float(top_ev[0] - top_ev[1]),
            float(top_logit[0] - top_logit[1]),
            float(top_prob[0] - top_prob[1]),
            float(summary.disagreement_entropy[i]),
            float(distinct_predictions),
            float(summary.mean_invalid_label_mass[i]),
            float(protected_competitor),
            float(obj["decision"].primary_margin[i]),
        ]
        rows.append(row)
    names = (
        "mean_worker_uncertainty",
        "max_worker_uncertainty",
        "min_worker_uncertainty",
        "worker_uncertainty_variance",
        "fraction_workers_decoded_uncertain",
        "candidate_support_fraction",
        "candidate_support_count",
        "candidate_mean_evidence",
        "candidate_sum_evidence",
        "candidate_max_evidence",
        "supporter_mean_evidence",
        "supporter_evidence_variance",
        "mean_evidence_margin_top1_top2",
        "mean_logit_margin_top1_top2",
        "mean_probability_margin_top1_top2",
        "disagreement_entropy",
        "distinct_worker_prediction_count",
        "mean_invalid_label_mass",
        "has_protected_competitor",
        "reducer_primary_margin",
    )
    validate_inference_feature_names(names)
    return torch.tensor(rows, dtype=torch.float32), names


def labels_to_target(cache: dict[str, Any]) -> torch.Tensor:
    return torch.tensor([1.0 if label == "UNCERTAIN" else 0.0 for label in cache["labels"]], dtype=torch.float32)


def split_indices(cache: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
    # index % 5 keeps the balanced task/difficulty stream distributed across both partitions.
    train = [i for i in range(len(cache["labels"])) if i % 5 != 4]
    calib = [i for i in range(len(cache["labels"])) if i % 5 == 4]
    return torch.tensor(train, dtype=torch.long), torch.tensor(calib, dtype=torch.long)


def train_logistic(x: torch.Tensor, y: torch.Tensor, train_idx: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    torch.manual_seed(2025)
    tx = x[train_idx]
    ty = y[train_idx]
    mean = tx.mean(dim=0)
    std = tx.std(dim=0).clamp_min(1e-6)
    sx = standardize_features(tx, mean, std)
    model = torch.nn.Linear(x.shape[1], 1)
    opt = torch.optim.Adam(model.parameters(), lr=0.05, weight_decay=1e-3)
    pos_weight = ((ty == 0).sum() / (ty == 1).sum()).clamp_min(1.0)
    for _ in range(300):
        opt.zero_grad()
        logits = model(sx).squeeze(-1)
        loss = F.binary_cross_entropy_with_logits(logits, ty, pos_weight=pos_weight)
        loss.backward()
        opt.step()
    with torch.no_grad():
        weights = model.weight.detach().flatten().clone()
        bias = model.bias.detach().clone()
    return weights, bias, mean, std


def logistic_scores(x: torch.Tensor, weights: torch.Tensor, bias: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    sx = standardize_features(x, mean, std)
    return torch.sigmoid(sx @ weights + bias)


def train_stump(x: torch.Tensor, y: torch.Tensor, train_idx: torch.Tensor, calib_idx: torch.Tensor) -> dict[str, Any]:
    best = {"score": -1.0}
    tx = x[train_idx]
    ty = y[train_idx]
    cx = x[calib_idx]
    cy = y[calib_idx]
    for feature in range(x.shape[1]):
        vals = torch.quantile(tx[:, feature], torch.tensor([0.2, 0.4, 0.6, 0.8]))
        for threshold in vals.tolist():
            for direction in ("ge", "lt"):
                pred = (cx[:, feature] >= threshold) if direction == "ge" else (cx[:, feature] < threshold)
                acc = float((pred.to(torch.float32) == cy).to(torch.float32).mean())
                if acc > best["score"]:
                    best = {"feature_index": feature, "threshold": float(threshold), "direction": direction, "score": acc}
    return best


def stump_scores(x: torch.Tensor, stump: dict[str, Any]) -> torch.Tensor:
    if stump["direction"] == "ge":
        return (x[:, stump["feature_index"]] >= stump["threshold"]).to(torch.float32)
    return (x[:, stump["feature_index"]] < stump["threshold"]).to(torch.float32)


def evaluate(cache: dict[str, Any], obj: dict[str, Any], source: str, scores: torch.Tensor, threshold: float, name: str) -> dict[str, Any]:
    candidates = obj["candidates"][source]
    preds = apply_abstention(candidates, scores, threshold=threshold)
    truth = cache["labels"]
    total = len(truth)
    pred_unc = [p == "UNCERTAIN" for p in preds]
    truth_unc = [t == "UNCERTAIN" for t in truth]
    correct = sum(p == t for p, t in zip(preds, truth))
    non_unc = [i for i, t in enumerate(truth) if t != "UNCERTAIN"]
    tp = sum(p and t for p, t in zip(pred_unc, truth_unc))
    fp = sum(p and not t for p, t in zip(pred_unc, truth_unc))
    fn = sum((not p) and t for p, t in zip(pred_unc, truth_unc))
    # Candidate-source oracle for utilization gap: at least one worker decode matches truth.
    worker_idx = _worker_prediction_indices(obj["output"], uncertainty_threshold=0.5)
    decoded_labels = (*NON_UNCERTAIN_LABELS, "UNCERTAIN")
    oracle = []
    for i, t in enumerate(truth):
        oracle.append(any(decoded_labels[int(worker_idx[w, i])] == t for w in range(worker_idx.shape[0])))
    return {
        "name": name,
        "candidate_source": source,
        "threshold": threshold,
        "overall_accuracy": correct / total,
        "non_uncertain_label_accuracy": sum(preds[i] == truth[i] for i in non_unc) / len(non_unc),
        "uncertain_precision": None if sum(pred_unc) == 0 else tp / sum(pred_unc),
        "uncertain_recall": tp / sum(truth_unc),
        "false_abstention_rate": fp / total,
        "false_abstention_count": fp,
        "missed_uncertain_rate": fn / sum(truth_unc),
        "missed_uncertain_count": fn,
        "uncertain_prediction_rate": sum(pred_unc) / total,
        "utilization_gap": (sum(oracle) - correct) / total,
    }


def threshold_grid(scores: torch.Tensor, y: torch.Tensor, calib_idx: torch.Tensor) -> list[float]:
    # Small deterministic set plus score quantiles from calibration split.
    vals = {0.3, 0.4, 0.5, 0.6, 0.7}
    qs = torch.quantile(scores[calib_idx], torch.tensor([0.2, 0.35, 0.5, 0.65, 0.8]))
    vals.update(float(v) for v in qs)
    return sorted(v for v in vals if 0.0 <= v <= 1.0)


def select_threshold(cache: dict[str, Any], obj: dict[str, Any], source: str, scores: torch.Tensor, calib_idx: torch.Tensor, name: str) -> tuple[float, list[dict[str, Any]]]:
    rows = []
    for threshold in threshold_grid(scores, labels_to_target(cache), calib_idx):
        full = evaluate(cache, obj, source, scores, threshold, name)
        # Recompute on calibration subset for model selection.
        subset_cache = {**cache, "labels": [cache["labels"][int(i)] for i in calib_idx]}
        subset_obj = {**obj, "candidates": {source: [obj["candidates"][source][int(i)] for i in calib_idx]}, "output": obj["output"]}
        preds = apply_abstention([obj["candidates"][source][int(i)] for i in calib_idx], scores[calib_idx], threshold=threshold)
        truth = subset_cache["labels"]
        truth_unc = [t == "UNCERTAIN" for t in truth]
        pred_unc = [p == "UNCERTAIN" for p in preds]
        acc = sum(p == t for p, t in zip(preds, truth)) / len(truth)
        recall = sum(p and t for p, t in zip(pred_unc, truth_unc)) / sum(truth_unc)
        precision = None if sum(pred_unc) == 0 else sum(p and t for p, t in zip(pred_unc, truth_unc)) / sum(pred_unc)
        rows.append({"threshold": threshold, "internal_accuracy": acc, "uncertain_recall": recall, "uncertain_precision": precision, "full_development": full})
    # Preserve recall reasonably: within 3pp of current v0 recall target, then maximize accuracy.
    eligible = [r for r in rows if r["uncertain_recall"] >= 0.91]
    if not eligible:
        eligible = rows
    eligible.sort(key=lambda r: (r["internal_accuracy"], r["uncertain_recall"]), reverse=True)
    return eligible[0]["threshold"], rows


def run_dataset(cache: dict[str, Any], obj: dict[str, Any], source: str, model_info: dict[str, Any], threshold: float, name: str) -> dict[str, Any]:
    x, _ = feature_matrix(cache, obj, source)
    if model_info["model_type"] == "logistic_regression":
        scores = logistic_scores(x, model_info["weights"], model_info["bias"], model_info["mean"], model_info["std"])
    elif model_info["model_type"] == "decision_stump":
        scores = stump_scores(x, model_info["stump"])
    else:
        scores = x[:, 0]
    return evaluate(cache, obj, source, scores, threshold, name)


def main() -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    dev = load_cache(DEV_CACHE)
    confirm = load_cache(CONFIRM_CACHE)
    dev_obj = base_objects(dev)
    confirm_obj = base_objects(confirm)
    train_idx, calib_idx = split_indices(dev)
    split_protocol = {"method": "index % 5 == 4 is internal calibration; all other samples are abstention-training", "train_count": int(train_idx.numel()), "calibration_count": int(calib_idx.numel())}

    baselines = []
    for source in LABEL_CANDIDATES:
        x, names = feature_matrix(dev, dev_obj, source)
        baselines.append(evaluate(dev, dev_obj, source, x[:, 0], 0.5, f"fixed_threshold_0_5_{source}"))
        baselines.append(evaluate(dev, dev_obj, source, x[:, 0], 0.6, f"fixed_threshold_0_6_{source}"))
    for current in ("reducer_v0", "mean_logit", "mean_probability"):
        preds = dev_obj["current_predictions"][current]
        # Encode current predictions as candidate source is irrelevant for display.
        truth = dev["labels"]
        acc = sum(p == t for p, t in zip(preds, truth)) / len(truth)
        baselines.append({"name": f"current_{current}", "overall_accuracy": acc})

    model_rows = []
    configs = []
    for source in LABEL_CANDIDATES:
        x, names = feature_matrix(dev, dev_obj, source)
        y = labels_to_target(dev)
        weights, bias, mean, std = train_logistic(x, y, train_idx)
        scores = logistic_scores(x, weights, bias, mean, std)
        threshold, frontier = select_threshold(dev, dev_obj, source, scores, calib_idx, f"logistic_{source}")
        model_info = {"model_type": "logistic_regression", "weights": weights, "bias": bias, "mean": mean, "std": std}
        model_rows.append({
            "candidate_source": source,
            "model_type": "logistic_regression",
            "selected_threshold": threshold,
            "development": evaluate(dev, dev_obj, source, scores, threshold, f"logistic_{source}"),
            "internal_frontier": frontier,
            "consumed_confirmation_1": run_dataset(confirm, confirm_obj, source, model_info, threshold, f"logistic_{source}"),
        })
        configs.append((model_rows[-1], AbstentionConfig(
            version="step2a5-abstention-diagnostic-v0",
            candidate_source=source,
            feature_names=names,
            model_type="logistic_regression",
            threshold=threshold,
            weights=tuple(float(v) for v in weights),
            bias=float(bias),
            feature_mean=tuple(float(v) for v in mean),
            feature_std=tuple(float(v) for v in std),
        )))

        stump = train_stump(x, y, train_idx, calib_idx)
        stump_scores_dev = stump_scores(x, stump)
        st_threshold, st_frontier = select_threshold(dev, dev_obj, source, stump_scores_dev, calib_idx, f"stump_{source}")
        stump_info = {"model_type": "decision_stump", "stump": stump}
        model_rows.append({
            "candidate_source": source,
            "model_type": "decision_stump",
            "selected_threshold": st_threshold,
            "development": evaluate(dev, dev_obj, source, stump_scores_dev, st_threshold, f"stump_{source}"),
            "internal_frontier": st_frontier,
            "consumed_confirmation_1": run_dataset(confirm, confirm_obj, source, stump_info, st_threshold, f"stump_{source}"),
            "stump": stump,
        })

    # Candidate must improve dev, improve confirmation direction, preserve recall >= 0.91 on confirmation.
    model_rows.sort(key=lambda r: (r["development"]["overall_accuracy"], r["development"]["uncertain_recall"]), reverse=True)
    best = model_rows[0]
    ready = (
        best["development"]["overall_accuracy"] > 0.95205
        and best["consumed_confirmation_1"]["overall_accuracy"] > 0.9548
        and best["consumed_confirmation_1"]["uncertain_recall"] >= 0.91
    )
    best_config = None
    for row, config in configs:
        if row is best:
            best_config = config
            break
    if best_config is not None and ready:
        best_config.save(ROOT / "candidate_abstention_config.json")
    summary = {
        "phase": "step2a5_calibrated_abstention_layer_protocol_development",
        "frozen_test_accessed": False,
        "workers_trained": False,
        "dev_confirmation_2_generated": False,
        "two_stage_contract": {
            "stage_1": "select exactly one non-UNCERTAIN label candidate",
            "stage_2": "abstention layer may accept that label or emit UNCERTAIN; it may not change label A into label B",
            "primary_target": "truth == UNCERTAIN",
        },
        "feature_names": list(feature_matrix(dev, dev_obj, LABEL_CANDIDATES[0])[1]),
        "development_split_protocol": split_protocol,
        "baselines": baselines,
        "models": model_rows,
        "best_model": best,
        "verdict": "READY_FOR_FRESH_CONFIRMATION" if ready else "ABSTENTION_CALIBRATION_NOT_CONFIRMED",
        "dev_confirmation_2_recommended": ready,
        "candidate_protocol": asdict(best_config) if best_config is not None and ready else None,
    }
    dump(ROOT / "summary.json", summary)
    dump(ROOT / "development_models.json", model_rows)
    dump(ROOT / "baselines.json", baselines)
    print(json.dumps({"event": "step2a5_complete", "verdict": summary["verdict"], "summary": str(ROOT / "summary.json")}, sort_keys=True))


if __name__ == "__main__":
    main()

