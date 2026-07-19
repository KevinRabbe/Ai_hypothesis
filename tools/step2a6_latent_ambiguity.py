"""Step 2A.6 latent ambiguity signal audit.

Diagnostic only: no frozen test access, no worker training, no fresh confirmation.
"""

from __future__ import annotations

import json
import math
import subprocess
from dataclasses import asdict
from pathlib import Path
from typing import Any, Sequence

import torch
import torch.nn.functional as F

from ai_hypothesis.step01.model import NON_UNCERTAIN_LABELS, Step01Unit, UnitConfig
from ai_hypothesis.step01.schema import BENCHMARK_VERSION
from ai_hypothesis.step01.torch_data import collate_samples, make_loader
from ai_hypothesis.step02.abstention import apply_abstention, standardize_features
from ai_hypothesis.step02.latent import (
    extract_pooled_latent,
    summarize_worker_local_scalar_scores,
)
from tools.step2a2_redesign import (
    COUNT,
    DEV_CONFIRM_BASE_SEED,
    checkpoint_paths,
    dev_confirmation_samples,
)
from tools.step2a4_uncertainty import CONFIRM_CACHE, DEV_CACHE, load_cache, objects
from tools.step2a5_abstention import (
    LABEL_CANDIDATES,
    base_objects,
    feature_matrix,
    logistic_scores,
    train_logistic,
)

ROOT = Path("results/step02/step2a6_latent_ambiguity_signal_audit")
DEV_LATENT_CACHE = ROOT / "cache" / "w5_validation_latents.pt"
CONFIRM_LATENT_CACHE = ROOT / "cache" / "w5_dev_confirmation_1_latents.pt"
BATCH_SIZE = 512
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LABELS = (*NON_UNCERTAIN_LABELS, "UNCERTAIN")


def dump(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def git_revision() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()


def load_worker_models(paths: Sequence[Path]) -> list[Step01Unit]:
    models: list[Step01Unit] = []
    for path in paths:
        checkpoint = torch.load(path, map_location=DEVICE)
        model = Step01Unit(UnitConfig(**checkpoint["unit_config"])).to(DEVICE)
        model.load_state_dict(checkpoint["model_state"], strict=True)
        model.eval()
        models.append(model)
    return models


def build_validation_latent_cache() -> dict[str, Any]:
    paths = checkpoint_paths()
    models = load_worker_models(paths)
    latent_chunks = [[] for _ in models]
    labels: list[str] = []
    tasks: list[str] = []
    difficulties: list[str] = []
    sample_seeds: list[int] = []
    loader = make_loader(
        split="validation",
        count=COUNT,
        batch_size=BATCH_SIZE,
        shuffle=False,
        seed=1,
        num_workers=0,
    )
    with torch.inference_mode():
        for batch in loader:
            features = batch["features"].to(DEVICE)
            mask = batch["mask"].to(DEVICE)
            for worker_index, model in enumerate(models):
                latent_chunks[worker_index].append(
                    extract_pooled_latent(model, features, mask).detach().cpu()
                )
            for sample in batch["samples"]:
                labels.append(sample.label)
                tasks.append(sample.task.value)
                difficulties.append(sample.difficulty.value)
                sample_seeds.append(sample.seed)
    payload = {
        "phase": "step2a6_validation_latent_cache",
        "split": "validation",
        "benchmark_version": BENCHMARK_VERSION,
        "count": COUNT,
        "device": DEVICE,
        "checkpoints": [str(path) for path in paths],
        "git_revision": git_revision(),
        "latent_contract": "worker-local shared pre-head masked pooled representation",
        "latents": torch.stack([torch.cat(chunks, dim=0) for chunks in latent_chunks], dim=0),
        "labels": labels,
        "tasks": tasks,
        "difficulties": difficulties,
        "sample_seeds": sample_seeds,
    }
    DEV_LATENT_CACHE.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, DEV_LATENT_CACHE)
    return payload


def build_confirmation_latent_cache() -> dict[str, Any]:
    paths = checkpoint_paths()
    models = load_worker_models(paths)
    latent_chunks = [[] for _ in models]
    labels: list[str] = []
    tasks: list[str] = []
    difficulties: list[str] = []
    sample_seeds: list[int] = []
    batch_samples = []
    with torch.inference_mode():
        for sample in dev_confirmation_samples(COUNT):
            batch_samples.append(sample)
            if len(batch_samples) == BATCH_SIZE:
                batch = collate_samples(batch_samples)
                features = batch["features"].to(DEVICE)
                mask = batch["mask"].to(DEVICE)
                for worker_index, model in enumerate(models):
                    latent_chunks[worker_index].append(
                        extract_pooled_latent(model, features, mask).detach().cpu()
                    )
                for item in batch_samples:
                    labels.append(item.label)
                    tasks.append(item.task.value)
                    difficulties.append(item.difficulty.value)
                    sample_seeds.append(item.seed)
                batch_samples = []
        if batch_samples:
            batch = collate_samples(batch_samples)
            features = batch["features"].to(DEVICE)
            mask = batch["mask"].to(DEVICE)
            for worker_index, model in enumerate(models):
                latent_chunks[worker_index].append(
                    extract_pooled_latent(model, features, mask).detach().cpu()
                )
            for item in batch_samples:
                labels.append(item.label)
                tasks.append(item.task.value)
                difficulties.append(item.difficulty.value)
                sample_seeds.append(item.seed)
    payload = {
        "phase": "step2a6_dev_confirmation_1_latent_cache",
        "split": "dev_confirmation_1_consumed",
        "benchmark_version": BENCHMARK_VERSION,
        "generation_contract": "balanced TASKS x DIFFICULTIES stream using generate_sample with already-consumed Step 2A.2 base seed",
        "base_seed": DEV_CONFIRM_BASE_SEED,
        "count": COUNT,
        "device": DEVICE,
        "checkpoints": [str(path) for path in paths],
        "git_revision": git_revision(),
        "latent_contract": "worker-local shared pre-head masked pooled representation",
        "latents": torch.stack([torch.cat(chunks, dim=0) for chunks in latent_chunks], dim=0),
        "labels": labels,
        "tasks": tasks,
        "difficulties": difficulties,
        "sample_seeds": sample_seeds,
    }
    CONFIRM_LATENT_CACHE.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, CONFIRM_LATENT_CACHE)
    return payload


def load_or_build_latents(path: Path, builder) -> dict[str, Any]:
    if path.exists():
        return torch.load(path, map_location="cpu", weights_only=False)
    return builder()


def assert_cache_alignment(output_cache: dict[str, Any], latent_cache: dict[str, Any]) -> None:
    for key in ("labels", "tasks", "difficulties", "sample_seeds"):
        if output_cache[key] != latent_cache[key]:
            raise RuntimeError(f"latent cache does not align with output cache for {key}")


def split_indices(count: int) -> tuple[torch.Tensor, torch.Tensor]:
    train = [i for i in range(count) if i % 5 != 4]
    calib = [i for i in range(count) if i % 5 == 4]
    return torch.tensor(train, dtype=torch.long), torch.tensor(calib, dtype=torch.long)


def target(labels: Sequence[str]) -> torch.Tensor:
    return torch.tensor([1.0 if label == "UNCERTAIN" else 0.0 for label in labels])


def rank_auc(scores: torch.Tensor, y: torch.Tensor) -> float | None:
    pos = y == 1
    neg = y == 0
    pos_count = int(pos.sum())
    neg_count = int(neg.sum())
    if pos_count == 0 or neg_count == 0:
        return None
    order = torch.argsort(scores)
    sorted_scores = scores[order]
    ranks = torch.empty_like(scores, dtype=torch.float64)
    start = 0
    while start < scores.numel():
        end = start + 1
        while end < scores.numel() and float(sorted_scores[end]) == float(sorted_scores[start]):
            end += 1
        avg_rank = (start + 1 + end) / 2.0
        ranks[order[start:end]] = avg_rank
        start = end
    rank_sum_pos = float(ranks[pos].sum())
    return (rank_sum_pos - pos_count * (pos_count + 1) / 2.0) / (pos_count * neg_count)


def average_precision(scores: torch.Tensor, y: torch.Tensor) -> float | None:
    pos_count = int((y == 1).sum())
    if pos_count == 0:
        return None
    order = torch.argsort(scores, descending=True)
    sorted_y = y[order]
    tp = torch.cumsum(sorted_y, dim=0)
    precision = tp / torch.arange(1, y.numel() + 1, dtype=torch.float32)
    return float((precision * sorted_y).sum() / pos_count)


def summarize_scores(values: torch.Tensor) -> dict[str, Any]:
    if values.numel() == 0:
        return {"count": 0}
    qs = torch.quantile(values.float(), torch.tensor([0.1, 0.25, 0.5, 0.75, 0.9]))
    return {
        "count": int(values.numel()),
        "mean": float(values.float().mean()),
        "std": float(values.float().std(unbiased=False)),
        "p10": float(qs[0]),
        "p25": float(qs[1]),
        "median": float(qs[2]),
        "p75": float(qs[3]),
        "p90": float(qs[4]),
    }


def calibration(scores: torch.Tensor, y: torch.Tensor, bins: int = 10) -> dict[str, float]:
    brier = float(torch.mean((scores - y) ** 2))
    ece = 0.0
    for index in range(bins):
        low = index / bins
        high = (index + 1) / bins
        mask = (scores >= low) & ((scores <= high) if index == bins - 1 else (scores < high))
        if bool(mask.any()):
            ece += float(mask.float().mean()) * abs(float(scores[mask].mean()) - float(y[mask].mean()))
    return {"brier": brier, "ece_10_bin": ece}


def threshold_metrics(scores: torch.Tensor, y: torch.Tensor, threshold: float) -> dict[str, Any]:
    pred = scores >= threshold
    truth = y == 1
    tp = int((pred & truth).sum())
    fp = int((pred & ~truth).sum())
    fn = int((~pred & truth).sum())
    tn = int((~pred & ~truth).sum())
    specificity = None if fp + tn == 0 else tn / (fp + tn)
    false_positive_rate = None if fp + tn == 0 else fp / (fp + tn)
    return {
        "threshold": threshold,
        "accuracy": (tp + tn) / y.numel(),
        "precision": None if tp + fp == 0 else tp / (tp + fp),
        "recall": None if tp + fn == 0 else tp / (tp + fn),
        "specificity": specificity,
        "false_positive_rate": false_positive_rate,
        "false_abstention_count": fp,
        "missed_uncertain_count": fn,
    }


def evaluate_scores(scores: torch.Tensor, y: torch.Tensor, threshold: float) -> dict[str, Any]:
    pos = scores[y == 1]
    neg = scores[y == 0]
    return {
        "roc_auc": rank_auc(scores, y),
        "pr_auc": average_precision(scores, y),
        **threshold_metrics(scores, y, threshold),
        **calibration(scores, y),
        "score_distribution": {
            "truth_uncertain": summarize_scores(pos),
            "truth_non_uncertain": summarize_scores(neg),
        },
    }


def select_threshold(scores: torch.Tensor, y: torch.Tensor, calib_idx: torch.Tensor) -> float:
    vals = {0.3, 0.4, 0.5, 0.6, 0.7}
    vals.update(float(v) for v in torch.quantile(scores[calib_idx], torch.tensor([0.2, 0.35, 0.5, 0.65, 0.8])))
    best = None
    for threshold in sorted(v for v in vals if 0 <= v <= 1):
        metrics = threshold_metrics(scores[calib_idx], y[calib_idx], threshold)
        recall = metrics["recall"] or 0.0
        precision = metrics["precision"] or 0.0
        specificity = metrics["specificity"] if metrics["specificity"] is not None else 0.0
        balanced = 0.5 * (recall + specificity)
        key = (balanced, metrics["accuracy"], recall, precision)
        if best is None or key > best[0]:
            best = (key, threshold)
    assert best is not None
    return float(best[1])


def train_binary_logistic(
    x: torch.Tensor,
    y: torch.Tensor,
    train_idx: torch.Tensor,
    *,
    epochs: int = 400,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    torch.manual_seed(2606)
    tx = x[train_idx]
    ty = y[train_idx]
    mean = tx.mean(dim=0)
    std = tx.std(dim=0).clamp_min(1e-6)
    sx = standardize_features(tx, mean, std)
    layer = torch.nn.Linear(x.shape[1], 1)
    opt = torch.optim.Adam(layer.parameters(), lr=0.05, weight_decay=1e-3)
    pos_weight = ((ty == 0).sum() / (ty == 1).sum()).clamp_min(1.0)
    for _ in range(epochs):
        opt.zero_grad()
        logits = layer(sx).squeeze(-1)
        loss = F.binary_cross_entropy_with_logits(logits, ty, pos_weight=pos_weight)
        loss.backward()
        opt.step()
    return (
        layer.weight.detach().flatten().clone(),
        layer.bias.detach().clone(),
        mean,
        std,
    )


def apply_binary_logistic(x: torch.Tensor, model: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]) -> torch.Tensor:
    weights, bias, mean, std = model
    return torch.sigmoid(standardize_features(x, mean, std) @ weights + bias)


def group_breakdown(scores: torch.Tensor, y: torch.Tensor, groups: Sequence[str], threshold: float) -> dict[str, Any]:
    rows = {}
    for group in sorted(set(groups)):
        idx = torch.tensor([i for i, value in enumerate(groups) if value == group], dtype=torch.long)
        rows[group] = evaluate_scores(scores[idx], y[idx], threshold)
    return rows


def evaluate_candidate_predictions(labels: Sequence[str], candidates: Sequence[str], scores: torch.Tensor, threshold: float) -> dict[str, Any]:
    preds = apply_abstention(candidates, scores, threshold=threshold)
    total = len(labels)
    truth_unc = [label == "UNCERTAIN" for label in labels]
    pred_unc = [pred == "UNCERTAIN" for pred in preds]
    correct = sum(pred == truth for pred, truth in zip(preds, labels))
    tp = sum(p and t for p, t in zip(pred_unc, truth_unc))
    fp = sum(p and not t for p, t in zip(pred_unc, truth_unc))
    fn = sum((not p) and t for p, t in zip(pred_unc, truth_unc))
    non_unc = [i for i, label in enumerate(labels) if label != "UNCERTAIN"]
    return {
        "threshold": threshold,
        "overall_accuracy": correct / total,
        "non_uncertain_label_accuracy": sum(preds[i] == labels[i] for i in non_unc) / len(non_unc),
        "uncertain_precision": None if sum(pred_unc) == 0 else tp / sum(pred_unc),
        "uncertain_recall": tp / sum(truth_unc),
        "false_abstention_count": fp,
        "missed_uncertain_count": fn,
        "uncertain_prediction_rate": sum(pred_unc) / total,
    }


def separation_effect(a: torch.Tensor, b: torch.Tensor) -> dict[str, Any]:
    if a.numel() == 0 or b.numel() == 0:
        return {"cohen_d": None, "pairwise_probability_a_greater_b": None}
    pooled = math.sqrt((float(a.var(unbiased=False)) + float(b.var(unbiased=False))) / 2.0)
    cohen = None if pooled == 0 else (float(a.mean()) - float(b.mean())) / pooled
    sorted_b = torch.sort(b).values
    greater = torch.searchsorted(sorted_b, a, right=False).float().mean() / b.numel()
    return {"cohen_d": cohen, "pairwise_probability_a_greater_b": float(greater)}


def current_reducer_categories(cache: dict[str, Any]) -> dict[str, list[int]]:
    obj = objects(cache)
    labels = cache["labels"]
    categories = {
        "genuine_uncertain_correctly_abstained": [],
        "false_abstention_counterfactual_label_correct": [],
        "useful_abstention_counterfactual_label_wrong": [],
    }
    for index, truth in enumerate(labels):
        if truth == "UNCERTAIN" and obj["reducer_v0"][index] == "UNCERTAIN":
            categories["genuine_uncertain_correctly_abstained"].append(index)
        elif truth != "UNCERTAIN" and obj["reducer_v0"][index] == "UNCERTAIN":
            if obj["reducer_label"][index] == truth:
                categories["false_abstention_counterfactual_label_correct"].append(index)
            else:
                categories["useful_abstention_counterfactual_label_wrong"].append(index)
    return categories


def category_distributions(scores_by_name: dict[str, torch.Tensor], categories: dict[str, list[int]]) -> dict[str, Any]:
    rows = {}
    a_idx = torch.tensor(categories["genuine_uncertain_correctly_abstained"], dtype=torch.long)
    b_idx = torch.tensor(categories["false_abstention_counterfactual_label_correct"], dtype=torch.long)
    c_idx = torch.tensor(categories["useful_abstention_counterfactual_label_wrong"], dtype=torch.long)
    for name, scores in scores_by_name.items():
        a = scores[a_idx]
        b = scores[b_idx]
        c = scores[c_idx]
        rows[name] = {
            "genuine_uncertain_correctly_abstained": summarize_scores(a),
            "false_abstention_counterfactual_label_correct": summarize_scores(b),
            "useful_abstention_counterfactual_label_wrong": summarize_scores(c),
            "effect_genuine_vs_false": separation_effect(a, b),
            "effect_genuine_vs_useful": separation_effect(a, c),
            "effect_useful_vs_false": separation_effect(c, b),
        }
    return rows


def step2a5_output_feature_scores(dev_output_cache: dict[str, Any], confirm_output_cache: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
    dev_obj = base_objects(dev_output_cache)
    confirm_obj = base_objects(confirm_output_cache)
    train_idx, _ = split_indices(len(dev_output_cache["labels"]))
    x_dev, _ = feature_matrix(dev_output_cache, dev_obj, "mean_logit")
    y_dev = target(dev_output_cache["labels"])
    weights, bias, mean, std = train_logistic(x_dev, y_dev, train_idx)
    model = (weights, bias, mean, std)
    x_confirm, _ = feature_matrix(confirm_output_cache, confirm_obj, "mean_logit")
    return logistic_scores(x_dev, *model), logistic_scores(x_confirm, *model)


def main() -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    dev_output = load_cache(DEV_CACHE)
    confirm_output = load_cache(CONFIRM_CACHE)
    dev_latent = load_or_build_latents(DEV_LATENT_CACHE, build_validation_latent_cache)
    confirm_latent = load_or_build_latents(CONFIRM_LATENT_CACHE, build_confirmation_latent_cache)
    assert_cache_alignment(dev_output, dev_latent)
    assert_cache_alignment(confirm_output, confirm_latent)

    y_dev = target(dev_output["labels"])
    y_confirm = target(confirm_output["labels"])
    train_idx, calib_idx = split_indices(len(dev_output["labels"]))

    worker_rows = []
    latent_dev_scores = []
    latent_confirm_scores = []
    latent_thresholds = []
    for worker in range(dev_latent["latents"].shape[0]):
        existing_dev = torch.sigmoid(dev_output["uncertainty_logits"][worker])
        existing_confirm = torch.sigmoid(confirm_output["uncertainty_logits"][worker])
        existing_threshold = 0.5

        scalar_model = train_binary_logistic(existing_dev.unsqueeze(1), y_dev, train_idx)
        scalar_dev = apply_binary_logistic(existing_dev.unsqueeze(1), scalar_model)
        scalar_confirm = apply_binary_logistic(existing_confirm.unsqueeze(1), scalar_model)
        scalar_threshold = select_threshold(scalar_dev, y_dev, calib_idx)

        latent_model = train_binary_logistic(dev_latent["latents"][worker], y_dev, train_idx)
        latent_dev = apply_binary_logistic(dev_latent["latents"][worker], latent_model)
        latent_confirm = apply_binary_logistic(confirm_latent["latents"][worker], latent_model)
        latent_threshold = select_threshold(latent_dev, y_dev, calib_idx)

        latent_dev_scores.append(latent_dev)
        latent_confirm_scores.append(latent_confirm)
        latent_thresholds.append(latent_threshold)
        worker_rows.append(
            {
                "worker_index": worker,
                "checkpoint": dev_latent["checkpoints"][worker],
                "existing_uncertainty_head": {
                    "development": evaluate_scores(existing_dev, y_dev, existing_threshold),
                    "consumed_confirmation_1": evaluate_scores(existing_confirm, y_confirm, existing_threshold),
                    "task_breakdown_development": group_breakdown(existing_dev, y_dev, dev_output["tasks"], existing_threshold),
                    "difficulty_breakdown_development": group_breakdown(existing_dev, y_dev, dev_output["difficulties"], existing_threshold),
                },
                "calibrated_uncertainty_scalar": {
                    "selected_threshold": scalar_threshold,
                    "development": evaluate_scores(scalar_dev, y_dev, scalar_threshold),
                    "consumed_confirmation_1": evaluate_scores(scalar_confirm, y_confirm, scalar_threshold),
                },
                "latent_linear_probe": {
                    "selected_threshold": latent_threshold,
                    "development": evaluate_scores(latent_dev, y_dev, latent_threshold),
                    "consumed_confirmation_1": evaluate_scores(latent_confirm, y_confirm, latent_threshold),
                },
            }
        )

    latent_dev_matrix = torch.stack(latent_dev_scores)
    latent_confirm_matrix = torch.stack(latent_confirm_scores)
    latent_summaries_dev = summarize_worker_local_scalar_scores(
        latent_dev_matrix,
        torch.tensor(latent_thresholds),
    )
    latent_summaries_confirm = summarize_worker_local_scalar_scores(
        latent_confirm_matrix,
        torch.tensor(latent_thresholds),
    )
    population_dev_scores = {
        "latent_mean_probability": latent_summaries_dev["mean_probability"],
        "latent_max_probability": latent_summaries_dev["max_probability"],
        "latent_fraction_above_worker_threshold": latent_summaries_dev["fraction_above_worker_threshold"],
        "mean_worker_uncertainty": torch.sigmoid(dev_output["uncertainty_logits"]).mean(dim=0),
    }
    population_confirm_scores = {
        "latent_mean_probability": latent_summaries_confirm["mean_probability"],
        "latent_max_probability": latent_summaries_confirm["max_probability"],
        "latent_fraction_above_worker_threshold": latent_summaries_confirm["fraction_above_worker_threshold"],
        "mean_worker_uncertainty": torch.sigmoid(confirm_output["uncertainty_logits"]).mean(dim=0),
    }

    dev_candidates = base_objects(dev_output)["candidates"]
    confirm_candidates = base_objects(confirm_output)["candidates"]
    population_rows = []
    for score_name, dev_scores in population_dev_scores.items():
        threshold = select_threshold(dev_scores, y_dev, calib_idx)
        for source in LABEL_CANDIDATES:
            population_rows.append(
                {
                    "score": score_name,
                    "candidate_source": source,
                    "selected_threshold": threshold,
                    "development": evaluate_candidate_predictions(
                        dev_output["labels"],
                        dev_candidates[source],
                        dev_scores,
                        threshold,
                    ),
                    "consumed_confirmation_1": evaluate_candidate_predictions(
                        confirm_output["labels"],
                        confirm_candidates[source],
                        population_confirm_scores[score_name],
                        threshold,
                    ),
                }
            )

    dev_step2a5_scores, confirm_step2a5_scores = step2a5_output_feature_scores(dev_output, confirm_output)
    false_abstention_separation = {
        "development": category_distributions(
            {
                "mean_worker_uncertainty": population_dev_scores["mean_worker_uncertainty"],
                "latent_mean_probability": population_dev_scores["latent_mean_probability"],
                "step2a5_output_feature_model": dev_step2a5_scores,
            },
            current_reducer_categories(dev_output),
        ),
        "consumed_confirmation_1": category_distributions(
            {
                "mean_worker_uncertainty": population_confirm_scores["mean_worker_uncertainty"],
                "latent_mean_probability": population_confirm_scores["latent_mean_probability"],
                "step2a5_output_feature_model": confirm_step2a5_scores,
            },
            current_reducer_categories(confirm_output),
        ),
    }

    mean_existing_auc = sum(row["existing_uncertainty_head"]["consumed_confirmation_1"]["roc_auc"] for row in worker_rows) / len(worker_rows)
    mean_latent_auc = sum(row["latent_linear_probe"]["consumed_confirmation_1"]["roc_auc"] for row in worker_rows) / len(worker_rows)
    best_latent_population = max(population_rows, key=lambda row: row["development"]["overall_accuracy"])
    best_confirm_acc = best_latent_population["consumed_confirmation_1"]["overall_accuracy"]
    current_mean_prob_acc = 0.95265
    if mean_latent_auc >= mean_existing_auc + 0.02 and best_confirm_acc > current_mean_prob_acc:
        verdict = "LATENT_SIGNAL_PRESENT"
    elif mean_latent_auc >= mean_existing_auc + 0.005:
        verdict = "OUTPUT_HEAD_CALIBRATION_LIMITED"
    else:
        verdict = "LATENT_SIGNAL_NOT_FOUND"

    summary = {
        "phase": "step2a6_latent_ambiguity_signal_audit",
        "frozen_test_accessed": False,
        "dev_confirmation_2_generated": False,
        "workers_trained": False,
        "full_population_scaling_started": False,
        "git_revision": git_revision(),
        "checkpoint_paths": dev_latent["checkpoints"],
        "latent_contract": {
            "path": "features -> input_projection -> position_embedding add -> TransformerEncoder -> final_norm -> masked mean pooling -> shared pooled vector -> label_head and uncertainty_head",
            "dimensionality": int(dev_latent["latents"].shape[-1]),
            "both_heads_receive_same_pooled_representation": True,
            "label_head": "single Linear(d_model, 11) applied directly to pooled vector",
            "uncertainty_head": "single Linear(d_model, 1) applied directly to pooled vector then squeezed",
            "raw_cross_worker_latent_averaging": False,
        },
        "development_split_protocol": {
            "train": "index % 5 != 4",
            "calibration": "index % 5 == 4",
            "selection_uses_consumed_confirmation": False,
        },
        "worker_local_results": worker_rows,
        "population_level_abstention_diagnostic": population_rows,
        "false_abstention_separation": false_abstention_separation,
        "mean_existing_uncertainty_confirm_roc_auc": mean_existing_auc,
        "mean_latent_probe_confirm_roc_auc": mean_latent_auc,
        "best_population_diagnostic_by_development_accuracy": best_latent_population,
        "secondary_candidate_correctness_diagnostic": None,
        "verdict": verdict,
        "recommended_next_step": (
            "Investigate a minimal richer uncertainty/evidence output contract without freezing a production reducer."
            if verdict != "LATENT_SIGNAL_NOT_FOUND"
            else "Treat current 50K workers as not exposing easy latent ambiguity signal; consider objectives only after finishing planned diagnostics."
        ),
    }
    dump(ROOT / "summary.json", summary)
    dump(ROOT / "worker_local_results.json", worker_rows)
    dump(ROOT / "population_abstention_diagnostic.json", population_rows)
    print(json.dumps({"event": "step2a6_complete", "verdict": verdict, "summary": str(ROOT / "summary.json")}, sort_keys=True))


if __name__ == "__main__":
    main()
