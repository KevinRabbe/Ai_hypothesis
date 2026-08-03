"""Development-only exhaustive-margin training for the Gate9D shared router."""
from __future__ import annotations

import importlib.util
import json
import pathlib
import sys
import time
from typing import Any

import torch
import torch.nn.functional as F

_ROOT = pathlib.Path(__file__).resolve().parent
_V2_PATH = _ROOT / "gate9d_learned_shared_router_zero_fp_v2.py"

VERSION = "gate9d-learned-shared-router-exhaustive-margin-v3"
STATUS = "DEVELOPMENT_ONLY_SUPERVISED_ROUTING_EXHAUSTIVE_MARGIN"
BRANCH = "agent/gate9d-learned-shared-router-exhaustive-margin-v3"
BASE_HEAD = "1f3ba1e73dfe13d065880237536f72ff488d64c1"
TRAIN_STEPS = 512
LEARNING_RATE = 0.003
TARGET_MARGIN = 2.0
PASS = "G9D_LEARNED_SHARED_ROUTER_EXHAUSTIVE_MARGIN_PASSES"
FAIL_NOT_SEPARABLE = "G9D_LEARNED_SHARED_ROUTER_EXHAUSTIVE_MARGIN_NOT_SEPARABLE"
FAIL_EXECUTION = "G9D_LEARNED_SHARED_ROUTER_EXHAUSTIVE_MARGIN_EXECUTION_FAILED"


def _load_v2():
    name = "gate9d_router_margin_v2_dependency"
    existing = sys.modules.get(name)
    if existing is not None:
        return existing
    spec = importlib.util.spec_from_file_location(name, _V2_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load zero-FP router dependency")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


v2 = _load_v2()
v1 = v2.v1
v0 = v2.v0
SharedRouter = v2.SharedRouter


def exhaustive_margin_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    losses = []
    for gate in range(2):
        positive = logits[targets[:, gate].bool(), gate]
        negative = logits[~targets[:, gate].bool(), gate]
        losses.append(F.softplus(TARGET_MARGIN - positive).mean())
        losses.append(F.softplus(TARGET_MARGIN + negative).mean())
    return torch.stack(losses).mean()


def worst_states(model: SharedRouter, device: torch.device) -> dict[str, Any]:
    worker, query, targets = v0.exhaustive_router_domain(device)
    with torch.no_grad():
        logits = model(worker, query)
    result: dict[str, Any] = {}
    for gate_index, gate_name in enumerate(("bias", "contribution")):
        pos_mask = targets[:, gate_index].bool()
        neg_mask = ~pos_mask
        pos_logits = logits[pos_mask, gate_index]
        neg_logits = logits[neg_mask, gate_index]
        pos_indices = torch.nonzero(pos_mask, as_tuple=False).squeeze(1)
        neg_indices = torch.nonzero(neg_mask, as_tuple=False).squeeze(1)
        pos_local = torch.argmin(pos_logits)
        neg_local = torch.argmax(neg_logits)
        pos_index = pos_indices[pos_local]
        neg_index = neg_indices[neg_local]
        result[gate_name] = {
            "min_positive_logit": float(pos_logits[pos_local].cpu()),
            "min_positive_worker": int(worker[pos_index].cpu()),
            "min_positive_query": int(query[pos_index].cpu()),
            "max_negative_logit": float(neg_logits[neg_local].cpu()),
            "max_negative_worker": int(worker[neg_index].cpu()),
            "max_negative_query": int(query[neg_index].cpu()),
            "margin": float((pos_logits[pos_local] - neg_logits[neg_local]).cpu()),
        }
    return result


def train_router(seed: int, device: torch.device) -> tuple[SharedRouter, list[dict[str, Any]]]:
    v0._configure(seed)
    model = SharedRouter().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.0)
    worker, query, targets = v0.exhaustive_router_domain(device)
    checkpoints = {1, 16, 64, 128, 256, 512}
    curves: list[dict[str, Any]] = []
    started = time.perf_counter()
    for step in range(1, TRAIN_STEPS + 1):
        optimizer.zero_grad(set_to_none=True)
        logits = model(worker, query)
        loss = exhaustive_margin_loss(logits, targets)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step in checkpoints:
            calibration = v2.calibrate_thresholds(model, device)
            curves.append({
                "step": step,
                "loss": float(loss.detach().cpu()),
                "gradient_norm": float(grad_norm.detach().cpu()),
                "seconds": time.perf_counter() - started,
                "separable": calibration["separable"],
                "bias_margin": calibration["gates"]["bias"]["margin"],
                "contribution_margin": calibration["gates"]["contribution"]["margin"],
            })
    return model, curves


def run(output_root: pathlib.Path, execution_head: str) -> dict[str, Any]:
    if output_root.exists():
        raise FileExistsError(f"output exists: {output_root}")
    device = torch.device("cuda", 0) if torch.cuda.is_available() else torch.device("cpu")
    support_inputs, support_outputs, queries, targets, counters = v0.materialize_population(device)
    output_root.mkdir(parents=True)
    final_rows: list[dict[str, Any]] = []
    curves_all: list[dict[str, Any]] = []
    calibrations: list[dict[str, Any]] = []
    worst: list[dict[str, Any]] = []
    any_not_separable = False
    expected_messages = int(targets.numel() + sum(int(q).bit_count() for q in queries.cpu().tolist()))

    for seed_index, seed in enumerate(v0.SEEDS):
        model, curves = train_router(seed, device)
        curves_all.extend({"seed_index": seed_index, **row} for row in curves)
        calibration = v2.calibrate_thresholds(model, device)
        calibrations.append({"seed_index": seed_index, **calibration})
        worst.append({"seed_index": seed_index, **worst_states(model, device)})
        if not calibration["separable"]:
            any_not_separable = True
            continue
        routing = v2.calibrated_class_metrics(model, device, calibration)
        for population_size in v0.POPULATION_SIZES:
            inputs, outputs = v0.sparse.augment_population(support_inputs, support_outputs, counters, population_size)
            predictions, stats = v2.calibrated_execute(model, calibration, inputs, outputs, queries)
            permutation = v0.sparse.deterministic_permutation(population_size).to(device)
            permuted, _ = v2.calibrated_execute(model, calibration, inputs[:, permutation], outputs[:, permutation], queries)
            shuffled_outputs = torch.roll(support_outputs, shifts=247, dims=0)
            _, shuffled_augmented = v0.sparse.augment_population(support_inputs, shuffled_outputs, counters, population_size)
            shuffled, _ = v2.calibrated_execute(model, calibration, inputs, shuffled_augmented, queries)
            full_metrics = v0.sparse.metrics(predictions, targets)
            permuted_metrics = v0.sparse.metrics(permuted, targets)
            shuffled_metrics = v0.sparse.metrics(shuffled, targets)
            observed_messages = stats["bias_messages"] + stats["contribution_messages"]
            final_rows.append({
                "seed_index": seed_index,
                "initialization_seed": seed,
                "population_size": population_size,
                "parameter_count": v0._parameter_count(model),
                **routing,
                "full_exact_accuracy": full_metrics["exact_accuracy"],
                "full_bit_accuracy": full_metrics["bit_accuracy"],
                "permuted_exact_accuracy": permuted_metrics["exact_accuracy"],
                "shuffled_exact_accuracy": shuffled_metrics["exact_accuracy"],
                "expected_message_count": expected_messages,
                "observed_message_count": observed_messages,
                "message_count_exact": observed_messages == expected_messages,
                **stats,
            })

    passes = (
        not any_not_separable
        and len(final_rows) == len(v0.SEEDS) * len(v0.POPULATION_SIZES)
        and all(
            row["zero_joint_accuracy"] == 1.0
            and row["selected_basis_joint_accuracy"] == 1.0
            and row["unselected_basis_joint_accuracy"] == 1.0
            and row["distractor_joint_accuracy"] == 1.0
            and row["message_count_exact"]
            and row["full_exact_accuracy"] == 1.0
            and row["permuted_exact_accuracy"] == 1.0
            and row["shuffled_exact_accuracy"] <= 0.02
            for row in final_rows
        )
    )
    diagnosis = PASS if passes else (FAIL_NOT_SEPARABLE if any_not_separable else FAIL_EXECUTION)
    summary = {
        "status": "G9D_LEARNED_SHARED_ROUTER_EXHAUSTIVE_MARGIN_COMPLETE_DEVELOPMENT_ONLY",
        "version": VERSION,
        "diagnosis": diagnosis,
        "execution_head": execution_head,
        "base_head": BASE_HEAD,
        "development_only": True,
        "supervised_routing_labels_used": True,
        "exhaustive_domain_training_used": True,
        "target_margin": TARGET_MARGIN,
        "parameter_count": v0._parameter_count(SharedRouter()),
        "training_steps": TRAIN_STEPS,
        "population_sizes": list(v0.POPULATION_SIZES),
        "calibrations": calibrations,
        "worst_states": worst,
        "rows": final_rows,
        "boundaries": {
            "end_to_end_answer_loss_used": False,
            "support_output_used_by_router": False,
            "operator_identity_visible": False,
            "automatic_coordinate_discovery_claimed": False,
            "later_stage_opened": False,
            "population_confirmation_claimed": False,
            "frozen_result_modified": False,
        },
    }
    (output_root / "aggregate-summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    for name, rows in (("final-rows.jsonl", final_rows), ("curves.jsonl", curves_all)):
        with (output_root / name).open("w", encoding="utf-8", newline="\n") as handle:
            for row in rows:
                handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
    return summary
