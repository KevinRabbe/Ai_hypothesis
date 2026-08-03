"""Development-only zero-false-positive calibration for the Gate9D router."""
from __future__ import annotations

import importlib.util
import json
import pathlib
import sys
from typing import Any

import torch
from torch import Tensor

_ROOT = pathlib.Path(__file__).resolve().parent
_V1_PATH = _ROOT / "gate9d_learned_shared_router_hard_negative_v1.py"

VERSION = "gate9d-learned-shared-router-zero-fp-v2"
STATUS = "DEVELOPMENT_ONLY_SUPERVISED_ROUTING_ZERO_FP"
BRANCH = "agent/gate9d-learned-shared-router-zero-fp-v2"
BASE_HEAD = "5e89fb42d6a84e32f163d3309abbb2294206f9a1"
PASS = "G9D_LEARNED_SHARED_ROUTER_ZERO_FP_PASSES"
FAIL_NOT_SEPARABLE = "G9D_LEARNED_SHARED_ROUTER_ZERO_FP_NOT_SEPARABLE"
FAIL_EXECUTION = "G9D_LEARNED_SHARED_ROUTER_ZERO_FP_EXECUTION_FAILED"


def _load_v1():
    name = "gate9d_router_zero_fp_v1_dependency"
    existing = sys.modules.get(name)
    if existing is not None:
        return existing
    spec = importlib.util.spec_from_file_location(name, _V1_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load hard-negative router dependency")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


v1 = _load_v1()
v0 = v1.v0
SharedRouter = v0.SharedRouter


def calibrate_thresholds(model: SharedRouter, device: torch.device) -> dict[str, Any]:
    worker, query, targets = v0.exhaustive_router_domain(device)
    with torch.no_grad():
        logits = model(worker, query)
    result: dict[str, Any] = {"separable": True, "gates": {}}
    for gate_index, gate_name in enumerate(("bias", "contribution")):
        positive = logits[targets[:, gate_index].bool(), gate_index]
        negative = logits[~targets[:, gate_index].bool(), gate_index]
        min_positive = float(positive.min().cpu())
        max_negative = float(negative.max().cpu())
        margin = min_positive - max_negative
        separable = margin > 0.0
        threshold = (min_positive + max_negative) / 2.0 if separable else None
        result["gates"][gate_name] = {
            "min_positive_logit": min_positive,
            "max_negative_logit": max_negative,
            "margin": margin,
            "threshold": threshold,
            "separable": separable,
            "positive_count": int(positive.numel()),
            "negative_count": int(negative.numel()),
        }
        result["separable"] = result["separable"] and separable
    return result


def threshold_predictions(logits: Tensor, calibration: dict[str, Any]) -> Tensor:
    if not calibration["separable"]:
        raise RuntimeError("router gates are not strictly separable")
    thresholds = torch.tensor(
        [
            calibration["gates"]["bias"]["threshold"],
            calibration["gates"]["contribution"]["threshold"],
        ],
        dtype=logits.dtype,
        device=logits.device,
    )
    return logits >= thresholds


def calibrated_class_metrics(
    model: SharedRouter, device: torch.device, calibration: dict[str, Any]
) -> dict[str, float]:
    result: dict[str, float] = {}
    with torch.no_grad():
        for name, (worker, query, targets) in v1.routing_strata(device).items():
            predictions = threshold_predictions(model(worker, query), calibration)
            expected = targets.bool()
            result[f"{name}_joint_accuracy"] = float(
                torch.all(predictions == expected, dim=1).float().mean().cpu()
            )
            result[f"{name}_bias_accuracy"] = float(
                (predictions[:, 0] == expected[:, 0]).float().mean().cpu()
            )
            result[f"{name}_contribution_accuracy"] = float(
                (predictions[:, 1] == expected[:, 1]).float().mean().cpu()
            )
    return result


def calibrated_execute(
    model: SharedRouter,
    calibration: dict[str, Any],
    worker_inputs: Tensor,
    worker_outputs: Tensor,
    query: Tensor,
) -> tuple[Tensor, dict[str, Any]]:
    batch, population = worker_inputs.shape
    flat_inputs = worker_inputs.reshape(-1)
    flat_queries = query.unsqueeze(1).expand(-1, population).reshape(-1)
    with torch.no_grad():
        gates = threshold_predictions(
            model(flat_inputs, flat_queries), calibration
        ).reshape(batch, population, 2)
    bias_gate = gates[:, :, 0]
    contribution_gate = gates[:, :, 1]
    bias_count = bias_gate.sum(dim=1)
    route_valid = bias_count == 1
    safe_slot = torch.argmax(bias_gate.long(), dim=1)
    bias_bytes = worker_outputs.gather(1, safe_slot.unsqueeze(1)).squeeze(1)
    bias_bits = v0.sparse.byte_bits(bias_bytes)
    output_bits = v0.sparse.byte_bits(worker_outputs)
    deltas = torch.bitwise_xor(output_bits, bias_bits.unsqueeze(1))
    parity = torch.remainder(
        torch.sum(deltas * contribution_gate.unsqueeze(-1).long(), dim=1), 2
    )
    predictions = v0.sparse.decode_bits(torch.bitwise_xor(parity, bias_bits))
    predictions = torch.where(route_valid, predictions, torch.full_like(predictions, -1))
    return predictions, {
        "valid_bias_routes": int(route_valid.sum()),
        "bias_messages": int(bias_gate.sum()),
        "contribution_messages": int(contribution_gate.sum()),
        "active_worker_count": int(torch.any(bias_gate | contribution_gate, dim=0).sum()),
    }


def run(output_root: pathlib.Path, execution_head: str) -> dict[str, Any]:
    if output_root.exists():
        raise FileExistsError(f"output exists: {output_root}")
    device = torch.device("cuda", 0) if torch.cuda.is_available() else torch.device("cpu")
    support_inputs, support_outputs, queries, targets, counters = v0.materialize_population(device)
    output_root.mkdir(parents=True)
    final_rows: list[dict[str, Any]] = []
    curves_all: list[dict[str, Any]] = []
    calibrations: list[dict[str, Any]] = []
    any_not_separable = False

    expected_messages = int(
        targets.numel() + sum(int(value).bit_count() for value in queries.cpu().tolist())
    )
    for seed_index, seed in enumerate(v0.SEEDS):
        model, curves = v1.train_router(seed, device)
        curves_all.extend({"seed_index": seed_index, **row} for row in curves)
        calibration = calibrate_thresholds(model, device)
        calibrations.append({"seed_index": seed_index, **calibration})
        if not calibration["separable"]:
            any_not_separable = True
            continue
        routing = calibrated_class_metrics(model, device, calibration)
        for population_size in v0.POPULATION_SIZES:
            inputs, outputs = v0.sparse.augment_population(
                support_inputs, support_outputs, counters, population_size
            )
            predictions, stats = calibrated_execute(
                model, calibration, inputs, outputs, queries
            )
            permutation = v0.sparse.deterministic_permutation(population_size).to(device)
            permuted_predictions, _ = calibrated_execute(
                model, calibration, inputs[:, permutation], outputs[:, permutation], queries
            )
            shuffled_outputs = torch.roll(support_outputs, shifts=247, dims=0)
            _, shuffled_augmented = v0.sparse.augment_population(
                support_inputs, shuffled_outputs, counters, population_size
            )
            shuffled_predictions, _ = calibrated_execute(
                model, calibration, inputs, shuffled_augmented, queries
            )
            full = v0.sparse.metrics(predictions, targets)
            permuted = v0.sparse.metrics(permuted_predictions, targets)
            shuffled = v0.sparse.metrics(shuffled_predictions, targets)
            observed_messages = stats["bias_messages"] + stats["contribution_messages"]
            final_rows.append({
                "seed_index": seed_index,
                "initialization_seed": seed,
                "population_size": population_size,
                "parameter_count": v0._parameter_count(model),
                **routing,
                "full_exact_accuracy": full["exact_accuracy"],
                "full_bit_accuracy": full["bit_accuracy"],
                "permuted_exact_accuracy": permuted["exact_accuracy"],
                "shuffled_exact_accuracy": shuffled["exact_accuracy"],
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
    diagnosis = PASS if passes else (
        FAIL_NOT_SEPARABLE if any_not_separable else FAIL_EXECUTION
    )
    summary = {
        "status": "G9D_LEARNED_SHARED_ROUTER_ZERO_FP_COMPLETE_DEVELOPMENT_ONLY",
        "version": VERSION,
        "diagnosis": diagnosis,
        "execution_head": execution_head,
        "base_head": BASE_HEAD,
        "development_only": True,
        "supervised_routing_labels_used": True,
        "exhaustive_threshold_calibration_used": True,
        "parameter_count": v0._parameter_count(SharedRouter()),
        "training_steps": v1.TRAIN_STEPS,
        "population_sizes": list(v0.POPULATION_SIZES),
        "calibrations": calibrations,
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
    (output_root / "aggregate-summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    for name, rows in (("final-rows.jsonl", final_rows), ("curves.jsonl", curves_all)):
        with (output_root / name).open("w", encoding="utf-8", newline="\n") as handle:
            for row in rows:
                handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
    return summary
