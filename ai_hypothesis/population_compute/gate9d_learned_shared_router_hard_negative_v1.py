"""Development-only balanced hard-negative training for the Gate9D shared router."""
from __future__ import annotations

import importlib.util
import json
import pathlib
import sys
import time
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor

_ROOT = pathlib.Path(__file__).resolve().parent
_V0_PATH = _ROOT / "gate9d_learned_shared_router.py"

VERSION = "gate9d-learned-shared-router-hard-negative-v1"
STATUS = "DEVELOPMENT_ONLY_SUPERVISED_ROUTING_HARD_NEGATIVE"
BRANCH = "agent/gate9d-learned-shared-router-hard-negative-v1"
BASE_HEAD = "d974277db6e270433876228b7534d44456ecee3e"
TRAIN_STEPS = 1024
STRATUM_BATCH = 1024
LEARNING_RATE = 0.003
PASS = "G9D_LEARNED_SHARED_ROUTER_HARD_NEGATIVE_PASSES"
FAIL = "G9D_LEARNED_SHARED_ROUTER_HARD_NEGATIVE_FAILED"


def _load_v0():
    name = "gate9d_learned_router_hard_negative_v0_dependency"
    existing = sys.modules.get(name)
    if existing is not None:
        return existing
    spec = importlib.util.spec_from_file_location(name, _V0_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load learned-router v0 dependency")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


v0 = _load_v0()
SharedRouter = v0.SharedRouter


def routing_strata(device: torch.device) -> dict[str, tuple[Tensor, Tensor, Tensor]]:
    worker, query, targets = v0.exhaustive_router_domain(device)
    basis = v0.sparse._is_basis_value(worker)
    zero = worker == 0
    contribution = targets[:, 1].bool()
    masks = {
        "zero": zero,
        "selected_basis": basis & contribution,
        "unselected_basis": basis & ~contribution,
        "distractor": ~basis & ~zero,
    }
    strata = {name: (worker[mask], query[mask], targets[mask]) for name, mask in masks.items()}
    expected = {
        "zero": 256,
        "selected_basis": 1024,
        "unselected_basis": 1024,
        "distractor": 63232,
    }
    observed = {name: int(values[0].numel()) for name, values in strata.items()}
    if observed != expected:
        raise RuntimeError(f"routing strata drifted: {observed}")
    return strata


def _sample_balanced(
    strata: dict[str, tuple[Tensor, Tensor, Tensor]],
    generator: torch.Generator,
    device: torch.device,
) -> tuple[Tensor, Tensor, Tensor]:
    workers, queries, targets = [], [], []
    for name in ("zero", "selected_basis", "unselected_basis", "distractor"):
        worker, query, target = strata[name]
        indices = torch.randint(0, worker.numel(), (STRATUM_BATCH,), generator=generator).to(device)
        workers.append(worker[indices])
        queries.append(query[indices])
        targets.append(target[indices])
    return torch.cat(workers), torch.cat(queries), torch.cat(targets)


def class_metrics(model: SharedRouter, device: torch.device) -> dict[str, float]:
    strata = routing_strata(device)
    result: dict[str, float] = {}
    with torch.no_grad():
        for name, (worker, query, targets) in strata.items():
            predictions = model(worker, query) >= 0
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


def train_router(seed: int, device: torch.device) -> tuple[SharedRouter, list[dict[str, Any]]]:
    v0._configure(seed)
    model = SharedRouter().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.0)
    strata = routing_strata(device)
    generator = torch.Generator(device="cpu").manual_seed(seed + 101)
    checkpoints = {1, 16, 64, 128, 256, 512, 1024}
    curves: list[dict[str, Any]] = []
    started = time.perf_counter()
    for step in range(1, TRAIN_STEPS + 1):
        worker, query, targets = _sample_balanced(strata, generator, device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(worker, query)
        loss = F.binary_cross_entropy_with_logits(logits, targets)
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step in checkpoints:
            metrics = class_metrics(model, device)
            curves.append({
                "step": step,
                "loss": float(loss.detach().cpu()),
                "gradient_norm": float(gradient_norm.detach().cpu()),
                "seconds": time.perf_counter() - started,
                **metrics,
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
    for seed_index, seed in enumerate(v0.SEEDS):
        model, curves = train_router(seed, device)
        for curve in curves:
            curves_all.append({"seed_index": seed_index, **curve})
        routing = class_metrics(model, device)
        for population_size in v0.POPULATION_SIZES:
            inputs, outputs = v0.sparse.augment_population(
                support_inputs, support_outputs, counters, population_size
            )
            predictions, message_stats = v0.learned_execute(model, inputs, outputs, queries)
            permutation = v0.sparse.deterministic_permutation(population_size).to(device)
            permuted_predictions, _ = v0.learned_execute(
                model, inputs[:, permutation], outputs[:, permutation], queries
            )
            shuffled_outputs = torch.roll(support_outputs, shifts=247, dims=0)
            _, shuffled_augmented = v0.sparse.augment_population(
                support_inputs, shuffled_outputs, counters, population_size
            )
            shuffled_predictions, _ = v0.learned_execute(
                model, inputs, shuffled_augmented, queries
            )
            full = v0.sparse.metrics(predictions, targets)
            permuted = v0.sparse.metrics(permuted_predictions, targets)
            shuffled = v0.sparse.metrics(shuffled_predictions, targets)
            expected_messages = int(targets.numel() + sum(int(q).bit_count() for q in queries.cpu().tolist()))
            observed_messages = message_stats["bias_messages"] + message_stats["contribution_messages"]
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
                **message_stats,
            })
    passes = all(
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
    summary = {
        "status": "G9D_LEARNED_SHARED_ROUTER_HARD_NEGATIVE_COMPLETE_DEVELOPMENT_ONLY",
        "version": VERSION,
        "diagnosis": PASS if passes else FAIL,
        "execution_head": execution_head,
        "base_head": BASE_HEAD,
        "development_only": True,
        "supervised_routing_labels_used": True,
        "balanced_hard_negative_curriculum": True,
        "parameter_count": v0._parameter_count(SharedRouter()),
        "training_steps": TRAIN_STEPS,
        "stratum_batch_size": STRATUM_BATCH,
        "population_sizes": list(v0.POPULATION_SIZES),
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
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n"
    )
    with (output_root / "final-rows.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
        for row in final_rows:
            handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
    with (output_root / "curves.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
        for row in curves_all:
            handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
    return summary
