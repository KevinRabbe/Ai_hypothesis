"""Development-only supervised shared routing for the sparse affine population.

A single small router is reused by every worker. It sees only the worker's local
support input and the query, and predicts two gates: bias broadcast and basis
contribution. Message payloads and XOR aggregation remain fixed. This tests
learned routing, not automatic coordinate discovery.
"""
from __future__ import annotations

import importlib.util
import json
import pathlib
import sys
import time
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn

_ROOT = pathlib.Path(__file__).resolve().parent
_SPARSE_PATH = _ROOT / "gate9d_sparse_affine_worker_population.py"

VERSION = "gate9d-learned-shared-router-v0"
STATUS = "DEVELOPMENT_ONLY_SUPERVISED_ROUTING"
BRANCH = "agent/gate9d-learned-shared-router-v0"
BASE_HEAD = "6ad02bd4f0907bafa6a1d202eb157d701e26cbe8"
TRAIN_STEPS = 1024
BATCH_SIZE = 4096
HIDDEN_WIDTH = 64
LEARNING_RATE = 0.003
SEEDS = (920900, 920901, 920902)
COUNTER_START = (1 << 57) + 0x3000
OPERATOR_COUNT = 64
POPULATION_SIZES = (9, 16, 64, 256)
PASS = "G9D_LEARNED_SHARED_ROUTER_PASSES"
FAIL = "G9D_LEARNED_SHARED_ROUTER_FAILED"


def _load_sparse():
    name = "gate9d_learned_router_sparse_dependency"
    existing = sys.modules.get(name)
    if existing is not None:
        return existing
    spec = importlib.util.spec_from_file_location(name, _SPARSE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load sparse population dependency")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


sparse = _load_sparse()


class SharedRouter(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.hidden = nn.Linear(16, HIDDEN_WIDTH)
        self.output = nn.Linear(HIDDEN_WIDTH, 2)

    def forward(self, worker_input: Tensor, query: Tensor) -> Tensor:
        features = torch.cat(
            (sparse.byte_bits(worker_input).to(torch.float32),
             sparse.byte_bits(query).to(torch.float32)),
            dim=-1,
        )
        return self.output(torch.relu(self.hidden(features)))


def routing_targets(worker_input: Tensor, query: Tensor) -> Tensor:
    if worker_input.shape != query.shape or worker_input.ndim != 1:
        raise ValueError("router inputs must be matching vectors")
    bias = worker_input == 0
    basis = sparse._is_basis_value(worker_input)
    selected = torch.zeros_like(basis)
    if bool(torch.any(basis)):
        basis_values = worker_input[basis]
        indices = sparse._basis_index(basis_values)
        query_bits = sparse.byte_bits(query[basis])
        selected[basis] = query_bits.gather(1, indices.unsqueeze(1)).squeeze(1).bool()
    return torch.stack((bias, selected), dim=1).to(torch.float32)


def exhaustive_router_domain(device: torch.device) -> tuple[Tensor, Tensor, Tensor]:
    worker = torch.arange(256, dtype=torch.long, device=device).repeat_interleave(256)
    query = torch.arange(256, dtype=torch.long, device=device).repeat(256)
    return worker, query, routing_targets(worker, query)


def _configure(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)


def _parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def train_router(seed: int, device: torch.device) -> tuple[SharedRouter, list[dict[str, Any]]]:
    _configure(seed)
    model = SharedRouter().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.0)
    worker, query, targets = exhaustive_router_domain(device)
    positive = targets.sum(dim=0)
    negative = targets.shape[0] - positive
    pos_weight = negative / positive
    generator = torch.Generator(device="cpu").manual_seed(seed + 17)
    curves: list[dict[str, Any]] = []
    checkpoints = {1, 16, 64, 128, 256, 512, 1024}
    started = time.perf_counter()
    for step in range(1, TRAIN_STEPS + 1):
        index = torch.randint(0, worker.numel(), (BATCH_SIZE,), generator=generator)
        index = index.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(worker[index], query[index])
        loss = F.binary_cross_entropy_with_logits(
            logits, targets[index], pos_weight=pos_weight
        )
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step in checkpoints:
            with torch.no_grad():
                predictions = model(worker, query) >= 0
                expected = targets.bool()
                per_gate = (predictions == expected).float().mean(dim=0)
                exact = torch.all(predictions == expected, dim=1).float().mean()
            curves.append({
                "step": step,
                "loss": float(loss.detach().cpu()),
                "gradient_norm": float(gradient_norm.detach().cpu()),
                "bias_accuracy": float(per_gate[0].cpu()),
                "contribution_accuracy": float(per_gate[1].cpu()),
                "joint_accuracy": float(exact.cpu()),
                "seconds": time.perf_counter() - started,
            })
    return model, curves


def learned_execute(
    model: SharedRouter,
    worker_inputs: Tensor,
    worker_outputs: Tensor,
    query: Tensor,
) -> tuple[Tensor, dict[str, Any]]:
    batch, population = worker_inputs.shape
    flat_inputs = worker_inputs.reshape(-1)
    flat_queries = query.unsqueeze(1).expand(-1, population).reshape(-1)
    with torch.no_grad():
        gates = (model(flat_inputs, flat_queries) >= 0).reshape(batch, population, 2)
    bias_gate = gates[:, :, 0]
    contribution_gate = gates[:, :, 1]
    bias_count = bias_gate.sum(dim=1)
    route_valid = bias_count == 1
    safe_slot = torch.argmax(bias_gate.long(), dim=1)
    bias_bytes = worker_outputs.gather(1, safe_slot.unsqueeze(1)).squeeze(1)
    bias_bits = sparse.byte_bits(bias_bytes)
    output_bits = sparse.byte_bits(worker_outputs)
    deltas = torch.bitwise_xor(output_bits, bias_bits.unsqueeze(1))
    parity = torch.remainder(
        torch.sum(deltas * contribution_gate.unsqueeze(-1).long(), dim=1), 2
    )
    predictions = sparse.decode_bits(torch.bitwise_xor(parity, bias_bits))
    predictions = torch.where(route_valid, predictions, torch.full_like(predictions, -1))
    return predictions, {
        "valid_bias_routes": int(route_valid.sum()),
        "bias_messages": int(bias_gate.sum()),
        "contribution_messages": int(contribution_gate.sum()),
        "active_worker_count": int(torch.any(bias_gate | contribution_gate, dim=0).sum()),
    }


def materialize_population(device: torch.device):
    support_inputs, support_outputs, queries, targets, counters, dataset_sha = (
        sparse.materialize_evaluation()
    )
    keep = (counters >= COUNTER_START) & (counters < COUNTER_START + OPERATOR_COUNT)
    # The dependency materializes its own range, so create the fresh range directly.
    input_rows, output_rows, query_rows, target_rows, counter_rows = [], [], [], [], []
    for counter in range(COUNTER_START, COUNTER_START + OPERATOR_COUNT):
        operator = sparse.operators.operator_from_counter(counter)
        supports = sparse.operators.public_support_pairs(operator)
        inputs = tuple(source for source, _ in supports)
        outputs = tuple(target for _, target in supports)
        for query_value in sparse.QUERY_VALUES:
            input_rows.append(inputs)
            output_rows.append(outputs)
            query_rows.append(query_value)
            target_rows.append(operator.apply(query_value))
            counter_rows.append(counter)
    return (
        torch.tensor(input_rows, dtype=torch.long, device=device),
        torch.tensor(output_rows, dtype=torch.long, device=device),
        torch.tensor(query_rows, dtype=torch.long, device=device),
        torch.tensor(target_rows, dtype=torch.long, device=device),
        torch.tensor(counter_rows, dtype=torch.long, device=device),
    )


def run(output_root: pathlib.Path, execution_head: str) -> dict[str, Any]:
    if output_root.exists():
        raise FileExistsError(f"output exists: {output_root}")
    device = torch.device("cuda", 0) if torch.cuda.is_available() else torch.device("cpu")
    support_inputs, support_outputs, queries, targets, counters = materialize_population(device)
    output_root.mkdir(parents=True)
    final_rows: list[dict[str, Any]] = []
    curves_all: list[dict[str, Any]] = []
    for seed_index, seed in enumerate(SEEDS):
        model, curves = train_router(seed, device)
        for curve in curves:
            curves_all.append({"seed_index": seed_index, **curve})
        router_worker, router_query, router_targets = exhaustive_router_domain(device)
        with torch.no_grad():
            router_predictions = model(router_worker, router_query) >= 0
        router_joint = float(torch.all(router_predictions == router_targets.bool(), dim=1).float().mean().cpu())
        for population_size in POPULATION_SIZES:
            inputs, outputs = sparse.augment_population(
                support_inputs, support_outputs, counters, population_size
            )
            predictions, routing = learned_execute(model, inputs, outputs, queries)
            permutation = sparse.deterministic_permutation(population_size).to(device)
            permuted_predictions, _ = learned_execute(
                model, inputs[:, permutation], outputs[:, permutation], queries
            )
            shuffled_outputs = torch.roll(support_outputs, shifts=247, dims=0)
            _, shuffled_augmented = sparse.augment_population(
                support_inputs, shuffled_outputs, counters, population_size
            )
            shuffled_predictions, _ = learned_execute(
                model, inputs, shuffled_augmented, queries
            )
            full = sparse.metrics(predictions, targets)
            permuted = sparse.metrics(permuted_predictions, targets)
            shuffled = sparse.metrics(shuffled_predictions, targets)
            final_rows.append({
                "seed_index": seed_index,
                "initialization_seed": seed,
                "population_size": population_size,
                "parameter_count": _parameter_count(model),
                "router_joint_accuracy": router_joint,
                "full_exact_accuracy": full["exact_accuracy"],
                "full_bit_accuracy": full["bit_accuracy"],
                "permuted_exact_accuracy": permuted["exact_accuracy"],
                "shuffled_exact_accuracy": shuffled["exact_accuracy"],
                "messages_per_episode": (routing["bias_messages"] + routing["contribution_messages"]) / targets.numel(),
                **routing,
            })
    passes = all(
        row["router_joint_accuracy"] == 1.0
        and row["full_exact_accuracy"] == 1.0
        and row["permuted_exact_accuracy"] == 1.0
        and row["shuffled_exact_accuracy"] <= 0.02
        for row in final_rows
    )
    summary = {
        "status": "G9D_LEARNED_SHARED_ROUTER_COMPLETE_DEVELOPMENT_ONLY",
        "version": VERSION,
        "diagnosis": PASS if passes else FAIL,
        "execution_head": execution_head,
        "base_head": BASE_HEAD,
        "development_only": True,
        "supervised_routing_labels_used": True,
        "automatic_coordinate_discovery_claimed": False,
        "parameter_count": _parameter_count(SharedRouter()),
        "training_steps": TRAIN_STEPS,
        "operator_range": {"start": COUNTER_START, "count": OPERATOR_COUNT},
        "population_sizes": list(POPULATION_SIZES),
        "rows": final_rows,
        "boundaries": {
            "end_to_end_answer_loss_used": False,
            "support_output_used_by_router": False,
            "operator_identity_visible": False,
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
