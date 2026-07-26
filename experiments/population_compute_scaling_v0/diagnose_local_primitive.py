"""Development-only decomposition of the current relay neural primitive.

This script is not a benchmark gate. It asks where learning first fails:
1. random key/query equality inside the current local GRU/gate path;
2. one-hop population retrieval through the current shared aggregation/readout;
3. canonical relay-2 recurrence without mixed-difficulty interference.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from ai_hypothesis.population_compute.collective_relay import (
    RELAY_DIFFICULTIES,
    RelayDifficulty,
    generate_relay_dataset,
    generate_relay_world,
    relay_scope_thresholds,
)
from ai_hypothesis.population_compute.contract import CommunicationMode
from ai_hypothesis.population_compute.relay_experiment import evaluate_relay_condition
from ai_hypothesis.population_compute.relay_model import (
    NODE_BIT_WIDTH,
    RelayPopulationConfig,
    RelayPopulationModel,
    RelayTensorBatch,
    build_relay_tensor_batch,
    decode_node_logits,
    encode_node_bits,
)


NODE_COUNT = 1 << NODE_BIT_WIDTH
TRAINING_POPULATIONS = (4, 16, 64, 256)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--output", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    payload = {
        "diagnostic": "relay-local-primitive-v0",
        "seed": args.seed,
        "device": str(device),
        "key_match": diagnose_key_match(device),
        "one_hop": diagnose_one_hop(device),
        "relay_2_only": diagnose_relay_two(device),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def diagnose_key_match(device: torch.device) -> dict[str, object]:
    torch.manual_seed(101)
    model = RelayPopulationModel(RelayPopulationConfig(state_width=64, message_width=24)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    losses: list[float] = []
    model.train()
    for _ in range(1024):
        logits, targets = _key_match_batch(model, 512, device)
        loss = F.binary_cross_entropy_with_logits(logits, targets)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().item()))

    model.eval()
    with torch.no_grad():
        logits, targets = _key_match_batch(model, 20_000, device)
        predictions = logits >= 0
        truth = targets >= 0.5
        accuracy = float(predictions.eq(truth).float().mean().item())
        positive = truth
        negative = ~truth
        recall = float(predictions[positive].float().mean().item())
        specificity = float((~predictions[negative]).float().mean().item())
        eval_loss = float(F.binary_cross_entropy_with_logits(logits, targets).item())
    return {
        **_loss_summary(losses),
        "heldout_loss": eval_loss,
        "heldout_accuracy": accuracy,
        "heldout_match_recall": recall,
        "heldout_nonmatch_specificity": specificity,
    }


def diagnose_one_hop(device: torch.device) -> dict[str, object]:
    torch.manual_seed(202)
    model = RelayPopulationModel(RelayPopulationConfig(state_width=64, message_width=24)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    losses: list[float] = []
    model.train()
    for step in range(1024):
        active_workers = 4 if step % 2 == 0 else 16
        batch = _one_hop_batch(64, active_workers, device)
        output = model(
            batch,
            communication_mode=CommunicationMode.SPARSE_SHARED_V0,
            recurrent_rounds=1,
        )
        targets = batch.target_bits.gt(0).to(dtype=output.logits.dtype)
        loss = F.binary_cross_entropy_with_logits(output.logits, targets)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().item()))

    model.eval()
    evaluations = {}
    with torch.no_grad():
        for active_workers in (4, 16):
            batch = _one_hop_batch(4096, active_workers, device)
            by_mode = {}
            for mode in (
                CommunicationMode.NO_COMMUNICATION,
                CommunicationMode.SPARSE_SHARED_V0,
            ):
                output = model(batch, communication_mode=mode, recurrent_rounds=1)
                decoded = decode_node_logits(output.logits)
                exact = float(decoded.eq(batch.answer_keys).float().mean().item())
                bit_truth = batch.target_bits.gt(0)
                bit_accuracy = float(
                    output.logits.ge(0).eq(bit_truth).float().mean().item()
                )
                by_mode[mode.value] = {
                    "exact_solve_rate": exact,
                    "bit_accuracy": bit_accuracy,
                }
            evaluations[str(active_workers)] = by_mode
    return {
        **_loss_summary(losses),
        "heldout": evaluations,
    }


def diagnose_relay_two(device: torch.device) -> dict[str, object]:
    difficulty = RELAY_DIFFICULTIES[0]
    if difficulty.name != "relay-2":
        raise RuntimeError("expected relay-2 to be the first frozen difficulty")
    torch.manual_seed(303)
    model = RelayPopulationModel(RelayPopulationConfig(state_width=64, message_width=24)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    losses: list[float] = []
    model.train()
    for step in range(2048):
        active_workers = TRAINING_POPULATIONS[step % len(TRAINING_POPULATIONS)]
        population_round = step // len(TRAINING_POPULATIONS)
        thresholds = tuple(
            threshold
            for threshold in relay_scope_thresholds(difficulty)
            if threshold <= active_workers
        )
        threshold = thresholds[population_round % len(thresholds)]
        worlds = _relay_worlds_for_threshold(
            base_seed=5_000_000_000,
            step=step,
            batch_size=8,
            difficulty=difficulty,
            threshold=threshold,
        )
        batch = build_relay_tensor_batch(
            worlds,
            active_workers=active_workers,
            device=device,
        )
        if not bool(torch.all(batch.information_complete).item()):
            raise RuntimeError("relay-2 diagnostic training batch is not information-complete")
        output = model(
            batch,
            communication_mode=CommunicationMode.SPARSE_SHARED_V0,
            recurrent_rounds=2,
        )
        targets = batch.target_bits.gt(0).to(dtype=output.logits.dtype)
        loss = F.binary_cross_entropy_with_logits(output.logits, targets)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().item()))

    model.eval()
    worlds = generate_relay_dataset(
        start_seed=6_000_000_000,
        world_count=512,
        difficulty=difficulty,
    )
    fingerprint = model.parameter_fingerprint()
    parameter_count = model.trainable_parameter_count()
    evaluations = {}
    for population in TRAINING_POPULATIONS:
        by_mode = {}
        for mode in (
            CommunicationMode.NO_COMMUNICATION,
            CommunicationMode.SPARSE_SHARED_V0,
        ):
            metrics = evaluate_relay_condition(
                model,
                worlds,
                training_seed=303,
                benchmark_seed=6_000,
                difficulty=difficulty,
                population_size=population,
                communication_mode=mode,
                batch_size=32,
                expected_parameter_count=parameter_count,
                expected_fingerprint=fingerprint,
                device=device,
            )
            by_mode[mode.value] = {
                "solve_rate": metrics.solve_rate,
                "information_complete_rate": metrics.information_complete_rate,
                "solve_rate_given_information_complete": (
                    metrics.solve_rate_given_information_complete
                ),
                "solve_rate_given_information_incomplete": (
                    metrics.solve_rate_given_information_incomplete
                ),
            }
        evaluations[str(population)] = by_mode
    return {
        **_loss_summary(losses),
        "heldout": evaluations,
    }


def _key_match_batch(
    model: RelayPopulationModel,
    batch_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    query = torch.randint(0, NODE_COUNT, (batch_size,), device=device)
    match = torch.rand(batch_size, device=device) < 0.5
    key = torch.randint(0, NODE_COUNT, (batch_size,), device=device)
    collision = key.eq(query)
    key = torch.where(collision, (key + 1) % NODE_COUNT, key)
    key = torch.where(match, query, key)
    value = torch.randint(0, NODE_COUNT, (batch_size,), device=device)
    local = torch.cat((encode_node_bits(key), encode_node_bits(value)), dim=-1)
    state = torch.tanh(model.cell.input_projection(local))
    shared = torch.tanh(model.query_projection(encode_node_bits(query)))
    updated = model.cell.update(torch.cat((local, shared), dim=-1), state)
    logits = model.cell.message_gate(updated).squeeze(-1)
    return logits, match.to(dtype=logits.dtype)


def _one_hop_batch(
    batch_size: int,
    active_workers: int,
    device: torch.device,
) -> RelayTensorBatch:
    if active_workers not in {4, 16}:
        raise ValueError("one-hop diagnostic supports active populations 4 and 16")
    query = torch.randint(0, NODE_COUNT, (batch_size,), device=device)
    keys = torch.randint(0, NODE_COUNT, (batch_size, 256), device=device)
    query_matrix = query.unsqueeze(1).expand_as(keys)
    keys = torch.where(keys.eq(query_matrix), (keys + 1) % NODE_COUNT, keys)
    values = torch.randint(0, NODE_COUNT, (batch_size, 256), device=device)
    match_slot = torch.randint(0, active_workers, (batch_size,), device=device)
    rows = torch.arange(batch_size, device=device)
    keys[rows, match_slot] = query
    answer = values[rows, match_slot]
    active_mask = (
        torch.arange(256, device=device) < active_workers
    ).unsqueeze(0).expand(batch_size, 256).clone()
    batch = RelayTensorBatch(
        local_inputs=torch.cat((encode_node_bits(keys), encode_node_bits(values)), dim=-1),
        active_mask=active_mask,
        start_bits=encode_node_bits(query),
        target_bits=encode_node_bits(answer),
        answer_keys=answer,
        information_complete=torch.ones(batch_size, dtype=torch.bool, device=device),
        active_workers=active_workers,
        hop_count=2,
    )
    batch.validate()
    return batch


def _relay_worlds_for_threshold(
    *,
    base_seed: int,
    step: int,
    batch_size: int,
    difficulty: RelayDifficulty,
    threshold: int,
) -> tuple:
    thresholds = relay_scope_thresholds(difficulty)
    segment_size = batch_size * len(thresholds)
    start = base_seed + step * segment_size
    selected = tuple(
        seed
        for seed in range(start, start + segment_size)
        if thresholds[seed % len(thresholds)] == threshold
    )[:batch_size]
    if len(selected) != batch_size:
        raise RuntimeError("could not fill threshold-matched diagnostic batch")
    worlds = tuple(generate_relay_world(seed, difficulty) for seed in selected)
    if any(world.scope_threshold != threshold for world in worlds):
        raise RuntimeError("diagnostic threshold selection drifted")
    return worlds


def _loss_summary(losses: list[float]) -> dict[str, float]:
    if not losses or any(not math.isfinite(value) for value in losses):
        raise RuntimeError("diagnostic loss history is invalid")
    tail = losses[-min(100, len(losses)) :]
    return {
        "first_loss": losses[0],
        "best_loss": min(losses),
        "final_loss": losses[-1],
        "mean_last_100_loss": sum(tail) / len(tail),
    }


if __name__ == "__main__":
    raise SystemExit(main())
