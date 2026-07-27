"""Development-only width-256 relay diagnostic combining the two localized repairs.

Training and primary inference use parameter-free softmax-normalized population aggregation.
Training also uses the fixed auxiliary gate-selection objective from #69. The underlying learned
modules and parameter count are unchanged from corrected #64.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn.functional as F

from ai_hypothesis.population_compute.collective_relay import (
    RELAY_DIFFICULTIES,
    RelayWorld,
    generate_relay_world,
    relay_scope_thresholds,
)
from ai_hypothesis.population_compute.contract import CommunicationMode
from ai_hypothesis.population_compute.relay_model import (
    NODE_BIT_WIDTH,
    RelayPopulationConfig,
    RelayPopulationModel,
    build_relay_tensor_batch,
    decode_node_logits,
    encode_node_bits,
)

WIDTH = 256
TRAIN_BATCH_SIZE = 1
TRAINING_SEED = 903
TRAINING_SEED_BASE = 17_000_000_000
HELDOUT_SEED_BASE = 18_000_000_000
GATE_LOSS_WEIGHT = 1.0


@dataclass(frozen=True)
class DiagnosticResult:
    learned_parameter_count: int
    parameter_fingerprint: str
    steps: int
    heldout_world_count: int
    final_relay_loss: float
    final_gate_loss: float
    normalized_exact_solve_rate: float
    normalized_bit_accuracy: float
    standard_sigmoid_exact_solve_rate: float
    standard_sigmoid_bit_accuracy: float
    no_communication_exact_solve_rate: float
    no_communication_bit_accuracy: float
    hop1_gate_top1_rate: float
    hop1_gate_mean_margin: float
    hop2_model_query_gate_top1_rate: float
    hop2_model_query_gate_mean_margin: float
    hop1_shared_to_clean_next_query_mean_cosine: float
    hop1_shared_to_clean_next_query_mean_rmse: float
    mean_correct_softmax_weight: float


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--steps", type=int, default=2048)
    parser.add_argument("--heldout-world-count", type=int, default=512)
    parser.add_argument("--output", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.steps <= 0 or args.heldout_world_count <= 0:
        raise SystemExit("steps and heldout world count must be positive")
    difficulty = RELAY_DIFFICULTIES[0]
    if difficulty.name != "relay-2" or WIDTH not in relay_scope_thresholds(difficulty):
        raise RuntimeError("supervised-normalized diagnostic no longer matches relay-2 width 256")

    result = run_diagnostic(
        steps=args.steps,
        heldout_world_count=args.heldout_world_count,
        device=torch.device(args.device),
    )
    payload = {
        "diagnostic": "relay-supervised-normalized-width256-v0",
        "architecture": "corrected-relay+training-gate-supervision+softmax-normalized-aggregation",
        "learned_architecture_changed": False,
        "gate_loss_weight": GATE_LOSS_WEIGHT,
        "result": asdict(result),
        "confirmation_opened": False,
        "interpretation_frozen_before_result": {
            "strong_width256_solve": "credit assignment plus population-normalized transport jointly remove the observed width-256 failure",
            "strong_gates_and_query_but_poor_solve": "pooled recurrent state or readout remains independently limiting",
            "gates_or_query_degrade": "gate supervision and normalized transport interact poorly during optimization",
        },
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def run_diagnostic(*, steps: int, heldout_world_count: int, device: torch.device) -> DiagnosticResult:
    torch.manual_seed(TRAINING_SEED)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(TRAINING_SEED)
    model = RelayPopulationModel(RelayPopulationConfig(state_width=64, message_width=24)).to(device)
    parameter_count = model.trainable_parameter_count()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    final_relay_loss = 0.0
    final_gate_loss = 0.0

    model.train()
    train_base = TRAINING_SEED_BASE + WIDTH * 10_000_000
    for step in range(steps):
        worlds = _threshold_worlds(
            base_seed=train_base,
            offset=step * TRAIN_BATCH_SIZE,
            count=TRAIN_BATCH_SIZE,
        )
        batch = build_relay_tensor_batch(worlds, active_workers=WIDTH, device=device)
        logits, _, _ = _normalized_forward(model, batch)
        targets = batch.target_bits.gt(0).to(dtype=logits.dtype)
        relay_loss = F.binary_cross_entropy_with_logits(logits, targets)
        gate_loss = _gate_training_loss(model, batch, worlds, device=device)
        total_loss = relay_loss + GATE_LOSS_WEIGHT * gate_loss
        if not torch.isfinite(total_loss):
            raise RuntimeError("supervised-normalized training produced non-finite loss")
        optimizer.zero_grad(set_to_none=True)
        total_loss.backward()
        optimizer.step()
        final_relay_loss = float(relay_loss.detach().item())
        final_gate_loss = float(gate_loss.detach().item())

    fingerprint = model.parameter_fingerprint()
    if model.trainable_parameter_count() != parameter_count:
        raise RuntimeError("learned parameter count changed during combined diagnostic")

    heldout = _threshold_worlds(
        base_seed=HELDOUT_SEED_BASE + WIDTH * 10_000_000,
        offset=0,
        count=heldout_world_count,
    )
    metrics = _evaluate(model, heldout, device=device)
    if model.parameter_fingerprint() != fingerprint:
        raise RuntimeError("held-out evaluation mutated combined checkpoint")

    return DiagnosticResult(
        learned_parameter_count=parameter_count,
        parameter_fingerprint=fingerprint,
        steps=steps,
        heldout_world_count=heldout_world_count,
        final_relay_loss=final_relay_loss,
        final_gate_loss=final_gate_loss,
        **metrics,
    )


def _normalized_forward(model, batch):
    local = batch.local_inputs[:, :WIDTH, :]
    batch_size = local.shape[0]
    flat_local = local.reshape(batch_size * WIDTH, -1)
    initial = torch.tanh(model.cell.input_projection(flat_local))
    content = model.query_projection(local[..., NODE_BIT_WIDTH:])
    shared = torch.tanh(model.query_projection(batch.start_bits))
    states = initial
    first_shared = None
    first_weights = None

    for round_index in range(2):
        shared_flat = (
            shared.unsqueeze(1)
            .expand(batch_size, WIDTH, model.config.message_width)
            .reshape(batch_size * WIDTH, model.config.message_width)
        )
        states = model.cell.update(torch.cat((flat_local, shared_flat), dim=-1), initial)
        gate_logits = model.cell.message_gate(states).reshape(batch_size, WIDTH)
        weights = torch.softmax(gate_logits, dim=1)
        shared = torch.tanh((content * weights.unsqueeze(-1)).sum(dim=1))
        if round_index == 0:
            first_shared = shared
            first_weights = weights

    pooled = states.reshape(batch_size, WIDTH, model.config.state_width).mean(dim=1)
    logits = model.cell.output_head(model.cell.output_norm(torch.cat((pooled, shared), dim=-1)))
    assert first_shared is not None and first_weights is not None
    return logits, first_shared, first_weights


def _gate_training_loss(model, batch, worlds, *, device: torch.device):
    local = batch.local_inputs[:, :WIDTH, :]
    batch_size = local.shape[0]
    flat_local = local.reshape(batch_size * WIDTH, -1)
    initial = torch.tanh(model.cell.input_projection(flat_local))
    hop1_slots, hop2_slots, intermediate = _chain_targets(worlds, device=device)

    start = torch.tanh(model.query_projection(batch.start_bits))
    hop1_states = _update(model, flat_local, initial, start, batch_size)
    hop1_logits = model.cell.message_gate(hop1_states).reshape(batch_size, WIDTH)

    clean = torch.tanh(model.query_projection(encode_node_bits(intermediate)))
    hop2_states = _update(model, flat_local, initial, clean, batch_size)
    hop2_logits = model.cell.message_gate(hop2_states).reshape(batch_size, WIDTH)
    return 0.5 * (
        F.cross_entropy(hop1_logits, hop1_slots)
        + F.cross_entropy(hop2_logits, hop2_slots)
    )


def _evaluate(model, worlds, *, device: torch.device) -> dict[str, float]:
    model.eval()
    batch = build_relay_tensor_batch(worlds, active_workers=WIDTH, device=device)
    hop1_slots, hop2_slots, intermediate = _chain_targets(worlds, device=device)
    with torch.inference_mode():
        normalized_logits, first_shared, first_weights = _normalized_forward(model, batch)
        normalized_predictions = decode_node_logits(normalized_logits)
        normalized_exact = float(
            (normalized_predictions == batch.answer_keys).float().mean().item()
        )
        normalized_bits = float(
            ((normalized_logits >= 0) == (batch.target_bits > 0)).float().mean().item()
        )

        standard = model(
            batch,
            communication_mode=CommunicationMode.SPARSE_SHARED_V0,
            recurrent_rounds=2,
        )
        standard_predictions = decode_node_logits(standard.logits)
        standard_exact = float(
            (standard_predictions == batch.answer_keys).float().mean().item()
        )
        standard_bits = float(
            ((standard.logits >= 0) == (batch.target_bits > 0)).float().mean().item()
        )

        no_comm = model(
            batch,
            communication_mode=CommunicationMode.NO_COMMUNICATION,
            recurrent_rounds=2,
        )
        no_comm_predictions = decode_node_logits(no_comm.logits)
        no_comm_exact = float(
            (no_comm_predictions == batch.answer_keys).float().mean().item()
        )
        no_comm_bits = float(
            ((no_comm.logits >= 0) == (batch.target_bits > 0)).float().mean().item()
        )

        local = batch.local_inputs[:, :WIDTH, :]
        batch_size = local.shape[0]
        flat_local = local.reshape(batch_size * WIDTH, -1)
        initial = torch.tanh(model.cell.input_projection(flat_local))
        start = torch.tanh(model.query_projection(batch.start_bits))
        hop1_states = _update(model, flat_local, initial, start, batch_size)
        hop1_logits = model.cell.message_gate(hop1_states).reshape(batch_size, WIDTH)
        clean_next = torch.tanh(model.query_projection(encode_node_bits(intermediate)))
        hop2_states = _update(model, flat_local, initial, first_shared, batch_size)
        hop2_logits = model.cell.message_gate(hop2_states).reshape(batch_size, WIDTH)

        hop1_correct = hop1_logits.gather(1, hop1_slots.unsqueeze(1)).squeeze(1)
        hop1_best_other = _best_other(hop1_logits, hop1_slots)
        hop2_correct = hop2_logits.gather(1, hop2_slots.unsqueeze(1)).squeeze(1)
        hop2_best_other = _best_other(hop2_logits, hop2_slots)
        hop1_rank = 1 + (hop1_logits > hop1_correct.unsqueeze(1)).sum(dim=1)
        hop2_rank = 1 + (hop2_logits > hop2_correct.unsqueeze(1)).sum(dim=1)
        cosine = F.cosine_similarity(first_shared, clean_next, dim=-1)
        rmse = (first_shared - clean_next).square().mean(dim=-1).sqrt()
        correct_weight = first_weights.gather(1, hop1_slots.unsqueeze(1)).squeeze(1)

    return {
        "normalized_exact_solve_rate": normalized_exact,
        "normalized_bit_accuracy": normalized_bits,
        "standard_sigmoid_exact_solve_rate": standard_exact,
        "standard_sigmoid_bit_accuracy": standard_bits,
        "no_communication_exact_solve_rate": no_comm_exact,
        "no_communication_bit_accuracy": no_comm_bits,
        "hop1_gate_top1_rate": float((hop1_rank == 1).float().mean().item()),
        "hop1_gate_mean_margin": float((hop1_correct - hop1_best_other).mean().item()),
        "hop2_model_query_gate_top1_rate": float((hop2_rank == 1).float().mean().item()),
        "hop2_model_query_gate_mean_margin": float((hop2_correct - hop2_best_other).mean().item()),
        "hop1_shared_to_clean_next_query_mean_cosine": float(cosine.mean().item()),
        "hop1_shared_to_clean_next_query_mean_rmse": float(rmse.mean().item()),
        "mean_correct_softmax_weight": float(correct_weight.mean().item()),
    }


def _update(model, flat_local, initial, shared, batch_size: int):
    shared_flat = (
        shared.unsqueeze(1)
        .expand(batch_size, WIDTH, model.config.message_width)
        .reshape(batch_size * WIDTH, model.config.message_width)
    )
    return model.cell.update(torch.cat((flat_local, shared_flat), dim=-1), initial)


def _best_other(logits, correct_slots):
    positions = torch.arange(WIDTH, device=logits.device).unsqueeze(0)
    return logits.masked_fill(
        positions == correct_slots.unsqueeze(1), float("-inf")
    ).max(dim=1).values


def _chain_targets(worlds, *, device: torch.device):
    hop1_slots: list[int] = []
    hop2_slots: list[int] = []
    intermediate: list[int] = []
    for world in worlds:
        hop1 = _chain_record(world, world.start_key)
        hop2 = _chain_record(world, hop1.value)
        if hop1.worker_slot >= WIDTH or hop2.worker_slot >= WIDTH:
            raise RuntimeError("threshold-256 world escaped active scope")
        hop1_slots.append(hop1.worker_slot)
        hop2_slots.append(hop2.worker_slot)
        intermediate.append(hop1.value)
    return (
        torch.tensor(hop1_slots, dtype=torch.int64, device=device),
        torch.tensor(hop2_slots, dtype=torch.int64, device=device),
        torch.tensor(intermediate, dtype=torch.int64, device=device),
    )


def _chain_record(world: RelayWorld, key: int):
    matches = tuple(
        record for record in world.records if record.is_chain_edge and record.key == key
    )
    if len(matches) != 1:
        raise RuntimeError("expected one chain record for query")
    return matches[0]


def _threshold_worlds(*, base_seed: int, offset: int, count: int):
    difficulty = RELAY_DIFFICULTIES[0]
    thresholds = relay_scope_thresholds(difficulty)
    selected: list[RelayWorld] = []
    seed = base_seed + offset * len(thresholds)
    while len(selected) < count:
        if thresholds[seed % len(thresholds)] == WIDTH:
            world = generate_relay_world(seed, difficulty)
            if world.scope_threshold != WIDTH:
                raise RuntimeError("combined diagnostic threshold selection drifted")
            selected.append(world)
        seed += 1
    return tuple(selected)


if __name__ == "__main__":
    raise SystemExit(main())
