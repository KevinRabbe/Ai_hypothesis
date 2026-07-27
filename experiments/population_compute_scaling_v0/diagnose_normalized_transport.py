"""Development-only width-256 transport ablation after gate-supervised training.

One corrected width-256 relay-2 checkpoint is trained with #69's training-only gate objective.
The checkpoint is then evaluated unchanged with the ordinary independent-sigmoid sum and with a
parameter-free per-sample softmax normalization over the same gate logits and candidate messages.
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
class PathMetrics:
    exact_solve_rate: float
    bit_accuracy: float
    hop1_shared_to_clean_cosine: float
    hop1_shared_to_clean_rmse: float
    hop2_correct_gate_top1_rate: float
    hop2_correct_gate_mean_rank: float


@dataclass(frozen=True)
class DiagnosticResult:
    learned_parameter_count: int
    parameter_fingerprint: str
    steps: int
    heldout_world_count: int
    final_relay_loss: float
    final_gate_loss: float
    mean_correct_sigmoid_gate: float
    mean_total_nonmatch_sigmoid_gate_mass: float
    mean_nonmatch_to_correct_sigmoid_mass_ratio: float
    mean_correct_softmax_weight: float
    standard: PathMetrics
    normalized: PathMetrics


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
        raise RuntimeError("normalized-transport diagnostic no longer matches relay-2 width 256")

    result = run_diagnostic(
        steps=args.steps,
        heldout_world_count=args.heldout_world_count,
        device=torch.device(args.device),
    )
    payload = {
        "diagnostic": "relay-normalized-transport-v0",
        "architecture": "same-gate-supervised-width256-checkpoint",
        "ordinary_aggregation": "sigmoid(gate_logit) * candidate, summed over workers",
        "normalized_aggregation": "softmax(gate_logits) * candidate, summed over workers",
        "retraining_between_paths": False,
        "result": asdict(result),
        "confirmation_opened": False,
        "interpretation_frozen_before_result": {
            "normalized_query_and_solve_restore": "residual population message accumulation is the primary remaining bottleneck",
            "normalized_query_restores_but_solve_does_not": "readout is coupled to the old aggregation distribution",
            "normalized_query_does_not_restore": "candidate-message geometry or deeper communication representation remains limiting",
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
        ordinary = model(
            batch,
            communication_mode=CommunicationMode.SPARSE_SHARED_V0,
            recurrent_rounds=2,
        )
        targets = batch.target_bits.gt(0).to(dtype=ordinary.logits.dtype)
        relay_loss = F.binary_cross_entropy_with_logits(ordinary.logits, targets)
        gate_loss = _gate_training_loss(model, batch, worlds, device=device)
        loss = relay_loss + GATE_LOSS_WEIGHT * gate_loss
        if not torch.isfinite(loss):
            raise RuntimeError("normalized-transport training produced non-finite loss")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        final_relay_loss = float(relay_loss.detach().item())
        final_gate_loss = float(gate_loss.detach().item())

    fingerprint = model.parameter_fingerprint()
    if model.trainable_parameter_count() != parameter_count:
        raise RuntimeError("learned parameter count changed during diagnostic training")

    heldout = _threshold_worlds(
        base_seed=HELDOUT_SEED_BASE + WIDTH * 10_000_000,
        offset=0,
        count=heldout_world_count,
    )
    metrics = _compare_paths(model, heldout, device=device)
    if model.parameter_fingerprint() != fingerprint:
        raise RuntimeError("same-checkpoint transport evaluation mutated learned parameters")

    return DiagnosticResult(
        learned_parameter_count=parameter_count,
        parameter_fingerprint=fingerprint,
        steps=steps,
        heldout_world_count=heldout_world_count,
        final_relay_loss=final_relay_loss,
        final_gate_loss=final_gate_loss,
        **metrics,
    )


def _gate_training_loss(model, batch, worlds, *, device: torch.device):
    local, flat_local, initial = _active_state(model, batch)
    batch_size = local.shape[0]
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


def _compare_paths(model, worlds, *, device: torch.device) -> dict[str, object]:
    model.eval()
    batch = build_relay_tensor_batch(worlds, active_workers=WIDTH, device=device)
    hop1_slots, hop2_slots, intermediate = _chain_targets(worlds, device=device)

    with torch.inference_mode():
        local, flat_local, initial = _active_state(model, batch)
        batch_size = local.shape[0]
        content = model.query_projection(local[..., NODE_BIT_WIDTH:]).reshape(
            batch_size, WIDTH, model.config.message_width
        )
        clean_next = torch.tanh(model.query_projection(encode_node_bits(intermediate)))
        start = torch.tanh(model.query_projection(batch.start_bits))

        hop1_states = _update(model, flat_local, initial, start, batch_size)
        hop1_logits = model.cell.message_gate(hop1_states).reshape(batch_size, WIDTH)
        sigmoid_gates = torch.sigmoid(hop1_logits)
        softmax_weights = torch.softmax(hop1_logits, dim=1)

        correct_sigmoid = sigmoid_gates.gather(1, hop1_slots.unsqueeze(1)).squeeze(1)
        total_sigmoid = sigmoid_gates.sum(dim=1)
        nonmatch_sigmoid = total_sigmoid - correct_sigmoid
        ratio = nonmatch_sigmoid / correct_sigmoid.clamp_min(1e-12)
        correct_softmax = softmax_weights.gather(1, hop1_slots.unsqueeze(1)).squeeze(1)

        standard_shared = torch.tanh(
            (content * sigmoid_gates.unsqueeze(-1)).sum(dim=1)
        )
        normalized_shared = torch.tanh(
            (content * softmax_weights.unsqueeze(-1)).sum(dim=1)
        )

        standard = _second_hop_and_readout(
            model,
            batch,
            flat_local=flat_local,
            initial=initial,
            content=content,
            shared=standard_shared,
            clean_next=clean_next,
            hop2_slots=hop2_slots,
            normalized=False,
        )
        normalized = _second_hop_and_readout(
            model,
            batch,
            flat_local=flat_local,
            initial=initial,
            content=content,
            shared=normalized_shared,
            clean_next=clean_next,
            hop2_slots=hop2_slots,
            normalized=True,
        )

    return {
        "mean_correct_sigmoid_gate": float(correct_sigmoid.mean().item()),
        "mean_total_nonmatch_sigmoid_gate_mass": float(nonmatch_sigmoid.mean().item()),
        "mean_nonmatch_to_correct_sigmoid_mass_ratio": float(ratio.mean().item()),
        "mean_correct_softmax_weight": float(correct_softmax.mean().item()),
        "standard": standard,
        "normalized": normalized,
    }


def _second_hop_and_readout(
    model,
    batch,
    *,
    flat_local,
    initial,
    content,
    shared,
    clean_next,
    hop2_slots,
    normalized: bool,
) -> PathMetrics:
    batch_size = batch.local_inputs.shape[0]
    hop2_states = _update(model, flat_local, initial, shared, batch_size)
    hop2_logits = model.cell.message_gate(hop2_states).reshape(batch_size, WIDTH)
    if normalized:
        weights = torch.softmax(hop2_logits, dim=1)
    else:
        weights = torch.sigmoid(hop2_logits)
    final_shared = torch.tanh((content * weights.unsqueeze(-1)).sum(dim=1))

    pooled = hop2_states.reshape(batch_size, WIDTH, model.config.state_width).mean(dim=1)
    logits = model.cell.output_head(model.cell.output_norm(torch.cat((pooled, final_shared), dim=-1)))
    predictions = decode_node_logits(logits)
    exact = float((predictions == batch.answer_keys).float().mean().item())
    bit_accuracy = float(((logits >= 0) == (batch.target_bits > 0)).float().mean().item())

    correct = hop2_logits.gather(1, hop2_slots.unsqueeze(1)).squeeze(1)
    rank = 1 + (hop2_logits > correct.unsqueeze(1)).sum(dim=1)
    cosine = F.cosine_similarity(shared, clean_next, dim=-1)
    rmse = (shared - clean_next).square().mean(dim=-1).sqrt()
    return PathMetrics(
        exact_solve_rate=exact,
        bit_accuracy=bit_accuracy,
        hop1_shared_to_clean_cosine=float(cosine.mean().item()),
        hop1_shared_to_clean_rmse=float(rmse.mean().item()),
        hop2_correct_gate_top1_rate=float((rank == 1).float().mean().item()),
        hop2_correct_gate_mean_rank=float(rank.float().mean().item()),
    )


def _active_state(model, batch):
    local = batch.local_inputs[:, :WIDTH, :]
    batch_size = local.shape[0]
    flat_local = local.reshape(batch_size * WIDTH, -1)
    initial = torch.tanh(model.cell.input_projection(flat_local))
    return local, flat_local, initial


def _update(model, flat_local, initial, shared, batch_size: int):
    shared_flat = (
        shared.unsqueeze(1)
        .expand(batch_size, WIDTH, model.config.message_width)
        .reshape(batch_size * WIDTH, model.config.message_width)
    )
    return model.cell.update(torch.cat((flat_local, shared_flat), dim=-1), initial)


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
                raise RuntimeError("normalized-transport threshold selection drifted")
            selected.append(world)
        seed += 1
    return tuple(selected)


if __name__ == "__main__":
    raise SystemExit(main())
