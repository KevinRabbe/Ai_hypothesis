"""Development-only hop-level diagnostic for the corrected width-16 relay-2 failure.

This script trains exactly one corrected #64 relay-2 checkpoint at fixed width 16 using
#67's fixed-width protocol, then inspects the learned gate at each hop without changing
model parameters or opening confirmation data.
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

WIDTH = 16
TRAIN_BATCH_SIZE = 16
TRAINING_SEED = 901
TRAINING_SEED_BASE = 17_000_000_000
HELDOUT_SEED_BASE = 18_000_000_000


@dataclass(frozen=True)
class GateSummary:
    top1_rate: float
    mean_rank: float
    mean_margin: float
    median_margin: float
    mean_correct_logit: float
    mean_best_nonmatch_logit: float


@dataclass(frozen=True)
class DiagnosticResult:
    learned_parameter_count: int
    parameter_fingerprint: str
    steps: int
    training_seed: int
    train_batch_size: int
    heldout_world_count: int
    first_loss: float
    best_loss: float
    final_loss: float
    mean_last_100_loss: float
    exact_solve_rate: float
    bit_accuracy: float
    hop1_start_query_gate: GateSummary
    hop2_model_query_gate: GateSummary
    hop2_oracle_clean_query_gate: GateSummary
    hop1_shared_to_clean_next_query_mean_cosine: float
    hop1_shared_to_clean_next_query_mean_rmse: float
    hop1_shared_to_clean_next_query_mean_l2: float


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
        raise RuntimeError("width-16 gate diagnostic no longer matches frozen relay-2")
    device = torch.device(args.device)
    result = run_diagnostic(
        steps=args.steps,
        heldout_world_count=args.heldout_world_count,
        device=device,
    )
    payload = {
        "diagnostic": "relay-width16-gate-path-v0",
        "architecture": "exact-compositional-node-messages+hop-local-worker-state",
        "difficulty": difficulty.name,
        "active_workers": WIDTH,
        "device": str(device),
        "result": asdict(result),
        "confirmation_opened": False,
        "interpretation_frozen_before_result": {
            "poor_hop1_gate_ranking": "gate learning or end-to-end credit assignment failure",
            "good_hop1_and_good_oracle_hop2_but_poor_model_hop2": "first-hop query corruption",
            "poor_oracle_hop2": "gate does not generalize key/query discrimination at width 16 under relay training",
            "strong_gate_ranks_but_poor_solve": "inspect shared/readout transformation",
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
    losses: list[float] = []
    model.train()
    train_base = TRAINING_SEED_BASE + WIDTH * 10_000_000
    for step in range(steps):
        worlds = _threshold_worlds(
            base_seed=train_base,
            offset=step * TRAIN_BATCH_SIZE,
            count=TRAIN_BATCH_SIZE,
            threshold=WIDTH,
        )
        batch = build_relay_tensor_batch(worlds, active_workers=WIDTH, device=device)
        if not bool(torch.all(batch.information_complete).item()):
            raise RuntimeError("width-16 training batch is not information-complete")
        output = model(
            batch,
            communication_mode=CommunicationMode.SPARSE_SHARED_V0,
            recurrent_rounds=2,
        )
        targets = batch.target_bits.gt(0).to(dtype=output.logits.dtype)
        loss = F.binary_cross_entropy_with_logits(output.logits, targets)
        if not torch.isfinite(loss):
            raise RuntimeError("width-16 training produced non-finite loss")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().item()))
    fingerprint = model.parameter_fingerprint()
    if model.trainable_parameter_count() != parameter_count:
        raise RuntimeError("learned parameter count changed during training")
    heldout = _threshold_worlds(
        base_seed=HELDOUT_SEED_BASE + WIDTH * 10_000_000,
        offset=0,
        count=heldout_world_count,
        threshold=WIDTH,
    )
    metrics = _inspect_gate_path(model, heldout, device=device)
    if model.parameter_fingerprint() != fingerprint:
        raise RuntimeError("diagnostic inference mutated the checkpoint")
    tail = losses[-min(100, len(losses)) :]
    return DiagnosticResult(
        learned_parameter_count=parameter_count,
        parameter_fingerprint=fingerprint,
        steps=steps,
        training_seed=TRAINING_SEED,
        train_batch_size=TRAIN_BATCH_SIZE,
        heldout_world_count=heldout_world_count,
        first_loss=losses[0],
        best_loss=min(losses),
        final_loss=losses[-1],
        mean_last_100_loss=sum(tail) / len(tail),
        **metrics,
    )


def _inspect_gate_path(
    model: RelayPopulationModel,
    worlds: tuple[RelayWorld, ...],
    *,
    device: torch.device,
) -> dict[str, object]:
    model.eval()
    batch = build_relay_tensor_batch(worlds, active_workers=WIDTH, device=device)
    if not bool(torch.all(batch.information_complete).item()):
        raise RuntimeError("held-out gate diagnostic must be information-complete")
    hop1_slots: list[int] = []
    hop2_slots: list[int] = []
    intermediate_nodes: list[int] = []
    for world in worlds:
        hop1 = _chain_record_for_key(world, world.start_key)
        hop2 = _chain_record_for_key(world, hop1.value)
        if hop1.worker_slot >= WIDTH or hop2.worker_slot >= WIDTH:
            raise RuntimeError("threshold-16 held-out world escaped active scope")
        if hop2.value != world.answer_key:
            raise RuntimeError("relay-2 chain metadata does not terminate at answer")
        hop1_slots.append(hop1.worker_slot)
        hop2_slots.append(hop2.worker_slot)
        intermediate_nodes.append(hop1.value)
    with torch.inference_mode():
        active_local = batch.local_inputs[:, :WIDTH, :]
        batch_size = active_local.shape[0]
        flat_local = active_local.reshape(batch_size * WIDTH, -1)
        initial_states = torch.tanh(model.cell.input_projection(flat_local))
        value_bits = active_local[..., NODE_BIT_WIDTH:]
        message_content = model.query_projection(value_bits).reshape(
            batch_size * WIDTH,
            model.config.message_width,
        )
        start_query = torch.tanh(model.query_projection(batch.start_bits))
        hop1_states = _updated_states(
            model,
            flat_local=flat_local,
            initial_states=initial_states,
            shared=start_query,
            batch_size=batch_size,
        )
        hop1_logits = model.cell.message_gate(hop1_states).reshape(batch_size, WIDTH)
        hop1_shared = _aggregate_shared(
            model,
            states=hop1_states,
            message_content=message_content,
            batch_size=batch_size,
        )
        hop2_model_states = _updated_states(
            model,
            flat_local=flat_local,
            initial_states=initial_states,
            shared=hop1_shared,
            batch_size=batch_size,
        )
        hop2_model_logits = model.cell.message_gate(hop2_model_states).reshape(batch_size, WIDTH)
        intermediate_ids = torch.tensor(intermediate_nodes, dtype=torch.int64, device=device)
        clean_next_query = torch.tanh(model.query_projection(encode_node_bits(intermediate_ids)))
        hop2_oracle_states = _updated_states(
            model,
            flat_local=flat_local,
            initial_states=initial_states,
            shared=clean_next_query,
            batch_size=batch_size,
        )
        hop2_oracle_logits = model.cell.message_gate(hop2_oracle_states).reshape(batch_size, WIDTH)
        full_output = model(
            batch,
            communication_mode=CommunicationMode.SPARSE_SHARED_V0,
            recurrent_rounds=2,
        )
        predictions = decode_node_logits(full_output.logits)
        exact_solve_rate = float((predictions == batch.answer_keys).float().mean().item())
        bit_accuracy = float(
            ((full_output.logits >= 0) == (batch.target_bits > 0)).float().mean().item()
        )
        cosine = F.cosine_similarity(hop1_shared, clean_next_query, dim=-1)
        difference = hop1_shared - clean_next_query
        rmse = difference.square().mean(dim=-1).sqrt()
        l2 = difference.square().sum(dim=-1).sqrt()
    hop1_slot_tensor = torch.tensor(hop1_slots, dtype=torch.int64, device=device)
    hop2_slot_tensor = torch.tensor(hop2_slots, dtype=torch.int64, device=device)
    return {
        "exact_solve_rate": exact_solve_rate,
        "bit_accuracy": bit_accuracy,
        "hop1_start_query_gate": _summarize_gate(hop1_logits, hop1_slot_tensor),
        "hop2_model_query_gate": _summarize_gate(hop2_model_logits, hop2_slot_tensor),
        "hop2_oracle_clean_query_gate": _summarize_gate(hop2_oracle_logits, hop2_slot_tensor),
        "hop1_shared_to_clean_next_query_mean_cosine": float(cosine.mean().item()),
        "hop1_shared_to_clean_next_query_mean_rmse": float(rmse.mean().item()),
        "hop1_shared_to_clean_next_query_mean_l2": float(l2.mean().item()),
    }


def _updated_states(
    model: RelayPopulationModel,
    *,
    flat_local: torch.Tensor,
    initial_states: torch.Tensor,
    shared: torch.Tensor,
    batch_size: int,
) -> torch.Tensor:
    shared_for_active = (
        shared.unsqueeze(1)
        .expand(batch_size, WIDTH, model.config.message_width)
        .reshape(batch_size * WIDTH, model.config.message_width)
    )
    update_input = torch.cat((flat_local, shared_for_active), dim=-1)
    return model.cell.update(update_input, initial_states)


def _aggregate_shared(
    model: RelayPopulationModel,
    *,
    states: torch.Tensor,
    message_content: torch.Tensor,
    batch_size: int,
) -> torch.Tensor:
    gate = torch.sigmoid(model.cell.message_gate(states))
    messages = (message_content * gate).reshape(
        batch_size,
        WIDTH,
        model.config.message_width,
    )
    return torch.tanh(messages.sum(dim=1))


def _summarize_gate(logits: torch.Tensor, correct_slots: torch.Tensor) -> GateSummary:
    if logits.ndim != 2 or logits.shape[1] != WIDTH:
        raise ValueError("gate logits must have shape [batch, WIDTH]")
    if correct_slots.shape != (logits.shape[0],):
        raise ValueError("correct slot vector does not match gate batch")
    correct = logits.gather(1, correct_slots.unsqueeze(1)).squeeze(1)
    ranks = 1 + (logits > correct.unsqueeze(1)).sum(dim=1)
    worker_positions = torch.arange(WIDTH, device=logits.device).unsqueeze(0)
    others = logits.masked_fill(worker_positions == correct_slots.unsqueeze(1), float("-inf"))
    best_other = others.max(dim=1).values
    margins = correct - best_other
    return GateSummary(
        top1_rate=float((ranks == 1).float().mean().item()),
        mean_rank=float(ranks.float().mean().item()),
        mean_margin=float(margins.mean().item()),
        median_margin=float(margins.median().item()),
        mean_correct_logit=float(correct.mean().item()),
        mean_best_nonmatch_logit=float(best_other.mean().item()),
    )


def _chain_record_for_key(world: RelayWorld, key: int):
    matches = tuple(
        record for record in world.records if record.is_chain_edge and record.key == key
    )
    if len(matches) != 1:
        raise RuntimeError("expected exactly one chain record for diagnostic query")
    return matches[0]


def _threshold_worlds(
    *,
    base_seed: int,
    offset: int,
    count: int,
    threshold: int,
) -> tuple[RelayWorld, ...]:
    difficulty = RELAY_DIFFICULTIES[0]
    thresholds = relay_scope_thresholds(difficulty)
    selected: list[RelayWorld] = []
    seed = base_seed + offset * len(thresholds)
    while len(selected) < count:
        if thresholds[seed % len(thresholds)] == threshold:
            world = generate_relay_world(seed, difficulty)
            if world.scope_threshold != threshold:
                raise RuntimeError("width-16 gate threshold selection drifted")
            selected.append(world)
        seed += 1
    return tuple(selected)


if __name__ == "__main__":
    raise SystemExit(main())
