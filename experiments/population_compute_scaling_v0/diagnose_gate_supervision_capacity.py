"""Development-only fixed-width capacity diagnostic with training-only gate supervision.

Fresh corrected relay-2 checkpoints are trained independently at 4/16/64/256 active states.
Every checkpoint keeps the same architecture and learned parameter count. The only training change
versus #67 is the fixed auxiliary gate-selection objective proven useful at width 16 in #69.
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


WIDTHS = (4, 16, 64, 256)
TRAIN_BATCH_BY_WIDTH = {4: 64, 16: 16, 64: 4, 256: 1}
TRAINING_SEED_BASE = 17_000_000_000
HELDOUT_SEED_BASE = 18_000_000_000
GATE_LOSS_WEIGHT = 1.0


@dataclass(frozen=True)
class GateSummary:
    top1_rate: float
    mean_rank: float
    mean_margin: float


@dataclass(frozen=True)
class WidthResult:
    width: int
    training_seed: int
    train_batch_size: int
    steps: int
    learned_parameter_count: int
    parameter_fingerprint: str
    first_total_loss: float
    final_total_loss: float
    mean_last_100_total_loss: float
    final_relay_loss: float
    final_gate_loss: float
    sparse_exact_solve_rate: float
    sparse_bit_accuracy: float
    no_communication_exact_solve_rate: float
    no_communication_bit_accuracy: float
    hop1_start_query_gate: GateSummary
    hop2_model_query_gate: GateSummary
    hop2_oracle_clean_query_gate: GateSummary
    hop1_shared_to_clean_next_query_mean_cosine: float
    hop1_shared_to_clean_next_query_mean_rmse: float


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
    thresholds = relay_scope_thresholds(difficulty)
    if difficulty.name != "relay-2" or any(width not in thresholds for width in WIDTHS):
        raise RuntimeError("gate-supervision capacity diagnostic no longer matches relay-2")

    device = torch.device(args.device)
    results = tuple(
        _train_width(
            width=width,
            training_seed=900 + index,
            steps=args.steps,
            heldout_world_count=args.heldout_world_count,
            device=device,
        )
        for index, width in enumerate(WIDTHS)
    )
    if len({result.learned_parameter_count for result in results}) != 1:
        raise RuntimeError("learned parameter count changed across fixed-width checkpoints")

    payload = {
        "diagnostic": "relay-gate-supervision-capacity-v0",
        "architecture": "unchanged-inference-exact-compositional-relay",
        "difficulty": difficulty.name,
        "widths": list(WIDTHS),
        "device": str(device),
        "steps_per_width": args.steps,
        "heldout_world_count_per_width": args.heldout_world_count,
        "gate_loss_weight": GATE_LOSS_WEIGHT,
        "active_worker_updates_per_training_batch": {
            str(width): width * TRAIN_BATCH_BY_WIDTH[width] for width in WIDTHS
        },
        "results": [asdict(result) for result in results],
        "confirmation_opened": False,
        "interpretation_frozen_before_result": {
            "strong_through_256": "selectivity training scales; test mixed-population training with the auxiliary objective next",
            "degrades_despite_weak_gates": "population selectivity remains width-limited even with direct supervision",
            "strong_gates_but_solve_degrades": "inspect shared-field accumulation or readout at the failing width",
        },
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def _train_width(
    *,
    width: int,
    training_seed: int,
    steps: int,
    heldout_world_count: int,
    device: torch.device,
) -> WidthResult:
    batch_size = TRAIN_BATCH_BY_WIDTH[width]
    torch.manual_seed(training_seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(training_seed)

    model = RelayPopulationModel(RelayPopulationConfig(state_width=64, message_width=24)).to(device)
    parameter_count = model.trainable_parameter_count()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    total_losses: list[float] = []
    final_relay_loss = 0.0
    final_gate_loss = 0.0

    model.train()
    train_base = TRAINING_SEED_BASE + width * 10_000_000
    for step in range(steps):
        worlds = _threshold_worlds(
            base_seed=train_base,
            offset=step * batch_size,
            count=batch_size,
            threshold=width,
        )
        batch = build_relay_tensor_batch(worlds, active_workers=width, device=device)
        if not bool(torch.all(batch.information_complete).item()):
            raise RuntimeError(f"width {width} training batch is not information-complete")

        output = model(
            batch,
            communication_mode=CommunicationMode.SPARSE_SHARED_V0,
            recurrent_rounds=2,
        )
        targets = batch.target_bits.gt(0).to(dtype=output.logits.dtype)
        relay_loss = F.binary_cross_entropy_with_logits(output.logits, targets)
        gate_loss = _training_gate_loss(model, batch, worlds, width=width, device=device)
        total_loss = relay_loss + GATE_LOSS_WEIGHT * gate_loss
        if not torch.isfinite(total_loss):
            raise RuntimeError(f"width {width} produced non-finite total loss")

        optimizer.zero_grad(set_to_none=True)
        total_loss.backward()
        optimizer.step()
        total_losses.append(float(total_loss.detach().item()))
        final_relay_loss = float(relay_loss.detach().item())
        final_gate_loss = float(gate_loss.detach().item())

    fingerprint = model.parameter_fingerprint()
    if model.trainable_parameter_count() != parameter_count:
        raise RuntimeError(f"width {width} changed learned parameter count")

    heldout = _threshold_worlds(
        base_seed=HELDOUT_SEED_BASE + width * 10_000_000,
        offset=0,
        count=heldout_world_count,
        threshold=width,
    )
    sparse = _inspect(model, heldout, width=width, device=device)
    no_comm_exact, no_comm_bits = _evaluate_no_communication(
        model, heldout, width=width, device=device
    )
    if model.parameter_fingerprint() != fingerprint:
        raise RuntimeError(f"width {width} evaluation mutated the checkpoint")

    tail = total_losses[-min(100, len(total_losses)) :]
    return WidthResult(
        width=width,
        training_seed=training_seed,
        train_batch_size=batch_size,
        steps=steps,
        learned_parameter_count=parameter_count,
        parameter_fingerprint=fingerprint,
        first_total_loss=total_losses[0],
        final_total_loss=total_losses[-1],
        mean_last_100_total_loss=sum(tail) / len(tail),
        final_relay_loss=final_relay_loss,
        final_gate_loss=final_gate_loss,
        no_communication_exact_solve_rate=no_comm_exact,
        no_communication_bit_accuracy=no_comm_bits,
        **sparse,
    )


def _training_gate_loss(model, batch, worlds, *, width: int, device: torch.device):
    active_local, flat_local, initial_states = _active_local_state(model, batch, width=width)
    batch_size = active_local.shape[0]
    hop1_slots, hop2_slots, intermediate_nodes = _chain_targets(
        worlds, width=width, device=device
    )

    clean_start = torch.tanh(model.query_projection(batch.start_bits))
    hop1_states = _updated_states(
        model,
        flat_local=flat_local,
        initial_states=initial_states,
        shared=clean_start,
        batch_size=batch_size,
        width=width,
    )
    hop1_logits = model.cell.message_gate(hop1_states).reshape(batch_size, width)

    clean_intermediate = torch.tanh(
        model.query_projection(encode_node_bits(intermediate_nodes))
    )
    hop2_states = _updated_states(
        model,
        flat_local=flat_local,
        initial_states=initial_states,
        shared=clean_intermediate,
        batch_size=batch_size,
        width=width,
    )
    hop2_logits = model.cell.message_gate(hop2_states).reshape(batch_size, width)
    return 0.5 * (
        F.cross_entropy(hop1_logits, hop1_slots)
        + F.cross_entropy(hop2_logits, hop2_slots)
    )


def _inspect(model, worlds, *, width: int, device: torch.device) -> dict[str, object]:
    model.eval()
    batch = build_relay_tensor_batch(worlds, active_workers=width, device=device)
    hop1_slots, hop2_slots, intermediate_nodes = _chain_targets(
        worlds, width=width, device=device
    )
    with torch.inference_mode():
        active_local, flat_local, initial_states = _active_local_state(model, batch, width=width)
        batch_size = active_local.shape[0]
        message_content = model.query_projection(
            active_local[..., NODE_BIT_WIDTH:]
        ).reshape(batch_size * width, model.config.message_width)

        clean_start = torch.tanh(model.query_projection(batch.start_bits))
        hop1_states = _updated_states(
            model,
            flat_local=flat_local,
            initial_states=initial_states,
            shared=clean_start,
            batch_size=batch_size,
            width=width,
        )
        hop1_logits = model.cell.message_gate(hop1_states).reshape(batch_size, width)
        hop1_shared = _aggregate_shared(
            model,
            states=hop1_states,
            message_content=message_content,
            batch_size=batch_size,
            width=width,
        )

        hop2_model_states = _updated_states(
            model,
            flat_local=flat_local,
            initial_states=initial_states,
            shared=hop1_shared,
            batch_size=batch_size,
            width=width,
        )
        hop2_model_logits = model.cell.message_gate(hop2_model_states).reshape(
            batch_size, width
        )

        clean_intermediate = torch.tanh(
            model.query_projection(encode_node_bits(intermediate_nodes))
        )
        hop2_oracle_states = _updated_states(
            model,
            flat_local=flat_local,
            initial_states=initial_states,
            shared=clean_intermediate,
            batch_size=batch_size,
            width=width,
        )
        hop2_oracle_logits = model.cell.message_gate(hop2_oracle_states).reshape(
            batch_size, width
        )

        ordinary = model(
            batch,
            communication_mode=CommunicationMode.SPARSE_SHARED_V0,
            recurrent_rounds=2,
        )
        predictions = decode_node_logits(ordinary.logits)
        sparse_exact = float((predictions == batch.answer_keys).float().mean().item())
        sparse_bits = float(
            ((ordinary.logits >= 0) == (batch.target_bits > 0)).float().mean().item()
        )
        cosine = F.cosine_similarity(hop1_shared, clean_intermediate, dim=-1)
        rmse = (hop1_shared - clean_intermediate).square().mean(dim=-1).sqrt()

    return {
        "sparse_exact_solve_rate": sparse_exact,
        "sparse_bit_accuracy": sparse_bits,
        "hop1_start_query_gate": _summarize_gate(hop1_logits, hop1_slots, width=width),
        "hop2_model_query_gate": _summarize_gate(hop2_model_logits, hop2_slots, width=width),
        "hop2_oracle_clean_query_gate": _summarize_gate(
            hop2_oracle_logits, hop2_slots, width=width
        ),
        "hop1_shared_to_clean_next_query_mean_cosine": float(cosine.mean().item()),
        "hop1_shared_to_clean_next_query_mean_rmse": float(rmse.mean().item()),
    }


def _evaluate_no_communication(model, worlds, *, width: int, device: torch.device):
    model.eval()
    exact = 0
    correct_bits = 0
    total_bits = 0
    batch_size = max(1, 256 // width)
    with torch.inference_mode():
        for offset in range(0, len(worlds), batch_size):
            batch = build_relay_tensor_batch(
                worlds[offset : offset + batch_size],
                active_workers=width,
                device=device,
            )
            output = model(
                batch,
                communication_mode=CommunicationMode.NO_COMMUNICATION,
                recurrent_rounds=2,
            )
            predictions = decode_node_logits(output.logits)
            exact += int((predictions == batch.answer_keys).sum().item())
            predicted_bits = output.logits >= 0
            target_bits = batch.target_bits > 0
            correct_bits += int((predicted_bits == target_bits).sum().item())
            total_bits += int(target_bits.numel())
    return exact / len(worlds), correct_bits / total_bits


def _active_local_state(model, batch, *, width: int):
    active_local = batch.local_inputs[:, :width, :]
    batch_size = active_local.shape[0]
    flat_local = active_local.reshape(batch_size * width, -1)
    initial_states = torch.tanh(model.cell.input_projection(flat_local))
    return active_local, flat_local, initial_states


def _updated_states(model, *, flat_local, initial_states, shared, batch_size: int, width: int):
    shared_for_active = (
        shared.unsqueeze(1)
        .expand(batch_size, width, model.config.message_width)
        .reshape(batch_size * width, model.config.message_width)
    )
    return model.cell.update(
        torch.cat((flat_local, shared_for_active), dim=-1), initial_states
    )


def _aggregate_shared(model, *, states, message_content, batch_size: int, width: int):
    gate = torch.sigmoid(model.cell.message_gate(states))
    messages = (message_content * gate).reshape(
        batch_size, width, model.config.message_width
    )
    return torch.tanh(messages.sum(dim=1))


def _chain_targets(worlds, *, width: int, device: torch.device):
    hop1_slots: list[int] = []
    hop2_slots: list[int] = []
    intermediate_nodes: list[int] = []
    for world in worlds:
        hop1 = _chain_record_for_key(world, world.start_key)
        hop2 = _chain_record_for_key(world, hop1.value)
        if hop1.worker_slot >= width or hop2.worker_slot >= width:
            raise RuntimeError(f"threshold-{width} world escaped active scope")
        hop1_slots.append(hop1.worker_slot)
        hop2_slots.append(hop2.worker_slot)
        intermediate_nodes.append(hop1.value)
    return (
        torch.tensor(hop1_slots, dtype=torch.int64, device=device),
        torch.tensor(hop2_slots, dtype=torch.int64, device=device),
        torch.tensor(intermediate_nodes, dtype=torch.int64, device=device),
    )


def _summarize_gate(logits, correct_slots, *, width: int) -> GateSummary:
    correct = logits.gather(1, correct_slots.unsqueeze(1)).squeeze(1)
    ranks = 1 + (logits > correct.unsqueeze(1)).sum(dim=1)
    positions = torch.arange(width, device=logits.device).unsqueeze(0)
    best_other = logits.masked_fill(
        positions == correct_slots.unsqueeze(1), float("-inf")
    ).max(dim=1).values
    margins = correct - best_other
    return GateSummary(
        top1_rate=float((ranks == 1).float().mean().item()),
        mean_rank=float(ranks.float().mean().item()),
        mean_margin=float(margins.mean().item()),
    )


def _chain_record_for_key(world: RelayWorld, key: int):
    matches = tuple(
        record for record in world.records if record.is_chain_edge and record.key == key
    )
    if len(matches) != 1:
        raise RuntimeError("expected exactly one chain record for diagnostic query")
    return matches[0]


def _threshold_worlds(*, base_seed: int, offset: int, count: int, threshold: int):
    difficulty = RELAY_DIFFICULTIES[0]
    thresholds = relay_scope_thresholds(difficulty)
    selected: list[RelayWorld] = []
    seed = base_seed + offset * len(thresholds)
    while len(selected) < count:
        if thresholds[seed % len(thresholds)] == threshold:
            world = generate_relay_world(seed, difficulty)
            if world.scope_threshold != threshold:
                raise RuntimeError("gate-supervision capacity threshold selection drifted")
            selected.append(world)
        seed += 1
    return tuple(selected)


if __name__ == "__main__":
    raise SystemExit(main())
