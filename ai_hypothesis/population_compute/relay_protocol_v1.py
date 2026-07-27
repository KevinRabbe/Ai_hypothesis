"""Canonical repaired collective-relay protocol v1.

Inference uses parameter-free softmax-normalized population transport. Training may add
an oracle-derived selector loss, but oracle chain identity never enters the inference
inputs or forward path.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch

from .collective_relay import RelayWorld
from .relay_model import RelayPopulationModel, RelayTensorBatch, encode_node_bits
from .relay_serial_control import RelayScheduleOutput, normalized_parallel_forward


RELAY_PROTOCOL_VERSION = "relay-protocol-v1-normalized-gate-supervised"
DEFAULT_GATE_SUPERVISION_WEIGHT = 1.0


def forward_relay_v1(
    model: RelayPopulationModel,
    batch: RelayTensorBatch,
    *,
    recurrent_rounds: int | None = None,
) -> RelayScheduleOutput:
    """Oracle-free canonical inference for repaired relay v1."""

    return normalized_parallel_forward(
        model,
        batch,
        recurrent_rounds=recurrent_rounds,
    )


def gate_supervision_loss(
    model: RelayPopulationModel,
    batch: RelayTensorBatch,
    worlds: Sequence[RelayWorld],
) -> torch.Tensor:
    """Training-only loss selecting the unique correct record at every clean relay hop."""

    batch.validate()
    if len(worlds) != int(batch.local_inputs.shape[0]):
        raise ValueError("gate-supervision worlds must match the tensor batch")
    if not worlds:
        raise ValueError("gate supervision requires at least one relay world")

    active_workers = batch.active_workers
    local = batch.local_inputs[:, :active_workers, :]
    batch_size = int(local.shape[0])
    flat_local = local.reshape(batch_size * active_workers, -1)
    initial = torch.tanh(model.cell.input_projection(flat_local))
    queries, target_slots = oracle_hop_targets(
        worlds,
        active_workers=active_workers,
        device=local.device,
    )
    losses: list[torch.Tensor] = []

    for hop in range(queries.shape[1]):
        clean_query = torch.tanh(
            model.query_projection(encode_node_bits(queries[:, hop]))
        )
        shared_flat = (
            clean_query.unsqueeze(1)
            .expand(batch_size, active_workers, model.config.message_width)
            .reshape(batch_size * active_workers, model.config.message_width)
        )
        states = model.cell.update(
            torch.cat((flat_local, shared_flat), dim=-1),
            initial,
        )
        gate_logits = model.cell.message_gate(states).reshape(
            batch_size,
            active_workers,
        )
        losses.append(
            torch.nn.functional.cross_entropy(
                gate_logits,
                target_slots[:, hop],
            )
        )

    return torch.stack(losses).mean()


def oracle_hop_targets(
    worlds: Sequence[RelayWorld],
    *,
    active_workers: int,
    device: torch.device | str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return training labels for the clean query and matching worker at each hop."""

    if not worlds:
        raise ValueError("oracle hop targets require at least one world")
    if active_workers <= 0:
        raise ValueError("active_workers must be positive")

    hop_count = worlds[0].difficulty.hop_count
    query_rows: list[list[int]] = []
    slot_rows: list[list[int]] = []

    for world in worlds:
        world.validate()
        if world.difficulty.hop_count != hop_count:
            raise ValueError("gate-supervision batch mixed relay depths")

        query = world.start_key
        queries: list[int] = []
        slots: list[int] = []
        for _ in range(hop_count):
            matches = tuple(
                record
                for record in world.records
                if record.is_chain_edge and record.key == query
            )
            if len(matches) != 1:
                raise RuntimeError("expected one chain record per supervised relay hop")
            record = matches[0]
            if record.worker_slot >= active_workers:
                raise ValueError("gate-supervision target lies outside active scope")
            queries.append(query)
            slots.append(record.worker_slot)
            query = record.value

        if query != world.answer_key:
            raise RuntimeError("gate-supervision chain does not terminate at answer")
        query_rows.append(queries)
        slot_rows.append(slots)

    target_device = torch.device(device)
    return (
        torch.tensor(query_rows, dtype=torch.int64, device=target_device),
        torch.tensor(slot_rows, dtype=torch.int64, device=target_device),
    )
