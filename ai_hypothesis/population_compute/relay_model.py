"""Neural encoding and shared-weight model for collective-relay-v0."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Iterable

import torch
from torch import nn

from .collective_relay import RELAY_WORLD_SIZE, RelayWorld
from .contract import CommunicationMode
from .model import PopulationForwardOutput, SharedPopulationCell, SharedPopulationConfig


NODE_BIT_WIDTH = 12


@dataclass(frozen=True, slots=True)
class RelayPopulationConfig:
    state_width: int = 64
    message_width: int = 24

    def validate(self) -> None:
        if self.state_width <= 0:
            raise ValueError("state_width must be positive")
        if self.message_width <= 0:
            raise ValueError("message_width must be positive")


@dataclass(frozen=True, slots=True)
class RelayTensorBatch:
    """One model-ready batch with scope truth kept outside the neural input."""

    local_inputs: torch.Tensor
    active_mask: torch.Tensor
    start_bits: torch.Tensor
    target_bits: torch.Tensor
    answer_keys: torch.Tensor
    information_complete: torch.Tensor
    active_workers: int
    hop_count: int

    def validate(self) -> None:
        if self.local_inputs.ndim != 3:
            raise ValueError("local_inputs must be rank 3")
        batch_size, worker_count, local_width = self.local_inputs.shape
        if worker_count != RELAY_WORLD_SIZE:
            raise ValueError("relay batches must retain the frozen world width")
        if local_width != NODE_BIT_WIDTH * 2:
            raise ValueError("relay local input width is invalid")
        if self.active_mask.shape != (batch_size, worker_count):
            raise ValueError("active_mask shape does not match local_inputs")
        if self.start_bits.shape != (batch_size, NODE_BIT_WIDTH):
            raise ValueError("start_bits shape is invalid")
        if self.target_bits.shape != (batch_size, NODE_BIT_WIDTH):
            raise ValueError("target_bits shape is invalid")
        if self.answer_keys.shape != (batch_size,):
            raise ValueError("answer_keys shape is invalid")
        if self.information_complete.shape != (batch_size,):
            raise ValueError("information_complete shape is invalid")
        if not 1 <= self.active_workers <= RELAY_WORLD_SIZE:
            raise ValueError("active_workers is outside the relay world")
        if self.hop_count < 2:
            raise ValueError("relay hop_count must be at least 2")
        expected_mask = (
            torch.arange(worker_count, device=self.active_mask.device)
            < self.active_workers
        )
        if not torch.equal(
            self.active_mask,
            expected_mask.unsqueeze(0).expand(batch_size, worker_count),
        ):
            raise ValueError("active_mask must select the nested worker prefix")


class RelayPopulationModel(nn.Module):
    """Shared-weight neural population for the first collective-relay benchmark."""

    def __init__(self, config: RelayPopulationConfig = RelayPopulationConfig()) -> None:
        super().__init__()
        config.validate()
        self.config = config
        self.query_projection = nn.Linear(NODE_BIT_WIDTH, config.message_width)
        self.cell = SharedPopulationCell(
            SharedPopulationConfig(
                local_input_width=NODE_BIT_WIDTH * 2,
                state_width=config.state_width,
                message_width=config.message_width,
                output_width=NODE_BIT_WIDTH,
            )
        )

    def forward(
        self,
        batch: RelayTensorBatch,
        *,
        communication_mode: CommunicationMode,
        recurrent_rounds: int | None = None,
    ) -> PopulationForwardOutput:
        batch.validate()
        rounds = batch.hop_count if recurrent_rounds is None else recurrent_rounds
        if rounds <= 0:
            raise ValueError("recurrent_rounds must be positive")
        seed = torch.tanh(self.query_projection(batch.start_bits))
        value_bits = batch.local_inputs[..., NODE_BIT_WIDTH:]
        message_content = self.query_projection(value_bits)
        return self.cell(
            batch.local_inputs,
            batch.active_mask,
            recurrent_rounds=rounds,
            communication_mode=communication_mode,
            shared_seed=seed,
            message_content=message_content,
            reset_state_each_round=True,
        )

    def trainable_parameter_count(self) -> int:
        return sum(
            parameter.numel()
            for parameter in self.parameters()
            if parameter.requires_grad
        )

    def parameter_fingerprint(self) -> str:
        digest = hashlib.sha256()
        for name, tensor in sorted(self.state_dict().items()):
            detached = tensor.detach().cpu().contiguous().clone()
            digest.update(name.encode("utf-8"))
            digest.update(str(detached.dtype).encode("ascii"))
            digest.update(str(tuple(detached.shape)).encode("ascii"))
            digest.update(bytes(detached.untyped_storage()))
        return digest.hexdigest()


def encode_node_bits(node_ids: torch.Tensor) -> torch.Tensor:
    """Encode integer node identities as fixed {-1,+1} bits with no learned table."""

    if node_ids.dtype not in {
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.uint8,
    }:
        raise ValueError("node_ids must use an integer dtype")
    if torch.any(node_ids < 0) or torch.any(node_ids >= (1 << NODE_BIT_WIDTH)):
        raise ValueError("node id is outside the fixed bit encoding range")
    shifts = torch.arange(NODE_BIT_WIDTH, device=node_ids.device, dtype=torch.int64)
    bits = ((node_ids.to(torch.int64).unsqueeze(-1) >> shifts) & 1).to(torch.float32)
    return bits.mul(2.0).sub(1.0)


def decode_node_logits(logits: torch.Tensor) -> torch.Tensor:
    """Decode one logit per fixed identity bit into integer node identities."""

    if logits.ndim != 2 or logits.shape[-1] != NODE_BIT_WIDTH:
        raise ValueError("logits must have shape [batch, NODE_BIT_WIDTH]")
    bits = logits >= 0
    shifts = torch.arange(NODE_BIT_WIDTH, device=logits.device, dtype=torch.int64)
    weights = (torch.ones_like(shifts) << shifts).unsqueeze(0)
    return (bits.to(torch.int64) * weights).sum(dim=-1)


def build_relay_tensor_batch(
    worlds: Iterable[RelayWorld],
    *,
    active_workers: int,
    device: torch.device | str = "cpu",
) -> RelayTensorBatch:
    """Materialize nested-prefix local scope without leaking oracle chain labels."""

    materialized = tuple(worlds)
    if not materialized:
        raise ValueError("at least one relay world is required")
    if not 1 <= active_workers <= RELAY_WORLD_SIZE:
        raise ValueError("active_workers must be within the frozen relay world")

    first = materialized[0]
    first.validate()
    if first.difficulty.world_size != RELAY_WORLD_SIZE:
        raise ValueError("relay world does not use the frozen world size")
    for world in materialized[1:]:
        world.validate()
        if world.difficulty != first.difficulty:
            raise ValueError("one relay tensor batch must use one difficulty tier")

    target_device = torch.device(device)
    keys = torch.tensor(
        [[record.key for record in world.records] for world in materialized],
        dtype=torch.int64,
        device=target_device,
    )
    values = torch.tensor(
        [[record.value for record in world.records] for world in materialized],
        dtype=torch.int64,
        device=target_device,
    )
    starts = torch.tensor(
        [world.start_key for world in materialized],
        dtype=torch.int64,
        device=target_device,
    )
    answers = torch.tensor(
        [world.answer_key for world in materialized],
        dtype=torch.int64,
        device=target_device,
    )

    local_inputs = torch.cat((encode_node_bits(keys), encode_node_bits(values)), dim=-1)
    active_mask = (
        torch.arange(RELAY_WORLD_SIZE, device=target_device)
        < active_workers
    ).unsqueeze(0).expand(len(materialized), RELAY_WORLD_SIZE).clone()
    information_complete = torch.tensor(
        [
            all(
                (not record.is_chain_edge) or record.worker_slot < active_workers
                for record in world.records
            )
            for world in materialized
        ],
        dtype=torch.bool,
        device=target_device,
    )

    batch = RelayTensorBatch(
        local_inputs=local_inputs,
        active_mask=active_mask,
        start_bits=encode_node_bits(starts),
        target_bits=encode_node_bits(answers),
        answer_keys=answers,
        information_complete=information_complete,
        active_workers=active_workers,
        hop_count=first.difficulty.hop_count,
    )
    batch.validate()
    return batch
