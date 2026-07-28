"""Tensor encoding and matched persistent state-bank schedules for Gate 2.

The capability experiment is deliberately not trained here.  This module only establishes a
shared learned update rule, width-independent observation tensors, and parallel/serial
execution schedules whose capability output must agree.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Iterable

import torch
from torch import nn

from .gate2_persistent_state_capacity import (
    GATE2_EVIDENCE_ROUNDS,
    GATE2_INTERFERENCE_ROUNDS,
    GATE2_KEY_BIT_WIDTH,
    GATE2_PAYLOAD_BITS,
    GATE2_TOTAL_ROUNDS,
    Gate2ConditionPlan,
    Gate2ControlMode,
    Gate2World,
    build_gate2_condition_plan,
)

GATE2_ROUND_FEATURE_WIDTH = GATE2_TOTAL_ROUNDS
GATE2_BIT_POSITION_WIDTH = GATE2_PAYLOAD_BITS
GATE2_INTERFERENCE_TOKEN_BITS = 16
GATE2_OBSERVATION_WIDTH = (
    GATE2_KEY_BIT_WIDTH
    + GATE2_ROUND_FEATURE_WIDTH
    + GATE2_BIT_POSITION_WIDTH
    + 2  # evidence/type flag + evidence bit value
    + GATE2_INTERFERENCE_TOKEN_BITS
)


@dataclass(frozen=True, slots=True)
class Gate2PersistentModelConfig:
    state_width: int = 64
    query_width: int = 24

    def validate(self) -> None:
        if self.state_width <= 0:
            raise ValueError("state_width must be positive")
        if self.query_width <= 0:
            raise ValueError("query_width must be positive")


@dataclass(frozen=True, slots=True)
class Gate2TensorBatch:
    observations: torch.Tensor
    entity_order_by_round_slot_lane: torch.Tensor
    slot_by_round_entity: torch.Tensor
    query_bits: torch.Tensor
    answer_payloads: torch.Tensor
    query_entity_indices: torch.Tensor
    entity_count: int
    width: int
    mode: Gate2ControlMode

    def validate(self) -> None:
        if self.observations.ndim != 4:
            raise ValueError("observations must have shape [batch, rounds, entities, features]")
        batch_size, rounds, entity_count, feature_width = self.observations.shape
        if rounds != GATE2_TOTAL_ROUNDS:
            raise ValueError("Gate-2 tensor batch must contain exactly eight rounds")
        if entity_count != self.entity_count:
            raise ValueError("entity_count does not match observation tensor")
        if feature_width != GATE2_OBSERVATION_WIDTH:
            raise ValueError("observation feature width is invalid")
        if self.entity_count % self.width != 0:
            raise ValueError("frozen Gate-2 widths must divide entity_count")
        lane_count = self.entity_count // self.width
        if self.entity_order_by_round_slot_lane.shape != (
            batch_size,
            GATE2_TOTAL_ROUNDS,
            self.width,
            lane_count,
        ):
            raise ValueError("entity_order_by_round_slot_lane shape is invalid")
        if self.slot_by_round_entity.shape != (
            batch_size,
            GATE2_TOTAL_ROUNDS,
            self.entity_count,
        ):
            raise ValueError("slot_by_round_entity shape is invalid")
        if self.query_bits.shape != (batch_size, GATE2_KEY_BIT_WIDTH):
            raise ValueError("query_bits shape is invalid")
        if self.answer_payloads.shape != (batch_size,):
            raise ValueError("answer_payloads shape is invalid")
        if self.query_entity_indices.shape != (batch_size,):
            raise ValueError("query_entity_indices shape is invalid")
        if torch.any(self.answer_payloads < 0) or torch.any(
            self.answer_payloads >= (1 << GATE2_PAYLOAD_BITS)
        ):
            raise ValueError("answer payload is outside the frozen 4-bit answer space")
        if torch.any(self.query_entity_indices < 0) or torch.any(
            self.query_entity_indices >= self.entity_count
        ):
            raise ValueError("query entity index is outside the world")
        if torch.any(self.slot_by_round_entity < 0) or torch.any(
            self.slot_by_round_entity >= self.width
        ):
            raise ValueError("routing tensor contains an invalid slot")

        # Every round/slot lane table must be an exact permutation of all entities.
        canonical = torch.arange(self.entity_count, device=self.observations.device)
        flattened = self.entity_order_by_round_slot_lane.reshape(
            batch_size,
            GATE2_TOTAL_ROUNDS,
            self.entity_count,
        )
        sorted_entities = torch.sort(flattened, dim=-1).values
        if not torch.equal(
            sorted_entities,
            canonical.view(1, 1, -1).expand_as(sorted_entities),
        ):
            raise ValueError("slot/lane routing must cover every entity exactly once per round")

    @property
    def learned_updates_per_sample(self) -> int:
        return GATE2_TOTAL_ROUNDS * self.entity_count

    @property
    def collision_load(self) -> int:
        return self.entity_count // self.width


@dataclass(frozen=True, slots=True)
class Gate2ScheduleTelemetry:
    schedule: str
    entity_count: int
    width: int
    rounds: int
    learned_updates_per_sample: int
    peak_simultaneous_updates_per_sample: int
    persistent_state_vectors_per_sample: int
    collision_load: int

    def validate(self) -> None:
        if self.schedule not in {"parallel_persistent", "serial_persistent"}:
            raise ValueError("unknown Gate-2 schedule")
        if self.rounds != GATE2_TOTAL_ROUNDS:
            raise ValueError("Gate-2 schedule must preserve the frozen eight rounds")
        if self.learned_updates_per_sample != self.rounds * self.entity_count:
            raise ValueError("Gate-2 schedule learned-update accounting is invalid")
        if self.persistent_state_vectors_per_sample != self.width:
            raise ValueError("Gate-2 persistent state-bank size must equal runtime width")
        if self.collision_load != self.entity_count // self.width:
            raise ValueError("Gate-2 collision-load accounting is invalid")
        expected_peak = self.width if self.schedule == "parallel_persistent" else 1
        if self.peak_simultaneous_updates_per_sample != expected_peak:
            raise ValueError("Gate-2 simultaneous-update accounting is invalid")


@dataclass(frozen=True, slots=True)
class Gate2ScheduleOutput:
    logits: torch.Tensor
    final_states: torch.Tensor
    telemetry: Gate2ScheduleTelemetry


class Gate2PersistentStateModel(nn.Module):
    """One shared GRU update reused across every temporary Gate-2 runtime state slot."""

    def __init__(self, config: Gate2PersistentModelConfig = Gate2PersistentModelConfig()) -> None:
        super().__init__()
        config.validate()
        self.config = config
        self.update = nn.GRUCell(GATE2_OBSERVATION_WIDTH, config.state_width)
        self.query_projection = nn.Linear(GATE2_KEY_BIT_WIDTH, config.query_width)
        self.output_norm = nn.LayerNorm(config.state_width + config.query_width)
        self.output_head = nn.Linear(config.state_width + config.query_width, GATE2_PAYLOAD_BITS)

    def readout(self, target_state: torch.Tensor, query_bits: torch.Tensor) -> torch.Tensor:
        query = torch.tanh(self.query_projection(query_bits))
        combined = torch.cat((target_state, query), dim=-1)
        return self.output_head(self.output_norm(combined))

    def trainable_parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)

    def parameter_fingerprint(self) -> str:
        digest = hashlib.sha256()
        for name, tensor in sorted(self.state_dict().items()):
            detached = tensor.detach().cpu().contiguous().clone()
            digest.update(name.encode("utf-8"))
            digest.update(str(detached.dtype).encode("ascii"))
            digest.update(str(tuple(detached.shape)).encode("ascii"))
            digest.update(bytes(detached.untyped_storage()))
        return digest.hexdigest()



def build_gate2_tensor_batch(
    worlds: Iterable[Gate2World],
    *,
    width: int,
    mode: Gate2ControlMode,
    device: torch.device | str = "cpu",
) -> Gate2TensorBatch:
    materialized = tuple(worlds)
    if not materialized:
        raise ValueError("at least one Gate-2 world is required")
    first = materialized[0]
    first.validate()
    for world in materialized[1:]:
        world.validate()
        if world.entity_count != first.entity_count:
            raise ValueError("one Gate-2 tensor batch must use one entity count")

    plans = tuple(build_gate2_condition_plan(world, width=width, mode=mode) for world in materialized)
    target_device = torch.device(device)
    observations = torch.stack(
        [encode_gate2_world_observations(world, device=target_device) for world in materialized],
        dim=0,
    )
    routing = torch.tensor(
        [plan.slot_by_round_entity for plan in plans],
        dtype=torch.int64,
        device=target_device,
    )
    entity_order = _build_entity_order_by_round_slot_lane(routing, width=width)
    query_keys = torch.tensor(
        [world.query_key for world in materialized],
        dtype=torch.int64,
        device=target_device,
    )
    answers = torch.tensor(
        [world.answer_payload for world in materialized],
        dtype=torch.int64,
        device=target_device,
    )
    query_entities = torch.tensor(
        [world.query_entity_index for world in materialized],
        dtype=torch.int64,
        device=target_device,
    )

    batch = Gate2TensorBatch(
        observations=observations,
        entity_order_by_round_slot_lane=entity_order,
        slot_by_round_entity=routing,
        query_bits=encode_gate2_bits(query_keys, width=GATE2_KEY_BIT_WIDTH),
        answer_payloads=answers,
        query_entity_indices=query_entities,
        entity_count=first.entity_count,
        width=width,
        mode=mode,
    )
    batch.validate()
    return batch



def encode_gate2_world_observations(
    world: Gate2World,
    *,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """Encode the immutable world stream without any width, slot, or query information."""

    world.validate()
    target_device = torch.device(device)
    encoded = torch.zeros(
        GATE2_TOTAL_ROUNDS,
        world.entity_count,
        GATE2_OBSERVATION_WIDTH,
        dtype=torch.float32,
        device=target_device,
    )

    key_offset = 0
    round_offset = key_offset + GATE2_KEY_BIT_WIDTH
    bit_position_offset = round_offset + GATE2_ROUND_FEATURE_WIDTH
    evidence_flag_offset = bit_position_offset + GATE2_BIT_POSITION_WIDTH
    evidence_value_offset = evidence_flag_offset + 1
    interference_offset = evidence_value_offset + 1

    keys = torch.tensor(world.entity_keys, dtype=torch.int64, device=target_device)
    encoded[:, :, key_offset:round_offset] = encode_gate2_bits(
        keys,
        width=GATE2_KEY_BIT_WIDTH,
    ).unsqueeze(0)

    for observation in world.observations:
        row = encoded[observation.round_index, observation.entity_index]
        row[round_offset + observation.round_index] = 1.0
        if observation.is_evidence:
            assert observation.evidence_bit_index is not None
            assert observation.evidence_bit_value is not None
            row[bit_position_offset + observation.evidence_bit_index] = 1.0
            row[evidence_flag_offset] = 1.0
            row[evidence_value_offset] = 1.0 if observation.evidence_bit_value else -1.0
        else:
            assert observation.interference_token is not None
            row[evidence_flag_offset] = -1.0
            token = torch.tensor(
                observation.interference_token,
                dtype=torch.int64,
                device=target_device,
            )
            row[interference_offset:] = encode_gate2_bits(
                token,
                width=GATE2_INTERFERENCE_TOKEN_BITS,
            )

    return encoded



def encode_gate2_bits(values: torch.Tensor, *, width: int) -> torch.Tensor:
    if values.dtype not in {
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.uint8,
    }:
        raise ValueError("values must use an integer dtype")
    if torch.any(values < 0) or torch.any(values >= (1 << width)):
        raise ValueError("value is outside the requested fixed bit width")
    shifts = torch.arange(width, device=values.device, dtype=torch.int64)
    bits = ((values.to(torch.int64).unsqueeze(-1) >> shifts) & 1).to(torch.float32)
    return bits.mul(2.0).sub(1.0)



def decode_gate2_payload_logits(logits: torch.Tensor) -> torch.Tensor:
    if logits.ndim != 2 or logits.shape[-1] != GATE2_PAYLOAD_BITS:
        raise ValueError("logits must have shape [batch, 4]")
    bits = logits >= 0
    shifts = torch.arange(GATE2_PAYLOAD_BITS, device=logits.device, dtype=torch.int64)
    weights = (torch.ones_like(shifts) << shifts).unsqueeze(0)
    return (bits.to(torch.int64) * weights).sum(dim=-1)



def parallel_persistent_forward(
    model: Gate2PersistentStateModel,
    batch: Gate2TensorBatch,
) -> Gate2ScheduleOutput:
    """Update all independent state slots in parallel at each collision lane."""

    batch.validate()
    batch_size = int(batch.observations.shape[0])
    states = batch.observations.new_zeros(batch_size, batch.width, model.config.state_width)
    lane_count = batch.collision_load

    for round_index in range(GATE2_TOTAL_ROUNDS):
        if batch.mode is Gate2ControlMode.RESET_STATE and round_index > 0:
            states = torch.zeros_like(states)
        round_observations = batch.observations[:, round_index, :, :]
        for lane_index in range(lane_count):
            entity_indices = batch.entity_order_by_round_slot_lane[
                :, round_index, :, lane_index
            ]
            update_inputs = torch.gather(
                round_observations,
                1,
                entity_indices.unsqueeze(-1).expand(
                    batch_size,
                    batch.width,
                    GATE2_OBSERVATION_WIDTH,
                ),
            )
            states = model.update(
                update_inputs.reshape(batch_size * batch.width, GATE2_OBSERVATION_WIDTH),
                states.reshape(batch_size * batch.width, model.config.state_width),
            ).reshape(batch_size, batch.width, model.config.state_width)

    logits = _readout_from_final_target_slot(model, batch, states)
    telemetry = Gate2ScheduleTelemetry(
        schedule="parallel_persistent",
        entity_count=batch.entity_count,
        width=batch.width,
        rounds=GATE2_TOTAL_ROUNDS,
        learned_updates_per_sample=batch.learned_updates_per_sample,
        peak_simultaneous_updates_per_sample=batch.width,
        persistent_state_vectors_per_sample=batch.width,
        collision_load=batch.collision_load,
    )
    telemetry.validate()
    return Gate2ScheduleOutput(logits=logits, final_states=states, telemetry=telemetry)



def serial_persistent_forward(
    model: Gate2PersistentStateModel,
    batch: Gate2TensorBatch,
) -> Gate2ScheduleOutput:
    """Time-multiplex the exact same persistent state bank one slot at a time."""

    batch.validate()
    batch_size = int(batch.observations.shape[0])
    states = batch.observations.new_zeros(batch_size, batch.width, model.config.state_width)
    lane_count = batch.collision_load
    batch_indices = torch.arange(batch_size, device=batch.observations.device)

    for round_index in range(GATE2_TOTAL_ROUNDS):
        if batch.mode is Gate2ControlMode.RESET_STATE and round_index > 0:
            states = torch.zeros_like(states)
        round_observations = batch.observations[:, round_index, :, :]
        next_slot_states: list[torch.Tensor] = []
        for slot_index in range(batch.width):
            slot_state = states[:, slot_index, :]
            for lane_index in range(lane_count):
                entity_indices = batch.entity_order_by_round_slot_lane[
                    :, round_index, slot_index, lane_index
                ]
                update_inputs = round_observations[batch_indices, entity_indices, :]
                slot_state = model.update(update_inputs, slot_state)
            next_slot_states.append(slot_state)
        states = torch.stack(next_slot_states, dim=1)

    logits = _readout_from_final_target_slot(model, batch, states)
    telemetry = Gate2ScheduleTelemetry(
        schedule="serial_persistent",
        entity_count=batch.entity_count,
        width=batch.width,
        rounds=GATE2_TOTAL_ROUNDS,
        learned_updates_per_sample=batch.learned_updates_per_sample,
        peak_simultaneous_updates_per_sample=1,
        persistent_state_vectors_per_sample=batch.width,
        collision_load=batch.collision_load,
    )
    telemetry.validate()
    return Gate2ScheduleOutput(logits=logits, final_states=states, telemetry=telemetry)



def _readout_from_final_target_slot(
    model: Gate2PersistentStateModel,
    batch: Gate2TensorBatch,
    states: torch.Tensor,
) -> torch.Tensor:
    batch_size = int(states.shape[0])
    batch_indices = torch.arange(batch_size, device=states.device)
    target_slots = batch.slot_by_round_entity[
        batch_indices,
        GATE2_TOTAL_ROUNDS - 1,
        batch.query_entity_indices,
    ]
    target_states = states[batch_indices, target_slots, :]
    return model.readout(target_states, batch.query_bits)



def _build_entity_order_by_round_slot_lane(
    routing: torch.Tensor,
    *,
    width: int,
) -> torch.Tensor:
    if routing.ndim != 3:
        raise ValueError("routing must have shape [batch, rounds, entities]")
    batch_size, rounds, entity_count = routing.shape
    if rounds != GATE2_TOTAL_ROUNDS:
        raise ValueError("routing must contain the frozen eight rounds")
    if entity_count % width != 0:
        raise ValueError("width must divide entity_count")
    lane_count = entity_count // width
    result = torch.empty(
        batch_size,
        rounds,
        width,
        lane_count,
        dtype=torch.int64,
        device=routing.device,
    )
    for batch_index in range(batch_size):
        for round_index in range(rounds):
            for slot_index in range(width):
                entities = torch.nonzero(
                    routing[batch_index, round_index] == slot_index,
                    as_tuple=False,
                ).squeeze(1)
                if int(entities.numel()) != lane_count:
                    raise ValueError("routing is not exactly balanced for the frozen matrix")
                result[batch_index, round_index, slot_index] = entities
    return result
