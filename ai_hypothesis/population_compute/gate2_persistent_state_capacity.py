"""Deterministic mechanics for Gate 2 persistent-state capacity.

This module intentionally contains no training loop and no neural result.  It freezes the
benchmark mechanics that make population width an organization variable rather than an
information-coverage or learned-work variable.
"""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass
from enum import Enum

GATE2_PROTOCOL_VERSION = "gate2-persistent-state-capacity-v0"
GATE2_ENTITY_COUNTS = (16, 64, 256)
GATE2_WIDTHS = (1, 4, 16, 64, 256)
GATE2_PAYLOAD_BITS = 4
GATE2_EVIDENCE_ROUNDS = 4
GATE2_INTERFERENCE_ROUNDS = 4
GATE2_TOTAL_ROUNDS = GATE2_EVIDENCE_ROUNDS + GATE2_INTERFERENCE_ROUNDS
GATE2_KEY_BIT_WIDTH = 12
GATE2_KEY_SPACE = 1 << GATE2_KEY_BIT_WIDTH


class Gate2ControlMode(str, Enum):
    """Frozen Gate-2 capability controls."""

    STABLE_PERSISTENT = "stable_persistent"
    RESHUFFLED_LOCALITY = "reshuffled_locality"
    RESET_STATE = "reset_state"


@dataclass(frozen=True, slots=True)
class Gate2Observation:
    """One entity observation in the width-independent world stream."""

    round_index: int
    entity_index: int
    entity_key: int
    evidence_bit_index: int | None
    evidence_bit_value: int | None
    interference_token: int | None

    @property
    def is_evidence(self) -> bool:
        return self.round_index < GATE2_EVIDENCE_ROUNDS

    def validate(self, *, entity_count: int) -> None:
        if not 0 <= self.round_index < GATE2_TOTAL_ROUNDS:
            raise ValueError("round_index is outside the frozen Gate-2 round range")
        if not 0 <= self.entity_index < entity_count:
            raise ValueError("entity_index is outside the world")
        if not 0 <= self.entity_key < GATE2_KEY_SPACE:
            raise ValueError("entity_key is outside the fixed key encoding range")

        if self.is_evidence:
            if self.evidence_bit_index != self.round_index:
                raise ValueError("evidence round must reveal its matching payload bit")
            if self.evidence_bit_value not in {0, 1}:
                raise ValueError("evidence bit must be binary")
            if self.interference_token is not None:
                raise ValueError("evidence observations must not carry interference tokens")
        else:
            if self.evidence_bit_index is not None or self.evidence_bit_value is not None:
                raise ValueError("interference observations must not reveal payload bits")
            if self.interference_token is None or self.interference_token < 0:
                raise ValueError("interference observations require a non-negative token")

    def semantic_signature(self) -> tuple[int, int, int, int | None, int | None, int | None]:
        """Stable signature used to prove width/control observation identity."""

        return (
            self.round_index,
            self.entity_index,
            self.entity_key,
            self.evidence_bit_index,
            self.evidence_bit_value,
            self.interference_token,
        )


@dataclass(frozen=True, slots=True)
class Gate2World:
    """One delayed-keyed-traces world before runtime-width organization is applied."""

    seed: int
    entity_count: int
    entity_keys: tuple[int, ...]
    payloads: tuple[int, ...]
    stable_entity_order: tuple[int, ...]
    query_entity_index: int
    observations: tuple[Gate2Observation, ...]

    def validate(self) -> None:
        if self.entity_count not in GATE2_ENTITY_COUNTS:
            raise ValueError("entity_count is outside the frozen Gate-2 ladder")
        if len(self.entity_keys) != self.entity_count:
            raise ValueError("entity_keys length does not match entity_count")
        if len(set(self.entity_keys)) != self.entity_count:
            raise ValueError("entity keys must be unique within a world")
        if any(key < 0 or key >= GATE2_KEY_SPACE for key in self.entity_keys):
            raise ValueError("entity key is outside the fixed key space")
        if len(self.payloads) != self.entity_count:
            raise ValueError("payload count does not match entity_count")
        if any(payload < 0 or payload >= (1 << GATE2_PAYLOAD_BITS) for payload in self.payloads):
            raise ValueError("payload is outside the frozen 4-bit answer space")
        if tuple(sorted(self.stable_entity_order)) != tuple(range(self.entity_count)):
            raise ValueError("stable_entity_order must be a permutation of entity identities")
        if not 0 <= self.query_entity_index < self.entity_count:
            raise ValueError("query_entity_index is outside the world")
        if len(self.observations) != GATE2_TOTAL_ROUNDS * self.entity_count:
            raise ValueError("world must contain exactly eight observations per entity")

        cursor = 0
        for round_index in range(GATE2_TOTAL_ROUNDS):
            for entity_index in range(self.entity_count):
                observation = self.observations[cursor]
                cursor += 1
                observation.validate(entity_count=self.entity_count)
                if observation.round_index != round_index:
                    raise ValueError("observations must be ordered round-major")
                if observation.entity_index != entity_index:
                    raise ValueError("observations must preserve canonical entity order")
                if observation.entity_key != self.entity_keys[entity_index]:
                    raise ValueError("observation key does not match world entity key")
                if round_index < GATE2_EVIDENCE_ROUNDS:
                    expected = (self.payloads[entity_index] >> round_index) & 1
                    if observation.evidence_bit_value != expected:
                        raise ValueError("evidence observation disagrees with entity payload")

    @property
    def query_key(self) -> int:
        return self.entity_keys[self.query_entity_index]

    @property
    def answer_payload(self) -> int:
        return self.payloads[self.query_entity_index]

    @property
    def learned_update_count(self) -> int:
        return GATE2_TOTAL_ROUNDS * self.entity_count

    def observation_signature(self) -> tuple[tuple[int, int, int, int | None, int | None, int | None], ...]:
        return tuple(observation.semantic_signature() for observation in self.observations)


@dataclass(frozen=True, slots=True)
class Gate2ConditionPlan:
    """Width/control-specific runtime organization over one immutable Gate-2 world."""

    protocol_version: str
    world_seed: int
    entity_count: int
    width: int
    mode: Gate2ControlMode
    slot_by_round_entity: tuple[tuple[int, ...], ...]
    reset_state_each_round: bool
    learned_update_count: int
    inspected_entity_count: int
    observation_count: int
    observation_signature: tuple[
        tuple[int, int, int, int | None, int | None, int | None], ...
    ]

    def validate(self) -> None:
        if self.protocol_version != GATE2_PROTOCOL_VERSION:
            raise ValueError("unexpected Gate-2 protocol version")
        if self.entity_count not in GATE2_ENTITY_COUNTS:
            raise ValueError("entity_count is outside the frozen Gate-2 ladder")
        if self.width not in GATE2_WIDTHS or self.width > self.entity_count:
            raise ValueError("width is outside the frozen Gate-2 matrix")
        if len(self.slot_by_round_entity) != GATE2_TOTAL_ROUNDS:
            raise ValueError("routing must contain exactly eight rounds")
        for round_slots in self.slot_by_round_entity:
            if len(round_slots) != self.entity_count:
                raise ValueError("every routing round must cover every entity")
            if any(slot < 0 or slot >= self.width for slot in round_slots):
                raise ValueError("routing produced a slot outside the active population")
            loads = [0] * self.width
            for slot in round_slots:
                loads[slot] += 1
            if max(loads) - min(loads) > 1:
                raise ValueError("Gate-2 routing must remain balanced")

        expected_updates = GATE2_TOTAL_ROUNDS * self.entity_count
        if self.learned_update_count != expected_updates:
            raise ValueError("Gate-2 learned-work identity was violated")
        if self.inspected_entity_count != self.entity_count:
            raise ValueError("Gate-2 source coverage must remain complete")
        if self.observation_count != expected_updates:
            raise ValueError("Gate-2 observation count must equal learned update count")
        if len(self.observation_signature) != expected_updates:
            raise ValueError("Gate-2 observation signature is incomplete")
        if self.reset_state_each_round != (self.mode is Gate2ControlMode.RESET_STATE):
            raise ValueError("reset-state flag does not match the selected control")

        if self.mode in {Gate2ControlMode.STABLE_PERSISTENT, Gate2ControlMode.RESET_STATE}:
            if any(round_slots != self.slot_by_round_entity[0] for round_slots in self.slot_by_round_entity[1:]):
                raise ValueError("stable/reset routing must remain fixed across all rounds")

    @property
    def source_coverage(self) -> float:
        return self.inspected_entity_count / self.entity_count

    def slot_loads(self, round_index: int) -> tuple[int, ...]:
        round_slots = self.slot_by_round_entity[round_index]
        return tuple(round_slots.count(slot) for slot in range(self.width))

    def target_slot_load(self, world: Gate2World, round_index: int) -> int:
        if world.seed != self.world_seed or world.entity_count != self.entity_count:
            raise ValueError("condition plan does not belong to the supplied world")
        target_slot = self.slot_by_round_entity[round_index][world.query_entity_index]
        return self.slot_loads(round_index)[target_slot]



def generate_gate2_world(*, seed: int, entity_count: int) -> Gate2World:
    """Generate one deterministic world with domain-separated procedural randomness."""

    if entity_count not in GATE2_ENTITY_COUNTS:
        raise ValueError("entity_count is outside the frozen Gate-2 ladder")

    key_rng = _domain_rng(seed, "keys")
    payload_rng = _domain_rng(seed, "payloads")
    routing_rng = _domain_rng(seed, "stable-routing")
    query_rng = _domain_rng(seed, "query")

    entity_keys = tuple(key_rng.sample(range(GATE2_KEY_SPACE), entity_count))
    payloads = tuple(payload_rng.randrange(1 << GATE2_PAYLOAD_BITS) for _ in range(entity_count))
    stable_order = list(range(entity_count))
    routing_rng.shuffle(stable_order)
    query_entity_index = query_rng.randrange(entity_count)

    observations: list[Gate2Observation] = []
    for round_index in range(GATE2_TOTAL_ROUNDS):
        for entity_index, entity_key in enumerate(entity_keys):
            if round_index < GATE2_EVIDENCE_ROUNDS:
                bit_value = (payloads[entity_index] >> round_index) & 1
                observations.append(
                    Gate2Observation(
                        round_index=round_index,
                        entity_index=entity_index,
                        entity_key=entity_key,
                        evidence_bit_index=round_index,
                        evidence_bit_value=bit_value,
                        interference_token=None,
                    )
                )
            else:
                # The token is generated from its own domain and never from payload state.  It
                # exists only to make the later update non-empty while carrying no payload bit.
                token_rng = _domain_rng(seed, f"interference:{round_index}:{entity_index}")
                observations.append(
                    Gate2Observation(
                        round_index=round_index,
                        entity_index=entity_index,
                        entity_key=entity_key,
                        evidence_bit_index=None,
                        evidence_bit_value=None,
                        interference_token=token_rng.randrange(1 << 16),
                    )
                )

    world = Gate2World(
        seed=seed,
        entity_count=entity_count,
        entity_keys=entity_keys,
        payloads=payloads,
        stable_entity_order=tuple(stable_order),
        query_entity_index=query_entity_index,
        observations=tuple(observations),
    )
    world.validate()
    return world



def gate2_population_widths(entity_count: int) -> tuple[int, ...]:
    if entity_count not in GATE2_ENTITY_COUNTS:
        raise ValueError("entity_count is outside the frozen Gate-2 ladder")
    return tuple(width for width in GATE2_WIDTHS if width <= entity_count)



def build_gate2_condition_plan(
    world: Gate2World,
    *,
    width: int,
    mode: Gate2ControlMode,
) -> Gate2ConditionPlan:
    """Apply one frozen runtime organization without changing world information/work."""

    world.validate()
    if width not in gate2_population_widths(world.entity_count):
        raise ValueError("width is outside the frozen matrix for this entity count")
    if not isinstance(mode, Gate2ControlMode):
        raise TypeError("mode must be a Gate2ControlMode")

    if mode is Gate2ControlMode.RESHUFFLED_LOCALITY:
        routing = tuple(
            _reshuffled_slots(world, width=width, round_index=round_index)
            for round_index in range(GATE2_TOTAL_ROUNDS)
        )
    else:
        stable_slots = _stable_slots(world, width=width)
        routing = tuple(stable_slots for _ in range(GATE2_TOTAL_ROUNDS))

    plan = Gate2ConditionPlan(
        protocol_version=GATE2_PROTOCOL_VERSION,
        world_seed=world.seed,
        entity_count=world.entity_count,
        width=width,
        mode=mode,
        slot_by_round_entity=routing,
        reset_state_each_round=(mode is Gate2ControlMode.RESET_STATE),
        learned_update_count=world.learned_update_count,
        inspected_entity_count=world.entity_count,
        observation_count=len(world.observations),
        observation_signature=world.observation_signature(),
    )
    plan.validate()
    return plan



def build_gate2_condition_matrix(world: Gate2World) -> tuple[Gate2ConditionPlan, ...]:
    return tuple(
        build_gate2_condition_plan(world, width=width, mode=mode)
        for width in gate2_population_widths(world.entity_count)
        for mode in Gate2ControlMode
    )



def _stable_slots(world: Gate2World, *, width: int) -> tuple[int, ...]:
    rank_by_entity = [0] * world.entity_count
    for rank, entity_index in enumerate(world.stable_entity_order):
        rank_by_entity[entity_index] = rank
    return tuple(rank % width for rank in rank_by_entity)



def _reshuffled_slots(
    world: Gate2World,
    *,
    width: int,
    round_index: int,
) -> tuple[int, ...]:
    order = list(range(world.entity_count))
    _domain_rng(world.seed, f"reshuffled-routing:{round_index}").shuffle(order)
    rank_by_entity = [0] * world.entity_count
    for rank, entity_index in enumerate(order):
        rank_by_entity[entity_index] = rank
    return tuple(rank % width for rank in rank_by_entity)



def _domain_rng(seed: int, domain: str) -> random.Random:
    digest = hashlib.sha256(f"{seed}:{domain}".encode("utf-8")).digest()
    return random.Random(int.from_bytes(digest[:16], "big", signed=False))
