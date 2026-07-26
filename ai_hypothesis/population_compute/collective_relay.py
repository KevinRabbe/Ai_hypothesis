"""Deterministic worlds for the first population-compute benchmark family.

The world generator is intentionally model-agnostic. It freezes the information
structure before a neural encoding or population update cell is tuned.
"""

from __future__ import annotations

from dataclasses import dataclass
from random import Random


COLLECTIVE_RELAY_VERSION = "collective-relay-v0"


@dataclass(frozen=True, slots=True)
class RelayDifficulty:
    name: str
    world_size: int
    hop_count: int
    node_space: int = 4096

    def validate(self) -> None:
        if not self.name.strip():
            raise ValueError("difficulty name must be non-empty")
        if self.world_size <= 0:
            raise ValueError("world_size must be positive")
        if self.hop_count < 2:
            raise ValueError("hop_count must be at least 2 for a collective relay task")
        if self.hop_count > self.world_size:
            raise ValueError("hop_count cannot exceed world_size")
        if self.node_space < (self.world_size * 2 + self.hop_count + 1):
            raise ValueError("node_space is too small for collision-free generation")


RELAY_DIFFICULTIES: tuple[RelayDifficulty, ...] = (
    RelayDifficulty(name="relay-2", world_size=32, hop_count=2),
    RelayDifficulty(name="relay-4", world_size=64, hop_count=4),
    RelayDifficulty(name="relay-8", world_size=128, hop_count=8),
)


@dataclass(frozen=True, slots=True)
class RelayRecord:
    worker_slot: int
    key: int
    value: int
    is_chain_edge: bool


@dataclass(frozen=True, slots=True)
class RelayWorld:
    version: str
    seed: int
    difficulty: RelayDifficulty
    start_key: int
    answer_key: int
    records: tuple[RelayRecord, ...]

    def validate(self) -> None:
        if self.version != COLLECTIVE_RELAY_VERSION:
            raise ValueError("unexpected relay benchmark version")
        if self.seed < 0:
            raise ValueError("seed must be non-negative")
        self.difficulty.validate()
        if len(self.records) != self.difficulty.world_size:
            raise ValueError("record count must equal difficulty world_size")

        slots = tuple(record.worker_slot for record in self.records)
        if tuple(sorted(slots)) != tuple(range(self.difficulty.world_size)):
            raise ValueError("worker slots must be unique and cover the whole world")

        keys = tuple(record.key for record in self.records)
        if len(set(keys)) != len(keys):
            raise ValueError("relay record keys must be unique")

        chain_edges = sum(record.is_chain_edge for record in self.records)
        if chain_edges != self.difficulty.hop_count:
            raise ValueError("chain edge count does not match difficulty hop_count")

        if resolve_relay(self) != self.answer_key:
            raise ValueError("relay world does not resolve to its declared answer")


def generate_relay_world(seed: int, difficulty: RelayDifficulty) -> RelayWorld:
    """Generate one collision-free multi-hop world with shuffled local records."""

    if seed < 0:
        raise ValueError("seed must be non-negative")
    difficulty.validate()
    rng = Random(seed)

    required_unique = difficulty.world_size * 2 + difficulty.hop_count + 1
    candidates = rng.sample(range(1, difficulty.node_space), required_unique)
    cursor = 0

    chain_nodes = candidates[cursor : cursor + difficulty.hop_count + 1]
    cursor += difficulty.hop_count + 1
    chain_keys = set(chain_nodes[:-1])

    raw_records: list[tuple[int, int, bool]] = []
    for key, value in zip(chain_nodes, chain_nodes[1:], strict=True):
        raw_records.append((key, value, True))

    decoy_count = difficulty.world_size - difficulty.hop_count
    decoy_keys = candidates[cursor : cursor + decoy_count]
    cursor += decoy_count
    decoy_values = candidates[cursor : cursor + decoy_count]

    if chain_keys.intersection(decoy_keys):
        raise AssertionError("generator produced a chain/decoy key collision")

    raw_records.extend(
        (key, value, False)
        for key, value in zip(decoy_keys, decoy_values, strict=True)
    )
    rng.shuffle(raw_records)

    records = tuple(
        RelayRecord(
            worker_slot=slot,
            key=key,
            value=value,
            is_chain_edge=is_chain_edge,
        )
        for slot, (key, value, is_chain_edge) in enumerate(raw_records)
    )

    world = RelayWorld(
        version=COLLECTIVE_RELAY_VERSION,
        seed=seed,
        difficulty=difficulty,
        start_key=chain_nodes[0],
        answer_key=chain_nodes[-1],
        records=records,
    )
    world.validate()
    return world


def generate_relay_dataset(
    *,
    start_seed: int,
    world_count: int,
    difficulty: RelayDifficulty,
) -> tuple[RelayWorld, ...]:
    if start_seed < 0:
        raise ValueError("start_seed must be non-negative")
    if world_count <= 0:
        raise ValueError("world_count must be positive")
    return tuple(
        generate_relay_world(start_seed + offset, difficulty)
        for offset in range(world_count)
    )


def resolve_relay(world: RelayWorld) -> int:
    """Oracle resolver used only to validate/generated labels, never as a model path."""

    mapping = {record.key: record.value for record in world.records}
    current = world.start_key
    for _ in range(world.difficulty.hop_count):
        try:
            current = mapping[current]
        except KeyError as exc:
            raise ValueError("relay chain is incomplete") from exc
    return current
