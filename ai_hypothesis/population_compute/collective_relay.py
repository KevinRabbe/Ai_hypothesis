"""Deterministic worlds for the first population-compute benchmark family.

The world generator is intentionally model-agnostic. It freezes the information
structure before a neural encoding or population update cell is tuned.
"""

from __future__ import annotations

from dataclasses import dataclass
from random import Random

from .contract import DEVELOPMENT_POPULATION_SIZES


COLLECTIVE_RELAY_VERSION = "collective-relay-v0"
RELAY_WORLD_SIZE = 256


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
    RelayDifficulty(name="relay-2", world_size=RELAY_WORLD_SIZE, hop_count=2),
    RelayDifficulty(name="relay-4", world_size=RELAY_WORLD_SIZE, hop_count=4),
    RelayDifficulty(name="relay-8", world_size=RELAY_WORLD_SIZE, hop_count=8),
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
    scope_threshold: int
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

        chain_records = tuple(record for record in self.records if record.is_chain_edge)
        if len(chain_records) != self.difficulty.hop_count:
            raise ValueError("chain edge count does not match difficulty hop_count")

        allowed_thresholds = relay_scope_thresholds(self.difficulty)
        if self.scope_threshold not in allowed_thresholds:
            raise ValueError("scope_threshold is not a frozen relay population point")
        if not information_complete_at(self, self.scope_threshold):
            raise ValueError("scope_threshold must contain the complete relay chain")
        previous_points = tuple(
            size for size in DEVELOPMENT_POPULATION_SIZES if size < self.scope_threshold
        )
        if previous_points and information_complete_at(self, previous_points[-1]):
            raise ValueError("relay chain becomes complete before its declared scope_threshold")

        if resolve_relay(self) != self.answer_key:
            raise ValueError("relay world does not resolve to its declared answer")


def relay_scope_thresholds(difficulty: RelayDifficulty) -> tuple[int, ...]:
    """Population points at which this hop count can first become information-complete.

    One worker holds one key/value record, so a h-hop chain cannot be complete below h
    active records. Population size 1 is intentionally never complete for this multi-hop
    benchmark. The remaining frozen curve points are used as balanced scope thresholds.
    """

    difficulty.validate()
    thresholds = tuple(
        size
        for size in DEVELOPMENT_POPULATION_SIZES
        if size > 1 and size >= difficulty.hop_count and size <= difficulty.world_size
    )
    if not thresholds or thresholds[-1] != difficulty.world_size:
        raise ValueError("development population curve cannot cover relay difficulty")
    return thresholds


def information_complete_at(world: RelayWorld, active_workers: int) -> bool:
    """Whether the active nested prefix contains every required chain edge."""

    if not 1 <= active_workers <= world.difficulty.world_size:
        raise ValueError("active_workers is outside the relay world")
    return all(
        (not record.is_chain_edge) or record.worker_slot < active_workers
        for record in world.records
    )


def generate_relay_world(seed: int, difficulty: RelayDifficulty) -> RelayWorld:
    """Generate one collision-free multi-hop world with controlled scope availability.

    Consecutive seeds cycle over the admissible population thresholds. Required chain
    edges are placed so the chain is incomplete at the previous frozen population point
    and complete at the declared threshold. This prevents the 1/4/16/64/256 benchmark
    from degenerating into a nearly all-or-nothing information jump at 256 workers.
    """

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

    chain_pairs = list(zip(chain_nodes[:-1], chain_nodes[1:], strict=True))
    decoy_count = difficulty.world_size - difficulty.hop_count
    decoy_keys = candidates[cursor : cursor + decoy_count]
    cursor += decoy_count
    decoy_values = candidates[cursor : cursor + decoy_count]

    if chain_keys.intersection(decoy_keys):
        raise AssertionError("generator produced a chain/decoy key collision")

    thresholds = relay_scope_thresholds(difficulty)
    scope_threshold = thresholds[seed % len(thresholds)]
    previous_population = max(
        size for size in DEVELOPMENT_POPULATION_SIZES if size < scope_threshold
    )

    # At least one required edge lives beyond the previous frozen population point,
    # while all required edges remain inside the selected threshold.
    frontier_slot = rng.randrange(previous_population, scope_threshold)
    remaining_chain_slots = [
        slot for slot in range(scope_threshold) if slot != frontier_slot
    ]
    chain_slots = [
        frontier_slot,
        *rng.sample(remaining_chain_slots, difficulty.hop_count - 1),
    ]
    rng.shuffle(chain_slots)
    rng.shuffle(chain_pairs)

    records_by_slot: list[tuple[int, int, bool] | None] = [None] * difficulty.world_size
    for slot, (key, value) in zip(chain_slots, chain_pairs, strict=True):
        records_by_slot[slot] = (key, value, True)

    decoy_pairs = list(zip(decoy_keys, decoy_values, strict=True))
    rng.shuffle(decoy_pairs)
    decoy_slots = [
        slot for slot, record in enumerate(records_by_slot) if record is None
    ]
    for slot, (key, value) in zip(decoy_slots, decoy_pairs, strict=True):
        records_by_slot[slot] = (key, value, False)

    if any(record is None for record in records_by_slot):
        raise AssertionError("relay generator did not fill every worker slot")

    records = tuple(
        RelayRecord(
            worker_slot=slot,
            key=record[0],
            value=record[1],
            is_chain_edge=record[2],
        )
        for slot, record in enumerate(records_by_slot)
        if record is not None
    )

    world = RelayWorld(
        version=COLLECTIVE_RELAY_VERSION,
        seed=seed,
        difficulty=difficulty,
        start_key=chain_nodes[0],
        answer_key=chain_nodes[-1],
        scope_threshold=scope_threshold,
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
