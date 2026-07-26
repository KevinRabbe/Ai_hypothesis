"""Deterministic worlds for the fixed-parameter population-compute benchmark.

collective-relay-v1 removes two shortcuts found by the first development run of v0:
pre-threshold prefixes could expose the final answer-bearing target edge, and a small
complete prefix could contain the only complete chain. V1 instead builds many
structurally identical disjoint chains. The query selects one target chain; every
first-complete prefix contains the complete target chain plus at least one complete
distractor chain, while the target's final edge is outside the previous frozen
population point.
"""

from __future__ import annotations

from dataclasses import dataclass
from random import Random

from .contract import DEVELOPMENT_POPULATION_SIZES


COLLECTIVE_RELAY_VERSION = "collective-relay-v1"
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
        if self.world_size % self.hop_count != 0:
            raise ValueError("world_size must be divisible by hop_count")
        if self.world_size < self.hop_count * 2:
            raise ValueError("relay world must contain at least two complete chains")
        chain_count = self.world_size // self.hop_count
        required_unique_nodes = chain_count * (self.hop_count + 1)
        if self.node_space - 1 < required_unique_nodes:
            raise ValueError("node_space is too small for collision-free chain generation")


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
    chain_id: int
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

        chain_count = self.difficulty.world_size // self.difficulty.hop_count
        expected_chain_ids = set(range(chain_count))
        actual_chain_ids = {record.chain_id for record in self.records}
        if actual_chain_ids != expected_chain_ids:
            raise ValueError("relay chain IDs must cover the complete chain set")

        for chain_id in range(chain_count):
            chain_records = tuple(
                record for record in self.records if record.chain_id == chain_id
            )
            if len(chain_records) != self.difficulty.hop_count:
                raise ValueError("every relay chain must contain exactly hop_count edges")
            start, terminal = _chain_endpoints(chain_records, self.difficulty.hop_count)
            if chain_id == 0:
                if start != self.start_key or terminal != self.answer_key:
                    raise ValueError("target chain endpoints do not match declared query/answer")
                if not all(record.is_chain_edge for record in chain_records):
                    raise ValueError("target chain records must be marked as chain edges")
            elif any(record.is_chain_edge for record in chain_records):
                raise ValueError("distractor chains cannot be marked as target-chain edges")

        allowed_thresholds = relay_scope_thresholds(self.difficulty)
        if self.scope_threshold not in allowed_thresholds:
            raise ValueError("scope_threshold is not an admissible relay population point")
        if not information_complete_at(self, self.scope_threshold):
            raise ValueError("scope_threshold must contain the complete target chain")

        population_points = relay_population_points(self.difficulty)
        previous_population = max(
            size for size in population_points if size < self.scope_threshold
        )
        if information_complete_at(self, previous_population):
            raise ValueError("target chain becomes complete before its declared scope_threshold")

        final_target_edges = tuple(
            record
            for record in self.records
            if record.is_chain_edge and record.value == self.answer_key
        )
        if len(final_target_edges) != 1:
            raise ValueError("target chain must contain exactly one final answer edge")
        if final_target_edges[0].worker_slot < previous_population:
            raise ValueError("final target edge leaked before the previous population point")

        previous_records = self.records[:previous_population]
        if any(
            record.key == self.answer_key or record.value == self.answer_key
            for record in previous_records
        ):
            raise ValueError("answer identity leaked into the previous population prefix")

        complete_at_threshold = complete_chain_ids_at(self, self.scope_threshold)
        if 0 not in complete_at_threshold:
            raise ValueError("target chain is not complete at scope_threshold")
        if len(complete_at_threshold) < 2:
            raise ValueError("scope_threshold must contain a complete distractor chain")
        if len(complete_at_threshold) * self.difficulty.hop_count != self.scope_threshold:
            raise ValueError(
                "first-complete prefix must consist only of whole, structurally matched chains"
            )

        if resolve_relay(self) != self.answer_key:
            raise ValueError("relay world does not resolve to its declared answer")


def relay_population_points(difficulty: RelayDifficulty) -> tuple[int, ...]:
    """Return the nested population ladder for this world size."""

    difficulty.validate()
    points = {
        size for size in DEVELOPMENT_POPULATION_SIZES if size < difficulty.world_size
    }
    points.add(difficulty.world_size)
    points.add(1)
    return tuple(sorted(points))


def relay_scope_thresholds(difficulty: RelayDifficulty) -> tuple[int, ...]:
    """Frozen population points that can hold target + complete distractor chains.

    V1 requires at least two h-edge chains inside every first-complete prefix. That
    removes the v0 shortcut where a minimum complete prefix could consist solely of the
    target chain and expose its terminal node structurally without following the query.
    """

    difficulty.validate()
    minimum_scope = difficulty.hop_count * 2
    thresholds = tuple(
        size
        for size in relay_population_points(difficulty)
        if size >= minimum_scope and size % difficulty.hop_count == 0
    )
    if not thresholds or thresholds[-1] != difficulty.world_size:
        raise ValueError("population ladder cannot cover relay difficulty")
    return thresholds


def information_complete_at(world: RelayWorld, active_workers: int) -> bool:
    """Whether the active nested prefix contains every target-chain edge."""

    if not 1 <= active_workers <= world.difficulty.world_size:
        raise ValueError("active_workers is outside the relay world")
    return all(
        (not record.is_chain_edge) or record.worker_slot < active_workers
        for record in world.records
    )


def complete_chain_ids_at(world: RelayWorld, active_workers: int) -> tuple[int, ...]:
    """Return chain IDs whose complete h-edge structure is visible in the prefix."""

    if not 1 <= active_workers <= world.difficulty.world_size:
        raise ValueError("active_workers is outside the relay world")
    counts: dict[int, int] = {}
    for record in world.records[:active_workers]:
        counts[record.chain_id] = counts.get(record.chain_id, 0) + 1
    return tuple(
        sorted(
            chain_id
            for chain_id, count in counts.items()
            if count == world.difficulty.hop_count
        )
    )


def generate_relay_world(seed: int, difficulty: RelayDifficulty) -> RelayWorld:
    """Generate one shortcut-resistant multi-hop relay world.

    The world is a collection of disjoint h-edge chains with identical local structure.
    Chain 0 is selected by the query and every other chain is a distractor. At the
    declared first-complete population point, the prefix contains the full target chain
    plus one or more complete distractor chains. The final target edge is deliberately
    placed beyond the previous frozen population point, so earlier points neither have
    the full path nor contain the answer identity.
    """

    if seed < 0:
        raise ValueError("seed must be non-negative")
    difficulty.validate()
    rng = Random(seed)

    chain_count = difficulty.world_size // difficulty.hop_count
    nodes_per_chain = difficulty.hop_count + 1
    required_unique = chain_count * nodes_per_chain
    candidates = rng.sample(range(1, difficulty.node_space), required_unique)

    chains: list[tuple[int, ...]] = []
    for chain_id in range(chain_count):
        start = chain_id * nodes_per_chain
        chains.append(tuple(candidates[start : start + nodes_per_chain]))

    target_nodes = chains[0]
    target_pairs = tuple(zip(target_nodes[:-1], target_nodes[1:], strict=True))
    distractor_pairs: dict[int, tuple[tuple[int, int], ...]] = {
        chain_id: tuple(zip(nodes[:-1], nodes[1:], strict=True))
        for chain_id, nodes in enumerate(chains)
        if chain_id != 0
    }

    population_points = relay_population_points(difficulty)
    thresholds = relay_scope_thresholds(difficulty)
    scope_threshold = thresholds[seed % len(thresholds)]
    previous_population = max(
        size for size in population_points if size < scope_threshold
    )

    complete_chain_count = scope_threshold // difficulty.hop_count
    selected_distractor_ids = rng.sample(
        list(distractor_pairs),
        complete_chain_count - 1,
    )

    # The final target edge is the only record containing answer_key. Pin it beyond the
    # previous frozen population point so incomplete prefixes cannot solve by copying a
    # visible terminal value.
    frontier_slot = rng.randrange(previous_population, scope_threshold)
    records_by_slot: list[tuple[int, int, int, bool] | None] = [
        None
    ] * difficulty.world_size
    final_key, final_value = target_pairs[-1]
    records_by_slot[frontier_slot] = (final_key, final_value, 0, True)

    prefix_specs: list[tuple[int, int, int, bool]] = [
        (key, value, 0, True) for key, value in target_pairs[:-1]
    ]
    for chain_id in selected_distractor_ids:
        prefix_specs.extend(
            (key, value, chain_id, False)
            for key, value in distractor_pairs[chain_id]
        )
    if len(prefix_specs) != scope_threshold - 1:
        raise AssertionError("relay v1 prefix specification has the wrong size")
    rng.shuffle(prefix_specs)
    prefix_slots = [
        slot for slot in range(scope_threshold) if slot != frontier_slot
    ]
    for slot, spec in zip(prefix_slots, prefix_specs, strict=True):
        records_by_slot[slot] = spec

    remaining_specs: list[tuple[int, int, int, bool]] = []
    selected = set(selected_distractor_ids)
    for chain_id, pairs in distractor_pairs.items():
        if chain_id in selected:
            continue
        remaining_specs.extend(
            (key, value, chain_id, False) for key, value in pairs
        )
    rng.shuffle(remaining_specs)
    remaining_slots = list(range(scope_threshold, difficulty.world_size))
    if len(remaining_specs) != len(remaining_slots):
        raise AssertionError("relay v1 suffix specification has the wrong size")
    for slot, spec in zip(remaining_slots, remaining_specs, strict=True):
        records_by_slot[slot] = spec

    if any(record is None for record in records_by_slot):
        raise AssertionError("relay generator did not fill every worker slot")

    records = tuple(
        RelayRecord(
            worker_slot=slot,
            key=record[0],
            value=record[1],
            chain_id=record[2],
            is_chain_edge=record[3],
        )
        for slot, record in enumerate(records_by_slot)
        if record is not None
    )

    world = RelayWorld(
        version=COLLECTIVE_RELAY_VERSION,
        seed=seed,
        difficulty=difficulty,
        start_key=target_nodes[0],
        answer_key=target_nodes[-1],
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


def _chain_endpoints(
    records: tuple[RelayRecord, ...],
    hop_count: int,
) -> tuple[int, int]:
    mapping = {record.key: record.value for record in records}
    keys = set(mapping)
    values = set(mapping.values())
    starts = keys - values
    terminals = values - keys
    if len(starts) != 1 or len(terminals) != 1:
        raise ValueError("relay chain must have exactly one start and terminal")
    current = next(iter(starts))
    visited_keys: set[int] = set()
    for _ in range(hop_count):
        if current not in mapping:
            raise ValueError("relay chain terminates before hop_count")
        visited_keys.add(current)
        current = mapping[current]
    if len(visited_keys) != hop_count or current != next(iter(terminals)):
        raise ValueError("relay chain does not form one simple h-edge path")
    return next(iter(starts)), current
