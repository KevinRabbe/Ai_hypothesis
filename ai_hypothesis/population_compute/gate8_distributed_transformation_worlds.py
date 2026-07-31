"""Gate-8 distributed-transformation world generator and exact symbolic oracle.

This stage admits deterministic benchmark mechanics only. It contains no neural
model, training loop, tokenizer, prompt, 1B weights, or benchmark execution.
"""

from __future__ import annotations

import hashlib
import random
import re
from dataclasses import asdict, dataclass
from typing import Any

GATE8_WORLD_CONTRACT_VERSION = "gate8-distributed-transformation-world-contract-v0"
GATE8_WORLD_CONTRACT_PROTOCOL_HEAD = "e73541115e8ddd122f336463dc1a9ffdbf82df46"
GATE8_WORLD_CONTRACT_CORRECTION_HEAD = "124065691d257d483a37be4200452f1f7ca50063"
GATE8_WORLD_CONTRACT_STATUS = "GATE8_WORLD_GENERATOR_AND_SYMBOLIC_ORACLE_ADMITTED_EXECUTION_CLOSED"

GATE8_POPULATIONS = (32, 64, 128, 256, 512, 1_024)
GATE8_DEPTHS = (4, 8, 16, 32, 64, 128)
GATE8_VALID_CONDITIONS = tuple(
    (population, depth)
    for population in GATE8_POPULATIONS
    for depth in GATE8_DEPTHS
    if 8 * depth <= population
)
GATE8_ALPHABET_SIZE = 16
GATE8_TRANSFORM_COUNT = 8
GATE8_TEST_WORLDS_PER_CONDITION = 512
GATE8_TRAINING_WORLDS_PER_SEED = 262_144
GATE8_DEMONSTRATION_WORLDS = 8
GATE8_ALLOWED_SPLITS = ("contract", "train", "validation", "test", "demonstration")
GATE8_TRAINING_SEEDS = (0, 1, 2)
GATE8_NODE_LABEL_PATTERN = re.compile(r"^n_[0-9a-f]{20}$")

GATE8_TRANSFORM_PERMUTATIONS = (
    (13, 5, 6, 10, 15, 14, 8, 2, 7, 1, 12, 9, 3, 11, 4, 0),
    (6, 12, 8, 3, 2, 14, 9, 0, 7, 4, 10, 13, 15, 11, 1, 5),
    (7, 13, 14, 5, 11, 2, 0, 6, 1, 3, 12, 15, 9, 8, 10, 4),
    (9, 13, 3, 7, 12, 6, 8, 0, 2, 5, 15, 11, 1, 10, 14, 4),
    (12, 1, 9, 10, 14, 4, 5, 6, 2, 15, 11, 7, 3, 0, 13, 8),
    (8, 6, 7, 4, 1, 3, 2, 13, 5, 12, 0, 15, 10, 9, 14, 11),
    (4, 3, 6, 14, 9, 5, 15, 0, 2, 13, 12, 8, 1, 10, 7, 11),
    (10, 14, 2, 6, 3, 7, 0, 9, 15, 4, 13, 12, 8, 5, 1, 11),
)


def _seed_from_parts(*parts: object) -> int:
    digest = hashlib.sha256(":".join(str(part) for part in parts).encode("ascii")).digest()
    return int.from_bytes(digest[:8], "big") & ((1 << 63) - 1)


def _compose(left: tuple[int, ...], right: tuple[int, ...]) -> tuple[int, ...]:
    return tuple(left[right[index]] for index in range(GATE8_ALPHABET_SIZE))


def validate_gate8_transform_library() -> None:
    expected = tuple(range(GATE8_ALPHABET_SIZE))
    if len(GATE8_TRANSFORM_PERMUTATIONS) != GATE8_TRANSFORM_COUNT:
        raise RuntimeError("Gate8 transform count changed")
    if len(set(GATE8_TRANSFORM_PERMUTATIONS)) != GATE8_TRANSFORM_COUNT:
        raise RuntimeError("Gate8 transforms are not unique")
    for transform in GATE8_TRANSFORM_PERMUTATIONS:
        if tuple(sorted(transform)) != expected:
            raise RuntimeError("Gate8 transform is not a bijection")
    for left_index in range(GATE8_TRANSFORM_COUNT):
        for right_index in range(left_index + 1, GATE8_TRANSFORM_COUNT):
            left = GATE8_TRANSFORM_PERMUTATIONS[left_index]
            right = GATE8_TRANSFORM_PERMUTATIONS[right_index]
            if _compose(left, right) == _compose(right, left):
                raise RuntimeError("Gate8 primitive transforms must be pairwise non-commuting")


def apply_gate8_transform(transform_id: int, symbol: int) -> int:
    if not 0 <= transform_id < GATE8_TRANSFORM_COUNT:
        raise ValueError("Gate8 transform id is outside 0..7")
    if not 0 <= symbol < GATE8_ALPHABET_SIZE:
        raise ValueError("Gate8 symbol is outside 0..15")
    return GATE8_TRANSFORM_PERMUTATIONS[transform_id][symbol]


@dataclass(frozen=True, slots=True)
class Gate8EdgeShard:
    worker_index: int
    source_node: str
    target_node: str
    transform_id: int

    def validate(self, population: int) -> None:
        if not 0 <= self.worker_index < population:
            raise ValueError("Gate8 worker index is outside the population")
        if not GATE8_NODE_LABEL_PATTERN.fullmatch(self.source_node):
            raise ValueError("Gate8 source node label is not opaque")
        if not GATE8_NODE_LABEL_PATTERN.fullmatch(self.target_node):
            raise ValueError("Gate8 target node label is not opaque")
        if self.source_node == self.target_node:
            raise ValueError("Gate8 self edges are forbidden")
        if not 0 <= self.transform_id < GATE8_TRANSFORM_COUNT:
            raise ValueError("Gate8 edge transform is outside 0..7")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class Gate8PublicQuery:
    root_node: str
    target_node: str
    root_symbol: int

    def validate(self) -> None:
        if not GATE8_NODE_LABEL_PATTERN.fullmatch(self.root_node):
            raise ValueError("Gate8 root node label is not opaque")
        if not GATE8_NODE_LABEL_PATTERN.fullmatch(self.target_node):
            raise ValueError("Gate8 target node label is not opaque")
        if self.root_node == self.target_node:
            raise ValueError("Gate8 root and target must differ")
        if not 0 <= self.root_symbol < GATE8_ALPHABET_SIZE:
            raise ValueError("Gate8 root symbol is outside 0..15")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return asdict(self)


@dataclass(frozen=True, slots=True)
class Gate8PublicWorld:
    world_id: str
    split: str
    seed: int
    world_index: int
    population: int
    depth: int
    query: Gate8PublicQuery
    workers: tuple[Gate8EdgeShard, ...]

    def validate(self) -> None:
        _validate_identity(self.split, self.seed, self.world_index, self.population, self.depth)
        if not re.fullmatch(r"g8_[0-9a-f]{24}", self.world_id):
            raise ValueError("Gate8 world id is malformed")
        self.query.validate()
        if len(self.workers) != self.population:
            raise ValueError("Gate8 requires exactly one worker per edge")
        if tuple(worker.worker_index for worker in self.workers) != tuple(range(self.population)):
            raise ValueError("Gate8 worker indices must be contiguous and ordered")
        for worker in self.workers:
            worker.validate(self.population)
        edge_pairs = tuple((worker.source_node, worker.target_node) for worker in self.workers)
        if len(set(edge_pairs)) != self.population:
            raise ValueError("Gate8 duplicate directed edges are forbidden")
        transform_counts = [0] * GATE8_TRANSFORM_COUNT
        for worker in self.workers:
            transform_counts[worker.transform_id] += 1
        if tuple(transform_counts) != (self.population // GATE8_TRANSFORM_COUNT,) * GATE8_TRANSFORM_COUNT:
            raise ValueError("Gate8 transform marginals must be exactly balanced")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            "world_id": self.world_id,
            "split": self.split,
            "seed": self.seed,
            "world_index": self.world_index,
            "population": self.population,
            "depth": self.depth,
            "query": self.query.to_dict(),
            "workers": [worker.to_dict() for worker in self.workers],
        }


@dataclass(frozen=True, slots=True)
class Gate8WorldTruth:
    answer_symbol: int
    relevant_worker_indices: tuple[int, ...]

    def validate(self, population: int, depth: int) -> None:
        if not 0 <= self.answer_symbol < GATE8_ALPHABET_SIZE:
            raise ValueError("Gate8 answer symbol is outside 0..15")
        if len(self.relevant_worker_indices) != depth:
            raise ValueError("Gate8 truth path length changed")
        if len(set(self.relevant_worker_indices)) != depth:
            raise ValueError("Gate8 truth path repeats a worker")
        if any(not 0 <= index < population for index in self.relevant_worker_indices):
            raise ValueError("Gate8 truth worker index is outside the population")
        ordered = sorted(self.relevant_worker_indices)
        if ordered == list(range(ordered[0], ordered[0] + depth)):
            raise ValueError("Gate8 relevant workers may not form a contiguous serialization block")


@dataclass(frozen=True, slots=True)
class Gate8OracleResult:
    answer_symbol: int
    path_worker_indices: tuple[int, ...]
    path_transform_ids: tuple[int, ...]
    path_nodes: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class Gate8GeneratedWorld:
    public: Gate8PublicWorld
    truth: Gate8WorldTruth

    def validate(self) -> None:
        self.public.validate()
        self.truth.validate(self.public.population, self.public.depth)
        oracle = gate8_exact_symbolic_oracle(self.public)
        if oracle.answer_symbol != self.truth.answer_symbol:
            raise ValueError("Gate8 stored answer disagrees with the symbolic oracle")
        if oracle.path_worker_indices != self.truth.relevant_worker_indices:
            raise ValueError("Gate8 stored path disagrees with the symbolic oracle")


def _validate_identity(split: str, seed: int, world_index: int, population: int, depth: int) -> None:
    if split not in GATE8_ALLOWED_SPLITS:
        raise ValueError("Gate8 split is unknown")
    if (population, depth) not in GATE8_VALID_CONDITIONS:
        raise ValueError("Gate8 population/depth condition is outside the frozen matrix")
    if seed < 0 or world_index < 0:
        raise ValueError("Gate8 seed and world index must be non-negative")
    if split in ("train", "validation") and seed not in GATE8_TRAINING_SEEDS:
        raise ValueError("Gate8 train/validation seed must be 0, 1 or 2")
    if split == "train" and world_index >= GATE8_TRAINING_WORLDS_PER_SEED:
        raise ValueError("Gate8 training world index exceeds the frozen per-seed count")
    if split == "test" and world_index >= GATE8_TEST_WORLDS_PER_CONDITION:
        raise ValueError("Gate8 test world index exceeds the frozen condition count")
    if split == "demonstration" and world_index >= GATE8_DEMONSTRATION_WORLDS:
        raise ValueError("Gate8 demonstration index exceeds the frozen count")


def _namespace(split: str, seed: int) -> str:
    return f"gate8-distributed-transformation-{split}-v0:seed-{seed}"


def _node_labels(namespace: str, population: int, depth: int, world_index: int) -> tuple[str, ...]:
    labels = tuple(
        "n_" + hashlib.sha256(
            f"{namespace}:node:{population}:{depth}:{world_index}:{node_index}".encode("ascii")
        ).hexdigest()[:20]
        for node_index in range(population + 1)
    )
    if len(set(labels)) != population + 1:
        raise RuntimeError("Gate8 opaque node labels collided")
    return labels


def generate_gate8_world(
    *, split: str, seed: int, world_index: int, population: int, depth: int
) -> Gate8GeneratedWorld:
    _validate_identity(split, seed, world_index, population, depth)
    validate_gate8_transform_library()
    namespace = _namespace(split, seed)
    rng = random.Random(_seed_from_parts(namespace, population, depth, world_index))
    labels = _node_labels(namespace, population, depth, world_index)

    topology: list[tuple[int, int, bool]] = [
        (node_index, node_index + 1, True) for node_index in range(depth)
    ]
    existing_nodes = list(range(depth + 1))
    for new_node in range(depth + 1, population + 1):
        eligible = [node for node in existing_nodes if node != depth]
        parent = eligible[rng.randrange(len(eligible))]
        topology.append((parent, new_node, False))
        existing_nodes.append(new_node)

    transform_ids = [
        transform_id
        for transform_id in range(GATE8_TRANSFORM_COUNT)
        for _ in range(population // GATE8_TRANSFORM_COUNT)
    ]
    rng.shuffle(transform_ids)
    records = [
        (source, target, relevant, transform_id)
        for (source, target, relevant), transform_id in zip(topology, transform_ids, strict=True)
    ]
    for _ in range(128):
        rng.shuffle(records)
        relevant_positions = sorted(index for index, record in enumerate(records) if record[2])
        if relevant_positions != list(range(relevant_positions[0], relevant_positions[0] + depth)):
            break
    else:
        raise RuntimeError("Gate8 could not interleave relevant and distractor edges")

    workers = tuple(
        Gate8EdgeShard(
            worker_index=worker_index,
            source_node=labels[source],
            target_node=labels[target],
            transform_id=transform_id,
        )
        for worker_index, (source, target, _relevant, transform_id) in enumerate(records)
    )
    root_symbol = rng.randrange(GATE8_ALPHABET_SIZE)
    world_id = "g8_" + hashlib.sha256(
        f"{namespace}:world:{population}:{depth}:{world_index}".encode("ascii")
    ).hexdigest()[:24]
    public = Gate8PublicWorld(
        world_id=world_id,
        split=split,
        seed=seed,
        world_index=world_index,
        population=population,
        depth=depth,
        query=Gate8PublicQuery(
            root_node=labels[0],
            target_node=labels[depth],
            root_symbol=root_symbol,
        ),
        workers=workers,
    )
    oracle = gate8_exact_symbolic_oracle(public)
    result = Gate8GeneratedWorld(
        public=public,
        truth=Gate8WorldTruth(
            answer_symbol=oracle.answer_symbol,
            relevant_worker_indices=oracle.path_worker_indices,
        ),
    )
    result.validate()
    return result


def gate8_exact_symbolic_oracle(world: Gate8PublicWorld) -> Gate8OracleResult:
    world.validate()
    incoming: dict[str, Gate8EdgeShard] = {}
    nodes = {world.query.root_node}
    for worker in world.workers:
        nodes.add(worker.source_node)
        nodes.add(worker.target_node)
        if worker.target_node in incoming:
            raise ValueError("Gate8 graph is not a rooted tree: target has multiple parents")
        incoming[worker.target_node] = worker
    if world.query.root_node in incoming:
        raise ValueError("Gate8 root node has an incoming edge")
    if len(nodes) != world.population + 1:
        raise ValueError("Gate8 rooted tree must contain population + 1 nodes")
    if set(incoming) != nodes - {world.query.root_node}:
        raise ValueError("Gate8 graph contains an unreachable or parentless node")

    reverse_path: list[Gate8EdgeShard] = []
    current = world.query.target_node
    visited = set()
    while current != world.query.root_node:
        if current in visited:
            raise ValueError("Gate8 graph contains a cycle on the query path")
        visited.add(current)
        edge = incoming.get(current)
        if edge is None:
            raise ValueError("Gate8 target is unreachable from the root")
        reverse_path.append(edge)
        current = edge.source_node
        if len(reverse_path) > world.population:
            raise ValueError("Gate8 query path exceeded the population")
    path = tuple(reversed(reverse_path))
    if len(path) != world.depth:
        raise ValueError("Gate8 query path depth changed")

    symbol = world.query.root_symbol
    path_nodes = [world.query.root_node]
    for edge in path:
        if edge.source_node != path_nodes[-1]:
            raise ValueError("Gate8 query path is not ordered")
        symbol = apply_gate8_transform(edge.transform_id, symbol)
        path_nodes.append(edge.target_node)
    if path_nodes[-1] != world.query.target_node:
        raise ValueError("Gate8 query path did not reach the target")
    return Gate8OracleResult(
        answer_symbol=symbol,
        path_worker_indices=tuple(edge.worker_index for edge in path),
        path_transform_ids=tuple(edge.transform_id for edge in path),
        path_nodes=tuple(path_nodes),
    )


validate_gate8_transform_library()
