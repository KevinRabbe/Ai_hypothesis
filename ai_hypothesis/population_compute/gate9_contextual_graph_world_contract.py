"""Gate-9 contextual graph-world contract with scientific generation closed.

This module qualifies public/truth schemas, rooted-tree construction, opaque
identities, unique contextual operators per edge, exact public-support path
oracle, support-hit accounting, and post-checkpoint keyed operator allocation.
It generates contract worlds only. The scientific test assignment key and test
world generator remain unbound and closed.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import pathlib
import sys
from dataclasses import asdict, dataclass
from typing import Any

_ROOT = pathlib.Path(__file__).resolve().parent
_PROTOCOL_PATH = _ROOT / "gate9_contextual_operator_induction_protocol.py"
_OPERATOR_PATH = _ROOT / "gate9_contextual_operator_contract.py"
_CORRECTION_PATH = _ROOT / "gate9_graph_query_support_correction.py"


def _load(path: pathlib.Path, name: str):
    existing = sys.modules.get(name)
    if existing is not None:
        return existing
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load Gate9 dependency: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


protocol = _load(_PROTOCOL_PATH, "gate9_graph_contract_protocol_dependency")
operators = _load(_OPERATOR_PATH, "gate9_graph_contract_operator_dependency")
query_policy = _load(_CORRECTION_PATH, "gate9_graph_contract_query_policy_dependency")

GATE9_GRAPH_WORLD_CONTRACT_VERSION = "gate9-contextual-graph-world-contract-v0"
GATE9_GRAPH_WORLD_CONTRACT_STATUS = (
    "GATE9_CONTEXTUAL_GRAPH_WORLD_CONTRACT_QUALIFIED_SCIENTIFIC_GENERATION_CLOSED"
)
GATE9_QUERY_POLICY_CORRECTION_HEAD = "0dd7e5417bab5d3af074772a60725b95d22be76f"
GATE9_OPERATOR_CONTRACT_HEAD = "be6451e1af82b18749bd0313a9c02ca62c4eee5c"
GATE9_PROTOCOL_HEAD = "e5e20e8de6707d35f1a7a9315a5a9a67deacc9a1"

GATE9_CONTRACT_SPLIT = "contract"
GATE9_TEST_SPLIT = "test"
GATE9_CONTRACT_WORLDS_PER_CONDITION = 8
GATE9_CONTRACT_OPERATOR_COUNTER_START = 1 << 60
GATE9_CONTRACT_OPERATOR_COUNT = (
    sum(population for population, _ in protocol.GATE9_VALID_CONDITIONS)
    * GATE9_CONTRACT_WORLDS_PER_CONDITION
)
GATE9_CONTRACT_ASSIGNMENT_KEY = hashlib.sha256(
    b"gate9-contextual-graph-contract-assignment-key-v0"
).hexdigest()
GATE9_TEST_ASSIGNMENT_KEY_BOUND = False
GATE9_TEST_ASSIGNMENT_KEY_BINDING_TIME = (
    "after_all_three_checkpoint_admissions_before_any_test_world_generation"
)
GATE9_TEST_ASSIGNMENT_KEY_VISIBLE_TO_MODEL = False

_WORLD_NAMESPACE = "gate9-contextual-graph-contract-world-v0"
_NODE_NAMESPACE = "gate9-contextual-graph-contract-node-v0"
_TOPOLOGY_NAMESPACE = "gate9-contextual-graph-contract-topology-v0"
_WORKER_ORDER_NAMESPACE = "gate9-contextual-graph-contract-worker-order-v0"
_ROOT_SYMBOL_NAMESPACE = "gate9-contextual-graph-contract-root-symbol-v0"


def _hash_bytes(namespace: str, *parts: object) -> bytes:
    payload = ":".join((namespace, *(str(part) for part in parts)))
    return hashlib.sha256(payload.encode("ascii")).digest()


def _valid_assignment_key(value: str, *, allow_contract: bool) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise ValueError("Gate9 assignment key must be lowercase SHA-256 hex")
    if not allow_contract and value == GATE9_CONTRACT_ASSIGNMENT_KEY:
        raise ValueError("Gate9 scientific assignment key cannot reuse contract key")
    return value


def _permutation_parameters(key: str, size: int) -> tuple[int, int, int]:
    if size <= 1:
        raise ValueError("Gate9 operator assignment size must exceed one")
    key = _valid_assignment_key(key, allow_contract=True)
    digest = hashlib.sha256(
        f"gate9-contextual-operator-assignment-v0:{key}:{size}".encode("ascii")
    ).digest()
    multiplier = int.from_bytes(digest[:8], "big") % size
    if multiplier == 0:
        multiplier = 1
    while math.gcd(multiplier, size) != 1:
        multiplier = (multiplier + 1) % size
        if multiplier == 0:
            multiplier = 1
    offset = int.from_bytes(digest[8:16], "big") % size
    inverse = pow(multiplier, -1, size)
    return multiplier, offset, inverse


def permute_operator_ordinal(ordinal: int, size: int, assignment_key: str) -> int:
    if isinstance(ordinal, bool) or not isinstance(ordinal, int) or not 0 <= ordinal < size:
        raise ValueError("Gate9 operator ordinal lies outside assignment domain")
    multiplier, offset, _ = _permutation_parameters(assignment_key, size)
    return (multiplier * ordinal + offset) % size


def invert_operator_ordinal(permuted: int, size: int, assignment_key: str) -> int:
    if isinstance(permuted, bool) or not isinstance(permuted, int) or not 0 <= permuted < size:
        raise ValueError("Gate9 permuted ordinal lies outside assignment domain")
    _, offset, inverse = _permutation_parameters(assignment_key, size)
    return (inverse * (permuted - offset)) % size


def _condition_index(population: int, depth: int) -> int:
    try:
        return protocol.GATE9_VALID_CONDITIONS.index((population, depth))
    except ValueError as error:
        raise ValueError("Gate9 graph condition is outside the frozen matrix") from error


def _condition_edge_offset(population: int, depth: int, worlds: int) -> int:
    index = _condition_index(population, depth)
    return sum(
        prior_population * worlds
        for prior_population, _ in protocol.GATE9_VALID_CONDITIONS[:index]
    )


def graph_test_operator_counter(
    *,
    population: int,
    depth: int,
    world_index: int,
    canonical_edge_index: int,
    assignment_key: str,
) -> int:
    _valid_assignment_key(assignment_key, allow_contract=False)
    if not 0 <= world_index < protocol.GATE9_WORLDS_PER_CONDITION:
        raise ValueError("Gate9 graph-test world index lies outside 0..255")
    if not 0 <= canonical_edge_index < population:
        raise ValueError("Gate9 graph-test edge index lies outside population")
    ordinal = (
        _condition_edge_offset(
            population, depth, protocol.GATE9_WORLDS_PER_CONDITION
        )
        + world_index * population
        + canonical_edge_index
    )
    permuted = permute_operator_ordinal(
        ordinal,
        protocol.GATE9_GRAPH_TEST_OPERATOR_COUNT,
        assignment_key,
    )
    return protocol.GATE9_GRAPH_TEST_OPERATOR_COUNTER_START + permuted


def _contract_operator_counter(
    *, population: int, depth: int, world_index: int, canonical_edge_index: int
) -> int:
    if not 0 <= world_index < GATE9_CONTRACT_WORLDS_PER_CONDITION:
        raise ValueError("Gate9 contract world index lies outside 0..7")
    if not 0 <= canonical_edge_index < population:
        raise ValueError("Gate9 contract edge index lies outside population")
    ordinal = (
        _condition_edge_offset(
            population, depth, GATE9_CONTRACT_WORLDS_PER_CONDITION
        )
        + world_index * population
        + canonical_edge_index
    )
    permuted = permute_operator_ordinal(
        ordinal, GATE9_CONTRACT_OPERATOR_COUNT, GATE9_CONTRACT_ASSIGNMENT_KEY
    )
    return GATE9_CONTRACT_OPERATOR_COUNTER_START + permuted


def _node_label(population: int, depth: int, world_index: int, node_index: int) -> str:
    return _hash_bytes(
        _NODE_NAMESPACE, population, depth, world_index, node_index
    ).hex()[:24]


def _canonical_edges(
    population: int, depth: int, world_index: int
) -> tuple[tuple[int, int], ...]:
    _condition_index(population, depth)
    edges: list[tuple[int, int]] = [
        (path_index, path_index + 1) for path_index in range(depth)
    ]
    for target in range(depth + 1, population + 1):
        parent = int.from_bytes(
            _hash_bytes(
                _TOPOLOGY_NAMESPACE,
                population,
                depth,
                world_index,
                target,
            )[:8],
            "big",
        ) % target
        edges.append((parent, target))
    if len(edges) != population:
        raise RuntimeError("Gate9 canonical edge count drifted")
    return tuple(edges)


@dataclass(frozen=True, slots=True)
class Gate9GraphWorkerPublic:
    worker_index: int
    source_node: str
    target_node: str
    support_pairs: tuple[tuple[int, int], ...]

    def validate(self) -> None:
        if self.worker_index < 0:
            raise ValueError("Gate9 public worker index is negative")
        if not self.source_node or not self.target_node or self.source_node == self.target_node:
            raise ValueError("Gate9 public worker endpoints are invalid")
        operators.reconstruct_operator_from_support(self.support_pairs)

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            "worker_index": self.worker_index,
            "source_node": self.source_node,
            "target_node": self.target_node,
            "support_pairs": [list(pair) for pair in self.support_pairs],
        }


@dataclass(frozen=True, slots=True)
class Gate9GraphQueryPublic:
    root_node: str
    target_node: str
    root_symbol: int

    def validate(self) -> None:
        if not self.root_node or not self.target_node or self.root_node == self.target_node:
            raise ValueError("Gate9 public query endpoints are invalid")
        operators._valid_byte(self.root_symbol, "Gate9 graph root symbol")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return asdict(self)


@dataclass(frozen=True, slots=True)
class Gate9GraphWorldPublic:
    version: str
    split: str
    population: int
    depth: int
    world_index: int
    world_id: str
    workers: tuple[Gate9GraphWorkerPublic, ...]
    query: Gate9GraphQueryPublic

    def validate(self) -> None:
        if self.version != GATE9_GRAPH_WORLD_CONTRACT_VERSION:
            raise ValueError("Gate9 graph-world version drifted")
        if self.split != GATE9_CONTRACT_SPLIT:
            raise ValueError("Gate9 graph-world contract admits contract split only")
        _condition_index(self.population, self.depth)
        if not 0 <= self.world_index < GATE9_CONTRACT_WORLDS_PER_CONDITION:
            raise ValueError("Gate9 contract world index drifted")
        if len(self.workers) != self.population:
            raise ValueError("Gate9 graph world worker count drifted")
        if tuple(worker.worker_index for worker in self.workers) != tuple(
            range(self.population)
        ):
            raise ValueError("Gate9 public worker indices are not contiguous")
        self.query.validate()
        incoming: dict[str, int] = {}
        nodes = {self.query.root_node}
        for worker in self.workers:
            worker.validate()
            if worker.target_node in incoming:
                raise ValueError("Gate9 graph node has multiple incoming workers")
            incoming[worker.target_node] = worker.worker_index
            nodes.add(worker.source_node)
            nodes.add(worker.target_node)
        if self.query.root_node in incoming:
            raise ValueError("Gate9 graph root has an incoming worker")
        if self.query.target_node not in incoming:
            raise ValueError("Gate9 graph target lacks an incoming worker")
        if len(nodes) != self.population + 1 or len(incoming) != self.population:
            raise ValueError("Gate9 graph is not a population-edge tree")
        chain = _public_path_worker_indices(self)
        if len(chain) != self.depth:
            raise ValueError("Gate9 public target path depth drifted")
        if _public_world_id(self) != self.world_id:
            raise ValueError("Gate9 public world identity drifted")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            "version": self.version,
            "split": self.split,
            "population": self.population,
            "depth": self.depth,
            "world_index": self.world_index,
            "world_id": self.world_id,
            "workers": [worker.to_dict() for worker in self.workers],
            "query": self.query.to_dict(),
        }


@dataclass(frozen=True, slots=True)
class Gate9GraphWorldTruth:
    world_id: str
    path_worker_indices: tuple[int, ...]
    operator_counters_by_worker: tuple[int, ...]
    answer_symbol: int
    support_hits_by_path_position: tuple[bool, ...]

    def validate(self, public: Gate9GraphWorldPublic) -> None:
        public.validate()
        if self.world_id != public.world_id:
            raise ValueError("Gate9 truth world identity disagrees with public world")
        if self.path_worker_indices != _public_path_worker_indices(public):
            raise ValueError("Gate9 truth path disagrees with public topology")
        if len(self.operator_counters_by_worker) != public.population:
            raise ValueError("Gate9 truth operator-counter vector drifted")
        if len(set(self.operator_counters_by_worker)) != public.population:
            raise ValueError("Gate9 graph world reused an operator counter")
        lower = GATE9_CONTRACT_OPERATOR_COUNTER_START
        upper = lower + GATE9_CONTRACT_OPERATOR_COUNT
        if any(not lower <= value < upper for value in self.operator_counters_by_worker):
            raise ValueError("Gate9 contract truth counter lies outside contract range")
        operators._valid_byte(self.answer_symbol, "Gate9 graph truth answer")
        if len(self.support_hits_by_path_position) != public.depth:
            raise ValueError("Gate9 truth support-hit vector drifted")
        answer, hits = gate9_public_support_path_oracle(public)
        if answer != self.answer_symbol or hits != self.support_hits_by_path_position:
            raise ValueError("Gate9 private truth disagrees with public-support oracle")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _public_path_worker_indices(public: Gate9GraphWorldPublic) -> tuple[int, ...]:
    incoming = {worker.target_node: worker for worker in public.workers}
    reversed_path: list[int] = []
    node = public.query.target_node
    visited = set()
    while node != public.query.root_node:
        if node in visited or node not in incoming:
            raise ValueError("Gate9 public target path is cyclic or disconnected")
        visited.add(node)
        worker = incoming[node]
        reversed_path.append(worker.worker_index)
        node = worker.source_node
    return tuple(reversed(reversed_path))


def gate9_public_support_path_oracle(
    public: Gate9GraphWorldPublic,
) -> tuple[int, tuple[bool, ...]]:
    path = _public_path_worker_indices(public)
    value = public.query.root_symbol
    hits: list[bool] = []
    for worker_index in path:
        hits.append(value in protocol.GATE9_SUPPORT_INPUTS)
        value = operators.apply_public_support_oracle(
            public.workers[worker_index].support_pairs,
            value,
            require_novel_query=False,
        )
    return value, tuple(hits)


def _public_world_id(public: Gate9GraphWorldPublic) -> str:
    payload = {
        "version": public.version,
        "split": public.split,
        "population": public.population,
        "depth": public.depth,
        "world_index": public.world_index,
        "workers": [
            {
                "worker_index": worker.worker_index,
                "source_node": worker.source_node,
                "target_node": worker.target_node,
                "support_pairs": [list(pair) for pair in worker.support_pairs],
            }
            for worker in public.workers
        ],
        "query": public.query.to_dict(),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    return hashlib.sha256(encoded).hexdigest()


def generate_gate9_contract_world(
    *, population: int, depth: int, world_index: int
) -> tuple[Gate9GraphWorldPublic, Gate9GraphWorldTruth]:
    _condition_index(population, depth)
    if not 0 <= world_index < GATE9_CONTRACT_WORLDS_PER_CONDITION:
        raise ValueError("Gate9 contract world index lies outside 0..7")
    labels = tuple(
        _node_label(population, depth, world_index, node_index)
        for node_index in range(population + 1)
    )
    if len(set(labels)) != len(labels):
        raise RuntimeError("Gate9 opaque node-label collision")
    canonical = _canonical_edges(population, depth, world_index)
    ranked_indices = tuple(
        sorted(
            range(population),
            key=lambda edge_index: _hash_bytes(
                _WORKER_ORDER_NAMESPACE,
                population,
                depth,
                world_index,
                edge_index,
            ),
        )
    )
    workers: list[Gate9GraphWorkerPublic] = []
    counters: list[int] = []
    canonical_to_worker: dict[int, int] = {}
    for worker_index, canonical_edge_index in enumerate(ranked_indices):
        source, target = canonical[canonical_edge_index]
        counter = _contract_operator_counter(
            population=population,
            depth=depth,
            world_index=world_index,
            canonical_edge_index=canonical_edge_index,
        )
        operator = operators.operator_from_counter(counter)
        workers.append(
            Gate9GraphWorkerPublic(
                worker_index=worker_index,
                source_node=labels[source],
                target_node=labels[target],
                support_pairs=operators.public_support_pairs(operator),
            )
        )
        counters.append(counter)
        canonical_to_worker[canonical_edge_index] = worker_index
    root_symbol = _hash_bytes(
        _ROOT_SYMBOL_NAMESPACE, population, depth, world_index
    )[0]
    query = Gate9GraphQueryPublic(
        root_node=labels[0],
        target_node=labels[depth],
        root_symbol=root_symbol,
    )
    provisional = Gate9GraphWorldPublic(
        version=GATE9_GRAPH_WORLD_CONTRACT_VERSION,
        split=GATE9_CONTRACT_SPLIT,
        population=population,
        depth=depth,
        world_index=world_index,
        world_id="pending",
        workers=tuple(workers),
        query=query,
    )
    public = Gate9GraphWorldPublic(
        version=provisional.version,
        split=provisional.split,
        population=provisional.population,
        depth=provisional.depth,
        world_index=provisional.world_index,
        world_id=_public_world_id(provisional),
        workers=provisional.workers,
        query=provisional.query,
    )
    answer, hits = gate9_public_support_path_oracle(public)
    truth = Gate9GraphWorldTruth(
        world_id=public.world_id,
        path_worker_indices=tuple(
            canonical_to_worker[index] for index in range(depth)
        ),
        operator_counters_by_worker=tuple(counters),
        answer_symbol=answer,
        support_hits_by_path_position=hits,
    )
    public.validate()
    truth.validate(public)
    return public, truth


def gate9_graph_world_contract_plan() -> dict[str, Any]:
    multiplier, offset, inverse = _permutation_parameters(
        GATE9_CONTRACT_ASSIGNMENT_KEY, GATE9_CONTRACT_OPERATOR_COUNT
    )
    test_multiplier_placeholder = None
    return {
        "version": GATE9_GRAPH_WORLD_CONTRACT_VERSION,
        "status": GATE9_GRAPH_WORLD_CONTRACT_STATUS,
        "protocol_head": GATE9_PROTOCOL_HEAD,
        "operator_contract_head": GATE9_OPERATOR_CONTRACT_HEAD,
        "query_policy_correction_head": GATE9_QUERY_POLICY_CORRECTION_HEAD,
        "contract_split": GATE9_CONTRACT_SPLIT,
        "contract_worlds_per_condition": GATE9_CONTRACT_WORLDS_PER_CONDITION,
        "contract_operator_counter_start": GATE9_CONTRACT_OPERATOR_COUNTER_START,
        "contract_operator_count": GATE9_CONTRACT_OPERATOR_COUNT,
        "contract_assignment": {
            "key_sha256": hashlib.sha256(
                GATE9_CONTRACT_ASSIGNMENT_KEY.encode("ascii")
            ).hexdigest(),
            "multiplier": multiplier,
            "offset": offset,
            "inverse": inverse,
        },
        "graph_test_operator_counter_start": (
            protocol.GATE9_GRAPH_TEST_OPERATOR_COUNTER_START
        ),
        "graph_test_operator_count": protocol.GATE9_GRAPH_TEST_OPERATOR_COUNT,
        "test_assignment_key_bound": GATE9_TEST_ASSIGNMENT_KEY_BOUND,
        "test_assignment_key_binding_time": GATE9_TEST_ASSIGNMENT_KEY_BINDING_TIME,
        "test_assignment_key_visible_to_model": (
            GATE9_TEST_ASSIGNMENT_KEY_VISIBLE_TO_MODEL
        ),
        "test_assignment_multiplier": test_multiplier_placeholder,
        "test_operator_allocation_bijective_after_key_binding": True,
        "test_operator_counter_skipping_allowed": False,
        "test_operator_rejection_allowed": False,
        "public_worker_fields": [
            "worker_index",
            "source_node",
            "target_node",
            "support_pairs",
        ],
        "public_query_fields": ["root_node", "target_node", "root_symbol"],
        "public_operator_counter_exposed": False,
        "public_operator_key_exposed": False,
        "support_hit_reporting_required": True,
        "contract_world_generation_admitted": True,
        "scientific_test_world_generation_admitted": False,
        "architecture_admitted": False,
        "training_admitted": False,
        "checkpoint_loading_admitted": False,
        "scientific_execution_admitted": False,
        "result_classification_admitted": False,
    }
