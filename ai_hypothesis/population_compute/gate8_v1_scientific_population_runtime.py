"""Gate-8 v1 scientific population runtime with exact compiled transitions.

This module admits only the frozen population-organism side of the scientific
benchmark. It compiles each qualified neural checkpoint into the exhaustive
2,048-entry local transition table and executes the frozen graph semantics on
scientific-test worlds. It never reads world truth, loads the 1B reference, or
performs training.
"""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from typing import Any

import torch
from torch import Tensor, nn

GATE8_V1_SCIENTIFIC_POPULATION_RUNTIME_VERSION = (
    "gate8-v1-scientific-population-runtime-v0"
)
GATE8_V1_SCIENTIFIC_PROTOCOL_HEAD = "6bb89111a47713bea0a23bb1cae662ed5ec56b42"
GATE8_V1_GEMMA_BINDING_RESULT_HEAD = "8237732aecbec083c66668de9fae132e0cc4c1f9"
GATE8_V1_ARCHITECTURE_HEAD = "c3ab64008c816fa1eb6f9d6f8f0a1a99ed195ec8"
GATE8_V1_RUNTIME_HEAD = "333d88ac4fc52f1651741fba224e0b4605feedd3"
GATE8_V1_LEARNED_PARAMETER_COUNT = 19_649
GATE8_V1_MESSAGE_CODEBOOK_SIZE = 256
GATE8_V1_TRANSFORM_COUNT = 8
GATE8_V1_SYMBOL_COUNT = 16
GATE8_V1_MESSAGE_BITS = 8
GATE8_V1_WORKER_STATE_WIDTH = 65

GATE8_V1_FULL_MODE = "full"
GATE8_V1_NO_COMMUNICATION_MODE = "no_communication"
GATE8_V1_SHUFFLED_WORKER_MODE = "shuffled_worker"
GATE8_V1_SHUFFLED_MESSAGE_MODE = "shuffled_message"
GATE8_V1_TARGET_WORKER_ONLY_MODE = "target_worker_only"
GATE8_V1_POPULATION_MODES = (
    GATE8_V1_FULL_MODE,
    GATE8_V1_NO_COMMUNICATION_MODE,
    GATE8_V1_SHUFFLED_WORKER_MODE,
    GATE8_V1_SHUFFLED_MESSAGE_MODE,
    GATE8_V1_TARGET_WORKER_ONLY_MODE,
)


def _valid_sha256(value: str) -> bool:
    return len(value) == 64 and all(
        character in "0123456789abcdef" for character in value
    )


@dataclass(frozen=True, slots=True)
class Gate8V1CompiledTransitionTable:
    checkpoint_seed: int
    checkpoint_sha256: str
    message_codes: tuple[tuple[int, ...], ...]
    table_sha256: str

    def validate(self) -> None:
        if self.checkpoint_seed not in (0, 1, 2):
            raise ValueError("Gate8 v1 compiled table seed is outside 0..2")
        if not _valid_sha256(self.checkpoint_sha256):
            raise ValueError("Gate8 v1 compiled table checkpoint hash is malformed")
        if len(self.message_codes) != GATE8_V1_MESSAGE_CODEBOOK_SIZE:
            raise ValueError("Gate8 v1 compiled table inbox dimension drifted")
        flat: list[int] = []
        for row in self.message_codes:
            if len(row) != GATE8_V1_TRANSFORM_COUNT:
                raise ValueError("Gate8 v1 compiled table transform dimension drifted")
            if any(not 0 <= code < GATE8_V1_MESSAGE_CODEBOOK_SIZE for code in row):
                raise ValueError("Gate8 v1 compiled table contains an invalid code")
            flat.extend(row)
        if hashlib.sha256(bytes(flat)).hexdigest() != self.table_sha256:
            raise ValueError("Gate8 v1 compiled transition-table hash drifted")

    def lookup(self, inbox_code: int, transform_id: int) -> int:
        if not 0 <= inbox_code < GATE8_V1_MESSAGE_CODEBOOK_SIZE:
            raise ValueError("Gate8 v1 scientific inbox code is outside 0..255")
        if not 0 <= transform_id < GATE8_V1_TRANSFORM_COUNT:
            raise ValueError("Gate8 v1 scientific transform is outside 0..7")
        return self.message_codes[inbox_code][transform_id]

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return asdict(self)


@dataclass(frozen=True, slots=True)
class Gate8V1ScientificWorldPlan:
    world_id: str
    split: str
    population: int
    depth: int
    root_node: str
    root_symbol: int
    target_worker_index: int
    source_nodes: tuple[str, ...]
    target_nodes: tuple[str, ...]
    transform_ids: tuple[int, ...]
    outgoing: dict[str, tuple[int, ...]]

    def validate(self) -> None:
        if self.split not in ("contract", "test"):
            raise ValueError("Gate8 v1 scientific world-plan split is invalid")
        if not self.world_id or self.population <= 0 or self.depth <= 0:
            raise ValueError("Gate8 v1 scientific world-plan identity is invalid")
        if not 0 <= self.root_symbol < GATE8_V1_SYMBOL_COUNT:
            raise ValueError("Gate8 v1 scientific world-plan root symbol is invalid")
        if not 0 <= self.target_worker_index < self.population:
            raise ValueError("Gate8 v1 scientific world-plan target worker is invalid")
        for vector in (self.source_nodes, self.target_nodes, self.transform_ids):
            if len(vector) != self.population:
                raise ValueError("Gate8 v1 scientific world-plan vector is incomplete")
        if any(not 0 <= value < GATE8_V1_TRANSFORM_COUNT for value in self.transform_ids):
            raise ValueError("Gate8 v1 scientific world-plan transform is invalid")
        flattened = tuple(
            sorted(index for row in self.outgoing.values() for index in row)
        )
        if flattened != tuple(range(self.population)):
            raise ValueError("Gate8 v1 scientific world-plan outgoing index changed")


@dataclass(frozen=True, slots=True)
class Gate8V1ScientificPopulationResult:
    version: str
    mode: str
    world_id: str
    population: int
    depth: int
    checkpoint_seed: int
    transition_table_sha256: str
    target_reached: bool
    predicted_symbol: int | None
    rounds: int
    active_workers: int
    recurrent_updates: int
    delivered_messages: int
    communicated_bits: int

    def validate(self) -> None:
        if self.version != GATE8_V1_SCIENTIFIC_POPULATION_RUNTIME_VERSION:
            raise ValueError("Gate8 v1 scientific runtime version drifted")
        if self.mode not in GATE8_V1_POPULATION_MODES:
            raise ValueError("Gate8 v1 scientific runtime mode is unknown")
        if self.population <= 0 or self.depth <= 0:
            raise ValueError("Gate8 v1 scientific runtime dimensions are invalid")
        if self.checkpoint_seed not in (0, 1, 2):
            raise ValueError("Gate8 v1 scientific runtime seed is outside 0..2")
        if not _valid_sha256(self.transition_table_sha256):
            raise ValueError("Gate8 v1 scientific transition-table hash is malformed")
        if self.rounds < 0:
            raise ValueError("Gate8 v1 scientific round count is negative")
        counts = (
            self.active_workers,
            self.recurrent_updates,
            self.delivered_messages,
            self.communicated_bits,
        )
        if any(value < 0 for value in counts):
            raise ValueError("Gate8 v1 scientific resource count is negative")
        if self.active_workers != self.recurrent_updates:
            raise ValueError("Gate8 v1 one-update-per-active-worker invariant drifted")
        if self.communicated_bits != self.delivered_messages * GATE8_V1_MESSAGE_BITS:
            raise ValueError("Gate8 v1 scientific bit accounting drifted")
        if self.mode == GATE8_V1_NO_COMMUNICATION_MODE:
            if self.delivered_messages or self.communicated_bits:
                raise ValueError("Gate8 v1 no-communication control delivered a message")
        if self.mode == GATE8_V1_TARGET_WORKER_ONLY_MODE:
            if (self.rounds, self.active_workers, self.recurrent_updates) != (1, 1, 1):
                raise ValueError("Gate8 v1 target-worker-only accounting drifted")
            if self.delivered_messages:
                raise ValueError("Gate8 v1 target-worker-only control communicated")
        if self.target_reached:
            if self.predicted_symbol is None or not 0 <= self.predicted_symbol < 16:
                raise ValueError("Gate8 v1 reached target lacks a valid prediction")
        elif self.predicted_symbol is not None:
            raise ValueError("Gate8 v1 unreached target exposed a prediction")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return asdict(self)


def _validate_model(model: nn.Module) -> tuple[torch.device, torch.dtype]:
    if model.training:
        raise ValueError("Gate8 v1 transition compilation requires model.eval()")
    parameters = tuple(model.parameters())
    if sum(parameter.numel() for parameter in parameters) != 19_649:
        raise ValueError("Gate8 v1 transition compilation requires 19,649 parameters")
    devices = {parameter.device for parameter in parameters}
    dtypes = {parameter.dtype for parameter in parameters}
    if len(devices) != 1 or len(dtypes) != 1:
        raise ValueError("Gate8 v1 model parameters must share one device and dtype")
    device = next(iter(devices))
    dtype = next(iter(dtypes))
    if not dtype.is_floating_point:
        raise ValueError("Gate8 v1 model parameters must be floating point")
    for method in ("initial_hidden", "predicted_message_code", "predicted_symbol"):
        if not callable(getattr(model, method, None)):
            raise ValueError(f"Gate8 v1 model lacks required method: {method}")
    return device, dtype


def compile_gate8_v1_transition_table(
    *,
    model: nn.Module,
    checkpoint_seed: int,
    checkpoint_sha256: str,
) -> Gate8V1CompiledTransitionTable:
    """Compile the exact one-update neural worker into all 256 × 8 transitions."""

    if checkpoint_seed not in (0, 1, 2):
        raise ValueError("Gate8 v1 checkpoint seed is outside 0..2")
    if not _valid_sha256(checkpoint_sha256):
        raise ValueError("Gate8 v1 checkpoint SHA-256 is malformed")
    device, dtype = _validate_model(model)
    inbox = torch.arange(256, device=device, dtype=torch.long).repeat_interleave(8)
    transforms = torch.arange(8, device=device, dtype=torch.long).repeat(256)
    with torch.no_grad():
        hidden = model.initial_hidden(2_048, device=device, dtype=dtype)
        output = model(inbox_code=inbox, transform_id=transforms, hidden=hidden)
        codes = model.predicted_message_code(output)
        symbols = model.predicted_symbol(output)
    if not isinstance(codes, Tensor) or codes.dtype != torch.long:
        raise ValueError("Gate8 v1 compiled message vector is invalid")
    if tuple(codes.shape) != (2_048,):
        raise ValueError("Gate8 v1 compiled message-vector shape drifted")
    if not isinstance(symbols, Tensor) or symbols.dtype != torch.long:
        raise ValueError("Gate8 v1 compiled symbol vector is invalid")
    if tuple(symbols.shape) != (2_048,):
        raise ValueError("Gate8 v1 compiled symbol-vector shape drifted")
    codes_cpu = tuple(int(value) for value in codes.detach().cpu().tolist())
    symbols_cpu = tuple(int(value) for value in symbols.detach().cpu().tolist())
    if any(
        (code & 0x0F) != symbol
        for code, symbol in zip(codes_cpu, symbols_cpu, strict=True)
    ):
        raise ValueError("Gate8 v1 compiled message disagrees with symbol head")
    rows = tuple(
        tuple(codes_cpu[inbox_code * 8 + transform] for transform in range(8))
        for inbox_code in range(256)
    )
    table = Gate8V1CompiledTransitionTable(
        checkpoint_seed=checkpoint_seed,
        checkpoint_sha256=checkpoint_sha256,
        message_codes=rows,
        table_sha256=hashlib.sha256(bytes(codes_cpu)).hexdigest(),
    )
    table.validate()
    return table


def _validate_world(world: Any, *, admitted_split: str) -> None:
    if getattr(world, "split", None) != admitted_split:
        raise ValueError(f"Gate8 v1 scientific runtime admits {admitted_split} worlds only")
    validate = getattr(world, "validate", None)
    if not callable(validate):
        raise ValueError("Gate8 v1 scientific world lacks validation")
    validate()


def compile_gate8_v1_scientific_world_plan(
    world: Any,
    *,
    admitted_split: str,
) -> Gate8V1ScientificWorldPlan:
    _validate_world(world, admitted_split=admitted_split)
    incoming: dict[str, int] = {}
    outgoing_mutable: dict[str, list[int]] = {}
    nodes = {world.query.root_node}
    for expected_index, worker in enumerate(world.workers):
        if worker.worker_index != expected_index:
            raise ValueError("Gate8 v1 scientific workers are not contiguous")
        if worker.target_node in incoming:
            raise ValueError("Gate8 v1 scientific node has multiple incoming workers")
        incoming[worker.target_node] = worker.worker_index
        outgoing_mutable.setdefault(worker.source_node, []).append(worker.worker_index)
        nodes.add(worker.source_node)
        nodes.add(worker.target_node)
    if world.query.root_node in incoming or world.query.target_node not in incoming:
        raise ValueError("Gate8 v1 scientific root/target topology is invalid")
    if len(nodes) != world.population + 1:
        raise ValueError("Gate8 v1 scientific graph is not a population-edge tree")
    outgoing = {
        node: tuple(sorted(indices)) for node, indices in outgoing_mutable.items()
    }
    reachable = {world.query.root_node}
    remaining = set(range(world.population))
    while remaining:
        scheduled = tuple(
            index
            for node in tuple(reachable)
            for index in outgoing.get(node, ())
            if index in remaining
        )
        if not scheduled:
            raise ValueError("Gate8 v1 scientific graph contains an unreachable edge")
        for index in scheduled:
            reachable.add(world.workers[index].target_node)
            remaining.remove(index)
    plan = Gate8V1ScientificWorldPlan(
        world_id=world.world_id,
        split=admitted_split,
        population=world.population,
        depth=world.depth,
        root_node=world.query.root_node,
        root_symbol=world.query.root_symbol,
        target_worker_index=incoming[world.query.target_node],
        source_nodes=tuple(worker.source_node for worker in world.workers),
        target_nodes=tuple(worker.target_node for worker in world.workers),
        transform_ids=tuple(worker.transform_id for worker in world.workers),
        outgoing=outgoing,
    )
    plan.validate()
    return plan


def _ranked_permutation(
    *, namespace: str, world_id: str, round_index: int, size: int
) -> tuple[int, ...]:
    identity = tuple(range(size))
    if size <= 1:
        return identity
    ranked = tuple(
        sorted(
            identity,
            key=lambda index: hashlib.sha256(
                f"{namespace}:{world_id}:{round_index}:{index}".encode("ascii")
            ).digest(),
        )
    )
    return ranked[1:] + ranked[:1] if ranked == identity else ranked


def gate8_v1_scientific_shuffled_worker_permutation(
    world_id: str, population: int
) -> tuple[int, ...]:
    if not world_id or population <= 1:
        raise ValueError("Gate8 v1 shuffled-worker control requires a world and population")
    identity = tuple(range(population))
    ranked = tuple(
        sorted(
            identity,
            key=lambda worker_index: hashlib.sha256(
                f"gate8-v1-shuffled-worker:{world_id}:{worker_index}".encode("ascii")
            ).digest(),
        )
    )
    return ranked[1:] + ranked[:1] if ranked == identity else ranked


def _effective_transforms(
    plan: Gate8V1ScientificWorldPlan, mode: str
) -> tuple[int, ...]:
    original = plan.transform_ids
    if mode != GATE8_V1_SHUFFLED_WORKER_MODE:
        return original
    permutation = gate8_v1_scientific_shuffled_worker_permutation(
        plan.world_id, plan.population
    )
    transformed = tuple(original[source] for source in permutation)
    if sorted(transformed) != sorted(original):
        raise RuntimeError("Gate8 v1 shuffled-worker control changed transform marginals")
    return transformed


def run_gate8_v1_scientific_population_plan(
    *,
    table: Gate8V1CompiledTransitionTable,
    plan: Gate8V1ScientificWorldPlan,
    mode: str,
) -> Gate8V1ScientificPopulationResult:
    if mode not in GATE8_V1_POPULATION_MODES:
        raise ValueError("Gate8 v1 scientific population mode is unknown")
    if (
        table.checkpoint_seed not in (0, 1, 2)
        or not _valid_sha256(table.table_sha256)
        or len(table.message_codes) != 256
    ):
        raise ValueError("Gate8 v1 scientific transition-table identity is invalid")
    if (
        plan.split not in ("contract", "test")
        or len(plan.source_nodes) != plan.population
        or len(plan.target_nodes) != plan.population
        or len(plan.transform_ids) != plan.population
    ):
        raise ValueError("Gate8 v1 scientific world-plan identity is invalid")
    target_worker = plan.target_worker_index
    root_code = int(plan.root_symbol)

    if mode == GATE8_V1_TARGET_WORKER_ONLY_MODE:
        code = table.lookup(root_code, plan.transform_ids[target_worker])
        result = Gate8V1ScientificPopulationResult(
            version=GATE8_V1_SCIENTIFIC_POPULATION_RUNTIME_VERSION,
            mode=mode,
            world_id=plan.world_id,
            population=plan.population,
            depth=plan.depth,
            checkpoint_seed=table.checkpoint_seed,
            transition_table_sha256=table.table_sha256,
            target_reached=True,
            predicted_symbol=code & 0x0F,
            rounds=1,
            active_workers=1,
            recurrent_updates=1,
            delivered_messages=0,
            communicated_bits=0,
        )
        result.validate()
        return result

    transforms = _effective_transforms(plan, mode)
    mailboxes: dict[str, int] = {plan.root_node: root_code}
    reached = False
    prediction: int | None = None
    rounds = 0
    updates = 0
    deliveries = 0
    executed: set[int] = set()

    for round_index in range(plan.depth):
        scheduled = tuple(
            sorted(
                index
                for node in mailboxes
                for index in plan.outgoing.get(node, ())
            )
        )
        if not scheduled:
            break
        if any(index in executed for index in scheduled):
            raise RuntimeError("Gate8 v1 scientific worker executed more than once")
        executed.update(scheduled)
        emitted = tuple(
            table.lookup(mailboxes[plan.source_nodes[index]], transforms[index])
            for index in scheduled
        )
        rounds += 1
        updates += len(scheduled)

        if target_worker in scheduled:
            prediction = emitted[scheduled.index(target_worker)] & 0x0F
            reached = True

        if mode == GATE8_V1_NO_COMMUNICATION_MODE:
            next_mailboxes: dict[str, int] = {}
        else:
            delivered_codes = emitted
            if mode == GATE8_V1_SHUFFLED_MESSAGE_MODE:
                permutation = _ranked_permutation(
                    namespace="gate8-v1-scientific-shuffled-message-control-v0",
                    world_id=plan.world_id,
                    round_index=round_index,
                    size=len(scheduled),
                )
                delivered_codes = tuple(emitted[source] for source in permutation)
                if sorted(delivered_codes) != sorted(emitted):
                    raise RuntimeError(
                        "Gate8 v1 shuffled-message control changed code marginals"
                    )
            next_mailboxes = {}
            for local, worker_index in enumerate(scheduled):
                target = plan.target_nodes[worker_index]
                if target in next_mailboxes:
                    raise RuntimeError("Gate8 v1 scientific messages collided")
                next_mailboxes[target] = delivered_codes[local]
            deliveries += len(scheduled)
        mailboxes = next_mailboxes
        if reached:
            break

    result = Gate8V1ScientificPopulationResult(
        version=GATE8_V1_SCIENTIFIC_POPULATION_RUNTIME_VERSION,
        mode=mode,
        world_id=plan.world_id,
        population=plan.population,
        depth=plan.depth,
        checkpoint_seed=table.checkpoint_seed,
        transition_table_sha256=table.table_sha256,
        target_reached=reached,
        predicted_symbol=prediction,
        rounds=rounds,
        active_workers=updates,
        recurrent_updates=updates,
        delivered_messages=deliveries,
        communicated_bits=deliveries * GATE8_V1_MESSAGE_BITS,
    )
    result.validate()
    return result


def run_gate8_v1_scientific_population_runtime(
    *, table: Gate8V1CompiledTransitionTable, world: Any, mode: str
) -> Gate8V1ScientificPopulationResult:
    """Execute one frozen scientific-test world."""

    plan = compile_gate8_v1_scientific_world_plan(world, admitted_split="test")
    return run_gate8_v1_scientific_population_plan(
        table=table, plan=plan, mode=mode
    )


def run_gate8_v1_scientific_population_contract_probe(
    *, table: Gate8V1CompiledTransitionTable, world: Any, mode: str
) -> Gate8V1ScientificPopulationResult:
    """Contract-world-only equivalence probe for qualification CI."""

    plan = compile_gate8_v1_scientific_world_plan(world, admitted_split="contract")
    return run_gate8_v1_scientific_population_plan(
        table=table, plan=plan, mode=mode
    )


def gate8_v1_deterministic_random_answer(world_id: str) -> int:
    if not world_id:
        raise ValueError("Gate8 v1 random-control world ID is empty")
    digest = hashlib.sha256(
        f"gate8-v1-scientific-random-answer-control-v0:{world_id}".encode("ascii")
    ).digest()
    return int.from_bytes(digest[:8], "big") % GATE8_V1_SYMBOL_COUNT


def gate8_v1_scientific_population_runtime_plan() -> dict[str, Any]:
    return {
        "version": GATE8_V1_SCIENTIFIC_POPULATION_RUNTIME_VERSION,
        "scientific_protocol_head": GATE8_V1_SCIENTIFIC_PROTOCOL_HEAD,
        "gemma_binding_result_head": GATE8_V1_GEMMA_BINDING_RESULT_HEAD,
        "architecture_head": GATE8_V1_ARCHITECTURE_HEAD,
        "runtime_head": GATE8_V1_RUNTIME_HEAD,
        "learned_parameter_count": GATE8_V1_LEARNED_PARAMETER_COUNT,
        "transition_table_entries_per_checkpoint": 2_048,
        "transition_table_compilation_exact": True,
        "test_split_admitted": True,
        "test_seed": 0,
        "test_world_indices": [0, 511],
        "population_modes": list(GATE8_V1_POPULATION_MODES),
        "reads_world_truth": False,
        "reference_model_loaded": False,
        "reference_inference_performed": False,
        "training_performed": False,
    }
