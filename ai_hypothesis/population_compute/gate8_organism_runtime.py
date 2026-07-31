"""Deterministic Gate-8 organism runtime for contract-only qualification.

This stage admits synchronous mailbox scheduling and inference accounting only.
It rejects every split except ``contract`` and executes an already-qualified
19,649-parameter worker core in evaluation mode under ``torch.no_grad``.
Training, checkpointing, scientific-test worlds, and the 1B reference remain
closed.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn

GATE8_ORGANISM_RUNTIME_VERSION = "gate8-organism-runtime-contract-v0"
GATE8_ORGANISM_RUNTIME_STATUS = (
    "GATE8_DETERMINISTIC_CONTRACT_RUNTIME_ADMITTED_TRAINING_AND_SCIENCE_CLOSED"
)
GATE8_ORGANISM_RUNTIME_ARCHITECTURE_HEAD = (
    "2afdcc9f13f138e97c7b3821cc2a5a77bd87cf0c"
)
GATE8_LEARNED_PARAMETER_COUNT = 19_649
GATE8_MESSAGE_CODEBOOK_SIZE = 256
GATE8_MESSAGE_BITS = 8
GATE8_ROOT_SEED_CODE = 0
GATE8_ACTIVITY_THRESHOLD = 0.0
GATE8_RUNTIME_FULL = "full"
GATE8_RUNTIME_NO_COMMUNICATION = "no_communication"
GATE8_RUNTIME_SHUFFLED_WORKER = "shuffled_worker"
GATE8_RUNTIME_MODES = (
    GATE8_RUNTIME_FULL,
    GATE8_RUNTIME_NO_COMMUNICATION,
    GATE8_RUNTIME_SHUFFLED_WORKER,
)


@dataclass(frozen=True, slots=True)
class Gate8DeliveredMessage:
    worker_index: int
    target_node: str
    code: int


@dataclass(frozen=True, slots=True)
class Gate8RuntimeRound:
    round_index: int
    mailbox_nodes_before: tuple[str, ...]
    scheduled_worker_indices: tuple[int, ...]
    activity_positive_worker_indices: tuple[int, ...]
    delivered_messages: tuple[Gate8DeliveredMessage, ...]
    recurrent_updates: int
    communicated_bits: int


@dataclass(frozen=True, slots=True)
class Gate8RuntimeResult:
    version: str
    mode: str
    world_id: str
    population: int
    depth: int
    rounds_executed: int
    target_worker_index: int
    target_reached: bool
    predicted_symbol: int | None
    answer_logits: tuple[float, ...] | None
    recurrent_updates: int
    activity_positive_workers: int
    delivered_messages: int
    communicated_bits: int
    effective_transform_ids: tuple[int, ...]
    shuffled_worker_permutation: tuple[int, ...]
    rounds: tuple[Gate8RuntimeRound, ...]

    def validate(self) -> None:
        if self.version != GATE8_ORGANISM_RUNTIME_VERSION:
            raise ValueError("Gate8 runtime result version drifted")
        if self.mode not in GATE8_RUNTIME_MODES:
            raise ValueError("Gate8 runtime result mode is unknown")
        if len(self.effective_transform_ids) != self.population:
            raise ValueError("Gate8 runtime transform assignment is incomplete")
        if len(self.shuffled_worker_permutation) != self.population:
            raise ValueError("Gate8 runtime worker permutation is incomplete")
        if self.rounds_executed != len(self.rounds):
            raise ValueError("Gate8 runtime round count is inconsistent")
        if self.recurrent_updates != sum(row.recurrent_updates for row in self.rounds):
            raise ValueError("Gate8 runtime recurrent-update accounting drifted")
        if self.activity_positive_workers != sum(
            len(row.activity_positive_worker_indices) for row in self.rounds
        ):
            raise ValueError("Gate8 runtime activity accounting drifted")
        if self.delivered_messages != sum(
            len(row.delivered_messages) for row in self.rounds
        ):
            raise ValueError("Gate8 runtime message accounting drifted")
        if self.communicated_bits != self.delivered_messages * GATE8_MESSAGE_BITS:
            raise ValueError("Gate8 runtime bit accounting drifted")
        if self.communicated_bits != sum(row.communicated_bits for row in self.rounds):
            raise ValueError("Gate8 runtime per-round bit accounting drifted")
        if self.target_reached:
            if self.predicted_symbol is None or self.answer_logits is None:
                raise ValueError("Gate8 reached target lacks an answer")
            if not 0 <= self.predicted_symbol < 16:
                raise ValueError("Gate8 predicted symbol is outside 0..15")
            if len(self.answer_logits) != 16:
                raise ValueError("Gate8 answer-logit width changed")
        elif self.predicted_symbol is not None or self.answer_logits is not None:
            raise ValueError("Gate8 unreached target may not expose an answer")
        if self.mode == GATE8_RUNTIME_NO_COMMUNICATION:
            if self.delivered_messages != 0 or self.communicated_bits != 0:
                raise ValueError("Gate8 no-communication ablation delivered a message")


def _validate_core(model: nn.Module) -> tuple[torch.device, torch.dtype]:
    if model.training:
        raise ValueError("Gate8 contract runtime requires model.eval()")
    parameters = tuple(model.parameters())
    if not parameters:
        raise ValueError("Gate8 contract runtime requires a learned worker core")
    if sum(parameter.numel() for parameter in parameters) != GATE8_LEARNED_PARAMETER_COUNT:
        raise ValueError("Gate8 contract runtime requires exactly 19,649 parameters")
    devices = {parameter.device for parameter in parameters}
    dtypes = {parameter.dtype for parameter in parameters}
    if len(devices) != 1 or len(dtypes) != 1:
        raise ValueError("Gate8 worker-core parameters must share one device and dtype")
    dtype = next(iter(dtypes))
    if not dtype.is_floating_point:
        raise ValueError("Gate8 worker-core parameters must be floating point")
    for required in ("role_ids", "initial_hidden"):
        if not callable(getattr(model, required, None)):
            raise ValueError(f"Gate8 worker core lacks required method: {required}")
    return next(iter(devices)), dtype


def _validate_contract_world(world: Any) -> None:
    if getattr(world, "split", None) != "contract":
        raise ValueError("Gate8 runtime admits contract worlds only")
    validate = getattr(world, "validate", None)
    if not callable(validate):
        raise ValueError("Gate8 runtime world lacks validation")
    validate()


def _topology(world: Any) -> tuple[dict[str, tuple[int, ...]], int]:
    source_to_workers: dict[str, list[int]] = {}
    incoming: dict[str, int] = {}
    nodes = {world.query.root_node}
    for expected_index, worker in enumerate(world.workers):
        if worker.worker_index != expected_index:
            raise ValueError("Gate8 runtime requires ordered contiguous worker slots")
        source_to_workers.setdefault(worker.source_node, []).append(worker.worker_index)
        if worker.target_node in incoming:
            raise ValueError("Gate8 runtime node has multiple incoming workers")
        incoming[worker.target_node] = worker.worker_index
        nodes.add(worker.source_node)
        nodes.add(worker.target_node)
    if world.query.root_node in incoming:
        raise ValueError("Gate8 runtime root has an incoming worker")
    if world.query.target_node not in incoming:
        raise ValueError("Gate8 runtime target lacks an incoming worker")
    if len(nodes) != world.population + 1:
        raise ValueError("Gate8 runtime graph is not a population-edge tree")

    reachable = {world.query.root_node}
    remaining = set(range(world.population))
    while remaining:
        progressed = False
        for worker_index in tuple(sorted(remaining)):
            worker = world.workers[worker_index]
            if worker.source_node in reachable:
                reachable.add(worker.target_node)
                remaining.remove(worker_index)
                progressed = True
        if not progressed:
            raise ValueError("Gate8 runtime graph contains an unreachable edge")
    if len(reachable) != world.population + 1:
        raise ValueError("Gate8 runtime graph is not fully rooted")
    frozen_sources = {
        node: tuple(indices)
        for node, indices in source_to_workers.items()
    }
    return frozen_sources, incoming[world.query.target_node]


def gate8_shuffled_worker_permutation(world_id: str, population: int) -> tuple[int, ...]:
    if population <= 1:
        raise ValueError("Gate8 shuffled-worker ablation requires multiple workers")
    ranked = sorted(
        range(population),
        key=lambda worker_index: hashlib.sha256(
            f"gate8-shuffled-worker-v0:{world_id}:{worker_index}".encode("ascii")
        ).digest(),
    )
    identity = list(range(population))
    if ranked == identity:
        ranked = ranked[1:] + ranked[:1]
    permutation = tuple(ranked)
    if tuple(sorted(permutation)) != tuple(identity):
        raise RuntimeError("Gate8 shuffled-worker permutation is not bijective")
    return permutation


def _effective_transform_assignment(
    world: Any,
    mode: str,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    identity = tuple(range(world.population))
    if mode != GATE8_RUNTIME_SHUFFLED_WORKER:
        return (
            tuple(worker.transform_id for worker in world.workers),
            identity,
        )
    permutation = gate8_shuffled_worker_permutation(world.world_id, world.population)
    transforms = tuple(
        world.workers[source_worker_index].transform_id
        for source_worker_index in permutation
    )
    if sorted(transforms) != sorted(worker.transform_id for worker in world.workers):
        raise RuntimeError("Gate8 shuffled-worker ablation changed transform marginals")
    return transforms, permutation


def _validate_step_output(output: Any, active_count: int) -> None:
    expected = {
        "hidden": (active_count, 32),
        "message_logits": (active_count, GATE8_MESSAGE_CODEBOOK_SIZE),
        "activity_logit": (active_count,),
        "answer_logits": (active_count, 16),
    }
    for name, shape in expected.items():
        value = getattr(output, name, None)
        if not isinstance(value, Tensor) or tuple(value.shape) != shape:
            raise ValueError(f"Gate8 worker core returned invalid {name}")


def run_gate8_contract_runtime(
    *,
    model: nn.Module,
    world: Any,
    mode: str = GATE8_RUNTIME_FULL,
) -> Gate8RuntimeResult:
    """Execute one qualified contract world with deterministic argmax semantics."""

    if mode not in GATE8_RUNTIME_MODES:
        raise ValueError("Gate8 runtime mode is unknown")
    _validate_contract_world(world)
    device, _dtype = _validate_core(model)
    _source_to_workers, target_worker_index = _topology(world)
    effective_transform_ids, permutation = _effective_transform_assignment(world, mode)

    source_is_root = torch.tensor(
        [worker.source_node == world.query.root_node for worker in world.workers],
        dtype=torch.bool,
        device=device,
    )
    target_is_query = torch.tensor(
        [worker.target_node == world.query.target_node for worker in world.workers],
        dtype=torch.bool,
        device=device,
    )
    role_ids = model.role_ids(
        source_is_root=source_is_root,
        target_is_query=target_is_query,
    )

    with torch.no_grad():
        hidden = model.initial_hidden(role_ids)
        if tuple(hidden.shape) != (world.population, 32):
            raise ValueError("Gate8 worker core returned invalid initial hidden state")
        mailboxes: dict[str, int] = {
            world.query.root_node: GATE8_ROOT_SEED_CODE,
        }
        traces: list[Gate8RuntimeRound] = []
        predicted_symbol: int | None = None
        answer_logits: tuple[float, ...] | None = None
        target_reached = False

        for round_index in range(world.depth):
            scheduled = tuple(
                worker.worker_index
                for worker in world.workers
                if worker.source_node in mailboxes
            )
            if not scheduled:
                break
            active_indices = torch.tensor(scheduled, dtype=torch.long, device=device)
            inbox_code = torch.tensor(
                [mailboxes[world.workers[index].source_node] for index in scheduled],
                dtype=torch.long,
                device=device,
            )
            transform_id = torch.tensor(
                [effective_transform_ids[index] for index in scheduled],
                dtype=torch.long,
                device=device,
            )
            root_symbol = torch.full(
                (len(scheduled),),
                world.query.root_symbol,
                dtype=torch.long,
                device=device,
            )
            output = model(
                inbox_code=inbox_code,
                transform_id=transform_id,
                root_symbol=root_symbol,
                source_is_root=source_is_root.index_select(0, active_indices),
                target_is_query=target_is_query.index_select(0, active_indices),
                inbox_present=torch.ones(len(scheduled), dtype=torch.bool, device=device),
                round_is_zero=torch.full(
                    (len(scheduled),),
                    round_index == 0,
                    dtype=torch.bool,
                    device=device,
                ),
                hidden=hidden.index_select(0, active_indices),
            )
            _validate_step_output(output, len(scheduled))
            hidden = hidden.index_copy(0, active_indices, output.hidden)

            scheduled_position = {
                worker_index: local_index
                for local_index, worker_index in enumerate(scheduled)
            }
            if target_worker_index in scheduled_position:
                local_index = scheduled_position[target_worker_index]
                terminal_logits = output.answer_logits[local_index]
                predicted_symbol = int(torch.argmax(terminal_logits).item())
                answer_logits = tuple(
                    float(value)
                    for value in terminal_logits.detach().cpu().tolist()
                )
                target_reached = True

            activity_mask = output.activity_logit >= GATE8_ACTIVITY_THRESHOLD
            message_codes = torch.argmax(output.message_logits, dim=-1)
            activity_positive = tuple(
                worker_index
                for local_index, worker_index in enumerate(scheduled)
                if bool(activity_mask[local_index].item())
            )
            next_mailboxes: dict[str, int] = {}
            delivered: list[Gate8DeliveredMessage] = []
            if mode != GATE8_RUNTIME_NO_COMMUNICATION:
                for local_index, worker_index in enumerate(scheduled):
                    if not bool(activity_mask[local_index].item()):
                        continue
                    worker = world.workers[worker_index]
                    if worker.target_node in next_mailboxes:
                        raise RuntimeError("Gate8 synchronous delivery collided at one node")
                    code = int(message_codes[local_index].item())
                    if not 0 <= code < GATE8_MESSAGE_CODEBOOK_SIZE:
                        raise RuntimeError("Gate8 emitted message code is outside 0..255")
                    next_mailboxes[worker.target_node] = code
                    delivered.append(
                        Gate8DeliveredMessage(
                            worker_index=worker_index,
                            target_node=worker.target_node,
                            code=code,
                        )
                    )

            trace = Gate8RuntimeRound(
                round_index=round_index,
                mailbox_nodes_before=tuple(sorted(mailboxes)),
                scheduled_worker_indices=scheduled,
                activity_positive_worker_indices=activity_positive,
                delivered_messages=tuple(delivered),
                recurrent_updates=len(scheduled),
                communicated_bits=len(delivered) * GATE8_MESSAGE_BITS,
            )
            traces.append(trace)
            mailboxes = next_mailboxes
            if target_reached:
                break

    result = Gate8RuntimeResult(
        version=GATE8_ORGANISM_RUNTIME_VERSION,
        mode=mode,
        world_id=world.world_id,
        population=world.population,
        depth=world.depth,
        rounds_executed=len(traces),
        target_worker_index=target_worker_index,
        target_reached=target_reached,
        predicted_symbol=predicted_symbol,
        answer_logits=answer_logits,
        recurrent_updates=sum(row.recurrent_updates for row in traces),
        activity_positive_workers=sum(
            len(row.activity_positive_worker_indices) for row in traces
        ),
        delivered_messages=sum(len(row.delivered_messages) for row in traces),
        communicated_bits=sum(row.communicated_bits for row in traces),
        effective_transform_ids=effective_transform_ids,
        shuffled_worker_permutation=permutation,
        rounds=tuple(traces),
    )
    result.validate()
    return result


def gate8_organism_runtime_plan() -> dict[str, Any]:
    return {
        "version": GATE8_ORGANISM_RUNTIME_VERSION,
        "scientific_status": GATE8_ORGANISM_RUNTIME_STATUS,
        "architecture_head": GATE8_ORGANISM_RUNTIME_ARCHITECTURE_HEAD,
        "learned_parameter_count": GATE8_LEARNED_PARAMETER_COUNT,
        "admitted_split": "contract",
        "root_seed_code": GATE8_ROOT_SEED_CODE,
        "root_seed_communicated_bits": 0,
        "activity_threshold": GATE8_ACTIVITY_THRESHOLD,
        "message_selection": "argmax",
        "answer_selection": "argmax_unique_target_incoming_worker",
        "round_cap": "public_world_depth",
        "delivery_semantics": "synchronous_next_round_node_mailbox",
        "one_message_per_scheduled_worker_per_round": True,
        "message_bits": GATE8_MESSAGE_BITS,
        "no_communication_ablation": "suppress_all_worker_to_node_delivery",
        "shuffled_worker_ablation": (
            "deterministically_permute_transform_shards_across_fixed_topology_slots"
        ),
        "truth_used_by_runtime": False,
        "model_outputs_used_to_construct_ablation": False,
        "training_admitted": False,
        "checkpoint_admitted": False,
        "scientific_test_worlds_admitted": False,
        "reference_model_admitted": False,
    }
