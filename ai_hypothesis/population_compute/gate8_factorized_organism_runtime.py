"""Deterministic Gate-8 v1 factorized-message runtime contract.

This stage admits contract-world synchronous mailbox execution and exact
resource accounting for the qualified 19,649-parameter v1 worker core. It
keeps training, checkpointing, scientific-test worlds, seeds 1/2, and the 1B
reference model closed.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn

GATE8_V1_RUNTIME_VERSION = "gate8-factorized-message-runtime-contract-v1"
GATE8_V1_RUNTIME_STATUS = (
    "GATE8_V1_DETERMINISTIC_CONTRACT_RUNTIME_ADMITTED_"
    "TRAINING_AND_SCIENCE_CLOSED"
)
GATE8_V1_RUNTIME_ARCHITECTURE_HEAD = (
    "c3ab64008c816fa1eb6f9d6f8f0a1a99ed195ec8"
)

GATE8_V1_LEARNED_PARAMETER_COUNT = 19_649
GATE8_V1_MESSAGE_CODEBOOK_SIZE = 256
GATE8_V1_MESSAGE_BITS = 8
GATE8_V1_SYMBOL_COUNT = 16
GATE8_V1_WORKER_STATE_WIDTH = 65

GATE8_V1_RUNTIME_FULL = "full"
GATE8_V1_RUNTIME_NO_COMMUNICATION = "no_communication"
GATE8_V1_RUNTIME_SHUFFLED_WORKER = "shuffled_worker"
GATE8_V1_RUNTIME_MODES = (
    GATE8_V1_RUNTIME_FULL,
    GATE8_V1_RUNTIME_NO_COMMUNICATION,
    GATE8_V1_RUNTIME_SHUFFLED_WORKER,
)


@dataclass(frozen=True, slots=True)
class Gate8V1DeliveredMessage:
    worker_index: int
    target_node: str
    code: int

    def validate(self, *, population: int) -> None:
        if not 0 <= self.worker_index < population:
            raise ValueError("Gate8 v1 delivered-message worker index drifted")
        if not self.target_node:
            raise ValueError("Gate8 v1 delivered-message target is empty")
        if not 0 <= self.code < GATE8_V1_MESSAGE_CODEBOOK_SIZE:
            raise ValueError("Gate8 v1 delivered-message code is outside 0..255")


@dataclass(frozen=True, slots=True)
class Gate8V1RuntimeRound:
    round_index: int
    mailbox_nodes_before: tuple[str, ...]
    scheduled_worker_indices: tuple[int, ...]
    inbox_codes: tuple[int, ...]
    emitted_message_codes: tuple[int, ...]
    delivered_messages: tuple[Gate8V1DeliveredMessage, ...]
    recurrent_updates: int
    communicated_bits: int

    def validate(self, *, population: int, mode: str) -> None:
        if self.round_index < 0:
            raise ValueError("Gate8 v1 runtime round index is negative")
        if tuple(sorted(set(self.mailbox_nodes_before))) != self.mailbox_nodes_before:
            raise ValueError("Gate8 v1 mailbox-node trace is not sorted and unique")
        scheduled_count = len(self.scheduled_worker_indices)
        if scheduled_count == 0:
            raise ValueError("Gate8 v1 recorded an empty runtime round")
        if len(set(self.scheduled_worker_indices)) != scheduled_count:
            raise ValueError("Gate8 v1 scheduled one worker more than once in a round")
        if any(not 0 <= index < population for index in self.scheduled_worker_indices):
            raise ValueError("Gate8 v1 scheduled worker index is outside population")
        if len(self.inbox_codes) != scheduled_count:
            raise ValueError("Gate8 v1 inbox trace is not aligned to scheduled workers")
        if len(self.emitted_message_codes) != scheduled_count:
            raise ValueError("Gate8 v1 emission trace is not aligned to scheduled workers")
        if any(not 0 <= code < 256 for code in self.inbox_codes):
            raise ValueError("Gate8 v1 inbox trace contains an invalid code")
        if any(not 0 <= code < 256 for code in self.emitted_message_codes):
            raise ValueError("Gate8 v1 emission trace contains an invalid code")
        if self.recurrent_updates != scheduled_count:
            raise ValueError("Gate8 v1 recurrent-update count drifted")
        for delivered in self.delivered_messages:
            delivered.validate(population=population)
        if mode == GATE8_V1_RUNTIME_NO_COMMUNICATION:
            if self.delivered_messages or self.communicated_bits != 0:
                raise ValueError("Gate8 v1 no-communication round delivered a message")
        else:
            if len(self.delivered_messages) != scheduled_count:
                raise ValueError("Gate8 v1 deterministic runtime suppressed a delivery")
            delivered_indices = tuple(
                message.worker_index for message in self.delivered_messages
            )
            if delivered_indices != self.scheduled_worker_indices:
                raise ValueError("Gate8 v1 delivery order drifted from scheduling order")
            delivered_codes = tuple(message.code for message in self.delivered_messages)
            if delivered_codes != self.emitted_message_codes:
                raise ValueError("Gate8 v1 delivered code drifted from emitted code")
        if self.communicated_bits != len(self.delivered_messages) * 8:
            raise ValueError("Gate8 v1 per-round bit accounting drifted")


@dataclass(frozen=True, slots=True)
class Gate8V1RuntimeResult:
    version: str
    mode: str
    world_id: str
    population: int
    depth: int
    root_seed_code: int
    rounds_executed: int
    target_worker_index: int
    target_reached: bool
    predicted_symbol: int | None
    target_message_code: int | None
    symbol_logits: tuple[float, ...] | None
    recurrent_updates: int
    emitted_messages: int
    delivered_messages: int
    communicated_bits: int
    effective_transform_ids: tuple[int, ...]
    shuffled_worker_permutation: tuple[int, ...]
    rounds: tuple[Gate8V1RuntimeRound, ...]

    def validate(self) -> None:
        if self.version != GATE8_V1_RUNTIME_VERSION:
            raise ValueError("Gate8 v1 runtime result version drifted")
        if self.mode not in GATE8_V1_RUNTIME_MODES:
            raise ValueError("Gate8 v1 runtime result mode is unknown")
        if self.population <= 0 or self.depth <= 0:
            raise ValueError("Gate8 v1 runtime result dimensions are invalid")
        if not 0 <= self.root_seed_code < 256:
            raise ValueError("Gate8 v1 root seed code is invalid")
        if not 0 <= self.target_worker_index < self.population:
            raise ValueError("Gate8 v1 target worker index is invalid")
        if len(self.effective_transform_ids) != self.population:
            raise ValueError("Gate8 v1 transform assignment is incomplete")
        if any(not 0 <= value < 8 for value in self.effective_transform_ids):
            raise ValueError("Gate8 v1 transform assignment contains an invalid ID")
        identity = tuple(range(self.population))
        if tuple(sorted(self.shuffled_worker_permutation)) != identity:
            raise ValueError("Gate8 v1 worker permutation is not bijective")
        if self.rounds_executed != len(self.rounds):
            raise ValueError("Gate8 v1 round count is inconsistent")
        for expected_round, row in enumerate(self.rounds):
            if row.round_index != expected_round:
                raise ValueError("Gate8 v1 runtime round indices are not contiguous")
            row.validate(population=self.population, mode=self.mode)
        if self.recurrent_updates != sum(row.recurrent_updates for row in self.rounds):
            raise ValueError("Gate8 v1 recurrent-update accounting drifted")
        if self.emitted_messages != sum(
            len(row.emitted_message_codes) for row in self.rounds
        ):
            raise ValueError("Gate8 v1 emission accounting drifted")
        if self.emitted_messages != self.recurrent_updates:
            raise ValueError("Gate8 v1 one-emission-per-update invariant drifted")
        if self.delivered_messages != sum(
            len(row.delivered_messages) for row in self.rounds
        ):
            raise ValueError("Gate8 v1 delivery accounting drifted")
        if self.communicated_bits != self.delivered_messages * GATE8_V1_MESSAGE_BITS:
            raise ValueError("Gate8 v1 communicated-bit accounting drifted")
        if self.communicated_bits != sum(row.communicated_bits for row in self.rounds):
            raise ValueError("Gate8 v1 per-round communicated bits drifted")
        if self.mode == GATE8_V1_RUNTIME_NO_COMMUNICATION:
            if self.delivered_messages != 0 or self.communicated_bits != 0:
                raise ValueError("Gate8 v1 no-communication result delivered a message")
        elif self.delivered_messages != self.emitted_messages:
            raise ValueError("Gate8 v1 deterministic runtime lost an emitted message")

        if self.target_reached:
            if (
                self.predicted_symbol is None
                or self.target_message_code is None
                or self.symbol_logits is None
            ):
                raise ValueError("Gate8 v1 reached target lacks terminal evidence")
            if not 0 <= self.predicted_symbol < GATE8_V1_SYMBOL_COUNT:
                raise ValueError("Gate8 v1 predicted symbol is outside 0..15")
            if not 0 <= self.target_message_code < 256:
                raise ValueError("Gate8 v1 target message code is invalid")
            if len(self.symbol_logits) != GATE8_V1_SYMBOL_COUNT:
                raise ValueError("Gate8 v1 terminal symbol-logit width drifted")
            if (self.target_message_code & 0x0F) != self.predicted_symbol:
                raise ValueError(
                    "Gate8 v1 terminal answer disagrees with emitted message symbol"
                )
        elif any(
            value is not None
            for value in (
                self.predicted_symbol,
                self.target_message_code,
                self.symbol_logits,
            )
        ):
            raise ValueError("Gate8 v1 unreached target exposed terminal evidence")


def _validate_core(model: nn.Module) -> tuple[torch.device, torch.dtype]:
    if model.training:
        raise ValueError("Gate8 v1 contract runtime requires model.eval()")
    parameters = tuple(model.parameters())
    if not parameters:
        raise ValueError("Gate8 v1 contract runtime requires a learned worker core")
    if sum(parameter.numel() for parameter in parameters) != (
        GATE8_V1_LEARNED_PARAMETER_COUNT
    ):
        raise ValueError("Gate8 v1 contract runtime requires exactly 19,649 parameters")
    devices = {parameter.device for parameter in parameters}
    dtypes = {parameter.dtype for parameter in parameters}
    if len(devices) != 1 or len(dtypes) != 1:
        raise ValueError("Gate8 v1 core parameters must share one device and dtype")
    dtype = next(iter(dtypes))
    if not dtype.is_floating_point:
        raise ValueError("Gate8 v1 core parameters must be floating point")
    for required in (
        "initial_hidden",
        "root_message_code",
        "predicted_message_code",
        "predicted_symbol",
    ):
        if not callable(getattr(model, required, None)):
            raise ValueError(f"Gate8 v1 worker core lacks required method: {required}")
    return next(iter(devices)), dtype


def _validate_contract_world(world: Any) -> None:
    if getattr(world, "split", None) != "contract":
        raise ValueError("Gate8 v1 runtime admits contract worlds only")
    validate = getattr(world, "validate", None)
    if not callable(validate):
        raise ValueError("Gate8 v1 runtime world lacks validation")
    validate()


def _topology(world: Any) -> int:
    incoming: dict[str, int] = {}
    nodes = {world.query.root_node}
    for expected_index, worker in enumerate(world.workers):
        if worker.worker_index != expected_index:
            raise ValueError("Gate8 v1 runtime requires contiguous ordered worker slots")
        if worker.target_node in incoming:
            raise ValueError("Gate8 v1 runtime node has multiple incoming workers")
        incoming[worker.target_node] = worker.worker_index
        nodes.add(worker.source_node)
        nodes.add(worker.target_node)
    if world.query.root_node in incoming:
        raise ValueError("Gate8 v1 runtime root has an incoming worker")
    if world.query.target_node not in incoming:
        raise ValueError("Gate8 v1 runtime target lacks an incoming worker")
    if len(nodes) != world.population + 1:
        raise ValueError("Gate8 v1 runtime graph is not a population-edge tree")

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
            raise ValueError("Gate8 v1 runtime graph contains an unreachable edge")
    if len(reachable) != world.population + 1:
        raise ValueError("Gate8 v1 runtime graph is not fully rooted")
    return incoming[world.query.target_node]


def gate8_v1_shuffled_worker_permutation(
    world_id: str,
    population: int,
) -> tuple[int, ...]:
    if not world_id:
        raise ValueError("Gate8 v1 shuffled-worker world ID is empty")
    if population <= 1:
        raise ValueError("Gate8 v1 shuffled-worker ablation requires multiple workers")
    ranked = sorted(
        range(population),
        key=lambda worker_index: hashlib.sha256(
            f"gate8-v1-shuffled-worker:{world_id}:{worker_index}".encode("ascii")
        ).digest(),
    )
    identity = list(range(population))
    if ranked == identity:
        ranked = ranked[1:] + ranked[:1]
    permutation = tuple(ranked)
    if tuple(sorted(permutation)) != tuple(identity):
        raise RuntimeError("Gate8 v1 shuffled-worker permutation is not bijective")
    return permutation


def _effective_transform_assignment(
    world: Any,
    mode: str,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    identity = tuple(range(world.population))
    if mode != GATE8_V1_RUNTIME_SHUFFLED_WORKER:
        return tuple(worker.transform_id for worker in world.workers), identity
    permutation = gate8_v1_shuffled_worker_permutation(
        world.world_id,
        world.population,
    )
    transforms = tuple(
        world.workers[source_worker_index].transform_id
        for source_worker_index in permutation
    )
    original = tuple(worker.transform_id for worker in world.workers)
    if sorted(transforms) != sorted(original):
        raise RuntimeError("Gate8 v1 shuffled-worker changed transform marginals")
    return transforms, permutation


def _validate_step_output(output: Any, active_count: int) -> None:
    expected = {
        "hidden": (active_count, GATE8_V1_WORKER_STATE_WIDTH),
        "carrier_logits": (active_count, GATE8_V1_SYMBOL_COUNT),
        "symbol_logits": (active_count, GATE8_V1_SYMBOL_COUNT),
    }
    for name, shape in expected.items():
        value = getattr(output, name, None)
        if not isinstance(value, Tensor) or tuple(value.shape) != shape:
            raise ValueError(f"Gate8 v1 worker core returned invalid {name}")


def _validate_prediction_vector(
    name: str,
    value: Any,
    *,
    active_count: int,
    upper_bound: int,
) -> Tensor:
    if not isinstance(value, Tensor):
        raise ValueError(f"Gate8 v1 worker core returned non-tensor {name}")
    if value.dtype != torch.long or tuple(value.shape) != (active_count,):
        raise ValueError(f"Gate8 v1 worker core returned invalid {name}")
    if value.numel() and (
        int(value.min().item()) < 0
        or int(value.max().item()) >= upper_bound
    ):
        raise ValueError(f"Gate8 v1 worker core returned out-of-range {name}")
    return value


def run_gate8_v1_contract_runtime(
    *,
    model: nn.Module,
    world: Any,
    mode: str = GATE8_V1_RUNTIME_FULL,
) -> Gate8V1RuntimeResult:
    """Execute one contract world with synchronous deterministic delivery."""

    if mode not in GATE8_V1_RUNTIME_MODES:
        raise ValueError("Gate8 v1 runtime mode is unknown")
    _validate_contract_world(world)
    device, dtype = _validate_core(model)
    target_worker_index = _topology(world)
    effective_transform_ids, permutation = _effective_transform_assignment(world, mode)

    root_symbol = torch.tensor(
        [world.query.root_symbol],
        dtype=torch.long,
        device=device,
    )
    root_seed_tensor = model.root_message_code(root_symbol)
    root_seed_tensor = _validate_prediction_vector(
        "root seed code",
        root_seed_tensor,
        active_count=1,
        upper_bound=GATE8_V1_MESSAGE_CODEBOOK_SIZE,
    )
    root_seed_code = int(root_seed_tensor.item())
    if (root_seed_code >> 4) != 0 or (root_seed_code & 0x0F) != (
        world.query.root_symbol
    ):
        raise ValueError("Gate8 v1 core encoded the root symbol incorrectly")

    with torch.no_grad():
        hidden = model.initial_hidden(
            world.population,
            device=device,
            dtype=dtype,
        )
        if not isinstance(hidden, Tensor) or tuple(hidden.shape) != (
            world.population,
            GATE8_V1_WORKER_STATE_WIDTH,
        ):
            raise ValueError("Gate8 v1 core returned invalid initial hidden state")
        if hidden.device != device or hidden.dtype != dtype:
            raise ValueError("Gate8 v1 initial hidden state device or dtype drifted")

        mailboxes: dict[str, int] = {
            world.query.root_node: root_seed_code,
        }
        traces: list[Gate8V1RuntimeRound] = []
        target_reached = False
        predicted_symbol: int | None = None
        target_message_code: int | None = None
        symbol_logits: tuple[float, ...] | None = None

        for round_index in range(world.depth):
            scheduled = tuple(
                worker.worker_index
                for worker in world.workers
                if worker.source_node in mailboxes
            )
            if not scheduled:
                break
            active_indices = torch.tensor(
                scheduled,
                dtype=torch.long,
                device=device,
            )
            inbox_values = tuple(
                mailboxes[world.workers[index].source_node]
                for index in scheduled
            )
            inbox_code = torch.tensor(
                inbox_values,
                dtype=torch.long,
                device=device,
            )
            transform_id = torch.tensor(
                [effective_transform_ids[index] for index in scheduled],
                dtype=torch.long,
                device=device,
            )
            output = model(
                inbox_code=inbox_code,
                transform_id=transform_id,
                hidden=hidden.index_select(0, active_indices),
            )
            _validate_step_output(output, len(scheduled))
            if output.hidden.device != device or output.hidden.dtype != dtype:
                raise ValueError("Gate8 v1 output hidden device or dtype drifted")
            hidden = hidden.index_copy(0, active_indices, output.hidden)

            message_codes_tensor = _validate_prediction_vector(
                "message codes",
                model.predicted_message_code(output),
                active_count=len(scheduled),
                upper_bound=GATE8_V1_MESSAGE_CODEBOOK_SIZE,
            )
            symbol_predictions = _validate_prediction_vector(
                "symbol predictions",
                model.predicted_symbol(output),
                active_count=len(scheduled),
                upper_bound=GATE8_V1_SYMBOL_COUNT,
            )
            message_codes = tuple(
                int(value) for value in message_codes_tensor.detach().cpu().tolist()
            )

            for local_index, message_code in enumerate(message_codes):
                if (message_code & 0x0F) != int(symbol_predictions[local_index].item()):
                    raise ValueError(
                        "Gate8 v1 emitted message disagrees with symbol prediction"
                    )

            scheduled_position = {
                worker_index: local_index
                for local_index, worker_index in enumerate(scheduled)
            }
            if target_worker_index in scheduled_position:
                local_index = scheduled_position[target_worker_index]
                predicted_symbol = int(symbol_predictions[local_index].item())
                target_message_code = message_codes[local_index]
                terminal_logits = output.symbol_logits[local_index]
                symbol_logits = tuple(
                    float(value)
                    for value in terminal_logits.detach().cpu().tolist()
                )
                target_reached = True

            next_mailboxes: dict[str, int] = {}
            delivered: list[Gate8V1DeliveredMessage] = []
            if mode != GATE8_V1_RUNTIME_NO_COMMUNICATION:
                for local_index, worker_index in enumerate(scheduled):
                    worker = world.workers[worker_index]
                    if worker.target_node in next_mailboxes:
                        raise RuntimeError(
                            "Gate8 v1 synchronous delivery collided at one node"
                        )
                    code = message_codes[local_index]
                    next_mailboxes[worker.target_node] = code
                    delivered.append(
                        Gate8V1DeliveredMessage(
                            worker_index=worker_index,
                            target_node=worker.target_node,
                            code=code,
                        )
                    )

            trace = Gate8V1RuntimeRound(
                round_index=round_index,
                mailbox_nodes_before=tuple(sorted(mailboxes)),
                scheduled_worker_indices=scheduled,
                inbox_codes=inbox_values,
                emitted_message_codes=message_codes,
                delivered_messages=tuple(delivered),
                recurrent_updates=len(scheduled),
                communicated_bits=len(delivered) * GATE8_V1_MESSAGE_BITS,
            )
            trace.validate(population=world.population, mode=mode)
            traces.append(trace)
            mailboxes = next_mailboxes
            if target_reached:
                break

    result = Gate8V1RuntimeResult(
        version=GATE8_V1_RUNTIME_VERSION,
        mode=mode,
        world_id=world.world_id,
        population=world.population,
        depth=world.depth,
        root_seed_code=root_seed_code,
        rounds_executed=len(traces),
        target_worker_index=target_worker_index,
        target_reached=target_reached,
        predicted_symbol=predicted_symbol,
        target_message_code=target_message_code,
        symbol_logits=symbol_logits,
        recurrent_updates=sum(row.recurrent_updates for row in traces),
        emitted_messages=sum(len(row.emitted_message_codes) for row in traces),
        delivered_messages=sum(len(row.delivered_messages) for row in traces),
        communicated_bits=sum(row.communicated_bits for row in traces),
        effective_transform_ids=effective_transform_ids,
        shuffled_worker_permutation=permutation,
        rounds=tuple(traces),
    )
    result.validate()
    return result


def gate8_v1_runtime_plan() -> dict[str, Any]:
    return {
        "version": GATE8_V1_RUNTIME_VERSION,
        "scientific_status": GATE8_V1_RUNTIME_STATUS,
        "architecture_head": GATE8_V1_RUNTIME_ARCHITECTURE_HEAD,
        "learned_parameter_count": GATE8_V1_LEARNED_PARAMETER_COUNT,
        "admitted_split": "contract",
        "root_seed": "carrier_zero_plus_public_root_symbol",
        "synchronous_delivery": True,
        "deterministic_delivery_for_scheduled_workers": True,
        "one_emission_per_recurrent_update": True,
        "message_bits": GATE8_V1_MESSAGE_BITS,
        "terminal_answer": "target_worker_symbol_head_argmax",
        "terminal_answer_equals_message_low_nibble": True,
        "activity_gate": False,
        "runtime_modes": GATE8_V1_RUNTIME_MODES,
        "no_communication_ablation": True,
        "shuffled_worker_ablation": True,
        "reads_world_truth": False,
        "training_admitted": False,
        "checkpoint_loading_admitted": False,
        "scientific_test_worlds_admitted": False,
        "reference_model_admitted": False,
    }
