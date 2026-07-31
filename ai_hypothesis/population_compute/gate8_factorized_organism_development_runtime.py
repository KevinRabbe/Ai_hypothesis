"""Gate-8 v1 deterministic development runtime for training and validation.

This is a full-mode semantic mirror of the qualified v1 contract runtime. It
admits only contract, train and validation public worlds, and rejects
scientific-test and demonstration worlds before validation. It reads no truth
field and exposes no causal ablation or reference-model surface.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn

GATE8_V1_DEVELOPMENT_RUNTIME_VERSION = (
    "gate8-factorized-organism-development-runtime-v1"
)
GATE8_V1_DEVELOPMENT_RUNTIME_PROTOCOL_HEAD = (
    "a33dc123d090268a531d112251ea3ab53cb50062"
)
GATE8_V1_DEVELOPMENT_RUNTIME_QUALIFIED_RUNTIME_HEAD = (
    "333d88ac4fc52f1651741fba224e0b4605feedd3"
)
GATE8_V1_DEVELOPMENT_ALLOWED_SPLITS = ("contract", "train", "validation")
GATE8_V1_LEARNED_PARAMETER_COUNT = 19_649
GATE8_V1_MESSAGE_BITS = 8


@dataclass(frozen=True, slots=True)
class Gate8V1DevelopmentRuntimeResult:
    world_id: str
    split: str
    population: int
    depth: int
    root_seed_code: int
    target_worker_index: int
    target_reached: bool
    predicted_symbol: int | None
    target_message_code: int | None
    rounds_executed: int
    recurrent_updates: int
    emitted_messages: int
    delivered_messages: int
    communicated_bits: int

    def validate(self) -> None:
        if self.split not in GATE8_V1_DEVELOPMENT_ALLOWED_SPLITS:
            raise ValueError("Gate8 v1 development result split is not admitted")
        if not 0 <= self.root_seed_code < 256:
            raise ValueError("Gate8 v1 development root seed is outside 0..255")
        if self.rounds_executed < 0 or self.rounds_executed > self.depth:
            raise ValueError(
                "Gate8 v1 development round count is outside the horizon"
            )
        if min(
            self.recurrent_updates,
            self.emitted_messages,
            self.delivered_messages,
        ) < 0:
            raise ValueError("Gate8 v1 development accounting cannot be negative")
        if self.emitted_messages != self.recurrent_updates:
            raise ValueError("Gate8 v1 development emission accounting drifted")
        if self.delivered_messages != self.emitted_messages:
            raise ValueError("Gate8 v1 development delivery accounting drifted")
        if self.communicated_bits != self.delivered_messages * GATE8_V1_MESSAGE_BITS:
            raise ValueError("Gate8 v1 development bit accounting drifted")
        if self.target_reached:
            if self.predicted_symbol is None or not 0 <= self.predicted_symbol < 16:
                raise ValueError(
                    "Gate8 v1 reached development target lacks a valid answer"
                )
            if self.target_message_code is None or not 0 <= self.target_message_code < 256:
                raise ValueError(
                    "Gate8 v1 reached development target lacks a valid message"
                )
            if self.target_message_code & 0x0F != self.predicted_symbol:
                raise ValueError(
                    "Gate8 v1 development answer disagrees with target message"
                )
        elif (
            self.predicted_symbol is not None
            or self.target_message_code is not None
        ):
            raise ValueError(
                "Gate8 v1 unreached development target exposed an answer"
            )


def _public_world(world: Any) -> Any:
    public = getattr(world, "public", world)
    if getattr(public, "split", None) not in GATE8_V1_DEVELOPMENT_ALLOWED_SPLITS:
        raise ValueError("Gate8 v1 development runtime rejects this world split")
    validate = getattr(public, "validate", None)
    if not callable(validate):
        raise ValueError("Gate8 v1 development world lacks public validation")
    validate()
    return public


def _validate_model(model: nn.Module) -> torch.device:
    if model.training:
        raise ValueError("Gate8 v1 development runtime requires model.eval()")
    parameters = tuple(model.parameters())
    if not parameters:
        raise ValueError("Gate8 v1 development runtime requires learned parameters")
    if sum(parameter.numel() for parameter in parameters) != (
        GATE8_V1_LEARNED_PARAMETER_COUNT
    ):
        raise ValueError(
            "Gate8 v1 development runtime requires exactly 19,649 parameters"
        )
    devices = {parameter.device for parameter in parameters}
    dtypes = {parameter.dtype for parameter in parameters}
    if len(devices) != 1 or len(dtypes) != 1:
        raise ValueError(
            "Gate8 v1 development model parameters must share device and dtype"
        )
    if not next(iter(dtypes)).is_floating_point:
        raise ValueError("Gate8 v1 development model parameters must be floating point")
    if not callable(getattr(model, "initial_hidden", None)):
        raise ValueError("Gate8 v1 development model lacks initial_hidden")
    return next(iter(devices))


def _target_worker_and_rootedness(public: Any) -> int:
    incoming: dict[str, int] = {}
    reachable = {public.query.root_node}
    nodes = {public.query.root_node}
    for expected_index, worker in enumerate(public.workers):
        if worker.worker_index != expected_index:
            raise ValueError("Gate8 v1 development worker slots are not ordered")
        if worker.target_node in incoming:
            raise ValueError(
                "Gate8 v1 development node has multiple incoming workers"
            )
        incoming[worker.target_node] = worker.worker_index
        nodes.add(worker.source_node)
        nodes.add(worker.target_node)
    if public.query.root_node in incoming:
        raise ValueError("Gate8 v1 development root has an incoming worker")
    if public.query.target_node not in incoming:
        raise ValueError("Gate8 v1 development target lacks an incoming worker")
    if len(nodes) != public.population + 1:
        raise ValueError("Gate8 v1 development graph node count drifted")

    remaining = set(range(public.population))
    while remaining:
        progressed = False
        for worker_index in tuple(sorted(remaining)):
            worker = public.workers[worker_index]
            if worker.source_node in reachable:
                reachable.add(worker.target_node)
                remaining.remove(worker_index)
                progressed = True
        if not progressed:
            raise ValueError(
                "Gate8 v1 development graph contains unreachable edges"
            )
    return incoming[public.query.target_node]


def _validate_output(output: Any, count: int) -> None:
    expected = {
        "hidden": (count, 65),
        "carrier_logits": (count, 16),
        "symbol_logits": (count, 16),
    }
    for name, shape in expected.items():
        value = getattr(output, name, None)
        if not isinstance(value, Tensor) or tuple(value.shape) != shape:
            raise ValueError(
                f"Gate8 v1 development model returned invalid {name}"
            )


def run_gate8_v1_development_runtime(
    *,
    model: nn.Module,
    world: Any,
) -> Gate8V1DevelopmentRuntimeResult:
    public = _public_world(world)
    device = _validate_model(model)
    target_worker_index = _target_worker_and_rootedness(public)
    root_seed_code = public.query.root_symbol

    with torch.no_grad():
        hidden = model.initial_hidden(public.population)
        if tuple(hidden.shape) != (public.population, 65):
            raise ValueError("Gate8 v1 development initial hidden shape drifted")
        mailboxes: dict[str, int] = {
            public.query.root_node: root_seed_code,
        }
        rounds_executed = 0
        recurrent_updates = 0
        emitted_messages = 0
        delivered_messages = 0
        predicted_symbol: int | None = None
        target_message_code: int | None = None
        target_reached = False

        for _round_index in range(public.depth):
            scheduled = tuple(
                worker.worker_index
                for worker in public.workers
                if worker.source_node in mailboxes
            )
            if not scheduled:
                break
            indices = torch.tensor(scheduled, dtype=torch.long, device=device)
            output = model(
                inbox_code=torch.tensor(
                    [
                        mailboxes[public.workers[index].source_node]
                        for index in scheduled
                    ],
                    dtype=torch.long,
                    device=device,
                ),
                transform_id=torch.tensor(
                    [
                        public.workers[index].transform_id
                        for index in scheduled
                    ],
                    dtype=torch.long,
                    device=device,
                ),
                hidden=hidden.index_select(0, indices),
            )
            _validate_output(output, len(scheduled))
            hidden = hidden.index_copy(0, indices, output.hidden)
            recurrent_updates += len(scheduled)
            emitted_messages += len(scheduled)
            rounds_executed += 1

            carrier = torch.argmax(output.carrier_logits, dim=-1)
            symbol = torch.argmax(output.symbol_logits, dim=-1)
            codes = carrier * 16 + symbol
            positions = {
                worker_index: local_index
                for local_index, worker_index in enumerate(scheduled)
            }
            if target_worker_index in positions:
                local_index = positions[target_worker_index]
                target_reached = True
                predicted_symbol = int(symbol[local_index].item())
                target_message_code = int(codes[local_index].item())
                if target_message_code & 0x0F != predicted_symbol:
                    raise RuntimeError(
                        "Gate8 v1 development terminal message lost its symbol"
                    )

            next_mailboxes: dict[str, int] = {}
            for local_index, worker_index in enumerate(scheduled):
                target_node = public.workers[worker_index].target_node
                if target_node in next_mailboxes:
                    raise RuntimeError("Gate8 v1 development delivery collided")
                code = int(codes[local_index].item())
                if not 0 <= code < 256:
                    raise RuntimeError(
                        "Gate8 v1 development code is outside 0..255"
                    )
                next_mailboxes[target_node] = code
                delivered_messages += 1
            mailboxes = next_mailboxes
            if target_reached:
                break

    result = Gate8V1DevelopmentRuntimeResult(
        world_id=public.world_id,
        split=public.split,
        population=public.population,
        depth=public.depth,
        root_seed_code=root_seed_code,
        target_worker_index=target_worker_index,
        target_reached=target_reached,
        predicted_symbol=predicted_symbol,
        target_message_code=target_message_code,
        rounds_executed=rounds_executed,
        recurrent_updates=recurrent_updates,
        emitted_messages=emitted_messages,
        delivered_messages=delivered_messages,
        communicated_bits=delivered_messages * GATE8_V1_MESSAGE_BITS,
    )
    result.validate()
    return result


def gate8_v1_development_runtime_plan() -> dict[str, Any]:
    return {
        "version": GATE8_V1_DEVELOPMENT_RUNTIME_VERSION,
        "protocol_head": GATE8_V1_DEVELOPMENT_RUNTIME_PROTOCOL_HEAD,
        "qualified_runtime_head": (
            GATE8_V1_DEVELOPMENT_RUNTIME_QUALIFIED_RUNTIME_HEAD
        ),
        "allowed_splits": list(GATE8_V1_DEVELOPMENT_ALLOWED_SPLITS),
        "scientific_test_allowed": False,
        "demonstration_allowed": False,
        "truth_read": False,
        "mode": "full_only",
        "deterministic_delivery": True,
        "message_selection": "factorized_argmax",
        "answer_selection": "symbol_argmax",
        "answer_equals_message_low_nibble": True,
        "message_bits": 8,
        "root_seed": "carrier_zero_plus_public_root_symbol",
        "round_cap": "public_world_depth",
    }
