"""Gate-8 deterministic development runtime for training and validation.

This is a full-mode semantic mirror of the qualified contract runtime. It admits
only contract, train and validation public worlds, and rejects scientific-test
and demonstration worlds before validation. It reads no truth field and exposes
no causal ablation or reference-model surface.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn

GATE8_DEVELOPMENT_RUNTIME_VERSION = "gate8-organism-development-runtime-v0"
GATE8_DEVELOPMENT_RUNTIME_PROTOCOL_HEAD = (
    "869791e5b44089f9c79447b8ae212ce830f8496a"
)
GATE8_DEVELOPMENT_RUNTIME_QUALIFIED_RUNTIME_HEAD = (
    "1a2be148411bc71ba35fda12b035b724f06ec166"
)
GATE8_DEVELOPMENT_ALLOWED_SPLITS = ("contract", "train", "validation")
GATE8_LEARNED_PARAMETER_COUNT = 19_649
GATE8_ROOT_SEED_CODE = 0
GATE8_MESSAGE_BITS = 8


@dataclass(frozen=True, slots=True)
class Gate8DevelopmentRuntimeResult:
    world_id: str
    split: str
    population: int
    depth: int
    target_worker_index: int
    target_reached: bool
    predicted_symbol: int | None
    rounds_executed: int
    recurrent_updates: int
    delivered_messages: int
    communicated_bits: int

    def validate(self) -> None:
        if self.split not in GATE8_DEVELOPMENT_ALLOWED_SPLITS:
            raise ValueError("Gate8 development result split is not admitted")
        if self.rounds_executed < 0 or self.rounds_executed > self.depth:
            raise ValueError("Gate8 development round count is outside the horizon")
        if self.recurrent_updates < 0 or self.delivered_messages < 0:
            raise ValueError("Gate8 development accounting cannot be negative")
        if self.communicated_bits != self.delivered_messages * GATE8_MESSAGE_BITS:
            raise ValueError("Gate8 development bit accounting drifted")
        if self.target_reached:
            if self.predicted_symbol is None or not 0 <= self.predicted_symbol < 16:
                raise ValueError("Gate8 reached development target lacks a valid answer")
        elif self.predicted_symbol is not None:
            raise ValueError("Gate8 unreached development target exposed an answer")


def _public_world(world: Any) -> Any:
    public = getattr(world, "public", world)
    if getattr(public, "split", None) not in GATE8_DEVELOPMENT_ALLOWED_SPLITS:
        raise ValueError("Gate8 development runtime rejects this world split")
    validate = getattr(public, "validate", None)
    if not callable(validate):
        raise ValueError("Gate8 development world lacks public validation")
    validate()
    return public


def _validate_model(model: nn.Module) -> torch.device:
    if model.training:
        raise ValueError("Gate8 development runtime requires model.eval()")
    parameters = tuple(model.parameters())
    if sum(parameter.numel() for parameter in parameters) != GATE8_LEARNED_PARAMETER_COUNT:
        raise ValueError("Gate8 development runtime requires exactly 19,649 parameters")
    devices = {parameter.device for parameter in parameters}
    if len(devices) != 1:
        raise ValueError("Gate8 development model parameters must share one device")
    return next(iter(devices))


def _target_worker_and_rootedness(public: Any) -> int:
    incoming: dict[str, int] = {}
    reachable = {public.query.root_node}
    nodes = {public.query.root_node}
    for expected_index, worker in enumerate(public.workers):
        if worker.worker_index != expected_index:
            raise ValueError("Gate8 development worker slots are not ordered")
        if worker.target_node in incoming:
            raise ValueError("Gate8 development node has multiple incoming workers")
        incoming[worker.target_node] = worker.worker_index
        nodes.add(worker.source_node)
        nodes.add(worker.target_node)
    if public.query.root_node in incoming:
        raise ValueError("Gate8 development root has an incoming worker")
    if public.query.target_node not in incoming:
        raise ValueError("Gate8 development target lacks an incoming worker")
    if len(nodes) != public.population + 1:
        raise ValueError("Gate8 development graph node count drifted")

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
            raise ValueError("Gate8 development graph contains unreachable edges")
    return incoming[public.query.target_node]


def _validate_output(output: Any, count: int) -> None:
    expected = {
        "hidden": (count, 32),
        "message_logits": (count, 256),
        "activity_logit": (count,),
        "answer_logits": (count, 16),
    }
    for name, shape in expected.items():
        value = getattr(output, name, None)
        if not isinstance(value, Tensor) or tuple(value.shape) != shape:
            raise ValueError(f"Gate8 development model returned invalid {name}")


def run_gate8_development_runtime(
    *,
    model: nn.Module,
    world: Any,
) -> Gate8DevelopmentRuntimeResult:
    public = _public_world(world)
    device = _validate_model(model)
    target_worker_index = _target_worker_and_rootedness(public)
    source_is_root = torch.tensor(
        [worker.source_node == public.query.root_node for worker in public.workers],
        dtype=torch.bool,
        device=device,
    )
    target_is_query = torch.tensor(
        [worker.target_node == public.query.target_node for worker in public.workers],
        dtype=torch.bool,
        device=device,
    )
    role_ids = model.role_ids(
        source_is_root=source_is_root,
        target_is_query=target_is_query,
    )

    with torch.no_grad():
        hidden = model.initial_hidden(role_ids)
        if tuple(hidden.shape) != (public.population, 32):
            raise ValueError("Gate8 development initial hidden shape drifted")
        mailboxes: dict[str, int] = {public.query.root_node: GATE8_ROOT_SEED_CODE}
        rounds_executed = 0
        recurrent_updates = 0
        delivered_messages = 0
        predicted_symbol: int | None = None
        target_reached = False

        for round_index in range(public.depth):
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
                    [mailboxes[public.workers[index].source_node] for index in scheduled],
                    dtype=torch.long,
                    device=device,
                ),
                transform_id=torch.tensor(
                    [public.workers[index].transform_id for index in scheduled],
                    dtype=torch.long,
                    device=device,
                ),
                root_symbol=torch.full(
                    (len(scheduled),),
                    public.query.root_symbol,
                    dtype=torch.long,
                    device=device,
                ),
                source_is_root=source_is_root.index_select(0, indices),
                target_is_query=target_is_query.index_select(0, indices),
                inbox_present=torch.ones(len(scheduled), dtype=torch.bool, device=device),
                round_is_zero=torch.full(
                    (len(scheduled),),
                    round_index == 0,
                    dtype=torch.bool,
                    device=device,
                ),
                hidden=hidden.index_select(0, indices),
            )
            _validate_output(output, len(scheduled))
            hidden = hidden.index_copy(0, indices, output.hidden)
            recurrent_updates += len(scheduled)
            rounds_executed += 1

            positions = {
                worker_index: local_index
                for local_index, worker_index in enumerate(scheduled)
            }
            if target_worker_index in positions:
                target_reached = True
                predicted_symbol = int(
                    torch.argmax(output.answer_logits[positions[target_worker_index]]).item()
                )

            active = output.activity_logit >= 0.0
            codes = torch.argmax(output.message_logits, dim=-1)
            next_mailboxes: dict[str, int] = {}
            for local_index, worker_index in enumerate(scheduled):
                if not bool(active[local_index].item()):
                    continue
                target_node = public.workers[worker_index].target_node
                if target_node in next_mailboxes:
                    raise RuntimeError("Gate8 development delivery collided")
                code = int(codes[local_index].item())
                if not 0 <= code < 256:
                    raise RuntimeError("Gate8 development code is outside 0..255")
                next_mailboxes[target_node] = code
                delivered_messages += 1
            mailboxes = next_mailboxes
            if target_reached:
                break

    result = Gate8DevelopmentRuntimeResult(
        world_id=public.world_id,
        split=public.split,
        population=public.population,
        depth=public.depth,
        target_worker_index=target_worker_index,
        target_reached=target_reached,
        predicted_symbol=predicted_symbol,
        rounds_executed=rounds_executed,
        recurrent_updates=recurrent_updates,
        delivered_messages=delivered_messages,
        communicated_bits=delivered_messages * GATE8_MESSAGE_BITS,
    )
    result.validate()
    return result


def gate8_development_runtime_plan() -> dict[str, Any]:
    return {
        "version": GATE8_DEVELOPMENT_RUNTIME_VERSION,
        "protocol_head": GATE8_DEVELOPMENT_RUNTIME_PROTOCOL_HEAD,
        "qualified_runtime_head": GATE8_DEVELOPMENT_RUNTIME_QUALIFIED_RUNTIME_HEAD,
        "allowed_splits": list(GATE8_DEVELOPMENT_ALLOWED_SPLITS),
        "scientific_test_allowed": False,
        "demonstration_allowed": False,
        "truth_read": False,
        "mode": "full_only",
        "activity_threshold": 0.0,
        "message_selection": "argmax",
        "answer_selection": "argmax",
        "message_bits": 8,
        "root_seed_code": 0,
        "round_cap": "public_world_depth",
    }
