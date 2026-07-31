"""Controlled Gate-8 seed-0 diagnostic runtime interventions.

This runtime is a semantic mirror of the qualified development runtime with only
two preregistered interventions: force all scheduled activity gates open and/or
decode the terminal answer from the low four bits of the terminal message argmax.
It admits only contract, train, and validation worlds and never reads truth.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn

GATE8_SEED0_DIAGNOSTIC_RUNTIME_VERSION = (
    "gate8-seed0-causal-diagnostic-runtime-v0"
)
GATE8_SEED0_DIAGNOSTIC_PROTOCOL_HEAD = (
    "0fa9ec48c31b36c90d58da827139457fd812b98c"
)
GATE8_QUALIFIED_RUNTIME_HEAD = "1a2be148411bc71ba35fda12b035b724f06ec166"
GATE8_ALLOWED_SPLITS = ("contract", "train", "validation")
GATE8_LEARNED_PARAMETER_COUNT = 19_649
GATE8_ROOT_SEED_CODE = 0
GATE8_MESSAGE_BITS = 8
GATE8_RUNTIME_PROBES = (
    "baseline",
    "forced_active",
    "message_low4_decode",
    "forced_active_message_low4_decode",
)


@dataclass(frozen=True, slots=True)
class Gate8DiagnosticRuntimeResult:
    world_id: str
    split: str
    population: int
    depth: int
    probe: str
    target_worker_index: int
    target_reached: bool
    predicted_symbol: int | None
    target_message_code: int | None
    rounds_executed: int
    recurrent_updates: int
    delivered_messages: int
    communicated_bits: int

    def validate(self) -> None:
        if self.split not in GATE8_ALLOWED_SPLITS:
            raise ValueError("Gate8 diagnostic result split is not admitted")
        if self.probe not in GATE8_RUNTIME_PROBES:
            raise ValueError("Gate8 diagnostic result probe is unknown")
        if not 0 <= self.rounds_executed <= self.depth:
            raise ValueError("Gate8 diagnostic round count is outside the horizon")
        if self.recurrent_updates < 0 or self.delivered_messages < 0:
            raise ValueError("Gate8 diagnostic accounting cannot be negative")
        if self.communicated_bits != self.delivered_messages * GATE8_MESSAGE_BITS:
            raise ValueError("Gate8 diagnostic bit accounting drifted")
        if self.target_reached:
            if self.predicted_symbol is None or not 0 <= self.predicted_symbol < 16:
                raise ValueError("Gate8 reached diagnostic target lacks a valid answer")
            if self.target_message_code is None or not 0 <= self.target_message_code < 256:
                raise ValueError("Gate8 reached diagnostic target lacks a valid message")
        elif self.predicted_symbol is not None or self.target_message_code is not None:
            raise ValueError("Gate8 unreached diagnostic target exposed an output")


@dataclass(frozen=True, slots=True)
class Gate8RootInvarianceResult:
    cases: int
    message_invariant_cases: int
    answer_invariant_cases: int
    activity_invariant_cases: int

    def validate(self) -> None:
        if self.cases != 256 * 8:
            raise ValueError("Gate8 root-invariance case count drifted")
        for value in (
            self.message_invariant_cases,
            self.answer_invariant_cases,
            self.activity_invariant_cases,
        ):
            if not 0 <= value <= self.cases:
                raise ValueError("Gate8 root-invariance count is outside its range")

    @property
    def message_root_invariance(self) -> float:
        self.validate()
        return self.message_invariant_cases / self.cases

    @property
    def answer_root_invariance(self) -> float:
        self.validate()
        return self.answer_invariant_cases / self.cases

    @property
    def activity_root_invariance(self) -> float:
        self.validate()
        return self.activity_invariant_cases / self.cases


def _public_world(world: Any) -> Any:
    public = getattr(world, "public", world)
    if getattr(public, "split", None) not in GATE8_ALLOWED_SPLITS:
        raise ValueError("Gate8 diagnostic runtime rejects this world split")
    validate = getattr(public, "validate", None)
    if not callable(validate):
        raise ValueError("Gate8 diagnostic world lacks public validation")
    validate()
    return public


def _validate_model(model: nn.Module) -> torch.device:
    if model.training:
        raise ValueError("Gate8 diagnostic runtime requires model.eval()")
    parameters = tuple(model.parameters())
    if sum(parameter.numel() for parameter in parameters) != GATE8_LEARNED_PARAMETER_COUNT:
        raise ValueError("Gate8 diagnostic runtime requires exactly 19,649 parameters")
    devices = {parameter.device for parameter in parameters}
    if len(devices) != 1:
        raise ValueError("Gate8 diagnostic model parameters must share one device")
    return next(iter(devices))


def _target_worker(public: Any) -> int:
    incoming: dict[str, int] = {}
    reachable = {public.query.root_node}
    nodes = {public.query.root_node}
    for expected_index, worker in enumerate(public.workers):
        if worker.worker_index != expected_index:
            raise ValueError("Gate8 diagnostic worker slots are not ordered")
        if worker.target_node in incoming:
            raise ValueError("Gate8 diagnostic node has multiple incoming workers")
        incoming[worker.target_node] = worker.worker_index
        nodes.add(worker.source_node)
        nodes.add(worker.target_node)
    if public.query.root_node in incoming:
        raise ValueError("Gate8 diagnostic root has an incoming worker")
    if public.query.target_node not in incoming:
        raise ValueError("Gate8 diagnostic target lacks an incoming worker")
    if len(nodes) != public.population + 1:
        raise ValueError("Gate8 diagnostic graph node count drifted")
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
            raise ValueError("Gate8 diagnostic graph contains unreachable edges")
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
            raise ValueError(f"Gate8 diagnostic model returned invalid {name}")


def run_gate8_seed0_diagnostic_runtime(
    *,
    model: nn.Module,
    world: Any,
    probe: str,
) -> Gate8DiagnosticRuntimeResult:
    if probe not in GATE8_RUNTIME_PROBES:
        raise ValueError("Gate8 diagnostic runtime probe is unknown")
    force_active = probe in (
        "forced_active",
        "forced_active_message_low4_decode",
    )
    decode_message = probe in (
        "message_low4_decode",
        "forced_active_message_low4_decode",
    )
    public = _public_world(world)
    device = _validate_model(model)
    target_worker_index = _target_worker(public)
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
            raise ValueError("Gate8 diagnostic initial hidden shape drifted")
        mailboxes: dict[str, int] = {public.query.root_node: GATE8_ROOT_SEED_CODE}
        rounds_executed = 0
        recurrent_updates = 0
        delivered_messages = 0
        predicted_symbol: int | None = None
        target_message_code: int | None = None
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
            codes = torch.argmax(output.message_logits, dim=-1)
            positions = {
                worker_index: local_index
                for local_index, worker_index in enumerate(scheduled)
            }
            if target_worker_index in positions:
                position = positions[target_worker_index]
                target_reached = True
                target_message_code = int(codes[position].item())
                predicted_symbol = (
                    target_message_code % 16
                    if decode_message
                    else int(torch.argmax(output.answer_logits[position]).item())
                )

            active = (
                torch.ones(len(scheduled), dtype=torch.bool, device=device)
                if force_active
                else output.activity_logit >= 0.0
            )
            next_mailboxes: dict[str, int] = {}
            for local_index, worker_index in enumerate(scheduled):
                if not bool(active[local_index].item()):
                    continue
                target_node = public.workers[worker_index].target_node
                if target_node in next_mailboxes:
                    raise RuntimeError("Gate8 diagnostic delivery collided")
                code = int(codes[local_index].item())
                if not 0 <= code < 256:
                    raise RuntimeError("Gate8 diagnostic code is outside 0..255")
                next_mailboxes[target_node] = code
                delivered_messages += 1
            mailboxes = next_mailboxes
            if target_reached:
                break

    result = Gate8DiagnosticRuntimeResult(
        world_id=public.world_id,
        split=public.split,
        population=public.population,
        depth=public.depth,
        probe=probe,
        target_worker_index=target_worker_index,
        target_reached=target_reached,
        predicted_symbol=predicted_symbol,
        target_message_code=target_message_code,
        rounds_executed=rounds_executed,
        recurrent_updates=recurrent_updates,
        delivered_messages=delivered_messages,
        communicated_bits=delivered_messages * GATE8_MESSAGE_BITS,
    )
    result.validate()
    return result


def evaluate_gate8_nonroot_target_root_invariance(
    *,
    model: nn.Module,
) -> Gate8RootInvarianceResult:
    device = _validate_model(model)
    inbox_base = torch.arange(256, dtype=torch.long, device=device).repeat_interleave(8)
    transform_base = torch.arange(8, dtype=torch.long, device=device).repeat(256)
    cases = inbox_base.numel()
    inbox_code = inbox_base.repeat_interleave(16)
    transform_id = transform_base.repeat_interleave(16)
    root_symbol = torch.arange(16, dtype=torch.long, device=device).repeat(cases)
    source_is_root = torch.zeros(cases * 16, dtype=torch.bool, device=device)
    target_is_query = torch.ones(cases * 16, dtype=torch.bool, device=device)
    role_ids = model.role_ids(
        source_is_root=source_is_root,
        target_is_query=target_is_query,
    )
    with torch.no_grad():
        output = model(
            inbox_code=inbox_code,
            transform_id=transform_id,
            root_symbol=root_symbol,
            source_is_root=source_is_root,
            target_is_query=target_is_query,
            inbox_present=torch.ones(cases * 16, dtype=torch.bool, device=device),
            round_is_zero=torch.zeros(cases * 16, dtype=torch.bool, device=device),
            hidden=model.initial_hidden(role_ids),
        )
        _validate_output(output, cases * 16)
        messages = torch.argmax(output.message_logits, dim=-1).reshape(cases, 16)
        answers = torch.argmax(output.answer_logits, dim=-1).reshape(cases, 16)
        activities = (output.activity_logit >= 0.0).reshape(cases, 16)
        message_invariant = torch.all(messages == messages[:, :1], dim=1)
        answer_invariant = torch.all(answers == answers[:, :1], dim=1)
        activity_invariant = torch.all(activities == activities[:, :1], dim=1)
    result = Gate8RootInvarianceResult(
        cases=cases,
        message_invariant_cases=int(message_invariant.sum().item()),
        answer_invariant_cases=int(answer_invariant.sum().item()),
        activity_invariant_cases=int(activity_invariant.sum().item()),
    )
    result.validate()
    return result


def gate8_seed0_diagnostic_runtime_plan() -> dict[str, Any]:
    return {
        "version": GATE8_SEED0_DIAGNOSTIC_RUNTIME_VERSION,
        "protocol_head": GATE8_SEED0_DIAGNOSTIC_PROTOCOL_HEAD,
        "qualified_runtime_head": GATE8_QUALIFIED_RUNTIME_HEAD,
        "allowed_splits": list(GATE8_ALLOWED_SPLITS),
        "runtime_probes": list(GATE8_RUNTIME_PROBES),
        "truth_read": False,
        "scientific_test_allowed": False,
        "reference_model_allowed": False,
        "message_selection": "argmax",
        "message_decode": "terminal_code_mod16",
        "activity_intervention": "force_all_scheduled_true",
    }
