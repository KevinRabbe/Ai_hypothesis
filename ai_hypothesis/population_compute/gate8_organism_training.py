"""Gate-8 organism local-supervision training mechanics.

This module converts already-generated contract/train/validation public trees
into exact edge-level transition batches and computes the frozen losses. It
rejects scientific-test and demonstration worlds and contains no world generator,
checkpoint writer, optimizer construction, or reference-model path.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import torch
from torch import Tensor, nn
from torch.nn import functional as F

GATE8_ORGANISM_TRAINING_EXECUTION_VERSION = "gate8-organism-training-execution-v0"
GATE8_ORGANISM_TRAINING_EXECUTION_PROTOCOL_HEAD = (
    "869791e5b44089f9c79447b8ae212ce830f8496a"
)
GATE8_TRAINING_ALLOWED_SPLITS = ("contract", "train", "validation")
GATE8_LEARNED_PARAMETER_COUNT = 19_649


@dataclass(frozen=True, slots=True)
class Gate8LocalExample:
    worker_index: int
    inbox_code: int
    transform_id: int
    root_symbol: int
    source_is_root: bool
    target_is_query: bool
    message_target: int
    answer_target: int


@dataclass(frozen=True, slots=True)
class Gate8LocalTrainingBatch:
    inbox_code: Tensor
    transform_id: Tensor
    root_symbol: Tensor
    source_is_root: Tensor
    target_is_query: Tensor
    inbox_present: Tensor
    round_is_zero: Tensor
    message_target: Tensor
    answer_target: Tensor
    activity_target: Tensor
    world_count: int
    edge_count: int

    def validate(self) -> None:
        if self.world_count <= 0 or self.edge_count <= 0:
            raise ValueError("Gate8 local batch must contain worlds and edges")
        index_vectors = (
            self.inbox_code,
            self.transform_id,
            self.root_symbol,
            self.message_target,
            self.answer_target,
        )
        bool_vectors = (
            self.source_is_root,
            self.target_is_query,
            self.inbox_present,
            self.round_is_zero,
        )
        for value in index_vectors:
            if value.dtype != torch.long or value.shape != (self.edge_count,):
                raise ValueError("Gate8 local index vector has wrong dtype or shape")
        for value in bool_vectors:
            if value.dtype is not torch.bool or value.shape != (self.edge_count,):
                raise ValueError("Gate8 local bool vector has wrong dtype or shape")
        if not self.activity_target.is_floating_point() or self.activity_target.shape != (
            self.edge_count,
        ):
            raise ValueError("Gate8 local activity target has wrong dtype or shape")
        devices = {
            value.device
            for value in (*index_vectors, *bool_vectors, self.activity_target)
        }
        if len(devices) != 1:
            raise ValueError("Gate8 local batch tensors must share one device")
        if self.inbox_code.numel() and (
            int(self.inbox_code.min().item()) < 0
            or int(self.inbox_code.max().item()) >= 256
        ):
            raise ValueError("Gate8 inbox code is outside 0..255")
        if self.message_target.numel() and (
            int(self.message_target.min().item()) < 0
            or int(self.message_target.max().item()) >= 256
        ):
            raise ValueError("Gate8 message target is outside 0..255")
        if self.transform_id.numel() and (
            int(self.transform_id.min().item()) < 0
            or int(self.transform_id.max().item()) >= 8
        ):
            raise ValueError("Gate8 transform id is outside 0..7")
        for name, value in (
            ("root_symbol", self.root_symbol),
            ("answer_target", self.answer_target),
        ):
            if value.numel() and (
                int(value.min().item()) < 0 or int(value.max().item()) >= 16
            ):
                raise ValueError(f"Gate8 {name} is outside 0..15")
        if not bool(torch.all(self.inbox_present).item()):
            raise ValueError("Gate8 dense local supervision requires inbox-present edges")
        if not torch.equal(self.source_is_root, self.round_is_zero):
            raise ValueError("Gate8 round-zero flags must equal root-source flags")
        if not bool(torch.all(self.activity_target == 1).item()):
            raise ValueError("Gate8 frozen activity target must be one")


@dataclass(frozen=True, slots=True)
class Gate8LocalLoss:
    total: Tensor
    message: Tensor
    answer: Tensor
    activity: Tensor
    edge_count: int

    def detached_metrics(self) -> dict[str, float | int]:
        return {
            "total_loss": float(self.total.detach().item()),
            "message_loss": float(self.message.detach().item()),
            "answer_loss": float(self.answer.detach().item()),
            "activity_loss": float(self.activity.detach().item()),
            "edge_count": self.edge_count,
        }


@dataclass(frozen=True, slots=True)
class Gate8LocalEvaluation:
    edge_count: int
    total_loss_sum: float
    message_loss_sum: float
    answer_loss_sum: float
    activity_loss_sum: float
    message_correct: int
    answer_correct: int
    activity_correct: int
    inbox_codes: frozenset[int]
    target_codes: frozenset[int]

    def validate(self) -> None:
        if self.edge_count <= 0:
            raise ValueError("Gate8 local evaluation must contain edges")
        for name, value in (
            ("message_correct", self.message_correct),
            ("answer_correct", self.answer_correct),
            ("activity_correct", self.activity_correct),
        ):
            if not 0 <= value <= self.edge_count:
                raise ValueError(f"Gate8 {name} is outside the edge count")
        if any(not 0 <= code < 256 for code in self.inbox_codes):
            raise ValueError("Gate8 evaluated inbox code is outside 0..255")
        if any(not 0 <= code < 256 for code in self.target_codes):
            raise ValueError("Gate8 evaluated target code is outside 0..255")


def _public_world(world: Any) -> Any:
    public = getattr(world, "public", world)
    if getattr(public, "split", None) not in GATE8_TRAINING_ALLOWED_SPLITS:
        raise ValueError("Gate8 training mechanics reject this world split")
    validate = getattr(public, "validate", None)
    if not callable(validate):
        raise ValueError("Gate8 training world lacks public validation")
    validate()
    return public


def gate8_local_examples(
    *,
    world: Any,
    transform_permutations: tuple[tuple[int, ...], ...],
    protocol: Any,
) -> tuple[Gate8LocalExample, ...]:
    """Derive every edge transition without consulting stored world truth."""

    public = _public_world(world)
    if len(transform_permutations) != 8:
        raise ValueError("Gate8 training requires exactly eight primitive transforms")
    root = public.query.root_node
    node_state: dict[str, tuple[int, int]] = {
        root: (public.query.root_symbol, public.query.root_symbol)
    }
    unresolved = set(range(public.population))
    examples: list[Gate8LocalExample | None] = [None] * public.population

    while unresolved:
        progressed = False
        for worker_index in tuple(sorted(unresolved)):
            worker = public.workers[worker_index]
            state = node_state.get(worker.source_node)
            if state is None:
                continue
            input_carrier, input_symbol = state
            source_is_root = worker.source_node == root
            inbox_code = (
                protocol.GATE8_ROOT_SEED_CODE
                if source_is_root
                else protocol.gate8_encode_message_code(
                    carrier=input_carrier,
                    symbol=input_symbol,
                )
            )
            transform = transform_permutations[worker.transform_id]
            if len(transform) != 16:
                raise ValueError("Gate8 primitive transform width changed")
            output_symbol = transform[input_symbol]
            message_target = protocol.gate8_target_message_code(
                inbox_code=inbox_code,
                output_symbol=output_symbol,
                source_is_root=source_is_root,
                root_symbol=public.query.root_symbol,
            )
            output_carrier, decoded_output_symbol = (
                protocol.gate8_decode_message_code(message_target)
            )
            if decoded_output_symbol != output_symbol:
                raise RuntimeError("Gate8 message target lost its output symbol")
            if worker.target_node in node_state:
                raise ValueError("Gate8 training tree assigns one node twice")
            node_state[worker.target_node] = (
                output_carrier,
                decoded_output_symbol,
            )
            examples[worker_index] = Gate8LocalExample(
                worker_index=worker_index,
                inbox_code=inbox_code,
                transform_id=worker.transform_id,
                root_symbol=public.query.root_symbol,
                source_is_root=source_is_root,
                target_is_query=worker.target_node == public.query.target_node,
                message_target=message_target,
                answer_target=output_symbol,
            )
            unresolved.remove(worker_index)
            progressed = True
        if not progressed:
            raise ValueError("Gate8 training tree contains unreachable edges")

    if len(node_state) != public.population + 1:
        raise ValueError("Gate8 training labels did not cover the complete tree")
    if any(example is None for example in examples):
        raise RuntimeError("Gate8 training labels are incomplete")
    result = tuple(example for example in examples if example is not None)
    if tuple(example.worker_index for example in result) != tuple(range(public.population)):
        raise RuntimeError("Gate8 training examples lost worker order")
    return result


def collate_gate8_local_batch(
    *,
    worlds: Iterable[Any],
    transform_permutations: tuple[tuple[int, ...], ...],
    protocol: Any,
    device: torch.device | str,
) -> Gate8LocalTrainingBatch:
    world_list = tuple(worlds)
    if not world_list:
        raise ValueError("Gate8 local batch requires at least one world")
    examples = tuple(
        example
        for world in world_list
        for example in gate8_local_examples(
            world=world,
            transform_permutations=transform_permutations,
            protocol=protocol,
        )
    )
    target_device = torch.device(device)
    edge_count = len(examples)
    batch = Gate8LocalTrainingBatch(
        inbox_code=torch.tensor(
            [example.inbox_code for example in examples],
            dtype=torch.long,
            device=target_device,
        ),
        transform_id=torch.tensor(
            [example.transform_id for example in examples],
            dtype=torch.long,
            device=target_device,
        ),
        root_symbol=torch.tensor(
            [example.root_symbol for example in examples],
            dtype=torch.long,
            device=target_device,
        ),
        source_is_root=torch.tensor(
            [example.source_is_root for example in examples],
            dtype=torch.bool,
            device=target_device,
        ),
        target_is_query=torch.tensor(
            [example.target_is_query for example in examples],
            dtype=torch.bool,
            device=target_device,
        ),
        inbox_present=torch.ones(edge_count, dtype=torch.bool, device=target_device),
        round_is_zero=torch.tensor(
            [example.source_is_root for example in examples],
            dtype=torch.bool,
            device=target_device,
        ),
        message_target=torch.tensor(
            [example.message_target for example in examples],
            dtype=torch.long,
            device=target_device,
        ),
        answer_target=torch.tensor(
            [example.answer_target for example in examples],
            dtype=torch.long,
            device=target_device,
        ),
        activity_target=torch.ones(
            edge_count,
            dtype=torch.float32,
            device=target_device,
        ),
        world_count=len(world_list),
        edge_count=edge_count,
    )
    batch.validate()
    return batch


def gate8_local_forward(model: nn.Module, batch: Gate8LocalTrainingBatch) -> Any:
    batch.validate()
    if sum(parameter.numel() for parameter in model.parameters()) != GATE8_LEARNED_PARAMETER_COUNT:
        raise ValueError("Gate8 training requires exactly 19,649 learned parameters")
    role_ids = model.role_ids(
        source_is_root=batch.source_is_root,
        target_is_query=batch.target_is_query,
    )
    hidden = model.initial_hidden(role_ids)
    output = model(
        inbox_code=batch.inbox_code,
        transform_id=batch.transform_id,
        root_symbol=batch.root_symbol,
        source_is_root=batch.source_is_root,
        target_is_query=batch.target_is_query,
        inbox_present=batch.inbox_present,
        round_is_zero=batch.round_is_zero,
        hidden=hidden,
    )
    expected = {
        "hidden": (batch.edge_count, 32),
        "message_logits": (batch.edge_count, 256),
        "activity_logit": (batch.edge_count,),
        "answer_logits": (batch.edge_count, 16),
    }
    for name, shape in expected.items():
        value = getattr(output, name, None)
        if not isinstance(value, Tensor) or tuple(value.shape) != shape:
            raise ValueError(f"Gate8 training model returned invalid {name}")
    return output


def gate8_local_loss(
    *,
    output: Any,
    batch: Gate8LocalTrainingBatch,
    protocol: Any,
) -> Gate8LocalLoss:
    batch.validate()
    message = F.cross_entropy(output.message_logits, batch.message_target)
    answer = F.cross_entropy(output.answer_logits, batch.answer_target)
    activity = F.binary_cross_entropy_with_logits(
        output.activity_logit,
        batch.activity_target,
    )
    total = (
        protocol.GATE8_MESSAGE_LOSS_WEIGHT * message
        + protocol.GATE8_ANSWER_LOSS_WEIGHT * answer
        + protocol.GATE8_ACTIVITY_LOSS_WEIGHT * activity
    )
    return Gate8LocalLoss(
        total=total,
        message=message,
        answer=answer,
        activity=activity,
        edge_count=batch.edge_count,
    )


def evaluate_gate8_local_batch(
    *,
    model: nn.Module,
    batch: Gate8LocalTrainingBatch,
    protocol: Any,
) -> Gate8LocalEvaluation:
    with torch.no_grad():
        output = gate8_local_forward(model, batch)
        message_loss_sum = float(
            F.cross_entropy(
                output.message_logits,
                batch.message_target,
                reduction="sum",
            ).item()
        )
        answer_loss_sum = float(
            F.cross_entropy(
                output.answer_logits,
                batch.answer_target,
                reduction="sum",
            ).item()
        )
        activity_loss_sum = float(
            F.binary_cross_entropy_with_logits(
                output.activity_logit,
                batch.activity_target,
                reduction="sum",
            ).item()
        )
        total_loss_sum = (
            protocol.GATE8_MESSAGE_LOSS_WEIGHT * message_loss_sum
            + protocol.GATE8_ANSWER_LOSS_WEIGHT * answer_loss_sum
            + protocol.GATE8_ACTIVITY_LOSS_WEIGHT * activity_loss_sum
        )
        result = Gate8LocalEvaluation(
            edge_count=batch.edge_count,
            total_loss_sum=total_loss_sum,
            message_loss_sum=message_loss_sum,
            answer_loss_sum=answer_loss_sum,
            activity_loss_sum=activity_loss_sum,
            message_correct=int(
                (torch.argmax(output.message_logits, dim=-1) == batch.message_target)
                .sum()
                .item()
            ),
            answer_correct=int(
                (torch.argmax(output.answer_logits, dim=-1) == batch.answer_target)
                .sum()
                .item()
            ),
            activity_correct=int(
                ((output.activity_logit >= 0.0) == (batch.activity_target >= 0.5))
                .sum()
                .item()
            ),
            inbox_codes=frozenset(int(value) for value in batch.inbox_code.cpu().tolist()),
            target_codes=frozenset(
                int(value) for value in batch.message_target.cpu().tolist()
            ),
        )
    result.validate()
    return result


def merge_gate8_local_evaluations(
    evaluations: Iterable[Gate8LocalEvaluation],
) -> Gate8LocalEvaluation:
    rows = tuple(evaluations)
    if not rows:
        raise ValueError("Gate8 local evaluation merge requires rows")
    for row in rows:
        row.validate()
    result = Gate8LocalEvaluation(
        edge_count=sum(row.edge_count for row in rows),
        total_loss_sum=sum(row.total_loss_sum for row in rows),
        message_loss_sum=sum(row.message_loss_sum for row in rows),
        answer_loss_sum=sum(row.answer_loss_sum for row in rows),
        activity_loss_sum=sum(row.activity_loss_sum for row in rows),
        message_correct=sum(row.message_correct for row in rows),
        answer_correct=sum(row.answer_correct for row in rows),
        activity_correct=sum(row.activity_correct for row in rows),
        inbox_codes=frozenset().union(*(row.inbox_codes for row in rows)),
        target_codes=frozenset().union(*(row.target_codes for row in rows)),
    )
    result.validate()
    return result


def gate8_training_mechanics_plan() -> dict[str, Any]:
    return {
        "version": GATE8_ORGANISM_TRAINING_EXECUTION_VERSION,
        "protocol_head": GATE8_ORGANISM_TRAINING_EXECUTION_PROTOCOL_HEAD,
        "allowed_splits": list(GATE8_TRAINING_ALLOWED_SPLITS),
        "scientific_test_split_allowed": False,
        "demonstration_split_allowed": False,
        "truth_field_read": False,
        "world_generation_owned_here": False,
        "optimizer_owned_here": False,
        "checkpoint_write_owned_here": False,
        "reference_model_owned_here": False,
        "local_supervision": "every_edge",
        "learned_parameter_count": GATE8_LEARNED_PARAMETER_COUNT,
    }
