"""Gate-8 v1 factorized local-supervision training mechanics.

This module converts already-generated contract/train/validation public trees
into exact edge-level carrier and symbol transition batches. It rejects
scientific-test and demonstration worlds and contains no world generator,
optimizer construction, checkpoint writer, CUDA policy, or reference-model path.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import torch
from torch import Tensor, nn
from torch.nn import functional as F

GATE8_V1_TRAINING_EXECUTION_VERSION = (
    "gate8-factorized-message-training-execution-v1"
)
GATE8_V1_TRAINING_EXECUTION_PROTOCOL_HEAD = (
    "a33dc123d090268a531d112251ea3ab53cb50062"
)
GATE8_V1_TRAINING_ALLOWED_SPLITS = ("contract", "train", "validation")
GATE8_V1_LEARNED_PARAMETER_COUNT = 19_649
GATE8_V1_MESSAGE_CODEBOOK_SIZE = 256
GATE8_V1_COMPONENT_COUNT = 16


@dataclass(frozen=True, slots=True)
class Gate8V1LocalExample:
    worker_index: int
    inbox_code: int
    transform_id: int
    carrier_target: int
    symbol_target: int
    message_target: int


@dataclass(frozen=True, slots=True)
class Gate8V1LocalTrainingBatch:
    inbox_code: Tensor
    transform_id: Tensor
    carrier_target: Tensor
    symbol_target: Tensor
    message_target: Tensor
    world_count: int
    edge_count: int

    def validate(self) -> None:
        if self.world_count <= 0 or self.edge_count <= 0:
            raise ValueError("Gate8 v1 local batch must contain worlds and edges")
        vectors = (
            self.inbox_code,
            self.transform_id,
            self.carrier_target,
            self.symbol_target,
            self.message_target,
        )
        for value in vectors:
            if value.dtype != torch.long or value.shape != (self.edge_count,):
                raise ValueError(
                    "Gate8 v1 local index vector has wrong dtype or shape"
                )
        if len({value.device for value in vectors}) != 1:
            raise ValueError("Gate8 v1 local batch tensors must share one device")
        for name, value, upper_bound in (
            ("inbox_code", self.inbox_code, 256),
            ("transform_id", self.transform_id, 8),
            ("carrier_target", self.carrier_target, 16),
            ("symbol_target", self.symbol_target, 16),
            ("message_target", self.message_target, 256),
        ):
            if value.numel() and (
                int(value.min().item()) < 0
                or int(value.max().item()) >= upper_bound
            ):
                raise ValueError(
                    f"Gate8 v1 {name} is outside 0..{upper_bound - 1}"
                )
        reconstructed = self.carrier_target * 16 + self.symbol_target
        if not torch.equal(reconstructed, self.message_target):
            raise ValueError(
                "Gate8 v1 component targets do not reconstruct message targets"
            )


@dataclass(frozen=True, slots=True)
class Gate8V1LocalLoss:
    total: Tensor
    carrier: Tensor
    symbol: Tensor
    edge_count: int

    def detached_metrics(self) -> dict[str, float | int]:
        return {
            "total_loss": float(self.total.detach().item()),
            "carrier_loss": float(self.carrier.detach().item()),
            "symbol_loss": float(self.symbol.detach().item()),
            "edge_count": self.edge_count,
        }


@dataclass(frozen=True, slots=True)
class Gate8V1LocalEvaluation:
    edge_count: int
    total_loss_sum: float
    carrier_loss_sum: float
    symbol_loss_sum: float
    exact_message_correct: int
    carrier_correct: int
    symbol_correct: int
    inbox_codes: frozenset[int]
    target_codes: frozenset[int]
    target_carriers: frozenset[int]
    target_symbols: frozenset[int]

    def validate(self) -> None:
        if self.edge_count <= 0:
            raise ValueError("Gate8 v1 local evaluation must contain edges")
        for name, value in (
            ("exact_message_correct", self.exact_message_correct),
            ("carrier_correct", self.carrier_correct),
            ("symbol_correct", self.symbol_correct),
        ):
            if not 0 <= value <= self.edge_count:
                raise ValueError(f"Gate8 v1 {name} is outside the edge count")
        for name, values, upper_bound in (
            ("inbox_codes", self.inbox_codes, 256),
            ("target_codes", self.target_codes, 256),
            ("target_carriers", self.target_carriers, 16),
            ("target_symbols", self.target_symbols, 16),
        ):
            if any(not 0 <= value < upper_bound for value in values):
                raise ValueError(
                    f"Gate8 v1 {name} contains a value outside its frozen range"
                )
        for name, value in (
            ("total_loss_sum", self.total_loss_sum),
            ("carrier_loss_sum", self.carrier_loss_sum),
            ("symbol_loss_sum", self.symbol_loss_sum),
        ):
            if value < 0.0 or not torch.isfinite(torch.tensor(value)):
                raise ValueError(f"Gate8 v1 {name} must be finite and non-negative")


def _public_world(world: Any) -> Any:
    public = getattr(world, "public", world)
    if getattr(public, "split", None) not in GATE8_V1_TRAINING_ALLOWED_SPLITS:
        raise ValueError("Gate8 v1 training mechanics reject this world split")
    validate = getattr(public, "validate", None)
    if not callable(validate):
        raise ValueError("Gate8 v1 training world lacks public validation")
    validate()
    return public


def gate8_v1_local_examples(
    *,
    world: Any,
    transform_permutations: tuple[tuple[int, ...], ...],
    protocol: Any,
) -> tuple[Gate8V1LocalExample, ...]:
    """Derive every edge transition without consulting stored world truth."""

    public = _public_world(world)
    if len(transform_permutations) != 8:
        raise ValueError(
            "Gate8 v1 training requires exactly eight primitive transforms"
        )
    root = public.query.root_node
    root_code = protocol.gate8_v1_encode_message_code(
        carrier=0,
        symbol=public.query.root_symbol,
    )
    node_messages: dict[str, int] = {root: root_code}
    unresolved = set(range(public.population))
    examples: list[Gate8V1LocalExample | None] = [None] * public.population

    while unresolved:
        progressed = False
        for worker_index in tuple(sorted(unresolved)):
            worker = public.workers[worker_index]
            inbox_code = node_messages.get(worker.source_node)
            if inbox_code is None:
                continue
            if not 0 <= worker.transform_id < len(transform_permutations):
                raise ValueError("Gate8 v1 worker transform id is outside 0..7")
            transform = transform_permutations[worker.transform_id]
            carrier_target, symbol_target, message_target = (
                protocol.gate8_v1_target_transition(
                    inbox_code=inbox_code,
                    transform=transform,
                )
            )
            if worker.target_node in node_messages:
                raise ValueError("Gate8 v1 training tree assigns one node twice")
            node_messages[worker.target_node] = message_target
            examples[worker_index] = Gate8V1LocalExample(
                worker_index=worker_index,
                inbox_code=inbox_code,
                transform_id=worker.transform_id,
                carrier_target=carrier_target,
                symbol_target=symbol_target,
                message_target=message_target,
            )
            unresolved.remove(worker_index)
            progressed = True
        if not progressed:
            raise ValueError("Gate8 v1 training tree contains unreachable edges")

    if len(node_messages) != public.population + 1:
        raise ValueError("Gate8 v1 training labels did not cover the complete tree")
    if any(example is None for example in examples):
        raise RuntimeError("Gate8 v1 training labels are incomplete")
    result = tuple(example for example in examples if example is not None)
    if tuple(example.worker_index for example in result) != tuple(
        range(public.population)
    ):
        raise RuntimeError("Gate8 v1 training examples lost worker order")
    return result


def collate_gate8_v1_local_batch(
    *,
    worlds: Iterable[Any],
    transform_permutations: tuple[tuple[int, ...], ...],
    protocol: Any,
    device: torch.device | str,
) -> Gate8V1LocalTrainingBatch:
    world_list = tuple(worlds)
    if not world_list:
        raise ValueError("Gate8 v1 local batch requires at least one world")
    examples = tuple(
        example
        for world in world_list
        for example in gate8_v1_local_examples(
            world=world,
            transform_permutations=transform_permutations,
            protocol=protocol,
        )
    )
    target_device = torch.device(device)
    edge_count = len(examples)
    batch = Gate8V1LocalTrainingBatch(
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
        carrier_target=torch.tensor(
            [example.carrier_target for example in examples],
            dtype=torch.long,
            device=target_device,
        ),
        symbol_target=torch.tensor(
            [example.symbol_target for example in examples],
            dtype=torch.long,
            device=target_device,
        ),
        message_target=torch.tensor(
            [example.message_target for example in examples],
            dtype=torch.long,
            device=target_device,
        ),
        world_count=len(world_list),
        edge_count=edge_count,
    )
    batch.validate()
    return batch


def gate8_v1_local_forward(
    model: nn.Module,
    batch: Gate8V1LocalTrainingBatch,
) -> Any:
    batch.validate()
    if sum(parameter.numel() for parameter in model.parameters()) != (
        GATE8_V1_LEARNED_PARAMETER_COUNT
    ):
        raise ValueError(
            "Gate8 v1 training requires exactly 19,649 learned parameters"
        )
    initial_hidden = getattr(model, "initial_hidden", None)
    if not callable(initial_hidden):
        raise ValueError("Gate8 v1 model lacks initial_hidden")
    hidden = initial_hidden(batch.edge_count)
    output = model(
        inbox_code=batch.inbox_code,
        transform_id=batch.transform_id,
        hidden=hidden,
    )
    expected = {
        "hidden": (batch.edge_count, 65),
        "carrier_logits": (batch.edge_count, 16),
        "symbol_logits": (batch.edge_count, 16),
    }
    for name, shape in expected.items():
        value = getattr(output, name, None)
        if not isinstance(value, Tensor) or tuple(value.shape) != shape:
            raise ValueError(f"Gate8 v1 training model returned invalid {name}")
    return output


def gate8_v1_local_loss(
    *,
    output: Any,
    batch: Gate8V1LocalTrainingBatch,
    protocol: Any,
) -> Gate8V1LocalLoss:
    batch.validate()
    carrier = F.cross_entropy(output.carrier_logits, batch.carrier_target)
    symbol = F.cross_entropy(output.symbol_logits, batch.symbol_target)
    total = (
        protocol.GATE8_V1_CARRIER_LOSS_WEIGHT * carrier
        + protocol.GATE8_V1_SYMBOL_LOSS_WEIGHT * symbol
    )
    return Gate8V1LocalLoss(
        total=total,
        carrier=carrier,
        symbol=symbol,
        edge_count=batch.edge_count,
    )


def evaluate_gate8_v1_local_batch(
    *,
    model: nn.Module,
    batch: Gate8V1LocalTrainingBatch,
    protocol: Any,
) -> Gate8V1LocalEvaluation:
    with torch.no_grad():
        output = gate8_v1_local_forward(model, batch)
        carrier_loss_sum = float(
            F.cross_entropy(
                output.carrier_logits,
                batch.carrier_target,
                reduction="sum",
            ).item()
        )
        symbol_loss_sum = float(
            F.cross_entropy(
                output.symbol_logits,
                batch.symbol_target,
                reduction="sum",
            ).item()
        )
        total_loss_sum = (
            protocol.GATE8_V1_CARRIER_LOSS_WEIGHT * carrier_loss_sum
            + protocol.GATE8_V1_SYMBOL_LOSS_WEIGHT * symbol_loss_sum
        )
        predicted_carrier = torch.argmax(output.carrier_logits, dim=-1)
        predicted_symbol = torch.argmax(output.symbol_logits, dim=-1)
        predicted_code = predicted_carrier * 16 + predicted_symbol
        result = Gate8V1LocalEvaluation(
            edge_count=batch.edge_count,
            total_loss_sum=total_loss_sum,
            carrier_loss_sum=carrier_loss_sum,
            symbol_loss_sum=symbol_loss_sum,
            exact_message_correct=int(
                (predicted_code == batch.message_target).sum().item()
            ),
            carrier_correct=int(
                (predicted_carrier == batch.carrier_target).sum().item()
            ),
            symbol_correct=int(
                (predicted_symbol == batch.symbol_target).sum().item()
            ),
            inbox_codes=frozenset(
                int(value) for value in batch.inbox_code.cpu().tolist()
            ),
            target_codes=frozenset(
                int(value) for value in batch.message_target.cpu().tolist()
            ),
            target_carriers=frozenset(
                int(value) for value in batch.carrier_target.cpu().tolist()
            ),
            target_symbols=frozenset(
                int(value) for value in batch.symbol_target.cpu().tolist()
            ),
        )
    result.validate()
    return result


def merge_gate8_v1_local_evaluations(
    evaluations: Iterable[Gate8V1LocalEvaluation],
) -> Gate8V1LocalEvaluation:
    rows = tuple(evaluations)
    if not rows:
        raise ValueError("Gate8 v1 local evaluation merge requires rows")
    for row in rows:
        row.validate()
    result = Gate8V1LocalEvaluation(
        edge_count=sum(row.edge_count for row in rows),
        total_loss_sum=sum(row.total_loss_sum for row in rows),
        carrier_loss_sum=sum(row.carrier_loss_sum for row in rows),
        symbol_loss_sum=sum(row.symbol_loss_sum for row in rows),
        exact_message_correct=sum(
            row.exact_message_correct for row in rows
        ),
        carrier_correct=sum(row.carrier_correct for row in rows),
        symbol_correct=sum(row.symbol_correct for row in rows),
        inbox_codes=frozenset().union(*(row.inbox_codes for row in rows)),
        target_codes=frozenset().union(*(row.target_codes for row in rows)),
        target_carriers=frozenset().union(
            *(row.target_carriers for row in rows)
        ),
        target_symbols=frozenset().union(
            *(row.target_symbols for row in rows)
        ),
    )
    result.validate()
    return result


def gate8_v1_training_mechanics_plan() -> dict[str, Any]:
    return {
        "version": GATE8_V1_TRAINING_EXECUTION_VERSION,
        "protocol_head": GATE8_V1_TRAINING_EXECUTION_PROTOCOL_HEAD,
        "allowed_splits": list(GATE8_V1_TRAINING_ALLOWED_SPLITS),
        "scientific_test_split_allowed": False,
        "demonstration_split_allowed": False,
        "truth_field_read": False,
        "world_generation_owned_here": False,
        "optimizer_owned_here": False,
        "checkpoint_write_owned_here": False,
        "reference_model_owned_here": False,
        "local_supervision": "every_edge_factorized_carrier_and_symbol",
        "root_special_case": False,
        "activity_target": False,
        "answer_target": False,
        "joint_256_way_target": False,
        "learned_parameter_count": GATE8_V1_LEARNED_PARAMETER_COUNT,
    }
