"""Gate-8 v1 factorized-message organism architecture contract.

This module admits only the fixed-19,649-parameter shared neural worker core.
It deliberately removes the v0 root-symbol feature, role features, activity
head, monolithic 256-way message head, and duplicate answer head. It does not
admit graph execution, training, checkpointing, scientific-test worlds, or the
1B reference model.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn

GATE8_V1_ARCHITECTURE_VERSION = (
    "gate8-factorized-message-architecture-contract-v1"
)
GATE8_V1_ARCHITECTURE_STATUS = (
    "GATE8_V1_FIXED_19649_PARAMETER_FACTORIZED_ARCHITECTURE_"
    "ADMITTED_EXECUTION_AND_TRAINING_CLOSED"
)
GATE8_V1_ARCHITECTURE_BASE_RESULT_HEAD = (
    "ad54c8daa7617d54e15932da76da08212d0d1444"
)

GATE8_V1_LEARNED_PARAMETER_COUNT = 19_649
GATE8_V1_MESSAGE_CODEBOOK_SIZE = 256
GATE8_V1_MESSAGE_BITS = 8
GATE8_V1_NIBBLE_SIZE = 16
GATE8_V1_TRANSFORM_COUNT = 8

GATE8_V1_CARRIER_FEATURE_WIDTH = 7
GATE8_V1_SYMBOL_FEATURE_WIDTH = 11
GATE8_V1_TRANSFORM_FEATURE_WIDTH = 3
GATE8_V1_WORKER_INPUT_WIDTH = 21
GATE8_V1_WORKER_STATE_WIDTH = 65


@dataclass(frozen=True, slots=True)
class Gate8V1WorkerStepOutput:
    """One shared worker update with factorized 4-bit outputs."""

    hidden: Tensor
    carrier_logits: Tensor
    symbol_logits: Tensor


class Gate8V1SharedWorkerCore(nn.Module):
    """One parameter-shared local transform core reused by every edge."""

    def __init__(self) -> None:
        super().__init__()
        self.carrier_embedding = nn.Embedding(
            GATE8_V1_NIBBLE_SIZE,
            GATE8_V1_CARRIER_FEATURE_WIDTH,
        )
        self.symbol_embedding = nn.Embedding(
            GATE8_V1_NIBBLE_SIZE,
            GATE8_V1_SYMBOL_FEATURE_WIDTH,
        )
        self.transform_embedding = nn.Embedding(
            GATE8_V1_TRANSFORM_COUNT,
            GATE8_V1_TRANSFORM_FEATURE_WIDTH,
        )
        self.initial_hidden_state = nn.Parameter(
            torch.zeros(GATE8_V1_WORKER_STATE_WIDTH)
        )
        self.worker_update = nn.GRUCell(
            input_size=GATE8_V1_WORKER_INPUT_WIDTH,
            hidden_size=GATE8_V1_WORKER_STATE_WIDTH,
        )
        self.carrier_head = nn.Linear(
            GATE8_V1_WORKER_STATE_WIDTH,
            GATE8_V1_NIBBLE_SIZE,
        )
        self.symbol_head = nn.Linear(
            GATE8_V1_WORKER_STATE_WIDTH,
            GATE8_V1_NIBBLE_SIZE,
        )
        self.validate_parameter_contract()

    @staticmethod
    def split_message_code(inbox_code: Tensor) -> tuple[Tensor, Tensor]:
        """Split an 8-bit message into high carrier and low symbol nibbles."""

        _validate_index_vector(
            "inbox_code",
            inbox_code,
            upper_bound=GATE8_V1_MESSAGE_CODEBOOK_SIZE,
        )
        carrier = torch.bitwise_right_shift(inbox_code, 4)
        symbol = torch.bitwise_and(inbox_code, 0x0F)
        return carrier, symbol

    @staticmethod
    def compose_message_code(
        *,
        carrier: Tensor,
        symbol: Tensor,
    ) -> Tensor:
        """Compose high carrier and low symbol nibbles into one 8-bit code."""

        _validate_index_vector(
            "carrier",
            carrier,
            upper_bound=GATE8_V1_NIBBLE_SIZE,
        )
        _validate_index_vector(
            "symbol",
            symbol,
            upper_bound=GATE8_V1_NIBBLE_SIZE,
        )
        if carrier.shape != symbol.shape:
            raise ValueError("Gate8 v1 carrier and symbol shapes must match")
        return torch.bitwise_or(torch.bitwise_left_shift(carrier, 4), symbol)

    @staticmethod
    def root_message_code(root_symbol: Tensor) -> Tensor:
        """Encode the public root symbol directly into the initial message."""

        _validate_index_vector(
            "root_symbol",
            root_symbol,
            upper_bound=GATE8_V1_NIBBLE_SIZE,
        )
        carrier = torch.zeros_like(root_symbol)
        return Gate8V1SharedWorkerCore.compose_message_code(
            carrier=carrier,
            symbol=root_symbol,
        )

    def initial_hidden(
        self,
        batch_size: int,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> Tensor:
        """Return one shared learned initial state expanded across workers."""

        if isinstance(batch_size, bool) or not isinstance(batch_size, int):
            raise TypeError("Gate8 v1 batch_size must be an integer")
        if batch_size < 0:
            raise ValueError("Gate8 v1 batch_size must be non-negative")
        target_device = self.initial_hidden_state.device if device is None else device
        target_dtype = self.initial_hidden_state.dtype if dtype is None else dtype
        if not torch.empty((), dtype=target_dtype).is_floating_point():
            raise ValueError("Gate8 v1 initial hidden dtype must be floating point")
        state = self.initial_hidden_state.to(
            device=target_device,
            dtype=target_dtype,
        )
        return state.unsqueeze(0).expand(batch_size, -1)

    def forward(
        self,
        *,
        inbox_code: Tensor,
        transform_id: Tensor,
        hidden: Tensor,
    ) -> Gate8V1WorkerStepOutput:
        batch = _validate_step_inputs(
            inbox_code=inbox_code,
            transform_id=transform_id,
            hidden=hidden,
        )
        carrier, symbol = self.split_message_code(inbox_code)
        worker_input = torch.cat(
            (
                self.carrier_embedding(carrier),
                self.symbol_embedding(symbol),
                self.transform_embedding(transform_id),
            ),
            dim=-1,
        )
        if worker_input.shape != (batch, GATE8_V1_WORKER_INPUT_WIDTH):
            raise RuntimeError("Gate8 v1 worker input width drifted")
        next_hidden = self.worker_update(worker_input, hidden)
        return Gate8V1WorkerStepOutput(
            hidden=next_hidden,
            carrier_logits=self.carrier_head(next_hidden),
            symbol_logits=self.symbol_head(next_hidden),
        )

    @classmethod
    def predicted_message_code(
        cls,
        output: Gate8V1WorkerStepOutput,
    ) -> Tensor:
        _validate_output(output)
        return cls.compose_message_code(
            carrier=output.carrier_logits.argmax(dim=-1),
            symbol=output.symbol_logits.argmax(dim=-1),
        )

    @staticmethod
    def predicted_symbol(output: Gate8V1WorkerStepOutput) -> Tensor:
        """The terminal answer is exactly the factorized symbol prediction."""

        _validate_output(output)
        return output.symbol_logits.argmax(dim=-1)

    def validate_parameter_contract(self) -> None:
        observed = sum(parameter.numel() for parameter in self.parameters())
        if observed != GATE8_V1_LEARNED_PARAMETER_COUNT:
            raise RuntimeError(
                f"Gate8 v1 architecture parameter drift: {observed} "
                f"!= {GATE8_V1_LEARNED_PARAMETER_COUNT}"
            )


def _validate_index_vector(
    name: str,
    value: Tensor,
    *,
    upper_bound: int,
) -> None:
    if value.dtype != torch.long or value.ndim != 1:
        raise ValueError(f"Gate8 v1 {name} must be a rank-1 int64 tensor")
    if value.numel() and (
        int(value.min().item()) < 0
        or int(value.max().item()) >= upper_bound
    ):
        raise ValueError(f"Gate8 v1 {name} is outside its frozen range")


def _validate_step_inputs(
    *,
    inbox_code: Tensor,
    transform_id: Tensor,
    hidden: Tensor,
) -> int:
    _validate_index_vector(
        "inbox_code",
        inbox_code,
        upper_bound=GATE8_V1_MESSAGE_CODEBOOK_SIZE,
    )
    _validate_index_vector(
        "transform_id",
        transform_id,
        upper_bound=GATE8_V1_TRANSFORM_COUNT,
    )
    batch = inbox_code.shape[0]
    if transform_id.shape[0] != batch:
        raise ValueError("Gate8 v1 worker-step vectors must share one batch size")
    if hidden.ndim != 2 or hidden.shape != (
        batch,
        GATE8_V1_WORKER_STATE_WIDTH,
    ):
        raise ValueError("Gate8 v1 hidden state has the wrong shape")
    if not hidden.is_floating_point():
        raise ValueError("Gate8 v1 hidden state must be floating point")
    if inbox_code.device != transform_id.device or inbox_code.device != hidden.device:
        raise ValueError("Gate8 v1 worker-step tensors must share one device")
    return batch


def _validate_output(output: Gate8V1WorkerStepOutput) -> None:
    if not isinstance(output, Gate8V1WorkerStepOutput):
        raise TypeError("Gate8 v1 output has the wrong contract type")
    if output.hidden.ndim != 2:
        raise ValueError("Gate8 v1 output hidden state must be rank 2")
    batch = output.hidden.shape[0]
    if output.hidden.shape[1] != GATE8_V1_WORKER_STATE_WIDTH:
        raise ValueError("Gate8 v1 output hidden width drifted")
    expected = (batch, GATE8_V1_NIBBLE_SIZE)
    if output.carrier_logits.shape != expected:
        raise ValueError("Gate8 v1 carrier-logit shape drifted")
    if output.symbol_logits.shape != expected:
        raise ValueError("Gate8 v1 symbol-logit shape drifted")


def gate8_v1_architecture_parameter_ledger() -> dict[str, int]:
    """Return the exact no-padding learned-parameter allocation."""

    ledger = {
        "carrier_embedding": 16 * 7,
        "symbol_embedding": 16 * 11,
        "transform_embedding": 8 * 3,
        "initial_hidden_state": 65,
        "worker_update_weight_ih": 3 * 65 * 21,
        "worker_update_weight_hh": 3 * 65 * 65,
        "worker_update_biases": 2 * 3 * 65,
        "carrier_head": 65 * 16 + 16,
        "symbol_head": 65 * 16 + 16,
    }
    if sum(ledger.values()) != GATE8_V1_LEARNED_PARAMETER_COUNT:
        raise RuntimeError("Gate8 v1 static parameter ledger drifted")
    return ledger


def gate8_v1_observed_parameter_ledger(
    model: Gate8V1SharedWorkerCore,
) -> dict[str, int]:
    observed = {
        name: parameter.numel()
        for name, parameter in model.named_parameters()
    }
    if len(observed) != 12:
        raise RuntimeError("Gate8 v1 parameter tensor count drifted")
    if sum(observed.values()) != GATE8_V1_LEARNED_PARAMETER_COUNT:
        raise RuntimeError("Gate8 v1 observed parameter ledger drifted")
    return observed


def gate8_v1_architecture_plan() -> dict[str, Any]:
    return {
        "version": GATE8_V1_ARCHITECTURE_VERSION,
        "scientific_status": GATE8_V1_ARCHITECTURE_STATUS,
        "base_result_head": GATE8_V1_ARCHITECTURE_BASE_RESULT_HEAD,
        "learned_parameter_count": GATE8_V1_LEARNED_PARAMETER_COUNT,
        "shared_across_workers": True,
        "shared_across_populations": True,
        "shared_across_rounds": True,
        "message_bits": GATE8_V1_MESSAGE_BITS,
        "message_codebook_size": GATE8_V1_MESSAGE_CODEBOOK_SIZE,
        "carrier_classes": GATE8_V1_NIBBLE_SIZE,
        "symbol_classes": GATE8_V1_NIBBLE_SIZE,
        "carrier_feature_width": GATE8_V1_CARRIER_FEATURE_WIDTH,
        "symbol_feature_width": GATE8_V1_SYMBOL_FEATURE_WIDTH,
        "transform_feature_width": GATE8_V1_TRANSFORM_FEATURE_WIDTH,
        "worker_input_width": GATE8_V1_WORKER_INPUT_WIDTH,
        "worker_state_width": GATE8_V1_WORKER_STATE_WIDTH,
        "worker_observation": "carrier_nibble_plus_symbol_nibble_plus_transform_id",
        "root_symbol_in_initial_message": True,
        "root_symbol_feature": False,
        "role_feature": False,
        "runtime_flag_feature": False,
        "monolithic_message_head": False,
        "factorized_message_heads": True,
        "duplicate_answer_head": False,
        "terminal_answer_is_symbol_head": True,
        "activity_head": False,
        "deterministic_delivery_required_by_future_runtime": True,
        "padding_parameters": 0,
        "node_identity_parameters": False,
        "worker_identity_parameters": False,
        "population_specific_parameters": False,
        "depth_specific_parameters": False,
        "graph_scheduler_admitted": False,
        "training_admitted": False,
        "checkpoint_admitted": False,
        "scientific_test_worlds_admitted": False,
        "reference_model_admitted": False,
        "parameter_ledger": gate8_v1_architecture_parameter_ledger(),
    }
