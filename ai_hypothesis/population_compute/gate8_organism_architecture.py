"""Gate-8 fixed-parameter shared organism architecture contract.

This module admits only the shared neural worker core and exact parameter
accounting. It does not admit graph execution, training, checkpointing,
scientific-test world generation, or the 1B reference model.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn

GATE8_ORGANISM_ARCHITECTURE_VERSION = "gate8-organism-architecture-contract-v0"
GATE8_ORGANISM_ARCHITECTURE_STATUS = (
    "GATE8_FIXED_19649_PARAMETER_ORGANISM_ARCHITECTURE_ADMITTED_TRAINING_CLOSED"
)
GATE8_ORGANISM_ARCHITECTURE_BASE_RESULT_HEAD = (
    "c7f5260189ef9ac1a1beb73596446316631090c7"
)

GATE8_LEARNED_PARAMETER_COUNT = 19_649
GATE8_MESSAGE_CODEBOOK_SIZE = 256
GATE8_MESSAGE_BITS = 8
GATE8_TRANSFORM_COUNT = 8
GATE8_SYMBOL_COUNT = 16
GATE8_ROLE_COUNT = 4
GATE8_FEATURE_WIDTH = 12
GATE8_WORKER_STATE_WIDTH = 32
GATE8_WORKER_INPUT_WIDTH = 40
GATE8_RUNTIME_FLAG_WIDTH = 4

GATE8_ROLE_ORDINARY = 0
GATE8_ROLE_TARGET_INCOMING = 1
GATE8_ROLE_ROOT_OUTGOING = 2
GATE8_ROLE_ROOT_AND_TARGET = 3


@dataclass(frozen=True, slots=True)
class Gate8WorkerStepOutput:
    hidden: Tensor
    message_logits: Tensor
    activity_logit: Tensor
    answer_logits: Tensor


class Gate8SharedWorkerCore(nn.Module):
    """One parameter-shared neural core reused by every edge worker."""

    def __init__(self) -> None:
        super().__init__()
        self.message_code_embedding = nn.Embedding(
            GATE8_MESSAGE_CODEBOOK_SIZE,
            GATE8_FEATURE_WIDTH,
        )
        self.transform_embedding = nn.Embedding(
            GATE8_TRANSFORM_COUNT,
            GATE8_FEATURE_WIDTH,
        )
        self.root_symbol_embedding = nn.Embedding(
            GATE8_SYMBOL_COUNT,
            GATE8_FEATURE_WIDTH,
        )
        self.role_embedding = nn.Embedding(
            GATE8_ROLE_COUNT,
            GATE8_FEATURE_WIDTH,
        )
        self.initial_hidden_by_role = nn.Parameter(
            torch.empty(GATE8_ROLE_COUNT, GATE8_WORKER_STATE_WIDTH)
        )
        self.worker_update = nn.GRUCell(
            input_size=GATE8_WORKER_INPUT_WIDTH,
            hidden_size=GATE8_WORKER_STATE_WIDTH,
        )
        self.message_head = nn.Linear(
            GATE8_WORKER_STATE_WIDTH,
            GATE8_MESSAGE_CODEBOOK_SIZE,
        )
        self.activity_head = nn.Linear(GATE8_WORKER_STATE_WIDTH, 1)
        self.answer_head = nn.Linear(
            GATE8_WORKER_STATE_WIDTH,
            GATE8_SYMBOL_COUNT,
        )
        nn.init.zeros_(self.initial_hidden_by_role)
        self.validate_parameter_contract()

    @staticmethod
    def role_ids(
        *,
        source_is_root: Tensor,
        target_is_query: Tensor,
    ) -> Tensor:
        _validate_bool_vector("source_is_root", source_is_root)
        _validate_bool_vector("target_is_query", target_is_query)
        if source_is_root.shape != target_is_query.shape:
            raise ValueError("Gate8 role flags must have identical shapes")
        return (
            source_is_root.to(dtype=torch.long) * 2
            + target_is_query.to(dtype=torch.long)
        )

    def initial_hidden(self, role_ids: Tensor) -> Tensor:
        _validate_index_vector(
            "role_ids",
            role_ids,
            upper_bound=GATE8_ROLE_COUNT,
        )
        return self.initial_hidden_by_role.index_select(0, role_ids)

    def forward(
        self,
        *,
        inbox_code: Tensor,
        transform_id: Tensor,
        root_symbol: Tensor,
        source_is_root: Tensor,
        target_is_query: Tensor,
        inbox_present: Tensor,
        round_is_zero: Tensor,
        hidden: Tensor,
    ) -> Gate8WorkerStepOutput:
        batch = _validate_step_inputs(
            inbox_code=inbox_code,
            transform_id=transform_id,
            root_symbol=root_symbol,
            source_is_root=source_is_root,
            target_is_query=target_is_query,
            inbox_present=inbox_present,
            round_is_zero=round_is_zero,
            hidden=hidden,
        )
        role_ids = self.role_ids(
            source_is_root=source_is_root,
            target_is_query=target_is_query,
        )
        message_features = self.message_code_embedding(inbox_code)
        edge_features = (
            self.transform_embedding(transform_id)
            + self.role_embedding(role_ids)
        )
        query_features = self.root_symbol_embedding(root_symbol)
        runtime_flags = torch.stack(
            (
                source_is_root,
                target_is_query,
                inbox_present,
                round_is_zero,
            ),
            dim=-1,
        ).to(dtype=hidden.dtype)
        worker_input = torch.cat(
            (
                message_features,
                edge_features,
                query_features,
                runtime_flags,
            ),
            dim=-1,
        )
        if worker_input.shape != (batch, GATE8_WORKER_INPUT_WIDTH):
            raise RuntimeError("Gate8 worker input width drifted")
        next_hidden = self.worker_update(worker_input, hidden)
        return Gate8WorkerStepOutput(
            hidden=next_hidden,
            message_logits=self.message_head(next_hidden),
            activity_logit=self.activity_head(next_hidden).squeeze(-1),
            answer_logits=self.answer_head(next_hidden),
        )

    def validate_parameter_contract(self) -> None:
        observed = sum(parameter.numel() for parameter in self.parameters())
        if observed != GATE8_LEARNED_PARAMETER_COUNT:
            raise RuntimeError(
                f"Gate8 architecture parameter drift: {observed} "
                f"!= {GATE8_LEARNED_PARAMETER_COUNT}"
            )


def _validate_bool_vector(name: str, value: Tensor) -> None:
    if value.dtype is not torch.bool or value.ndim != 1:
        raise ValueError(f"Gate8 {name} must be a rank-1 bool tensor")


def _validate_index_vector(
    name: str,
    value: Tensor,
    *,
    upper_bound: int,
) -> None:
    if value.dtype != torch.long or value.ndim != 1:
        raise ValueError(f"Gate8 {name} must be a rank-1 int64 tensor")
    if value.numel() and (
        int(value.min().item()) < 0
        or int(value.max().item()) >= upper_bound
    ):
        raise ValueError(f"Gate8 {name} is outside its frozen range")


def _validate_step_inputs(
    *,
    inbox_code: Tensor,
    transform_id: Tensor,
    root_symbol: Tensor,
    source_is_root: Tensor,
    target_is_query: Tensor,
    inbox_present: Tensor,
    round_is_zero: Tensor,
    hidden: Tensor,
) -> int:
    _validate_index_vector(
        "inbox_code",
        inbox_code,
        upper_bound=GATE8_MESSAGE_CODEBOOK_SIZE,
    )
    _validate_index_vector(
        "transform_id",
        transform_id,
        upper_bound=GATE8_TRANSFORM_COUNT,
    )
    _validate_index_vector(
        "root_symbol",
        root_symbol,
        upper_bound=GATE8_SYMBOL_COUNT,
    )
    for name, value in (
        ("source_is_root", source_is_root),
        ("target_is_query", target_is_query),
        ("inbox_present", inbox_present),
        ("round_is_zero", round_is_zero),
    ):
        _validate_bool_vector(name, value)
    batch = inbox_code.shape[0]
    vectors = (
        transform_id,
        root_symbol,
        source_is_root,
        target_is_query,
        inbox_present,
        round_is_zero,
    )
    if any(value.shape[0] != batch for value in vectors):
        raise ValueError("Gate8 worker-step vectors must share one batch size")
    if hidden.ndim != 2 or hidden.shape != (
        batch,
        GATE8_WORKER_STATE_WIDTH,
    ):
        raise ValueError("Gate8 hidden state has the wrong shape")
    if not hidden.is_floating_point():
        raise ValueError("Gate8 hidden state must be floating point")
    return batch


def gate8_architecture_parameter_ledger() -> dict[str, int]:
    ledger = {
        "message_code_embedding": 256 * 12,
        "transform_embedding": 8 * 12,
        "root_symbol_embedding": 16 * 12,
        "role_embedding": 4 * 12,
        "initial_hidden_by_role": 4 * 32,
        "worker_update_weight_ih": 3 * 32 * 40,
        "worker_update_weight_hh": 3 * 32 * 32,
        "worker_update_biases": 2 * 3 * 32,
        "message_head": 32 * 256 + 256,
        "activity_head": 32 + 1,
        "answer_head": 32 * 16 + 16,
    }
    if sum(ledger.values()) != GATE8_LEARNED_PARAMETER_COUNT:
        raise RuntimeError("Gate8 static parameter ledger drifted")
    return ledger


def gate8_observed_parameter_ledger(
    model: Gate8SharedWorkerCore,
) -> dict[str, int]:
    observed = {
        name: parameter.numel()
        for name, parameter in model.named_parameters()
    }
    if sum(observed.values()) != GATE8_LEARNED_PARAMETER_COUNT:
        raise RuntimeError("Gate8 observed parameter ledger drifted")
    return observed


def gate8_organism_architecture_plan() -> dict[str, Any]:
    return {
        "version": GATE8_ORGANISM_ARCHITECTURE_VERSION,
        "scientific_status": GATE8_ORGANISM_ARCHITECTURE_STATUS,
        "base_result_head": GATE8_ORGANISM_ARCHITECTURE_BASE_RESULT_HEAD,
        "learned_parameter_count": GATE8_LEARNED_PARAMETER_COUNT,
        "shared_across_workers": True,
        "shared_across_populations": True,
        "shared_across_rounds": True,
        "message_codebook_size": GATE8_MESSAGE_CODEBOOK_SIZE,
        "message_bits": GATE8_MESSAGE_BITS,
        "worker_state_width": GATE8_WORKER_STATE_WIDTH,
        "feature_width": GATE8_FEATURE_WIDTH,
        "worker_input_width": GATE8_WORKER_INPUT_WIDTH,
        "worker_observation": (
            "one_transform_shard_plus_public_query_flags_and_one_inbox_code"
        ),
        "runtime_topology_state_learned": False,
        "node_identity_parameters": False,
        "worker_identity_parameters": False,
        "population_specific_parameters": False,
        "depth_specific_parameters": False,
        "graph_scheduler_admitted": False,
        "training_admitted": False,
        "checkpoint_admitted": False,
        "scientific_test_worlds_admitted": False,
        "reference_model_admitted": False,
        "parameter_ledger": gate8_architecture_parameter_ledger(),
    }
