"""Gate-9 exact-budget contextual worker architecture.

This module freezes one 19,649-parameter shared worker that maps nine public
byte-to-byte support examples plus one incoming byte query to eight output-bit
logits. It admits synthetic contract forward/backward checks only. It contains
no optimizer, training loop, checkpoint loader, graph runtime, scientific test
world generator, or result surface.
"""
from __future__ import annotations

import importlib.util
import pathlib
import sys
from typing import Any, Iterable, Sequence

import torch
from torch import Tensor, nn
import torch.nn.functional as F

_OPERATOR_PATH = pathlib.Path(__file__).with_name(
    "gate9_contextual_operator_contract.py"
)


def _load_operator_contract():
    name = "gate9_contextual_worker_operator_dependency"
    existing = sys.modules.get(name)
    if existing is not None:
        return existing
    spec = importlib.util.spec_from_file_location(name, _OPERATOR_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load Gate9 operator contract")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


operators = _load_operator_contract()

GATE9_CONTEXTUAL_WORKER_VERSION = "gate9-contextual-worker-architecture-v0"
GATE9_CONTEXTUAL_WORKER_STATUS = (
    "GATE9_CONTEXTUAL_WORKER_ARCHITECTURE_QUALIFIED_TRAINING_CLOSED"
)
GATE9_GRAPH_WORLD_CONTRACT_HEAD = "b6688aa8bf305f099ec320ea60dd5ccdffce4d51"
GATE9_LEARNED_PARAMETER_COUNT = 19_649
GATE9_SUPPORT_EXAMPLES = 9
GATE9_BYTE_BITS = 8
GATE9_PAIR_FEATURES = 16
GATE9_EMBED_DIM = 48
GATE9_ATTENTION_HEADS = 4
GATE9_FFN_DIM = 64
GATE9_FUSION_DIM = 24
GATE9_POSITION_MODULATION_DIM = 24
GATE9_OUTPUT_BITS = 8
GATE9_SUPPORT_ORDER = tuple(operators.GATE9_GLOBAL_SUPPORT_ORDER)

GATE9_PARAMETER_BREAKDOWN = {
    "pair_projection": 816,
    "query_projection": 432,
    "support_slot_modulation": 216,
    "support_attention": 9_408,
    "support_feedforward": 6_256,
    "query_support_fusion": 2_328,
    "output_bit_head": 192,
    "output_scale": 1,
}
if sum(GATE9_PARAMETER_BREAKDOWN.values()) != GATE9_LEARNED_PARAMETER_COUNT:
    raise RuntimeError("Gate9 contextual-worker parameter arithmetic drifted")


def _valid_byte(value: int, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value < 256:
        raise ValueError(f"{label} must be an integer within 0..255")
    return value


def _byte_bits(values: Tensor) -> Tensor:
    if values.dtype != torch.long:
        raise ValueError("Gate9 byte tensors must use torch.long")
    shifts = torch.arange(8, device=values.device, dtype=torch.long)
    return ((values.unsqueeze(-1) >> shifts) & 1).to(dtype=torch.float32)


def _validate_worker_tensors(
    support_inputs: Tensor,
    support_outputs: Tensor,
    query: Tensor,
) -> None:
    if support_inputs.dtype != torch.long or support_outputs.dtype != torch.long:
        raise ValueError("Gate9 support tensors must use torch.long")
    if query.dtype != torch.long:
        raise ValueError("Gate9 query tensor must use torch.long")
    if support_inputs.ndim != 2 or support_outputs.ndim != 2:
        raise ValueError("Gate9 support tensors must be rank two")
    if support_inputs.shape != support_outputs.shape:
        raise ValueError("Gate9 support input/output shapes disagree")
    if support_inputs.shape[1] != GATE9_SUPPORT_EXAMPLES:
        raise ValueError("Gate9 worker requires exactly nine support rows")
    if support_inputs.shape[0] <= 0:
        raise ValueError("Gate9 worker batch cannot be empty")
    if query.shape != (support_inputs.shape[0],):
        raise ValueError("Gate9 query tensor must contain one byte per worker")
    if support_inputs.device != support_outputs.device or query.device != support_inputs.device:
        raise ValueError("Gate9 worker tensors must share one device")
    for tensor, label in (
        (support_inputs, "support input"),
        (support_outputs, "support output"),
        (query, "query"),
    ):
        if bool(torch.any((tensor < 0) | (tensor > 255))):
            raise ValueError(f"Gate9 {label} byte lies outside 0..255")
    expected = torch.tensor(
        GATE9_SUPPORT_ORDER,
        dtype=torch.long,
        device=support_inputs.device,
    ).expand_as(support_inputs)
    if not torch.equal(support_inputs, expected):
        raise ValueError("Gate9 support inputs are not in the qualified global order")


def serialize_gate9_worker_batch(
    support_sets: Sequence[Iterable[tuple[int, int]]],
    queries: Sequence[int],
    *,
    device: torch.device | str | None = None,
) -> tuple[Tensor, Tensor, Tensor]:
    """Serialize only model-visible support pairs and incoming query bytes."""

    if len(support_sets) != len(queries) or len(queries) == 0:
        raise ValueError("Gate9 worker batch support/query counts disagree or are empty")
    input_rows: list[list[int]] = []
    output_rows: list[list[int]] = []
    for batch_index, support in enumerate(support_sets):
        pairs = tuple(support)
        if len(pairs) != GATE9_SUPPORT_EXAMPLES:
            raise ValueError("Gate9 worker support must contain exactly nine pairs")
        inputs: list[int] = []
        outputs: list[int] = []
        for pair_index, pair in enumerate(pairs):
            if not isinstance(pair, tuple) or len(pair) != 2:
                raise ValueError("Gate9 worker support row must be one input/output tuple")
            inputs.append(_valid_byte(pair[0], f"support input {batch_index}:{pair_index}"))
            outputs.append(_valid_byte(pair[1], f"support output {batch_index}:{pair_index}"))
        if tuple(inputs) != GATE9_SUPPORT_ORDER:
            raise ValueError("Gate9 worker support rows are not in qualified order")
        input_rows.append(inputs)
        output_rows.append(outputs)
    query_values = [_valid_byte(value, f"query {index}") for index, value in enumerate(queries)]
    support_inputs = torch.tensor(input_rows, dtype=torch.long, device=device)
    support_outputs = torch.tensor(output_rows, dtype=torch.long, device=device)
    query_tensor = torch.tensor(query_values, dtype=torch.long, device=device)
    _validate_worker_tensors(support_inputs, support_outputs, query_tensor)
    return support_inputs, support_outputs, query_tensor


class Gate9ContextualWorker(nn.Module):
    """One shared exact-budget contextual byte operator."""

    def __init__(self) -> None:
        super().__init__()
        self.pair_projection = nn.Linear(GATE9_PAIR_FEATURES, GATE9_EMBED_DIM)
        self.query_projection = nn.Linear(GATE9_BYTE_BITS, GATE9_EMBED_DIM)
        self.support_slot_modulation = nn.Parameter(
            torch.empty(GATE9_SUPPORT_EXAMPLES, GATE9_POSITION_MODULATION_DIM)
        )
        self.support_attention = nn.MultiheadAttention(
            embed_dim=GATE9_EMBED_DIM,
            num_heads=GATE9_ATTENTION_HEADS,
            batch_first=True,
        )
        self.support_ff_in = nn.Linear(GATE9_EMBED_DIM, GATE9_FFN_DIM)
        self.support_ff_out = nn.Linear(GATE9_FFN_DIM, GATE9_EMBED_DIM)
        self.query_support_fusion = nn.Linear(
            2 * GATE9_EMBED_DIM, GATE9_FUSION_DIM
        )
        self.output_bit_head = nn.Linear(
            GATE9_FUSION_DIM, GATE9_OUTPUT_BITS, bias=False
        )
        self.output_scale = nn.Parameter(torch.ones(()))
        self.reset_parameters()
        if self.learned_parameter_count() != GATE9_LEARNED_PARAMETER_COUNT:
            raise RuntimeError("Gate9 contextual-worker parameter count drifted")

    def reset_parameters(self) -> None:
        self.pair_projection.reset_parameters()
        self.query_projection.reset_parameters()
        nn.init.normal_(self.support_slot_modulation, mean=0.0, std=0.02)
        self.support_attention._reset_parameters()
        self.support_ff_in.reset_parameters()
        self.support_ff_out.reset_parameters()
        self.query_support_fusion.reset_parameters()
        self.output_bit_head.reset_parameters()
        with torch.no_grad():
            self.output_scale.fill_(1.0)

    def learned_parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def parameter_breakdown(self) -> dict[str, int]:
        observed = {
            "pair_projection": sum(
                parameter.numel() for parameter in self.pair_projection.parameters()
            ),
            "query_projection": sum(
                parameter.numel() for parameter in self.query_projection.parameters()
            ),
            "support_slot_modulation": self.support_slot_modulation.numel(),
            "support_attention": sum(
                parameter.numel() for parameter in self.support_attention.parameters()
            ),
            "support_feedforward": sum(
                parameter.numel()
                for module in (self.support_ff_in, self.support_ff_out)
                for parameter in module.parameters()
            ),
            "query_support_fusion": sum(
                parameter.numel() for parameter in self.query_support_fusion.parameters()
            ),
            "output_bit_head": sum(
                parameter.numel() for parameter in self.output_bit_head.parameters()
            ),
            "output_scale": self.output_scale.numel(),
        }
        if observed != GATE9_PARAMETER_BREAKDOWN:
            raise RuntimeError("Gate9 contextual-worker parameter breakdown drifted")
        return observed

    def forward(
        self,
        support_inputs: Tensor,
        support_outputs: Tensor,
        query: Tensor,
    ) -> Tensor:
        _validate_worker_tensors(support_inputs, support_outputs, query)
        pair_features = torch.cat(
            (_byte_bits(support_inputs), _byte_bits(support_outputs)), dim=-1
        )
        support = self.pair_projection(pair_features)
        modulation = F.pad(
            self.support_slot_modulation,
            (0, GATE9_EMBED_DIM - GATE9_POSITION_MODULATION_DIM),
        )
        support = support + modulation.unsqueeze(0)
        attended, _ = self.support_attention(
            support,
            support,
            support,
            need_weights=False,
        )
        support = F.layer_norm(support + attended, (GATE9_EMBED_DIM,))
        feedforward = self.support_ff_out(F.gelu(self.support_ff_in(support)))
        support = F.layer_norm(support + feedforward, (GATE9_EMBED_DIM,))
        support_summary = support.mean(dim=1)
        return self._output_from_summary(support_summary, query)

    def _output_from_summary(self, support_summary: Tensor, query: Tensor) -> Tensor:
        if support_summary.shape != (query.shape[0], GATE9_EMBED_DIM):
            raise ValueError("Gate9 support summary shape drifted")
        query_state = self.query_projection(_byte_bits(query))
        fused = torch.tanh(
            self.query_support_fusion(
                torch.cat((support_summary, query_state), dim=-1)
            )
        )
        logits = self.output_bit_head(fused) * self.output_scale
        if logits.shape != (query.shape[0], GATE9_OUTPUT_BITS):
            raise RuntimeError("Gate9 contextual-worker output shape drifted")
        return logits

    def forward_query_only(self, query: Tensor) -> Tensor:
        """Evaluate the frozen query-only control with no support rows available."""

        if query.dtype != torch.long:
            raise ValueError("Gate9 query tensor must use torch.long")
        if query.ndim != 1 or query.shape[0] <= 0:
            raise ValueError("Gate9 query-only tensor must be one nonempty byte vector")
        if bool(torch.any((query < 0) | (query > 255))):
            raise ValueError("Gate9 query byte lies outside 0..255")
        empty_support = torch.zeros(
            query.shape[0],
            GATE9_EMBED_DIM,
            device=query.device,
            dtype=self.query_projection.weight.dtype,
        )
        return self._output_from_summary(empty_support, query)

    @staticmethod
    def decode_bytes(bit_logits: Tensor) -> Tensor:
        if bit_logits.ndim != 2 or bit_logits.shape[1] != GATE9_OUTPUT_BITS:
            raise ValueError("Gate9 bit logits must have shape [batch,8]")
        if not torch.is_floating_point(bit_logits):
            raise ValueError("Gate9 bit logits must be floating point")
        weights = (1 << torch.arange(8, device=bit_logits.device, dtype=torch.long))
        return ((bit_logits >= 0).to(torch.long) * weights).sum(dim=-1)


def gate9_contextual_worker_architecture_plan() -> dict[str, Any]:
    model = Gate9ContextualWorker()
    return {
        "version": GATE9_CONTEXTUAL_WORKER_VERSION,
        "status": GATE9_CONTEXTUAL_WORKER_STATUS,
        "graph_world_contract_head": GATE9_GRAPH_WORLD_CONTRACT_HEAD,
        "learned_parameter_count": model.learned_parameter_count(),
        "parameter_breakdown": model.parameter_breakdown(),
        "input": {
            "support_examples": GATE9_SUPPORT_EXAMPLES,
            "support_order": list(GATE9_SUPPORT_ORDER),
            "support_pair_features": GATE9_PAIR_FEATURES,
            "query_features": GATE9_BYTE_BITS,
            "operator_counter_visible": False,
            "operator_key_visible": False,
            "world_id_visible": False,
            "worker_id_visible": False,
            "node_id_visible": False,
            "population_visible": False,
            "depth_visible": False,
            "round_visible": False,
            "target_flag_visible": False,
        },
        "architecture": {
            "pair_projection": [16, 48],
            "query_projection": [8, 48],
            "support_slot_modulation": [9, 24],
            "support_attention": {"embed_dim": 48, "heads": 4},
            "support_feedforward": [48, 64, 48],
            "query_support_fusion": [96, 24],
            "output_bit_head": [24, 8],
            "normalization": "parameter_free_layer_norm",
            "byte_decoding": "eight_independent_zero_threshold_bits",
            "query_only_control": "zero_support_summary_no_support_rows",
        },
        "shared_across_workers": True,
        "shared_across_populations": True,
        "shared_across_rounds": True,
        "per_operator_parameters": 0,
        "padding_parameters": 0,
        "contract_forward_backward_admitted": True,
        "optimizer_admitted": False,
        "training_admitted": False,
        "checkpoint_serialization_admitted": False,
        "checkpoint_loading_admitted": False,
        "scientific_test_world_generation_admitted": False,
        "scientific_execution_admitted": False,
        "result_classification_admitted": False,
    }
