"""Bounded neural adapter for Population Language Post-Training Learning L0."""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
import math
from typing import Mapping

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from . import l0_protocol
from . import post_training_learning_l0_protocol as protocol
from . import post_training_learning_l0_world as world

VERSION = "population-language-post-training-learning-l0-adapter-v0"
BRANCH = "agent/population-language-post-training-learning-l0-adapter-v0"
STATUS = "ADAPTER_IMPLEMENTATION_ONLY_NO_CALIBRATION_OR_FINAL_RESULT"
SUPPORTED_RANKS = (1, 2, 4, 6)
NAMES = (
    "operator_embedding_delta", "encoder_down", "encoder_up",
    "decoder_down", "decoder_up", "value_logit_bias",
)


@dataclass(frozen=True)
class AdapterConfig:
    rank: int = 6
    alpha: float | None = None

    def validate(self) -> "AdapterConfig":
        if type(self.rank) is not int or self.rank not in SUPPORTED_RANKS:
            raise ValueError(f"rank must be one of {SUPPORTED_RANKS}")
        alpha = float(self.rank if self.alpha is None else self.alpha)
        if not math.isfinite(alpha) or alpha <= 0:
            raise ValueError("alpha must be finite and positive")
        return self

    @property
    def scale(self) -> float:
        self.validate()
        return float(self.rank if self.alpha is None else self.alpha) / self.rank


@dataclass(frozen=True)
class AdaptedPopulationForward:
    logits: Tensor
    final_state: Tensor
    routed_messages: int


def parameter_count(rank: int) -> int:
    if rank not in SUPPORTED_RANKS:
        raise ValueError(f"rank must be one of {SUPPORTED_RANKS}")
    return 2 * rank * (14_544 + 128) + 8 * 512 + 16


def raw_fp32_bytes(rank: int) -> int:
    return 4 * parameter_count(rank)


def _random(shape: tuple[int, ...], seed: int, fan_in: int) -> Tensor:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    bound = 1.0 / math.sqrt(fan_in)
    return torch.empty(shape).uniform_(-bound, bound, generator=generator)


class BoundedPopulationAdapter(nn.Module):
    """Six trainable tensors around an immutable PopulationLanguageOrganism."""

    def __init__(self, base: nn.Module, *, model_seed: int, config: AdapterConfig = AdapterConfig()) -> None:
        super().__init__()
        self.config = config.validate()
        self.model_seed = model_seed
        self.initialization_seed = protocol.adapter_initialization_seed(model_seed)
        self.base = base
        self._validate_base()
        for parameter in base.parameters():
            parameter.requires_grad_(False)

        rank, seed = self.config.rank, self.initialization_seed
        device = base.token_embedding.weight.device
        self.operator_embedding_delta = nn.Parameter(torch.zeros(8, 512, device=device))
        self.encoder_down = nn.Parameter(_random((rank, 14_544), seed + 1, 14_544).to(device))
        self.encoder_up = nn.Parameter(torch.zeros(128, rank, device=device))
        self.decoder_down = nn.Parameter(_random((rank, 128), seed + 2, 128).to(device))
        self.decoder_up = nn.Parameter(torch.zeros(14_544, rank, device=device))
        self.value_logit_bias = nn.Parameter(torch.zeros(16, device=device))
        if self.trainable_parameter_count() != parameter_count(rank):
            raise RuntimeError("adapter parameter count drifted")
        if self.trainable_parameter_count() > protocol.MAX_TRAINABLE_ADAPTATION_PARAMETERS:
            raise RuntimeError("adapter exceeds parameter budget")
        if self.raw_fp32_tensor_bytes() > protocol.MAX_PERSISTED_ADAPTATION_BYTES:
            raise RuntimeError("adapter exceeds artifact budget")

    def _validate_base(self) -> None:
        required = (
            "config", "token_embedding", "position_embedding", "lexical_encoder",
            "initializer", "_communication_round", "lexical_decoder", "final_norm",
            "lm_bias", "communication_rounds", "top_k", "forward_with_state",
        )
        missing = [name for name in required if not hasattr(self.base, name)]
        if missing:
            raise TypeError(f"population base is missing: {missing}")
        expected = {
            "vocab_size": 64, "max_sequence_length": 32, "token_width": 512,
            "lexical_encoder_width": 14_544, "worker_width": 128,
            "lexical_decoder_width": 14_544,
        }
        for name, value in expected.items():
            if getattr(self.base.config, name, None) != value:
                raise ValueError(f"population base {name} drifted")
        if sum(p.numel() for p in self.base.parameters()) != protocol.BASE_PARAMETER_COUNT:
            raise ValueError("population base parameter count drifted")

    def declared_parameters(self) -> OrderedDict[str, nn.Parameter]:
        return OrderedDict((name, getattr(self, name)) for name in NAMES)

    def declared_adaptation_parameters(self) -> OrderedDict[str, nn.Parameter]:
        return self.declared_parameters()

    def trainable_parameter_count(self) -> int:
        return sum(p.numel() for p in self.declared_parameters().values())

    def raw_fp32_tensor_bytes(self) -> int:
        return 4 * self.trainable_parameter_count()

    def adaptation_state_dict(self) -> OrderedDict[str, Tensor]:
        return OrderedDict(
            (name, value.detach().cpu().float().contiguous().clone())
            for name, value in self.declared_parameters().items()
        )

    def load_adaptation_state_dict(self, state: Mapping[str, Tensor]) -> None:
        if tuple(state) != NAMES:
            raise ValueError("artifact tensor names or order drifted")
        with torch.no_grad():
            for name, target in self.declared_parameters().items():
                source = state[name]
                if not isinstance(source, Tensor):
                    raise TypeError(f"artifact entry {name!r} is not a tensor")
                if source.shape != target.shape or source.dtype != torch.float32:
                    raise ValueError(f"artifact tensor contract drifted for {name}")
                if not bool(torch.isfinite(source).all()):
                    raise ValueError(f"artifact contains non-finite values for {name}")
                target.copy_(source.to(target.device))

    def _validate_ids(self, input_ids: Tensor) -> None:
        if input_ids.dtype != torch.long or input_ids.ndim != 2:
            raise ValueError("input IDs must be rank-2 torch.long")
        if input_ids.numel() == 0 or not 0 < input_ids.shape[1] <= 32:
            raise ValueError("input shape lies outside the model contract")
        if bool(torch.any((input_ids < 0) | (input_ids >= 64))):
            raise ValueError("input ID lies outside the vocabulary")

    def _task_mask(self, input_ids: Tensor) -> Tensor:
        self._validate_ids(input_ids)
        if not 5 <= input_ids.shape[1] <= 8:
            return torch.zeros(input_ids.shape[0], dtype=torch.bool, device=input_ids.device)
        ids = l0_protocol.TOKEN_TO_ID
        operators = torch.tensor([ids[t] for t in world.OPERATOR_TOKENS], device=input_ids.device)
        values = torch.tensor([ids[t] for t in world.VALUE_TOKENS], device=input_ids.device)
        return (
            (input_ids[:, 0] == ids["<bos>"])
            & (input_ids[:, 1] == ids["<query>"])
            & (input_ids[:, -1] == ids["<answer>"])
            & torch.isin(input_ids[:, 2:-2], operators).all(dim=1)
            & torch.isin(input_ids[:, -2], values)
        )

    def _operator_delta(self, token_ids: Tensor) -> Tensor:
        ids = l0_protocol.TOKEN_TO_ID
        result = torch.zeros((token_ids.shape[0], 512), device=token_ids.device)
        for index, token in enumerate(world.OPERATOR_TOKENS):
            result = result + (token_ids == ids[token]).unsqueeze(-1) * self.operator_embedding_delta[index]
        return result

    def forward_with_state(self, input_ids: Tensor, *, worker_count: int | None = None) -> AdaptedPopulationForward:
        mask = self._task_mask(input_ids)
        if not bool(torch.any(mask)):
            return self.base.forward_with_state(input_ids, worker_count=worker_count)
        if not bool(torch.all(mask)):
            raise ValueError("mixed original-L0 and adaptation-task batches are forbidden")

        workers = self.base.config.training_workers if worker_count is None else worker_count
        if type(workers) is not int or workers <= 0:
            raise ValueError("worker count must be positive")
        batch, sequence = input_ids.shape
        dtype, device = self.base.token_embedding.weight.dtype, input_ids.device
        state = torch.zeros((batch, workers, 128), device=device, dtype=dtype)
        from .l0_models import deterministic_worker_coordinates
        coordinates = deterministic_worker_coordinates(workers, 128, device=device, dtype=dtype)
        outputs: list[Tensor] = []
        routed_messages = 0

        for position in range(sequence):
            token_ids = input_ids[:, position]
            token = self.base.token_embedding(token_ids)
            token = token + self._operator_delta(token_ids).to(token.dtype)
            positions = torch.full((batch,), position, dtype=torch.long, device=device)
            hidden = self.base.lexical_encoder[0](token + self.base.position_embedding(positions))
            hidden = F.gelu(hidden, approximate="tanh")
            lexical = self.base.lexical_encoder[2](hidden)
            residual = F.linear(F.linear(hidden, self.encoder_down.to(hidden.dtype)), self.encoder_up.to(hidden.dtype))
            lexical = lexical + self.config.scale * residual
            injected = self.base.initializer(lexical.unsqueeze(1) + coordinates.unsqueeze(0))
            for _ in range(self.base.communication_rounds):
                state = self.base._communication_round(state, injected)
                routed_messages += batch * workers * min(self.base.top_k, workers)

            pooled = state.mean(dim=1)
            decoded = self.base.lexical_decoder[0](pooled)
            residual = F.linear(F.linear(pooled, self.decoder_down.to(pooled.dtype)), self.decoder_up.to(pooled.dtype))
            decoded = F.gelu(decoded + self.config.scale * residual, approximate="tanh")
            decoded = self.base.final_norm(self.base.lexical_decoder[2](decoded))
            logits = F.linear(decoded, self.base.token_embedding.weight, self.base.lm_bias)
            if position == sequence - 1:
                value_ids = torch.tensor([l0_protocol.TOKEN_TO_ID[t] for t in world.VALUE_TOKENS], device=device)
                bias = torch.zeros(64, device=device, dtype=logits.dtype).scatter(
                    0, value_ids, self.value_logit_bias.to(logits.dtype)
                )
                logits = logits + bias
            outputs.append(logits)
        return AdaptedPopulationForward(torch.stack(outputs, dim=1), state, routed_messages)

    def forward(self, input_ids: Tensor, *, worker_count: int | None = None) -> Tensor:
        return self.forward_with_state(input_ids, worker_count=worker_count).logits


def validate_adapter_contract() -> dict[str, object]:
    counts = {rank: parameter_count(rank) for rank in SUPPORTED_RANKS}
    sizes = {rank: raw_fp32_bytes(rank) for rank in SUPPORTED_RANKS}
    checks = {
        "rank_six_parameter_count": counts[6] == 180_176,
        "rank_six_raw_bytes": sizes[6] == 720_704,
        "all_parameter_budgets": all(v <= protocol.MAX_TRAINABLE_ADAPTATION_PARAMETERS for v in counts.values()),
        "all_artifact_budgets": all(v <= protocol.MAX_PERSISTED_ADAPTATION_BYTES for v in sizes.values()),
        "six_declared_tensors": len(NAMES) == 6,
    }
    return {
        "status": STATUS, "version": VERSION, "supported_ranks": list(SUPPORTED_RANKS),
        "parameters_by_rank": counts, "raw_fp32_bytes_by_rank": sizes,
        "declared_tensor_names": list(NAMES), "checks": checks, "valid": all(checks.values()),
    }
