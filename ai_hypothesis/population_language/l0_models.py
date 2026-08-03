"""Matched Population Language L0 model implementations."""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from . import l0_protocol as protocol


def count_parameters(module: nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters())


def _validate_input_ids(input_ids: Tensor, vocab_size: int, maximum_length: int) -> None:
    if input_ids.dtype != torch.long or input_ids.ndim != 2:
        raise ValueError("Population Language input IDs must be rank-2 torch.long")
    if input_ids.shape[1] == 0 or input_ids.shape[1] > maximum_length:
        raise ValueError("Population Language sequence length is outside the model contract")
    if input_ids.numel() == 0:
        raise ValueError("Population Language input IDs cannot be empty")
    if bool(torch.any((input_ids < 0) | (input_ids >= vocab_size))):
        raise ValueError("Population Language input ID lies outside the vocabulary")


class CausalSelfAttention(nn.Module):
    def __init__(self, width: int, heads: int) -> None:
        super().__init__()
        if width % heads:
            raise ValueError("attention width must divide evenly across heads")
        self.width = width
        self.heads = heads
        self.head_width = width // heads
        self.qkv = nn.Linear(width, 3 * width)
        self.output = nn.Linear(width, width)

    def forward(self, values: Tensor) -> Tensor:
        batch, sequence, width = values.shape
        projected = self.qkv(values).view(
            batch, sequence, 3, self.heads, self.head_width
        )
        query, key, value = projected.permute(2, 0, 3, 1, 4).unbind(0)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.head_width)
        future = torch.ones(
            (sequence, sequence), dtype=torch.bool, device=values.device
        ).triu(diagonal=1)
        scores = scores.masked_fill(future, torch.finfo(scores.dtype).min)
        weights = torch.softmax(scores.to(torch.float32), dim=-1).to(values.dtype)
        mixed = torch.matmul(weights, value)
        mixed = mixed.transpose(1, 2).contiguous().view(batch, sequence, width)
        return self.output(mixed)


class TransformerBlock(nn.Module):
    def __init__(self, config: protocol.TransformerConfig) -> None:
        super().__init__()
        self.attention_norm = nn.LayerNorm(config.d_model)
        self.attention = CausalSelfAttention(config.d_model, config.heads)
        self.feed_forward_norm = nn.LayerNorm(config.d_model)
        self.feed_forward_input = nn.Linear(config.d_model, config.feed_forward)
        self.feed_forward_output = nn.Linear(config.feed_forward, config.d_model)

    def forward(self, values: Tensor) -> Tensor:
        values = values + self.attention(self.attention_norm(values))
        hidden = self.feed_forward_input(self.feed_forward_norm(values))
        hidden = F.gelu(hidden, approximate="tanh")
        return values + self.feed_forward_output(hidden)


class MatchedCausalTransformer(nn.Module):
    def __init__(self, config: protocol.TransformerConfig = protocol.TransformerConfig()) -> None:
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.position_embedding = nn.Embedding(config.max_sequence_length, config.d_model)
        self.blocks = nn.ModuleList(TransformerBlock(config) for _ in range(config.layers))
        self.final_norm = nn.LayerNorm(config.d_model)
        self.lm_bias = nn.Parameter(torch.zeros(config.vocab_size))

    def forward(self, input_ids: Tensor) -> Tensor:
        _validate_input_ids(
            input_ids, self.config.vocab_size, self.config.max_sequence_length
        )
        positions = torch.arange(input_ids.shape[1], device=input_ids.device)
        values = self.token_embedding(input_ids) + self.position_embedding(positions)
        for block in self.blocks:
            values = block(values)
        values = self.final_norm(values)
        return F.linear(values, self.token_embedding.weight, self.lm_bias)


class SharedGRUCell(nn.Module):
    """GRU-like cell with one shared three-gate bias, matching the protocol count."""

    def __init__(self, width: int) -> None:
        super().__init__()
        self.width = width
        self.input_weight = nn.Parameter(torch.empty(3 * width, width))
        self.hidden_weight = nn.Parameter(torch.empty(3 * width, width))
        self.bias = nn.Parameter(torch.zeros(3 * width))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        bound = 1.0 / math.sqrt(self.width)
        nn.init.uniform_(self.input_weight, -bound, bound)
        nn.init.uniform_(self.hidden_weight, -bound, bound)
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, drive: Tensor, state: Tensor) -> Tensor:
        input_parts = F.linear(drive, self.input_weight, self.bias).chunk(3, dim=-1)
        hidden_parts = F.linear(state, self.hidden_weight).chunk(3, dim=-1)
        reset = torch.sigmoid(input_parts[0] + hidden_parts[0])
        update = torch.sigmoid(input_parts[1] + hidden_parts[1])
        candidate = torch.tanh(input_parts[2] + reset * hidden_parts[2])
        return (1.0 - update) * candidate + update * state


@dataclass(frozen=True)
class PopulationForward:
    logits: Tensor
    final_state: Tensor
    routed_messages: int


def deterministic_worker_coordinates(
    worker_count: int,
    width: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> Tensor:
    if type(worker_count) is not int or worker_count <= 0:
        raise ValueError("worker count must be a positive integer")
    ranks = (torch.arange(worker_count, device=device, dtype=torch.float32) + 0.5) / worker_count
    channels = torch.arange(1, width + 1, device=device, dtype=torch.float32)
    phase = 2.0 * math.pi * ranks.unsqueeze(1) * channels.unsqueeze(0)
    coordinates = 0.5 * (torch.sin(phase) + torch.cos(phase * 0.5))
    return coordinates.to(dtype=dtype)


class PopulationLanguageOrganism(nn.Module):
    def __init__(
        self,
        config: protocol.PopulationConfig = protocol.PopulationConfig(),
        *,
        communication_rounds: int = 6,
        top_k: int = 4,
    ) -> None:
        super().__init__()
        if communication_rounds <= 0:
            raise ValueError("communication rounds must be positive")
        if top_k <= 0:
            raise ValueError("router top-k must be positive")
        self.config = config
        self.communication_rounds = communication_rounds
        self.top_k = top_k

        self.token_embedding = nn.Embedding(config.vocab_size, config.token_width)
        self.position_embedding = nn.Embedding(
            config.max_sequence_length, config.token_width
        )
        self.lexical_encoder = nn.Sequential(
            nn.Linear(config.token_width, config.lexical_encoder_width),
            nn.GELU(approximate="tanh"),
            nn.Linear(config.lexical_encoder_width, config.worker_width),
        )
        self.initializer = nn.Linear(config.worker_width, config.worker_width)
        self.gru = SharedGRUCell(config.worker_width)
        self.message_encoder = nn.Linear(config.worker_width, config.worker_width)
        self.router_query = nn.Linear(config.worker_width, config.router_dim)
        self.router_key = nn.Linear(config.worker_width, config.router_dim)
        self.worker_ff_input = nn.Linear(
            config.worker_width, config.worker_feed_forward
        )
        self.worker_ff_output = nn.Linear(
            config.worker_feed_forward, config.worker_width
        )
        self.lexical_decoder = nn.Sequential(
            nn.Linear(config.worker_width, config.lexical_decoder_width),
            nn.GELU(approximate="tanh"),
            nn.Linear(config.lexical_decoder_width, config.token_width),
        )
        self.final_norm = nn.LayerNorm(config.token_width)
        self.lm_bias = nn.Parameter(torch.zeros(config.vocab_size))

    def _communication_round(self, state: Tensor, injected: Tensor) -> Tensor:
        batch, workers, width = state.shape
        normalized = F.layer_norm(state, (width,))
        query = self.router_query(normalized)
        key = self.router_key(normalized)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(
            self.config.router_dim
        )
        sender_tie_break = -torch.arange(
            workers, device=state.device, dtype=scores.dtype
        ) * 1e-7
        scores = scores + sender_tie_break.view(1, 1, workers)
        selected_scores, selected_indices = torch.topk(
            scores, k=min(self.top_k, workers), dim=-1
        )
        weights = torch.softmax(
            selected_scores.to(torch.float32), dim=-1
        ).to(state.dtype)

        encoded = self.message_encoder(normalized)
        batch_indices = torch.arange(batch, device=state.device).view(batch, 1, 1)
        batch_indices = batch_indices.expand_as(selected_indices)
        selected_messages = encoded[batch_indices, selected_indices]
        message = torch.sum(selected_messages * weights.unsqueeze(-1), dim=-2)

        state = self.gru(injected + message, state)
        feed_forward = self.worker_ff_input(F.layer_norm(state, (width,)))
        feed_forward = F.gelu(feed_forward, approximate="tanh")
        return state + self.worker_ff_output(feed_forward)

    def forward_with_state(
        self,
        input_ids: Tensor,
        *,
        worker_count: int | None = None,
    ) -> PopulationForward:
        _validate_input_ids(
            input_ids, self.config.vocab_size, self.config.max_sequence_length
        )
        workers = self.config.training_workers if worker_count is None else worker_count
        if type(workers) is not int or workers <= 0:
            raise ValueError("worker count must be a positive integer")

        batch, sequence = input_ids.shape
        state = torch.zeros(
            (batch, workers, self.config.worker_width),
            device=input_ids.device,
            dtype=self.token_embedding.weight.dtype,
        )
        coordinates = deterministic_worker_coordinates(
            workers,
            self.config.worker_width,
            device=input_ids.device,
            dtype=state.dtype,
        )
        logits: list[Tensor] = []
        routed_messages = 0

        for position in range(sequence):
            token = self.token_embedding(input_ids[:, position])
            position_ids = torch.full(
                (batch,), position, dtype=torch.long, device=input_ids.device
            )
            token = token + self.position_embedding(position_ids)
            lexical = self.lexical_encoder(token)
            injected = self.initializer(
                lexical.unsqueeze(1) + coordinates.unsqueeze(0)
            )

            for _ in range(self.communication_rounds):
                state = self._communication_round(state, injected)
                routed_messages += batch * workers * min(self.top_k, workers)

            pooled = torch.mean(state, dim=1)
            decoded = self.lexical_decoder(pooled)
            decoded = self.final_norm(decoded)
            logits.append(
                F.linear(decoded, self.token_embedding.weight, self.lm_bias)
            )

        return PopulationForward(
            logits=torch.stack(logits, dim=1),
            final_state=state,
            routed_messages=routed_messages,
        )

    def forward(
        self,
        input_ids: Tensor,
        *,
        worker_count: int | None = None,
    ) -> Tensor:
        return self.forward_with_state(
            input_ids, worker_count=worker_count
        ).logits


def validate_model_parameter_counts() -> dict[str, int | bool]:
    transformer = MatchedCausalTransformer()
    population = PopulationLanguageOrganism()
    transformer_actual = count_parameters(transformer)
    population_actual = count_parameters(population)
    return {
        "transformer_actual": transformer_actual,
        "transformer_expected": protocol.transformer_parameter_count(),
        "population_actual": population_actual,
        "population_expected": protocol.population_parameter_count(),
        "valid": (
            transformer_actual == protocol.transformer_parameter_count()
            and population_actual == protocol.population_parameter_count()
        ),
    }
