"""Minimal shared-weight recurrent population cell for Gate 0B.

This is deliberately generic. The collective-relay encoder/runner can sit on top of
it without changing the population-compute contract.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import torch
from torch import nn

from .contract import CommunicationMode


@dataclass(frozen=True, slots=True)
class SharedPopulationConfig:
    local_input_width: int
    state_width: int
    message_width: int
    output_width: int

    def validate(self) -> None:
        if self.local_input_width <= 0:
            raise ValueError("local_input_width must be positive")
        if self.state_width <= 0:
            raise ValueError("state_width must be positive")
        if self.message_width <= 0:
            raise ValueError("message_width must be positive")
        if self.output_width <= 0:
            raise ValueError("output_width must be positive")


@dataclass(frozen=True, slots=True)
class PopulationTelemetry:
    active_state_updates: int
    messages_emitted: int
    communicated_scalar_count: int


class PopulationForwardOutput(tuple):
    """Tuple-like output without importing a heavier result abstraction."""

    __slots__ = ()

    def __new__(
        cls,
        logits: torch.Tensor,
        final_states: torch.Tensor,
        telemetry: PopulationTelemetry,
    ) -> "PopulationForwardOutput":
        return tuple.__new__(cls, (logits, final_states, telemetry))

    @property
    def logits(self) -> torch.Tensor:
        return self[0]

    @property
    def final_states(self) -> torch.Tensor:
        return self[1]

    @property
    def telemetry(self) -> PopulationTelemetry:
        return self[2]


class SharedPopulationCell(nn.Module):
    """One learned update rule reused by every runtime worker state.

    Communication is O(active_workers * message_width) per recurrent round rather
    than pairwise O(N^2): active workers write bounded messages into one shared
    mean field and read that same bounded field on the next update.
    """

    def __init__(self, config: SharedPopulationConfig) -> None:
        super().__init__()
        config.validate()
        self.config = config

        self.input_projection = nn.Linear(
            config.local_input_width,
            config.state_width,
        )
        self.message_projection = nn.Linear(
            config.state_width,
            config.message_width,
        )
        self.update = nn.GRUCell(
            config.local_input_width + config.message_width,
            config.state_width,
        )
        self.output_norm = nn.LayerNorm(config.state_width)
        self.output_head = nn.Linear(config.state_width, config.output_width)

    def forward(
        self,
        local_inputs: torch.Tensor,
        active_mask: torch.Tensor,
        *,
        recurrent_rounds: int,
        communication_mode: CommunicationMode,
    ) -> PopulationForwardOutput:
        if local_inputs.ndim != 3:
            raise ValueError(
                "local_inputs must have shape [batch, workers, local_input_width]"
            )
        if active_mask.ndim != 2:
            raise ValueError("active_mask must have shape [batch, workers]")
        if local_inputs.shape[:2] != active_mask.shape:
            raise ValueError("local_inputs and active_mask worker dimensions must match")
        if local_inputs.shape[-1] != self.config.local_input_width:
            raise ValueError(
                f"expected local input width {self.config.local_input_width}, "
                f"got {local_inputs.shape[-1]}"
            )
        if recurrent_rounds <= 0:
            raise ValueError("recurrent_rounds must be positive")
        if communication_mode not in {
            CommunicationMode.NO_COMMUNICATION,
            CommunicationMode.SPARSE_SHARED_V0,
        }:
            raise ValueError(
                "SharedPopulationCell v0 supports only no_communication and "
                "sparse_shared_v0"
            )

        mask = active_mask.to(dtype=torch.bool)
        if not torch.all(mask.any(dim=1)):
            raise ValueError("every sample must activate at least one worker state")

        state = torch.tanh(self.input_projection(local_inputs))
        state = state * mask.unsqueeze(-1).to(dtype=state.dtype)

        batch_size, worker_count, _ = local_inputs.shape
        active_states_per_batch = int(mask.sum().item())
        messages_emitted = 0
        communicated_scalars = 0

        for _ in range(recurrent_rounds):
            if communication_mode is CommunicationMode.SPARSE_SHARED_V0:
                messages = self.message_projection(state)
                message_mask = mask.unsqueeze(-1).to(dtype=messages.dtype)
                messages = messages * message_mask
                denominator = message_mask.sum(dim=1).clamp_min(1.0)
                shared = messages.sum(dim=1) / denominator
                shared_per_worker = shared.unsqueeze(1).expand(
                    batch_size,
                    worker_count,
                    self.config.message_width,
                )
                messages_emitted += active_states_per_batch
                # Count one bounded write and one bounded read per active state.
                communicated_scalars += (
                    2 * active_states_per_batch * self.config.message_width
                )
            else:
                shared_per_worker = local_inputs.new_zeros(
                    batch_size,
                    worker_count,
                    self.config.message_width,
                )

            update_input = torch.cat((local_inputs, shared_per_worker), dim=-1)
            flat_input = update_input.reshape(batch_size * worker_count, -1)
            flat_state = state.reshape(batch_size * worker_count, -1)
            candidate = self.update(flat_input, flat_state).reshape_as(state)
            state = torch.where(mask.unsqueeze(-1), candidate, torch.zeros_like(candidate))

        weights = mask.unsqueeze(-1).to(dtype=state.dtype)
        pooled = (state * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
        logits = self.output_head(self.output_norm(pooled))

        telemetry = PopulationTelemetry(
            active_state_updates=active_states_per_batch * recurrent_rounds,
            messages_emitted=messages_emitted,
            communicated_scalar_count=communicated_scalars,
        )
        return PopulationForwardOutput(logits, state, telemetry)

    def trainable_parameter_count(self) -> int:
        return sum(
            parameter.numel()
            for parameter in self.parameters()
            if parameter.requires_grad
        )

    def parameter_fingerprint(self) -> str:
        """Stable SHA-256 over exact state-dict names, metadata and tensor bytes.

        The byte path intentionally uses PyTorch storage directly so the scientific
        identity check does not acquire a hidden NumPy runtime dependency.
        """

        digest = hashlib.sha256()
        for name, tensor in sorted(self.state_dict().items()):
            # Clone after making the tensor contiguous so the storage contains
            # exactly this tensor's bytes rather than a potentially larger shared
            # backing allocation from a view.
            detached = tensor.detach().cpu().contiguous().clone()
            digest.update(name.encode("utf-8"))
            digest.update(str(detached.dtype).encode("ascii"))
            digest.update(str(tuple(detached.shape)).encode("ascii"))
            digest.update(bytes(detached.untyped_storage()))
        return digest.hexdigest()
