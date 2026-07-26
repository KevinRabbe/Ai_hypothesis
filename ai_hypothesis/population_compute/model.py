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
    """Tuple-like result for one recurrent population execution."""

    __slots__ = ()

    def __new__(
        cls,
        logits: torch.Tensor,
        final_states: torch.Tensor,
        final_shared: torch.Tensor,
        telemetry: PopulationTelemetry,
    ) -> "PopulationForwardOutput":
        return tuple.__new__(cls, (logits, final_states, final_shared, telemetry))

    @property
    def logits(self) -> torch.Tensor:
        return self[0]

    @property
    def final_states(self) -> torch.Tensor:
        return self[1]

    @property
    def final_shared(self) -> torch.Tensor:
        return self[2]

    @property
    def telemetry(self) -> PopulationTelemetry:
        return self[3]


class SharedPopulationCell(nn.Module):
    """One learned update rule reused by every runtime worker state.

    `sparse_shared_v0` uses a bounded gated shared field. Every active worker reads
    one message-width field, updates locally, then produces one gated candidate
    message. Candidate messages are summed and bounded with tanh. Communication is
    therefore O(active_workers * message_width), not pairwise O(N^2), and one rare
    useful emission is not divided by the total population size.

    Inactive slots are not merely zeroed after execution: they are gathered out before
    every learned projection, GRU update and message projection. Thus the neural compute
    path scales with the active-state count rather than the padded tensor width used by
    the surrounding benchmark representation.
    """

    def __init__(self, config: SharedPopulationConfig) -> None:
        super().__init__()
        config.validate()
        self.config = config

        self.input_projection = nn.Linear(
            config.local_input_width,
            config.state_width,
        )
        self.update = nn.GRUCell(
            config.local_input_width + config.message_width,
            config.state_width,
        )
        self.message_projection = nn.Linear(
            config.state_width,
            config.message_width,
        )
        self.message_gate = nn.Linear(config.state_width, 1)
        self.output_norm = nn.LayerNorm(config.state_width + config.message_width)
        self.output_head = nn.Linear(
            config.state_width + config.message_width,
            config.output_width,
        )

    def forward(
        self,
        local_inputs: torch.Tensor,
        active_mask: torch.Tensor,
        *,
        recurrent_rounds: int,
        communication_mode: CommunicationMode,
        shared_seed: torch.Tensor | None = None,
        message_content: torch.Tensor | None = None,
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

        batch_size, worker_count, _ = local_inputs.shape
        if shared_seed is None:
            seed = local_inputs.new_zeros(batch_size, self.config.message_width)
        else:
            if shared_seed.ndim != 2:
                raise ValueError("shared_seed must have shape [batch, message_width]")
            if shared_seed.shape != (batch_size, self.config.message_width):
                raise ValueError(
                    "shared_seed must match the local-input batch and message width"
                )
            seed = shared_seed.to(device=local_inputs.device, dtype=local_inputs.dtype)

        if message_content is not None:
            expected_message_shape = (
                batch_size,
                worker_count,
                self.config.message_width,
            )
            if message_content.shape != expected_message_shape:
                raise ValueError(
                    "message_content must have shape "
                    "[batch, workers, message_width]"
                )
            message_content = message_content.to(
                device=local_inputs.device,
                dtype=local_inputs.dtype,
            )

        flat_mask = mask.reshape(-1)
        active_flat_indices = torch.nonzero(flat_mask, as_tuple=False).squeeze(1)
        flat_local_inputs = local_inputs.reshape(
            batch_size * worker_count,
            self.config.local_input_width,
        )
        active_local_inputs = flat_local_inputs.index_select(0, active_flat_indices)
        active_message_content = None
        if message_content is not None:
            flat_message_content = message_content.reshape(
                batch_size * worker_count,
                self.config.message_width,
            )
            active_message_content = flat_message_content.index_select(
                0,
                active_flat_indices,
            )

        flat_batch_indices = (
            torch.arange(batch_size, device=local_inputs.device)
            .unsqueeze(1)
            .expand(batch_size, worker_count)
            .reshape(-1)
        )
        active_batch_indices = flat_batch_indices.index_select(0, active_flat_indices)
        active_states = torch.tanh(self.input_projection(active_local_inputs))
        shared = seed

        active_states_per_batch = int(active_states.shape[0])
        messages_emitted = 0
        communicated_scalars = 0

        for _ in range(recurrent_rounds):
            shared_for_active = shared.index_select(0, active_batch_indices)
            update_input = torch.cat((active_local_inputs, shared_for_active), dim=-1)
            active_states = self.update(update_input, active_states)

            if communication_mode is CommunicationMode.SPARSE_SHARED_V0:
                gate = torch.sigmoid(self.message_gate(active_states))
                if active_message_content is None:
                    content = torch.tanh(self.message_projection(active_states))
                else:
                    content = active_message_content
                messages = content * gate
                message_sum = messages.new_zeros(
                    batch_size,
                    self.config.message_width,
                ).index_add(0, active_batch_indices, messages)
                shared = torch.tanh(message_sum)
                messages_emitted += active_states_per_batch
                # One bounded field read and one candidate write per active state.
                communicated_scalars += (
                    2 * active_states_per_batch * self.config.message_width
                )
            else:
                # The externally supplied seed is available to each local worker,
                # but population-produced state never moves between workers.
                shared = seed

        pooled_sum = active_states.new_zeros(
            batch_size,
            self.config.state_width,
        ).index_add(0, active_batch_indices, active_states)
        active_counts = torch.bincount(
            active_batch_indices,
            minlength=batch_size,
        ).to(dtype=active_states.dtype).unsqueeze(1)
        pooled = pooled_sum / active_counts.clamp_min(1.0)
        readout = torch.cat((pooled, shared), dim=-1)
        logits = self.output_head(self.output_norm(readout))

        # Preserve the public padded diagnostic shape without carrying padded recurrent
        # state through the learned hot path.
        flat_final_states = active_states.new_zeros(
            batch_size * worker_count,
            self.config.state_width,
        ).index_copy(0, active_flat_indices, active_states)
        final_states = flat_final_states.reshape(
            batch_size,
            worker_count,
            self.config.state_width,
        )

        telemetry = PopulationTelemetry(
            active_state_updates=active_states_per_batch * recurrent_rounds,
            messages_emitted=messages_emitted,
            communicated_scalar_count=communicated_scalars,
        )
        return PopulationForwardOutput(logits, final_states, shared, telemetry)

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
