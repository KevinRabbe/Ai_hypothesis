"""Matched parallel-versus-serial execution control for repaired relay transport.

The repaired relay computation resets record-local neural state on every relay hop. That
makes one normalized population hop separable across local records: every record applies
the same learned update to the same shared query, then a parameter-free softmax reducer
combines candidate values.

This module evaluates that exact computation in two schedules:

* ``normalized_parallel_forward`` keeps all active record states resident together.
* ``normalized_serial_forward`` time-multiplexes the same learned machinery through one
  record state per sample and computes the same softmax-weighted shared field with a
  numerically stable online accumulator.

Both schedules expose the same source prefix and perform exactly ``N * relay_hops``
learned worker updates per sample. The control therefore isolates simultaneous population
state / parallel execution from the trivial effect of spending more learned updates.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from .relay_model import NODE_BIT_WIDTH, RelayPopulationModel, RelayTensorBatch


@dataclass(frozen=True, slots=True)
class RelayScheduleTelemetry:
    schedule: str
    active_workers: int
    recurrent_rounds: int
    worker_updates_per_sample: int
    candidate_evaluations_per_sample: int
    inter_state_communicated_scalars_per_sample: int
    peak_active_neural_states_per_sample: int

    def validate(self) -> None:
        if self.schedule not in {"parallel_normalized", "serial_normalized"}:
            raise ValueError("unknown relay execution schedule")
        if self.active_workers <= 0:
            raise ValueError("active_workers must be positive")
        if self.recurrent_rounds <= 0:
            raise ValueError("recurrent_rounds must be positive")
        if self.worker_updates_per_sample != self.active_workers * self.recurrent_rounds:
            raise ValueError("worker-update accounting does not match relay schedule")
        if self.candidate_evaluations_per_sample != self.worker_updates_per_sample:
            raise ValueError("every relay worker update must produce one candidate evaluation")
        if self.inter_state_communicated_scalars_per_sample < 0:
            raise ValueError("communicated scalar count must be non-negative")
        if not 1 <= self.peak_active_neural_states_per_sample <= self.active_workers:
            raise ValueError("peak active state count is outside the nominal population")


@dataclass(frozen=True, slots=True)
class RelayScheduleOutput:
    logits: torch.Tensor
    final_shared: torch.Tensor
    telemetry: RelayScheduleTelemetry


def normalized_parallel_forward(
    model: RelayPopulationModel,
    batch: RelayTensorBatch,
    *,
    recurrent_rounds: int | None = None,
) -> RelayScheduleOutput:
    """Execute the repaired normalized relay with all active record states in parallel."""

    batch.validate()
    rounds = _resolved_rounds(batch, recurrent_rounds)
    active_workers = batch.active_workers
    local = batch.local_inputs[:, :active_workers, :]
    batch_size = int(local.shape[0])

    flat_local = local.reshape(batch_size * active_workers, -1)
    initial = torch.tanh(model.cell.input_projection(flat_local))
    value_content = model.query_projection(local[..., NODE_BIT_WIDTH:])
    shared = torch.tanh(model.query_projection(batch.start_bits))
    states = initial

    for _ in range(rounds):
        shared_flat = (
            shared.unsqueeze(1)
            .expand(batch_size, active_workers, model.config.message_width)
            .reshape(batch_size * active_workers, model.config.message_width)
        )
        states = model.cell.update(torch.cat((flat_local, shared_flat), dim=-1), initial)
        gate_logits = model.cell.message_gate(states).reshape(batch_size, active_workers)
        weights = torch.softmax(gate_logits, dim=1)
        shared = torch.tanh((value_content * weights.unsqueeze(-1)).sum(dim=1))

    pooled = states.reshape(batch_size, active_workers, model.config.state_width).mean(dim=1)
    logits = model.cell.output_head(model.cell.output_norm(torch.cat((pooled, shared), dim=-1)))
    telemetry = RelayScheduleTelemetry(
        schedule="parallel_normalized",
        active_workers=active_workers,
        recurrent_rounds=rounds,
        worker_updates_per_sample=active_workers * rounds,
        candidate_evaluations_per_sample=active_workers * rounds,
        # Match the existing population-cell accounting: one bounded shared-field read and
        # one candidate-value write per active state per hop.
        inter_state_communicated_scalars_per_sample=(
            2 * active_workers * rounds * model.config.message_width
        ),
        peak_active_neural_states_per_sample=active_workers,
    )
    telemetry.validate()
    return RelayScheduleOutput(logits=logits, final_shared=shared, telemetry=telemetry)


def normalized_serial_forward(
    model: RelayPopulationModel,
    batch: RelayTensorBatch,
    *,
    recurrent_rounds: int | None = None,
) -> RelayScheduleOutput:
    """Serialize the exact normalized relay computation through one neural state.

    The softmax reducer is accumulated online, so this path never needs to retain N gate
    logits, N candidate messages, or N hidden states simultaneously. Source records remain
    available as benchmark input, but only one learned record-state update is live per
    sample at any instant.
    """

    batch.validate()
    rounds = _resolved_rounds(batch, recurrent_rounds)
    active_workers = batch.active_workers
    local = batch.local_inputs[:, :active_workers, :]
    batch_size = int(local.shape[0])
    shared = torch.tanh(model.query_projection(batch.start_bits))
    final_state_sum = local.new_zeros(batch_size, model.config.state_width)

    for _ in range(rounds):
        running_max = local.new_full((batch_size, 1), -math.inf)
        running_weight_sum = local.new_zeros((batch_size, 1))
        running_weighted_content = local.new_zeros(
            batch_size,
            model.config.message_width,
        )
        state_sum = local.new_zeros(batch_size, model.config.state_width)

        for worker_index in range(active_workers):
            record = local[:, worker_index, :]
            initial = torch.tanh(model.cell.input_projection(record))
            state = model.cell.update(torch.cat((record, shared), dim=-1), initial)
            gate_logit = model.cell.message_gate(state)
            content = model.query_projection(record[:, NODE_BIT_WIDTH:])

            new_max = torch.maximum(running_max, gate_logit)
            previous_scale = torch.exp(running_max - new_max)
            current_scale = torch.exp(gate_logit - new_max)
            running_weighted_content = (
                running_weighted_content * previous_scale
                + content * current_scale
            )
            running_weight_sum = (
                running_weight_sum * previous_scale
                + current_scale
            )
            running_max = new_max
            state_sum = state_sum + state

        if not torch.all(torch.isfinite(running_weight_sum)):
            raise RuntimeError("serial relay softmax accumulator became non-finite")
        if torch.any(running_weight_sum <= 0):
            raise RuntimeError("serial relay softmax accumulator has zero mass")
        shared = torch.tanh(running_weighted_content / running_weight_sum)
        final_state_sum = state_sum

    pooled = final_state_sum / float(active_workers)
    logits = model.cell.output_head(model.cell.output_norm(torch.cat((pooled, shared), dim=-1)))
    telemetry = RelayScheduleTelemetry(
        schedule="serial_normalized",
        active_workers=active_workers,
        recurrent_rounds=rounds,
        worker_updates_per_sample=active_workers * rounds,
        candidate_evaluations_per_sample=active_workers * rounds,
        # There is only one live learned state. The online reducer is local deterministic
        # state, not transfer between simultaneously active neural workers.
        inter_state_communicated_scalars_per_sample=0,
        peak_active_neural_states_per_sample=1,
    )
    telemetry.validate()
    return RelayScheduleOutput(logits=logits, final_shared=shared, telemetry=telemetry)


def _resolved_rounds(batch: RelayTensorBatch, recurrent_rounds: int | None) -> int:
    rounds = batch.hop_count if recurrent_rounds is None else int(recurrent_rounds)
    if rounds <= 0:
        raise ValueError("recurrent_rounds must be positive")
    return rounds
