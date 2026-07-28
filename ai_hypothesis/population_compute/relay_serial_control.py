"""Matched execution controls for repaired relay transport.

The repaired relay computation resets record-local neural state on every relay hop. One
normalized population hop is separable across local records: every record applies the same
learned recurrent update to the same shared query, then a parameter-free softmax reducer
combines candidate values.

Three schedules are exposed:

* ``normalized_parallel_forward`` keeps all active record states resident together and
  computes immutable record projections once.
* ``normalized_serial_forward`` is the minimum-live-state control. It time-multiplexes one
  record state and recomputes immutable record projections each hop to avoid an O(N) cache.
* ``normalized_serial_cached_forward`` computes the same immutable record projections once
  as the parallel path, then serializes only the recurrent record updates. It trades O(N)
  cached projection state for matched static learned-projection work.

All schedules preserve source scope and perform exactly ``N * relay_hops`` recurrent worker
updates. The two serial schedules make the time/memory trade-off explicit instead of hiding
projection recomputation inside the equal-worker-update claim.
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
    input_projection_evaluations_per_sample: int
    value_projection_evaluations_per_sample: int
    inter_state_communicated_scalars_per_sample: int
    peak_active_neural_states_per_sample: int
    cached_state_vectors_per_sample: int = 0
    cached_message_vectors_per_sample: int = 0

    @property
    def static_projection_evaluations_per_sample(self) -> int:
        return (
            self.input_projection_evaluations_per_sample
            + self.value_projection_evaluations_per_sample
        )

    def validate(self) -> None:
        if self.schedule not in {
            "parallel_normalized",
            "serial_normalized",
            "serial_cached_normalized",
        }:
            raise ValueError("unknown relay execution schedule")
        if self.active_workers <= 0:
            raise ValueError("active_workers must be positive")
        if self.recurrent_rounds <= 0:
            raise ValueError("recurrent_rounds must be positive")
        if self.worker_updates_per_sample != self.active_workers * self.recurrent_rounds:
            raise ValueError("worker-update accounting does not match relay schedule")
        if self.candidate_evaluations_per_sample != self.worker_updates_per_sample:
            raise ValueError("every relay worker update must produce one candidate evaluation")
        if self.input_projection_evaluations_per_sample <= 0:
            raise ValueError("input projection count must be positive")
        if self.value_projection_evaluations_per_sample <= 0:
            raise ValueError("value projection count must be positive")
        if self.inter_state_communicated_scalars_per_sample < 0:
            raise ValueError("communicated scalar count must be non-negative")
        if not 1 <= self.peak_active_neural_states_per_sample <= self.active_workers:
            raise ValueError("peak active state count is outside the nominal population")
        if self.cached_state_vectors_per_sample < 0 or self.cached_message_vectors_per_sample < 0:
            raise ValueError("cached vector counts must be non-negative")


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
        input_projection_evaluations_per_sample=active_workers,
        value_projection_evaluations_per_sample=active_workers,
        inter_state_communicated_scalars_per_sample=(
            2 * active_workers * rounds * model.config.message_width
        ),
        peak_active_neural_states_per_sample=active_workers,
        cached_state_vectors_per_sample=active_workers,
        cached_message_vectors_per_sample=active_workers,
    )
    telemetry.validate()
    return RelayScheduleOutput(logits=logits, final_shared=shared, telemetry=telemetry)


def normalized_serial_forward(
    model: RelayPopulationModel,
    batch: RelayTensorBatch,
    *,
    recurrent_rounds: int | None = None,
) -> RelayScheduleOutput:
    """Minimum-live-state serial control.

    Immutable input/value projections are recomputed per hop so only one learned record state
    is live and no O(N) projection cache is retained. This preserves recurrent worker-update
    count but performs more static learned projection work than the parallel schedule.
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
            running_max, running_weight_sum, running_weighted_content = _accumulate_candidate(
                running_max,
                running_weight_sum,
                running_weighted_content,
                gate_logit,
                content,
            )
            state_sum = state_sum + state

        _validate_reducer_mass(running_weight_sum)
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
        input_projection_evaluations_per_sample=active_workers * rounds,
        value_projection_evaluations_per_sample=active_workers * rounds,
        inter_state_communicated_scalars_per_sample=0,
        peak_active_neural_states_per_sample=1,
    )
    telemetry.validate()
    return RelayScheduleOutput(logits=logits, final_shared=shared, telemetry=telemetry)


def normalized_serial_cached_forward(
    model: RelayPopulationModel,
    batch: RelayTensorBatch,
    *,
    recurrent_rounds: int | None = None,
) -> RelayScheduleOutput:
    """Serial recurrent schedule with parallel-equivalent static projection work.

    Immutable per-record initial-state and candidate-value projections are computed once and
    cached. Recurrent record updates and normalized reduction remain serial. This is a stronger
    compute-matched serial baseline than ``normalized_serial_forward`` at the cost of O(N)
    cached state/message vectors.
    """

    batch.validate()
    rounds = _resolved_rounds(batch, recurrent_rounds)
    active_workers = batch.active_workers
    local = batch.local_inputs[:, :active_workers, :]
    batch_size = int(local.shape[0])
    flat_local = local.reshape(batch_size * active_workers, -1)
    initial_cache = torch.tanh(model.cell.input_projection(flat_local)).reshape(
        batch_size,
        active_workers,
        model.config.state_width,
    )
    value_cache = model.query_projection(local[..., NODE_BIT_WIDTH:])
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
            initial = initial_cache[:, worker_index, :]
            state = model.cell.update(torch.cat((record, shared), dim=-1), initial)
            gate_logit = model.cell.message_gate(state)
            content = value_cache[:, worker_index, :]
            running_max, running_weight_sum, running_weighted_content = _accumulate_candidate(
                running_max,
                running_weight_sum,
                running_weighted_content,
                gate_logit,
                content,
            )
            state_sum = state_sum + state

        _validate_reducer_mass(running_weight_sum)
        shared = torch.tanh(running_weighted_content / running_weight_sum)
        final_state_sum = state_sum

    pooled = final_state_sum / float(active_workers)
    logits = model.cell.output_head(model.cell.output_norm(torch.cat((pooled, shared), dim=-1)))
    telemetry = RelayScheduleTelemetry(
        schedule="serial_cached_normalized",
        active_workers=active_workers,
        recurrent_rounds=rounds,
        worker_updates_per_sample=active_workers * rounds,
        candidate_evaluations_per_sample=active_workers * rounds,
        input_projection_evaluations_per_sample=active_workers,
        value_projection_evaluations_per_sample=active_workers,
        inter_state_communicated_scalars_per_sample=0,
        peak_active_neural_states_per_sample=1,
        cached_state_vectors_per_sample=active_workers,
        cached_message_vectors_per_sample=active_workers,
    )
    telemetry.validate()
    return RelayScheduleOutput(logits=logits, final_shared=shared, telemetry=telemetry)


def _accumulate_candidate(
    running_max: torch.Tensor,
    running_weight_sum: torch.Tensor,
    running_weighted_content: torch.Tensor,
    gate_logit: torch.Tensor,
    content: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    new_max = torch.maximum(running_max, gate_logit)
    previous_scale = torch.exp(running_max - new_max)
    current_scale = torch.exp(gate_logit - new_max)
    return (
        new_max,
        running_weight_sum * previous_scale + current_scale,
        running_weighted_content * previous_scale + content * current_scale,
    )


def _validate_reducer_mass(running_weight_sum: torch.Tensor) -> None:
    if not torch.all(torch.isfinite(running_weight_sum)):
        raise RuntimeError("serial relay softmax accumulator became non-finite")
    if torch.any(running_weight_sum <= 0):
        raise RuntimeError("serial relay softmax accumulator has zero mass")


def _resolved_rounds(batch: RelayTensorBatch, recurrent_rounds: int | None) -> int:
    rounds = batch.hop_count if recurrent_rounds is None else int(recurrent_rounds)
    if rounds <= 0:
        raise ValueError("recurrent_rounds must be positive")
    return rounds
