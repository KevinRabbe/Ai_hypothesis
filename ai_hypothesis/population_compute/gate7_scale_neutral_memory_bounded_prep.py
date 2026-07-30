"""Memory-bounded exact recurrent execution for the frozen Gate-7 scale-neutral scorer.

The scorer's reference `advance` method expands one projected input into a contiguous [batch,repeats,32]
sequence before calling the GRU.  That is convenient at low scale but creates an avoidable 8x activation
allocation at the high-scale frontier.  This preparation helper applies the same frozen GRU one step at a
time while retaining one projected input and one recurrent state batch.

No training, checkpoint loading, scientific worlds, outcome assignment, compiler, CUDA graphs, or mixed
precision live here.
"""

from __future__ import annotations

import torch

from .gate7_scale_neutral_model_prep import (
    GATE7_SCALE_NEUTRAL_INPUT_WIDTH,
    GATE7_SCALE_NEUTRAL_STATE_WIDTH,
    Gate7ScaleNeutralScorer,
)

GATE7_SCALE_NEUTRAL_MEMORY_BOUNDED_PREPARATION_ONLY = True


def advance_gate7_scale_neutral_memory_bounded(
    model: Gate7ScaleNeutralScorer,
    state: torch.Tensor,
    phase_input: torch.Tensor,
    *,
    repeats: int,
) -> torch.Tensor:
    """Apply the exact frozen GRU update repeatedly without materializing a repeated sequence."""

    if repeats <= 0:
        raise ValueError("repeats must be positive")
    if state.ndim != 2 or phase_input.ndim != 2 or state.shape[0] != phase_input.shape[0]:
        raise ValueError("state/input tensors must be matching rank-two batches")
    if state.shape[1] != GATE7_SCALE_NEUTRAL_STATE_WIDTH:
        raise ValueError("state width differs from the frozen scale-neutral scorer")
    if phase_input.shape[1] != GATE7_SCALE_NEUTRAL_INPUT_WIDTH:
        raise ValueError("input width differs from the frozen scale-neutral scorer")
    if state.device != phase_input.device:
        raise ValueError("state and phase input must share one device")
    if state.dtype != torch.float32 or phase_input.dtype != torch.float32:
        raise ValueError("memory-bounded scale-neutral execution remains FP32")

    projected = torch.nn.functional.silu(model.input_projection(phase_input))
    recurrent_input = projected.unsqueeze(1)
    current = state
    for _ in range(repeats):
        _, final_state = model.update(recurrent_input, current.unsqueeze(0))
        current = final_state[0]
    return current
