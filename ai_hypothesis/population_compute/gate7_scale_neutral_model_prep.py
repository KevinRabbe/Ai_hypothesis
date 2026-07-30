"""Preparation-only scale-neutral recurrent scorer substrate for Gate-7.

No training runner, scientific world namespace, high-scale evaluator, or Gate-7 outcome lives here.
The module only defines the frozen 19-input scale-neutral representation and the same recurrent
architecture shape used by Gate-3 v1.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass

import torch
from torch import nn

from .gate3_v1_sparse_active_reserve import GATE3_V1_RECURRENT_UPDATES_PER_CHILD

GATE7_SCALE_NEUTRAL_MODEL_PREPARATION_ONLY = True
GATE7_SCALE_NEUTRAL_POSITION_WIDTH = 13
GATE7_SCALE_NEUTRAL_HINT_WIDTH = 3
GATE7_SCALE_NEUTRAL_ACTION_WIDTH = 3
GATE7_SCALE_NEUTRAL_INPUT_WIDTH = (
    GATE7_SCALE_NEUTRAL_POSITION_WIDTH
    + GATE7_SCALE_NEUTRAL_HINT_WIDTH
    + GATE7_SCALE_NEUTRAL_ACTION_WIDTH
)
GATE7_SCALE_NEUTRAL_STATE_WIDTH = 64
GATE7_SCALE_NEUTRAL_PARAMETER_COUNT = 19_649


@dataclass(frozen=True, slots=True)
class Gate7ScaleNeutralModelConfig:
    input_projection_width: int = 32
    state_width: int = GATE7_SCALE_NEUTRAL_STATE_WIDTH

    def validate(self) -> None:
        if self.input_projection_width <= 0 or self.state_width <= 0:
            raise ValueError("Gate-7 scale-neutral model widths must be positive")


class Gate7ScaleNeutralScorer(nn.Module):
    """Same layer geometry as Gate-3 v1 with a scale-neutral 19-input encoding."""

    def __init__(self, config: Gate7ScaleNeutralModelConfig = Gate7ScaleNeutralModelConfig()) -> None:
        super().__init__()
        config.validate()
        self.config = config
        self.input_projection = nn.Linear(
            GATE7_SCALE_NEUTRAL_INPUT_WIDTH,
            config.input_projection_width,
        )
        self.update = nn.GRU(
            config.input_projection_width,
            config.state_width,
            batch_first=True,
        )
        self.output_norm = nn.LayerNorm(config.state_width)
        self.score_head = nn.Linear(config.state_width, 1)

    def initial_state(self, count: int, *, device: torch.device | str) -> torch.Tensor:
        if count <= 0:
            raise ValueError("initial-state count must be positive")
        return torch.zeros((count, self.config.state_width), dtype=torch.float32, device=device)

    def advance(
        self,
        state: torch.Tensor,
        phase_input: torch.Tensor,
        *,
        repeats: int = GATE3_V1_RECURRENT_UPDATES_PER_CHILD,
    ) -> torch.Tensor:
        if repeats <= 0:
            raise ValueError("repeats must be positive")
        if state.ndim != 2 or phase_input.ndim != 2 or state.shape[0] != phase_input.shape[0]:
            raise ValueError("state/input tensors must be matching rank-two batches")
        if phase_input.shape[1] != GATE7_SCALE_NEUTRAL_INPUT_WIDTH:
            raise ValueError("phase input width differs from frozen Gate-7 scale-neutral width")
        projected = torch.nn.functional.silu(self.input_projection(phase_input))
        sequence = projected.unsqueeze(1).expand(-1, repeats, -1).contiguous()
        _, final_state = self.update(sequence, state.unsqueeze(0))
        return final_state[0]

    def score(self, state: torch.Tensor) -> torch.Tensor:
        return self.score_head(self.output_norm(state)).squeeze(-1)

    def trainable_parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)

    def parameter_fingerprint(self) -> str:
        digest = hashlib.sha256()
        for name, tensor in sorted(self.state_dict().items()):
            detached = tensor.detach().cpu().contiguous().clone()
            digest.update(name.encode("utf-8"))
            digest.update(str(detached.dtype).encode("ascii"))
            digest.update(str(tuple(detached.shape)).encode("ascii"))
            digest.update(bytes(detached.untyped_storage()))
        return digest.hexdigest()


def gate7_scale_neutral_position_features(
    *,
    child_depth: int,
    world_depth: int,
    device: torch.device | str,
) -> torch.Tensor:
    """Return the frozen 13 bounded depth/progress features without a maximum-depth constant."""

    if world_depth <= 0:
        raise ValueError("world depth must be positive")
    if not 1 <= child_depth <= world_depth:
        raise ValueError("child depth must lie within the public world")

    d = float(child_depth)
    total = float(world_depth)
    remaining = total - d
    fraction = d / total
    remaining_fraction = remaining / total

    values = (
        fraction,
        remaining_fraction,
        1.0 / total,
        1.0 / d,
        1.0 / (remaining + 1.0),
        d / (d + 1.0),
        total / (total + 1.0),
        0.0 if remaining == 0.0 else remaining / (remaining + 1.0),
        math.sin(math.pi * fraction),
        math.cos(math.pi * fraction),
        math.sin(2.0 * math.pi * fraction),
        math.cos(2.0 * math.pi * fraction),
        2.0 * fraction - 1.0,
    )
    result = torch.tensor(values, dtype=torch.float32, device=device)
    if result.shape != (GATE7_SCALE_NEUTRAL_POSITION_WIDTH,) or not bool(torch.isfinite(result).all()):
        raise RuntimeError("scale-neutral positional encoding must remain finite and width 13")
    return result


def gate7_scale_neutral_position_features_batch(
    *,
    child_depths: torch.Tensor,
    world_depths: torch.Tensor,
) -> torch.Tensor:
    """Vectorized frozen positional encoding for an already-validated public depth batch.

    Value validation belongs at the public-world trust boundary before these tensors enter the CUDA
    execution path. This hot function performs only tensor metadata checks so it never synchronizes
    CUDA values back to Python.
    """

    if child_depths.ndim != 1 or world_depths.shape != child_depths.shape:
        raise ValueError("child/world depths must be matching rank-one tensors")
    if child_depths.dtype != torch.int64 or world_depths.dtype != torch.int64:
        raise ValueError("child/world depths must use int64")
    if child_depths.device != world_depths.device:
        raise ValueError("child/world depths must share one device")

    child = child_depths.to(dtype=torch.float32)
    total = world_depths.to(dtype=torch.float32)
    remaining = total - child
    fraction = child / total

    return torch.stack(
        (
            fraction,
            remaining / total,
            total.reciprocal(),
            child.reciprocal(),
            (remaining + 1.0).reciprocal(),
            child / (child + 1.0),
            total / (total + 1.0),
            remaining / (remaining + 1.0),
            torch.sin(math.pi * fraction),
            torch.cos(math.pi * fraction),
            torch.sin(2.0 * math.pi * fraction),
            torch.cos(2.0 * math.pi * fraction),
            2.0 * fraction - 1.0,
        ),
        dim=1,
    )


def encode_gate7_scale_neutral_child_input(
    *,
    world_depth: int,
    child_depth: int,
    observed_hint: int | None,
    branch_action: int | None,
    sink: bool,
    device: torch.device | str,
) -> torch.Tensor:
    """Encode one public child transition with no population/reserve/K/hidden-answer input."""

    if sink:
        if observed_hint is not None or branch_action is not None:
            raise ValueError("sink input cannot carry hint or branch action")
        hint_index = 2
        action_index = 2
    else:
        if observed_hint not in (0, 1) or branch_action not in (0, 1):
            raise ValueError("productive input requires binary hint/action")
        hint_index = int(observed_hint)
        action_index = int(branch_action)

    vector = torch.zeros(GATE7_SCALE_NEUTRAL_INPUT_WIDTH, dtype=torch.float32, device=device)
    vector[:GATE7_SCALE_NEUTRAL_POSITION_WIDTH] = gate7_scale_neutral_position_features(
        child_depth=child_depth,
        world_depth=world_depth,
        device=device,
    )
    hint_offset = GATE7_SCALE_NEUTRAL_POSITION_WIDTH
    action_offset = hint_offset + GATE7_SCALE_NEUTRAL_HINT_WIDTH
    vector[hint_offset + hint_index] = 1.0
    vector[action_offset + action_index] = 1.0
    return vector


def encode_gate7_scale_neutral_child_inputs_batch(
    *,
    world_depths: torch.Tensor,
    child_depths: torch.Tensor,
    observed_hints: torch.Tensor,
    branch_actions: torch.Tensor,
    sink: torch.Tensor,
) -> torch.Tensor:
    """Vectorized public child-input encoder with no CUDA-to-Python scalar extraction."""

    if world_depths.ndim != 1:
        raise ValueError("batched encoder inputs must be rank-one")
    expected_shape = world_depths.shape
    for tensor in (child_depths, observed_hints, branch_actions, sink):
        if tensor.shape != expected_shape:
            raise ValueError("batched encoder tensors must have matching shapes")
        if tensor.device != world_depths.device:
            raise ValueError("batched encoder tensors must share one device")
    if world_depths.dtype != torch.int64 or child_depths.dtype != torch.int64:
        raise ValueError("world/child depths must use int64")
    if observed_hints.dtype != torch.int64 or branch_actions.dtype != torch.int64:
        raise ValueError("hint/action tensors must use int64")
    if sink.dtype != torch.bool:
        raise ValueError("sink tensor must use bool")

    positions = gate7_scale_neutral_position_features_batch(
        child_depths=child_depths,
        world_depths=world_depths,
    )
    sink_index = torch.full_like(observed_hints, 2)
    hint_indices = torch.where(sink, sink_index, observed_hints)
    action_indices = torch.where(sink, sink_index, branch_actions)
    hints = torch.nn.functional.one_hot(
        hint_indices,
        num_classes=GATE7_SCALE_NEUTRAL_HINT_WIDTH,
    ).to(dtype=torch.float32)
    actions = torch.nn.functional.one_hot(
        action_indices,
        num_classes=GATE7_SCALE_NEUTRAL_ACTION_WIDTH,
    ).to(dtype=torch.float32)
    return torch.cat((positions, hints, actions), dim=1)
