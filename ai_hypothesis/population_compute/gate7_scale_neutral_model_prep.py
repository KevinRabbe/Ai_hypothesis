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
