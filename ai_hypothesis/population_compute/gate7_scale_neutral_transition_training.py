"""Frozen checkpoint-training substrate for the Gate-7 scale-neutral scorer transition.

This module trains only the three preregistered transition checkpoints. It contains no bridge evaluator,
Gate-7 high-scale scientific world generator, population treatment, routing K, or scientific outcome.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import asdict, dataclass
from typing import Callable

import torch
from torch import nn

from .gate3_v1_sparse_active_reserve import (
    GATE3_V1_HINT_RELIABILITY,
    GATE3_V1_RECURRENT_UPDATES_PER_CHILD,
)
from .gate7_scale_neutral_model_prep import (
    GATE7_SCALE_NEUTRAL_PARAMETER_COUNT,
    Gate7ScaleNeutralModelConfig,
    Gate7ScaleNeutralScorer,
    encode_gate7_scale_neutral_child_inputs_batch,
)

GATE7_SCALE_NEUTRAL_TRANSITION_VERSION = "gate7-scale-neutral-scorer-transition-v0"
GATE7_SCALE_NEUTRAL_TRAINING_SEEDS = (0, 1, 2)
GATE7_SCALE_NEUTRAL_TRAINING_DEPTHS = tuple(range(6, 19))
GATE7_SCALE_NEUTRAL_TRAINING_STEPS = 1_200
GATE7_SCALE_NEUTRAL_TRAINING_BATCH_SIZE = 256
GATE7_SCALE_NEUTRAL_LEARNING_RATE = 3e-4
GATE7_SCALE_NEUTRAL_WEIGHT_DECAY = 1e-4
GATE7_SCALE_NEUTRAL_GRADIENT_CLIP_NORM = 1.0
GATE7_SCALE_NEUTRAL_SIGNED_HINT_EVIDENCE = math.log(
    GATE3_V1_HINT_RELIABILITY / (1.0 - GATE3_V1_HINT_RELIABILITY)
)

Progress = Callable[[int, int, int, float], None]


@dataclass(frozen=True, slots=True)
class Gate7ScaleNeutralTrainingSummary:
    training_seed: int
    steps: int
    examples_seen: int
    depth_schedule: tuple[int, ...]
    initial_loss: float
    final_loss: float
    mean_last_50_loss: float
    learned_parameter_count: int
    parameter_fingerprint: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _seed_from_parts(*parts: object) -> int:
    digest = hashlib.sha256(":".join(str(part) for part in parts).encode("ascii")).digest()
    return int.from_bytes(digest[:8], "big") & ((1 << 63) - 1)


def gate7_scale_neutral_training_batch(
    *,
    training_seed: int,
    step: int,
    depth: int,
    batch_size: int = GATE7_SCALE_NEUTRAL_TRAINING_BATCH_SIZE,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate deterministic public hints and independent candidate actions on CPU.

    A latent binary path is used only to generate noisy public hints. Candidate actions use a disjoint
    deterministic stream and are therefore independent of the latent path and the observed hints.
    """

    if training_seed not in GATE7_SCALE_NEUTRAL_TRAINING_SEEDS:
        raise ValueError("training seed is outside the frozen transition seed set")
    if step < 0:
        raise ValueError("training step must be non-negative")
    if depth not in GATE7_SCALE_NEUTRAL_TRAINING_DEPTHS:
        raise ValueError("training depth is outside the frozen transition schedule")
    if batch_size <= 0:
        raise ValueError("training batch size must be positive")

    latent_generator = torch.Generator(device="cpu")
    latent_generator.manual_seed(
        _seed_from_parts("gate7-scale-neutral-transition-train-latent", training_seed, step, depth)
    )
    reliability_generator = torch.Generator(device="cpu")
    reliability_generator.manual_seed(
        _seed_from_parts("gate7-scale-neutral-transition-train-hints", training_seed, step, depth)
    )
    candidate_generator = torch.Generator(device="cpu")
    candidate_generator.manual_seed(
        _seed_from_parts("gate7-scale-neutral-transition-train-candidate", training_seed, step, depth)
    )

    latent = torch.randint(0, 2, (batch_size, depth), dtype=torch.int64, generator=latent_generator)
    reliable = torch.rand((batch_size, depth), generator=reliability_generator) < GATE3_V1_HINT_RELIABILITY
    hints = torch.where(reliable, latent, 1 - latent)
    actions = torch.randint(0, 2, (batch_size, depth), dtype=torch.int64, generator=candidate_generator)
    return hints, actions


def build_gate7_scale_neutral_training_inputs(
    *,
    world_depth: int,
    child_depth: int,
    hints: torch.Tensor,
    actions: torch.Tensor,
    device: torch.device | str,
) -> torch.Tensor:
    """Build one productive training phase without synchronizing CUDA token values to Python.

    The only caller feeds tokens produced by `gate7_scale_neutral_training_batch`, which is the
    CPU trust boundary that guarantees binary hints/actions. This function validates tensor metadata
    and public depth integers, then keeps the complete encoding path on the target device.
    """

    if world_depth not in GATE7_SCALE_NEUTRAL_TRAINING_DEPTHS:
        raise ValueError("world depth is outside the frozen transition training schedule")
    if not 1 <= child_depth <= world_depth:
        raise ValueError("child depth must lie within the training world")
    if hints.ndim != 1 or actions.shape != hints.shape:
        raise ValueError("hints/actions must be matching rank-one tensors")
    if hints.dtype != torch.int64 or actions.dtype != torch.int64:
        raise ValueError("hints/actions must use int64 tokens")

    target = torch.device(device)
    count = hints.shape[0]
    hint_tokens = hints.to(target)
    action_tokens = actions.to(target)
    return encode_gate7_scale_neutral_child_inputs_batch(
        world_depths=torch.full((count,), world_depth, dtype=torch.int64, device=target),
        child_depths=torch.full((count,), child_depth, dtype=torch.int64, device=target),
        observed_hints=hint_tokens,
        branch_actions=action_tokens,
        sink=torch.zeros(count, dtype=torch.bool, device=target),
    )


def train_gate7_scale_neutral_checkpoint(
    *,
    training_seed: int,
    device: torch.device | str,
    progress: Progress | None = None,
) -> tuple[Gate7ScaleNeutralScorer, Gate7ScaleNeutralTrainingSummary]:
    """Train one exact frozen transition checkpoint."""

    if training_seed not in GATE7_SCALE_NEUTRAL_TRAINING_SEEDS:
        raise ValueError("training seed is outside the frozen transition seed set")
    target_device = torch.device(device)
    torch.manual_seed(training_seed)
    if target_device.type == "cuda":
        torch.cuda.manual_seed_all(training_seed)

    model = Gate7ScaleNeutralScorer(Gate7ScaleNeutralModelConfig()).to(target_device)
    if model.trainable_parameter_count() != GATE7_SCALE_NEUTRAL_PARAMETER_COUNT:
        raise RuntimeError("scale-neutral model parameter count differs from frozen protocol")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=GATE7_SCALE_NEUTRAL_LEARNING_RATE,
        weight_decay=GATE7_SCALE_NEUTRAL_WEIGHT_DECAY,
    )
    loss_fn = nn.SmoothL1Loss()
    losses: list[float] = []
    model.train()

    for step in range(GATE7_SCALE_NEUTRAL_TRAINING_STEPS):
        depth = GATE7_SCALE_NEUTRAL_TRAINING_DEPTHS[
            step % len(GATE7_SCALE_NEUTRAL_TRAINING_DEPTHS)
        ]
        hints_cpu, actions_cpu = gate7_scale_neutral_training_batch(
            training_seed=training_seed,
            step=step,
            depth=depth,
        )
        hints = hints_cpu.to(target_device)
        actions = actions_cpu.to(target_device)
        states = model.initial_state(GATE7_SCALE_NEUTRAL_TRAINING_BATCH_SIZE, device=target_device)
        cumulative = torch.zeros(
            GATE7_SCALE_NEUTRAL_TRAINING_BATCH_SIZE,
            dtype=torch.float32,
            device=target_device,
        )
        phase_losses: list[torch.Tensor] = []

        for prefix_index in range(depth):
            hint = hints[:, prefix_index]
            action = actions[:, prefix_index]
            inputs = build_gate7_scale_neutral_training_inputs(
                world_depth=depth,
                child_depth=prefix_index + 1,
                hints=hint,
                actions=action,
                device=target_device,
            )
            states = model.advance(
                states,
                inputs,
                repeats=GATE3_V1_RECURRENT_UPDATES_PER_CHILD,
            )
            predictions = model.score(states)
            signed = torch.where(
                action == hint,
                torch.full_like(cumulative, GATE7_SCALE_NEUTRAL_SIGNED_HINT_EVIDENCE),
                torch.full_like(cumulative, -GATE7_SCALE_NEUTRAL_SIGNED_HINT_EVIDENCE),
            )
            cumulative = cumulative + signed
            target_values = cumulative / float(depth)
            phase_losses.append(loss_fn(predictions, target_values))

        loss = torch.stack(phase_losses).mean()
        scalar_loss = float(loss.detach().item())
        if not math.isfinite(scalar_loss):
            raise RuntimeError("scale-neutral transition training produced non-finite loss")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GATE7_SCALE_NEUTRAL_GRADIENT_CLIP_NORM)
        optimizer.step()

        losses.append(scalar_loss)
        if progress is not None:
            progress(step + 1, GATE7_SCALE_NEUTRAL_TRAINING_STEPS, depth, scalar_loss)

    summary = Gate7ScaleNeutralTrainingSummary(
        training_seed=training_seed,
        steps=GATE7_SCALE_NEUTRAL_TRAINING_STEPS,
        examples_seen=GATE7_SCALE_NEUTRAL_TRAINING_STEPS * GATE7_SCALE_NEUTRAL_TRAINING_BATCH_SIZE,
        depth_schedule=GATE7_SCALE_NEUTRAL_TRAINING_DEPTHS,
        initial_loss=losses[0],
        final_loss=losses[-1],
        mean_last_50_loss=sum(losses[-50:]) / len(losses[-50:]),
        learned_parameter_count=model.trainable_parameter_count(),
        parameter_fingerprint=model.parameter_fingerprint(),
    )
    return model, summary
