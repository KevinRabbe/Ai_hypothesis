"""Data-blind terminal Stage-B executor for Gate-7 high-scale preparation.

This executor operates on one immutable complete Stage-A frontier plus a condition-local live index bank.
Each activation selects one frontier candidate, computes both depth-(d+1) terminal children, records their
public path identities/scores, and swap-deletes only the selected index.  Terminal children are never
retained.  No hidden answer, scientific world generator, checkpoint loader, result classifier, or admitted
runner lives here.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .gate3_v1_sparse_active_reserve import GATE3_V1_RECURRENT_UPDATES_PER_CHILD
from .gate7_high_scale_index_bank_prep import (
    Gate7HighScaleImmutableFrontier,
    Gate7HighScaleLiveIndexBank,
    clone_gate7_high_scale_live_index_bank,
    delete_gate7_high_scale_selected,
    gather_gate7_high_scale_selected_states,
    initialize_gate7_high_scale_live_index_bank,
    select_gate7_high_scale_bounded_hash,
    select_gate7_high_scale_bounded_score,
    select_gate7_high_scale_global_hash,
    select_gate7_high_scale_global_score,
)
from .gate7_high_scale_routing_bandwidth_protocol import (
    GATE7_HIGH_SCALE_K_LADDER,
    GATE7_HIGH_SCALE_LEARNED_PARAMETER_COUNT,
    GATE7_HIGH_SCALE_STAGE_B_PARENT_SLOTS,
    build_gate7_high_scale_tier_plan,
)
from .gate7_scale_neutral_model_prep import (
    Gate7ScaleNeutralScorer,
    encode_gate7_scale_neutral_child_inputs_batch,
)

GATE7_HIGH_SCALE_TERMINAL_STAGE_B_PREPARATION_ONLY = True
GATE7_HIGH_SCALE_STAGE_B_GLOBAL_SCORE = "global_score"
GATE7_HIGH_SCALE_STAGE_B_GLOBAL_HASH = "global_hash"
GATE7_HIGH_SCALE_STAGE_B_BOUNDED_SCORE = "bounded_score"
GATE7_HIGH_SCALE_STAGE_B_BOUNDED_HASH = "bounded_hash"
GATE7_HIGH_SCALE_STAGE_B_MODES = (
    GATE7_HIGH_SCALE_STAGE_B_GLOBAL_SCORE,
    GATE7_HIGH_SCALE_STAGE_B_GLOBAL_HASH,
    GATE7_HIGH_SCALE_STAGE_B_BOUNDED_SCORE,
    GATE7_HIGH_SCALE_STAGE_B_BOUNDED_HASH,
)


@dataclass(frozen=True, slots=True)
class Gate7HighScaleTerminalStageBTranscript:
    mode: str
    k: int | None
    stage_b_slots: int
    selected_frontier_indices: torch.Tensor
    terminal_path_ids: torch.Tensor
    terminal_child_scores: torch.Tensor
    neural_score_observations_by_slot: torch.Tensor
    final_bank: Gate7HighScaleLiveIndexBank

    def validate(self, *, batch_size: int, population: int) -> None:
        if self.mode not in GATE7_HIGH_SCALE_STAGE_B_MODES:
            raise ValueError("terminal Stage-B transcript mode is invalid")
        if not 1 <= self.stage_b_slots <= GATE7_HIGH_SCALE_STAGE_B_PARENT_SLOTS:
            raise ValueError("terminal Stage-B slot count is invalid")
        if self.selected_frontier_indices.shape != (batch_size, self.stage_b_slots):
            raise ValueError("selected frontier transcript shape changed")
        if self.terminal_path_ids.shape != (batch_size, self.stage_b_slots, 2):
            raise ValueError("terminal path transcript shape changed")
        if self.terminal_child_scores.shape != (batch_size, self.stage_b_slots, 2):
            raise ValueError("terminal child-score transcript shape changed")
        if self.neural_score_observations_by_slot.shape != (batch_size, self.stage_b_slots):
            raise ValueError("score-observation transcript shape changed")
        if self.selected_frontier_indices.dtype != torch.int64:
            raise ValueError("selected frontier indices must use int64")
        if self.terminal_path_ids.dtype != torch.int64:
            raise ValueError("terminal path IDs must use int64")
        if self.terminal_child_scores.dtype != torch.float32:
            raise ValueError("terminal child scores must remain FP32")
        if self.neural_score_observations_by_slot.dtype != torch.int64:
            raise ValueError("score-observation transcript must use int64")
        if self.final_bank.population != population:
            raise ValueError("final index bank population changed")
        self.final_bank.validate()

    def total_neural_score_observations_per_world(self) -> torch.Tensor:
        return self.neural_score_observations_by_slot.sum(dim=1)


def validate_gate7_high_scale_terminal_hints(
    *, terminal_hints_by_world: tuple[int, ...], batch_size: int
) -> None:
    if len(terminal_hints_by_world) != batch_size:
        raise ValueError("terminal hint count differs from physical world batch")
    if any(value not in (0, 1) for value in terminal_hints_by_world):
        raise ValueError("terminal hints must remain binary public values")


def _terminal_child_inputs(
    *,
    world_depth: int,
    terminal_hints_by_world: tuple[int, ...],
    device: torch.device,
) -> torch.Tensor:
    batch = len(terminal_hints_by_world)
    count = batch * 2
    hints = torch.tensor(terminal_hints_by_world, dtype=torch.int64, device=device)
    actions = torch.tensor((0, 1), dtype=torch.int64, device=device)
    return encode_gate7_scale_neutral_child_inputs_batch(
        world_depths=torch.full((count,), world_depth, dtype=torch.int64, device=device),
        child_depths=torch.full((count,), world_depth, dtype=torch.int64, device=device),
        observed_hints=hints[:, None].expand(batch, 2).reshape(count),
        branch_actions=actions[None, :].expand(batch, 2).reshape(count),
        sink=torch.zeros(count, dtype=torch.bool, device=device),
    )


def _select(
    *,
    mode: str,
    k: int | None,
    frontier: Gate7HighScaleImmutableFrontier,
    bank: Gate7HighScaleLiveIndexBank,
    public_seeds: torch.Tensor,
    slot_index: int,
):
    if mode == GATE7_HIGH_SCALE_STAGE_B_GLOBAL_SCORE:
        return select_gate7_high_scale_global_score(
            frontier,
            bank,
            public_seeds=public_seeds,
            slot_index=slot_index,
        )
    if mode == GATE7_HIGH_SCALE_STAGE_B_GLOBAL_HASH:
        return select_gate7_high_scale_global_hash(
            frontier,
            bank,
            public_seeds=public_seeds,
            slot_index=slot_index,
        )
    if mode == GATE7_HIGH_SCALE_STAGE_B_BOUNDED_SCORE:
        assert k is not None
        return select_gate7_high_scale_bounded_score(
            frontier,
            bank,
            k=k,
            public_seeds=public_seeds,
            slot_index=slot_index,
        )
    if mode == GATE7_HIGH_SCALE_STAGE_B_BOUNDED_HASH:
        assert k is not None
        return select_gate7_high_scale_bounded_hash(
            frontier,
            bank,
            k=k,
            public_seeds=public_seeds,
            slot_index=slot_index,
        )
    raise ValueError("unsupported terminal Stage-B mode")


def run_gate7_high_scale_terminal_stage_b_preparation(
    model: Gate7ScaleNeutralScorer,
    frontier: Gate7HighScaleImmutableFrontier,
    *,
    terminal_hints_by_world: tuple[int, ...],
    public_seeds: torch.Tensor,
    mode: str,
    k: int | None,
    stage_b_slots: int = GATE7_HIGH_SCALE_STAGE_B_PARENT_SLOTS,
    initial_bank: Gate7HighScaleLiveIndexBank | None = None,
) -> Gate7HighScaleTerminalStageBTranscript:
    """Run generic terminal Stage B without any scientific-world or hidden-answer dependency."""

    frontier.validate()
    plan = build_gate7_high_scale_tier_plan(frontier.population)
    if model.trainable_parameter_count() != GATE7_HIGH_SCALE_LEARNED_PARAMETER_COUNT:
        raise ValueError("terminal Stage-B model must contain exactly 19,649 learned parameters")
    if mode not in GATE7_HIGH_SCALE_STAGE_B_MODES:
        raise ValueError("unsupported terminal Stage-B mode")
    bounded = mode in {
        GATE7_HIGH_SCALE_STAGE_B_BOUNDED_SCORE,
        GATE7_HIGH_SCALE_STAGE_B_BOUNDED_HASH,
    }
    if bounded:
        if k not in GATE7_HIGH_SCALE_K_LADDER or k >= frontier.population:
            raise ValueError("bounded terminal Stage B requires one frozen K below N")
    elif k is not None:
        raise ValueError("global terminal Stage B cannot carry K")
    if not 1 <= stage_b_slots <= GATE7_HIGH_SCALE_STAGE_B_PARENT_SLOTS:
        raise ValueError("stage_b_slots is outside the frozen 1..128 preparation horizon")
    if stage_b_slots >= frontier.population:
        raise ValueError("terminal Stage B must leave at least one frontier candidate live")
    validate_gate7_high_scale_terminal_hints(
        terminal_hints_by_world=terminal_hints_by_world,
        batch_size=frontier.batch_size,
    )
    if public_seeds.shape != (frontier.batch_size,) or public_seeds.dtype != torch.int64:
        raise ValueError("public seeds must use int64 [batch]")
    if public_seeds.device != frontier.states.device:
        raise ValueError("public seeds must share the immutable frontier device")

    if initial_bank is None:
        bank = initialize_gate7_high_scale_live_index_bank(
            batch_size=frontier.batch_size,
            population=frontier.population,
            device=frontier.states.device,
        )
    else:
        if initial_bank.batch_size != frontier.batch_size or initial_bank.population != frontier.population:
            raise ValueError("initial index bank geometry differs from the immutable frontier")
        bank = clone_gate7_high_scale_live_index_bank(initial_bank)

    target = frontier.states.device
    model = model.to(target)
    model.eval()
    child_inputs = _terminal_child_inputs(
        world_depth=plan.world_depth,
        terminal_hints_by_world=terminal_hints_by_world,
        device=target,
    )
    selected_rows: list[torch.Tensor] = []
    terminal_rows: list[torch.Tensor] = []
    score_rows: list[torch.Tensor] = []
    observation_rows: list[torch.Tensor] = []
    actions = torch.tensor((0, 1), dtype=torch.int64, device=target)

    with torch.inference_mode():
        for slot_index in range(stage_b_slots):
            selection = _select(
                mode=mode,
                k=k,
                frontier=frontier,
                bank=bank,
                public_seeds=public_seeds,
                slot_index=slot_index,
            )
            selected_state = gather_gate7_high_scale_selected_states(frontier, selection)
            child_states = model.advance(
                selected_state[:, None, :].expand(frontier.batch_size, 2, 64).reshape(
                    frontier.batch_size * 2, 64
                ),
                child_inputs,
                repeats=GATE3_V1_RECURRENT_UPDATES_PER_CHILD,
            )
            child_scores = model.score(child_states).reshape(frontier.batch_size, 2)
            terminal_paths = selection.selected_original_indices[:, None] * 2 + actions[None, :]

            selected_rows.append(selection.selected_original_indices)
            terminal_rows.append(terminal_paths)
            score_rows.append(child_scores)
            observation_rows.append(selection.neural_scores_observed_per_world)
            bank = delete_gate7_high_scale_selected(bank, selection)

    transcript = Gate7HighScaleTerminalStageBTranscript(
        mode=mode,
        k=k,
        stage_b_slots=stage_b_slots,
        selected_frontier_indices=torch.stack(selected_rows, dim=1),
        terminal_path_ids=torch.stack(terminal_rows, dim=1),
        terminal_child_scores=torch.stack(score_rows, dim=1),
        neural_score_observations_by_slot=torch.stack(observation_rows, dim=1),
        final_bank=bank,
    )
    transcript.validate(batch_size=frontier.batch_size, population=frontier.population)
    return transcript
