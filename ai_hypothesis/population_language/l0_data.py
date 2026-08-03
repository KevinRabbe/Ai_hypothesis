"""Tensor materialization for the deterministic Population Language L0 world."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch
from torch import Tensor

from . import l0_protocol as protocol


@dataclass(frozen=True)
class LanguageBatch:
    input_ids: Tensor
    target_ids: Tensor
    loss_mask: Tensor
    answer_mask: Tensor
    ordinals: tuple[int, ...]

    def to(self, device: torch.device | str) -> "LanguageBatch":
        return LanguageBatch(
            input_ids=self.input_ids.to(device),
            target_ids=self.target_ids.to(device),
            loss_mask=self.loss_mask.to(device),
            answer_mask=self.answer_mask.to(device),
            ordinals=self.ordinals,
        )


def materialize_batch(
    split: protocol.Split,
    ordinals: Iterable[int],
    *,
    device: torch.device | str | None = None,
) -> LanguageBatch:
    locked_ordinals = tuple(ordinals)
    if not locked_ordinals:
        raise ValueError("Population Language L0 batch cannot be empty")
    if any(type(value) is not int or value < 0 for value in locked_ordinals):
        raise ValueError("Population Language L0 ordinals must be nonnegative integers")

    pad_id = protocol.TOKEN_TO_ID["<pad>"]
    full = torch.full(
        (len(locked_ordinals), protocol.MAX_SEQUENCE_LENGTH),
        pad_id,
        dtype=torch.long,
        device=device,
    )
    answer_positions = torch.zeros_like(full, dtype=torch.bool)

    for row, ordinal in enumerate(locked_ordinals):
        episode = protocol.make_episode(split, ordinal)
        token_ids = torch.tensor(episode.token_ids, dtype=torch.long, device=device)
        full[row, : token_ids.numel()] = token_ids
        answer_positions[row, episode.answer_start : len(episode.tokens) - 1] = True

    input_ids = full[:, :-1].contiguous()
    target_ids = full[:, 1:].contiguous()
    loss_mask = target_ids != pad_id
    answer_mask = answer_positions[:, 1:].contiguous()

    if bool(torch.any(answer_mask & ~loss_mask)):
        raise RuntimeError("Population Language L0 answer mask selected padding")
    if not bool(torch.all(answer_mask.sum(dim=1) == 5)):
        raise RuntimeError("Population Language L0 answer span must contain five tokens")

    return LanguageBatch(
        input_ids=input_ids,
        target_ids=target_ids,
        loss_mask=loss_mask,
        answer_mask=answer_mask,
        ordinals=locked_ordinals,
    )
