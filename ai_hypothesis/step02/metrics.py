"""Metric helpers for Step 2 population diagnostic reporting."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ConditionalRate:
    numerator: int
    denominator: int
    rate: float | None


def conditional_rate(numerator: int, denominator: int) -> ConditionalRate:
    """Return a conditional rate, using None when the denominator is absent."""

    if numerator < 0:
        raise ValueError("numerator must be non-negative")
    if denominator < 0:
        raise ValueError("denominator must be non-negative")
    if numerator > denominator:
        raise ValueError("numerator cannot exceed denominator")
    return ConditionalRate(
        numerator=numerator,
        denominator=denominator,
        rate=None if denominator == 0 else numerator / denominator,
    )


def is_true_minority_opportunity(
    *,
    correct_worker_count: int,
    population_width: int,
    majority_is_correct: bool,
) -> bool:
    """Whether truth is present only in a numerical minority of workers."""

    if population_width <= 0:
        raise ValueError("population_width must be positive")
    if correct_worker_count < 0 or correct_worker_count > population_width:
        raise ValueError("correct_worker_count must be within population width")
    return (
        correct_worker_count > 0
        and correct_worker_count < (population_width / 2)
        and not majority_is_correct
    )
