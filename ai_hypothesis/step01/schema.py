"""Shared Step 1 benchmark schema.

The Step 1 benchmark intentionally uses only Python's standard library so the
generator and deterministic baselines remain easy to inspect and reproduce.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

BENCHMARK_VERSION = "step01-v0"
SEQUENCE_LENGTH = 32
FEATURE_WIDTH = 16
CONTROL_ROW = 0
DATA_START = 1


class TaskFamily(str, Enum):
    PATTERN = "A_PATTERN"
    CHANGE = "B_CHANGE"
    CONFLICT = "C_CONFLICT"
    RELATION = "D_RELATION"
    RELEVANCE = "E_RELEVANCE"


class Difficulty(str, Enum):
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"
    AMBIGUOUS = "ambiguous"


TASKS: tuple[TaskFamily, ...] = tuple(TaskFamily)
DIFFICULTIES: tuple[Difficulty, ...] = tuple(Difficulty)

TASK_INDEX = {task: index for index, task in enumerate(TASKS)}

VALID_LABELS: dict[TaskFamily, frozenset[str]] = {
    TaskFamily.PATTERN: frozenset({"SIGNAL", "NO_SIGNAL", "UNCERTAIN"}),
    TaskFamily.CHANGE: frozenset({"CHANGE", "NO_CHANGE", "UNCERTAIN"}),
    TaskFamily.CONFLICT: frozenset({"CONFLICT", "COMPATIBLE", "UNCERTAIN"}),
    TaskFamily.RELATION: frozenset(
        {"A_BEFORE_B", "B_BEFORE_A", "SAME_TIME", "UNCERTAIN"}
    ),
    TaskFamily.RELEVANCE: frozenset({"RELEVANT", "NOT_RELEVANT", "UNCERTAIN"}),
}


@dataclass(frozen=True, slots=True)
class BenchmarkSample:
    """One reproducible benchmark example."""

    task: TaskFamily
    difficulty: Difficulty
    seed: int
    features: tuple[tuple[float, ...], ...]
    mask: tuple[bool, ...]
    label: str
    metadata: dict[str, Any]

    def validate(self) -> None:
        if len(self.features) != SEQUENCE_LENGTH:
            raise ValueError(
                f"expected {SEQUENCE_LENGTH} rows, got {len(self.features)}"
            )
        for row in self.features:
            if len(row) != FEATURE_WIDTH:
                raise ValueError(
                    f"expected feature width {FEATURE_WIDTH}, got {len(row)}"
                )
        if len(self.mask) != SEQUENCE_LENGTH:
            raise ValueError(
                f"expected mask length {SEQUENCE_LENGTH}, got {len(self.mask)}"
            )
        if not self.mask[CONTROL_ROW]:
            raise ValueError("control row must always be valid")
        if self.label not in VALID_LABELS[self.task]:
            raise ValueError(f"invalid label {self.label!r} for task {self.task.value}")

        control = self.features[CONTROL_ROW]
        for index, task in enumerate(TASKS):
            expected = 1.0 if task == self.task else 0.0
            if control[index] != expected:
                raise ValueError("control-row task encoding is invalid")

    def to_dict(self) -> dict[str, Any]:
        return {
            "benchmark_version": BENCHMARK_VERSION,
            "task": self.task.value,
            "difficulty": self.difficulty.value,
            "seed": self.seed,
            "features": [list(row) for row in self.features],
            "mask": list(self.mask),
            "label": self.label,
            "metadata": self.metadata,
        }


def empty_canvas(task: TaskFamily) -> tuple[list[list[float]], list[bool]]:
    """Create the common 32x16 input with a one-hot task control row."""

    features = [[0.0] * FEATURE_WIDTH for _ in range(SEQUENCE_LENGTH)]
    mask = [False] * SEQUENCE_LENGTH
    mask[CONTROL_ROW] = True
    features[CONTROL_ROW][TASK_INDEX[task]] = 1.0
    return features, mask


def freeze_sample(
    *,
    task: TaskFamily,
    difficulty: Difficulty,
    seed: int,
    features: list[list[float]],
    mask: list[bool],
    label: str,
    metadata: dict[str, Any],
) -> BenchmarkSample:
    sample = BenchmarkSample(
        task=task,
        difficulty=difficulty,
        seed=seed,
        features=tuple(tuple(float(value) for value in row) for row in features),
        mask=tuple(bool(value) for value in mask),
        label=label,
        metadata=metadata,
    )
    sample.validate()
    return sample
