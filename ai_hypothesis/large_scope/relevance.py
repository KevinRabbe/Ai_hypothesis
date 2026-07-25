"""Deterministic large-scope relevance benchmark built from frozen Step 1 windows.

Each world contains many ordinary 32x16 ``E_RELEVANCE`` samples. At most one window
is truly RELEVANT; the rest are NOT_RELEVANT or controlled UNCERTAIN distractors.
A deterministic inspection permutation makes width comparisons nested prefixes, so
larger widths add scope without changing what smaller widths already inspected.
"""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass

from ai_hypothesis.step01.generator import generate_sample
from ai_hypothesis.step01.schema import BenchmarkSample, Difficulty, TaskFamily


LARGE_SCOPE_BENCHMARK_VERSION = "large-scope-relevance-v0"
LARGE_SCOPE_SEED_BASE = 3_000_000_000
LARGE_SCOPE_SEED_SPAN = 900_000_000


@dataclass(frozen=True, slots=True)
class LargeScopeRelevanceConfig:
    window_count: int = 16
    target_difficulty: Difficulty = Difficulty.HARD
    distractor_difficulty: Difficulty = Difficulty.HARD
    ambiguous_distractor_fraction: float = 0.125

    def validate(self) -> None:
        if self.window_count <= 1:
            raise ValueError("window_count must be greater than one")
        if self.target_difficulty is Difficulty.AMBIGUOUS:
            raise ValueError("target_difficulty cannot be ambiguous")
        if self.distractor_difficulty is Difficulty.AMBIGUOUS:
            raise ValueError(
                "distractor_difficulty must be answerable; use ambiguous_distractor_fraction"
            )
        if not 0.0 <= self.ambiguous_distractor_fraction <= 1.0:
            raise ValueError("ambiguous_distractor_fraction must be in [0, 1]")


@dataclass(frozen=True, slots=True)
class LargeScopeRelevanceSample:
    seed: int
    config: LargeScopeRelevanceConfig
    windows: tuple[BenchmarkSample, ...]
    window_seeds: tuple[int, ...]
    target_present: bool
    target_index: int | None

    def validate(self) -> None:
        self.config.validate()
        if len(self.windows) != self.config.window_count:
            raise ValueError("window count does not match config")
        if len(self.window_seeds) != self.config.window_count:
            raise ValueError("window seed count does not match config")
        if len(set(self.window_seeds)) != len(self.window_seeds):
            raise ValueError("window seeds must be unique inside one world")
        if any(seed < LARGE_SCOPE_SEED_BASE for seed in self.window_seeds):
            raise ValueError("large-scope window seed escaped the reserved seed range")

        relevant_indices: list[int] = []
        for index, window in enumerate(self.windows):
            window.validate()
            if window.task is not TaskFamily.RELEVANCE:
                raise ValueError("large-scope windows must use Step 1 relevance semantics")
            if window.label == "RELEVANT":
                relevant_indices.append(index)

        if self.target_present:
            if self.target_index is None:
                raise ValueError("target_present requires target_index")
            if not 0 <= self.target_index < self.config.window_count:
                raise ValueError("target_index is outside the world")
            if relevant_indices != [self.target_index]:
                raise ValueError("positive world must contain exactly one relevant target window")
        else:
            if self.target_index is not None:
                raise ValueError("negative world must not have target_index")
            if relevant_indices:
                raise ValueError("negative world must not contain a relevant window")


def generate_large_scope_relevance(
    seed: int,
    config: LargeScopeRelevanceConfig = LargeScopeRelevanceConfig(),
    *,
    target_present: bool | None = None,
) -> LargeScopeRelevanceSample:
    """Generate one deterministic world without changing Worker v1 local semantics."""

    if seed < 0:
        raise ValueError("seed must be non-negative")
    config.validate()
    resolved_target_present = seed % 2 == 0 if target_present is None else target_present
    layout_rng = random.Random(_scope_seed("layout", seed, config.window_count))
    target_index = (
        layout_rng.randrange(config.window_count) if resolved_target_present else None
    )

    windows: list[BenchmarkSample] = []
    window_seeds: list[int] = []
    used_seeds: set[int] = set()

    for window_index in range(config.window_count):
        if target_index == window_index:
            difficulty = config.target_difficulty
            desired_label = "RELEVANT"
        else:
            ambiguous = (
                layout_rng.random() < config.ambiguous_distractor_fraction
            )
            difficulty = Difficulty.AMBIGUOUS if ambiguous else config.distractor_difficulty
            desired_label = "UNCERTAIN" if ambiguous else "NOT_RELEVANT"

        window_seed = _window_seed(
            world_seed=seed,
            window_index=window_index,
            desired_label=desired_label,
            used_seeds=used_seeds,
        )
        window = generate_sample(TaskFamily.RELEVANCE, difficulty, window_seed)
        if window.label != desired_label:
            raise RuntimeError(
                f"Step 1 generator produced {window.label!r}, expected {desired_label!r}"
            )
        used_seeds.add(window_seed)
        window_seeds.append(window_seed)
        windows.append(window)

    sample = LargeScopeRelevanceSample(
        seed=seed,
        config=config,
        windows=tuple(windows),
        window_seeds=tuple(window_seeds),
        target_present=resolved_target_present,
        target_index=target_index,
    )
    sample.validate()
    return sample


def inspection_order(seed: int, window_count: int) -> tuple[int, ...]:
    """Return one deterministic no-duplicate order independent of world layout RNG."""

    if seed < 0:
        raise ValueError("seed must be non-negative")
    if window_count <= 0:
        raise ValueError("window_count must be positive")
    indices = list(range(window_count))
    random.Random(_scope_seed("inspection", seed, window_count)).shuffle(indices)
    return tuple(indices)


def inspection_prefix(sample: LargeScopeRelevanceSample, width: int) -> tuple[int, ...]:
    """Return the nested prefix inspected at one width."""

    sample.validate()
    if width <= 0:
        raise ValueError("width must be positive")
    if width > sample.config.window_count:
        raise ValueError("width cannot exceed world window_count")
    return inspection_order(sample.seed, sample.config.window_count)[:width]


def same_worker_indices(
    *,
    seed: int,
    width: int,
    population_width: int,
) -> tuple[int, ...]:
    """Scope-only control: one checkpoint inspects every selected window."""

    _validate_worker_plan(width=width, population_width=population_width)
    worker = _scope_seed("same-worker", seed, population_width) % population_width
    return (worker,) * width


def diverse_worker_indices(
    *,
    seed: int,
    width: int,
    population_width: int,
) -> tuple[int, ...]:
    """Scope + diversity: distinct checkpoints inspect the same window prefix."""

    _validate_worker_plan(width=width, population_width=population_width)
    if width > population_width:
        raise ValueError("diverse width cannot exceed available population width")
    start = _scope_seed("diverse-workers", seed, population_width) % population_width
    return tuple((start + offset) % population_width for offset in range(width))


def _validate_worker_plan(*, width: int, population_width: int) -> None:
    if width <= 0:
        raise ValueError("width must be positive")
    if population_width <= 0:
        raise ValueError("population_width must be positive")


def _window_seed(
    *,
    world_seed: int,
    window_index: int,
    desired_label: str,
    used_seeds: set[int],
) -> int:
    raw = LARGE_SCOPE_SEED_BASE + (
        _scope_seed("window", world_seed, window_index, desired_label)
        % LARGE_SCOPE_SEED_SPAN
    )
    if desired_label == "RELEVANT":
        raw += raw % 2
    elif desired_label == "NOT_RELEVANT":
        raw += 1 - (raw % 2)
    elif desired_label != "UNCERTAIN":
        raise ValueError(f"unsupported desired label {desired_label!r}")

    candidate = raw
    while candidate in used_seeds:
        candidate += 2 if desired_label != "UNCERTAIN" else 1
        if candidate >= LARGE_SCOPE_SEED_BASE + LARGE_SCOPE_SEED_SPAN:
            candidate = LARGE_SCOPE_SEED_BASE + (candidate % LARGE_SCOPE_SEED_SPAN)
            if desired_label == "RELEVANT" and candidate % 2 != 0:
                candidate += 1
            elif desired_label == "NOT_RELEVANT" and candidate % 2 == 0:
                candidate += 1
    return candidate


def _scope_seed(*parts: object) -> int:
    payload = "|".join(str(part) for part in (LARGE_SCOPE_BENCHMARK_VERSION, *parts))
    digest = hashlib.sha256(payload.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big", signed=False)
