"""Procedural Step 1 benchmark generator.

Every example is deterministic for (benchmark version, task, difficulty, seed).
The generator exposes latent metadata for validation and oracle scoring, but the
neural unit receives only ``features`` and ``mask``.
"""

from __future__ import annotations

import hashlib
import math
import random
from collections.abc import Iterator

from .schema import (
    BENCHMARK_VERSION,
    DATA_START,
    DIFFICULTIES,
    SEQUENCE_LENGTH,
    TASKS,
    BenchmarkSample,
    Difficulty,
    TaskFamily,
    empty_canvas,
    freeze_sample,
)

SPLIT_BASE_SEEDS = {
    "train": 0,
    "validation": 1_000_000_000,
    "test": 2_000_000_000,
}

_PATTERN = (
    (1.0, 0.25, -0.75, 0.0),
    (0.25, 1.0, 0.0, -0.75),
    (-0.75, 0.0, 1.0, 0.25),
    (0.0, -0.75, 0.25, 1.0),
)

_SIGNATURE_A = (1.0, -0.5, 0.75, 0.25)
_SIGNATURE_B = (-0.25, 1.0, -0.5, 0.75)


def _stable_seed(*parts: object) -> int:
    payload = "|".join(str(part) for part in (BENCHMARK_VERSION, *parts)).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big", signed=False)


def _rng(task: TaskFamily, difficulty: Difficulty, seed: int) -> random.Random:
    return random.Random(_stable_seed(task.value, difficulty.value, seed))


def _noise_scale(difficulty: Difficulty) -> float:
    return {
        Difficulty.EASY: 0.08,
        Difficulty.MEDIUM: 0.18,
        Difficulty.HARD: 0.32,
        Difficulty.AMBIGUOUS: 0.28,
    }[difficulty]


def _missing_probability(difficulty: Difficulty) -> float:
    return {
        Difficulty.EASY: 0.0,
        Difficulty.MEDIUM: 0.04,
        Difficulty.HARD: 0.12,
        Difficulty.AMBIGUOUS: 0.20,
    }[difficulty]


def _add_noise(rng: random.Random, value: float, sigma: float) -> float:
    return value + rng.gauss(0.0, sigma)


def _set_row(
    features: list[list[float]],
    mask: list[bool],
    row_index: int,
    values: list[float] | tuple[float, ...],
) -> None:
    if row_index >= SEQUENCE_LENGTH:
        raise IndexError("benchmark row exceeds fixed sequence length")
    for index, value in enumerate(values):
        features[row_index][index] = float(value)
    mask[row_index] = True


def generate_sample(
    task: TaskFamily | str,
    difficulty: Difficulty | str,
    seed: int,
) -> BenchmarkSample:
    """Generate exactly one deterministic benchmark example."""

    task = TaskFamily(task)
    difficulty = Difficulty(difficulty)

    if task is TaskFamily.PATTERN:
        return _generate_pattern(difficulty, seed)
    if task is TaskFamily.CHANGE:
        return _generate_change(difficulty, seed)
    if task is TaskFamily.CONFLICT:
        return _generate_conflict(difficulty, seed)
    if task is TaskFamily.RELATION:
        return _generate_relation(difficulty, seed)
    if task is TaskFamily.RELEVANCE:
        return _generate_relevance(difficulty, seed)
    raise AssertionError(f"unhandled task {task}")


def generate_dataset(split: str, count: int) -> Iterator[BenchmarkSample]:
    """Yield a balanced deterministic stream across task/difficulty pairs."""

    if split not in SPLIT_BASE_SEEDS:
        raise ValueError(f"unknown split {split!r}")
    if count < 0:
        raise ValueError("count must be non-negative")

    base = SPLIT_BASE_SEEDS[split]
    pair_count = len(TASKS) * len(DIFFICULTIES)
    for index in range(count):
        task = TASKS[index % len(TASKS)]
        difficulty = DIFFICULTIES[(index // len(TASKS)) % len(DIFFICULTIES)]
        cycle = index // pair_count
        seed = base + cycle * pair_count + index % pair_count
        yield generate_sample(task, difficulty, seed)


def _generate_pattern(difficulty: Difficulty, seed: int) -> BenchmarkSample:
    rng = _rng(TaskFamily.PATTERN, difficulty, seed)
    features, mask = empty_canvas(TaskFamily.PATTERN)

    rows = 24
    sigma = _noise_scale(difficulty)
    for offset in range(rows):
        values = [rng.gauss(0.0, sigma * 1.2) for _ in range(4)]
        _set_row(features, mask, DATA_START + offset, values)

    if difficulty is Difficulty.AMBIGUOUS:
        label = "UNCERTAIN"
        target_strength = rng.uniform(0.32, 0.58)
        occlusion = rng.uniform(0.35, 0.65)
    elif seed % 2 == 0:
        label = "SIGNAL"
        target_strength = {
            Difficulty.EASY: 1.30,
            Difficulty.MEDIUM: 1.00,
            Difficulty.HARD: 0.72,
        }[difficulty]
        occlusion = {
            Difficulty.EASY: 0.0,
            Difficulty.MEDIUM: 0.10,
            Difficulty.HARD: 0.25,
        }[difficulty]
    else:
        label = "NO_SIGNAL"
        target_strength = 0.0
        occlusion = 0.0

    target_start = rng.randint(0, rows - len(_PATTERN))
    if label != "NO_SIGNAL":
        for local_row, template_row in enumerate(_PATTERN):
            row_index = DATA_START + target_start + local_row
            if rng.random() < occlusion:
                continue
            for dim, template_value in enumerate(template_row):
                features[row_index][dim] += (
                    target_strength * template_value + rng.gauss(0.0, sigma)
                )
    elif difficulty is Difficulty.HARD:
        shuffled = list(_PATTERN)
        rng.shuffle(shuffled)
        for local_row, template_row in enumerate(shuffled):
            row_index = DATA_START + target_start + local_row
            for dim, template_value in enumerate(template_row):
                features[row_index][dim] += 0.45 * template_value

    metadata = {
        "target_start": target_start,
        "target_strength": target_strength,
        "occlusion": occlusion,
        "latent_present": label == "SIGNAL",
        "latent_ambiguous": label == "UNCERTAIN",
    }
    return freeze_sample(
        task=TaskFamily.PATTERN,
        difficulty=difficulty,
        seed=seed,
        features=features,
        mask=mask,
        label=label,
        metadata=metadata,
    )


def _state_observations(
    *,
    rng: random.Random,
    latent: list[float],
    count: int,
    sigma: float,
    missing_probability: float,
) -> list[tuple[list[float], float]]:
    observations: list[tuple[list[float], float]] = []
    for _ in range(count):
        local_sigma = sigma * rng.uniform(0.55, 1.65)
        reliability = 1.0 / max(local_sigma, 1e-6)
        values = [_add_noise(rng, value, local_sigma) for value in latent]
        if rng.random() < missing_probability:
            missing_dim = rng.randrange(len(values))
            values[missing_dim] = 0.0
            reliability = -reliability
        observations.append((values, reliability))
    return observations


def _generate_change(difficulty: Difficulty, seed: int) -> BenchmarkSample:
    rng = _rng(TaskFamily.CHANGE, difficulty, seed)
    features, mask = empty_canvas(TaskFamily.CHANGE)

    latent_a = [rng.uniform(-1.0, 1.0) for _ in range(4)]
    if difficulty is Difficulty.AMBIGUOUS:
        delta_norm = rng.uniform(0.30, 0.48)
        label = "UNCERTAIN"
    elif seed % 2 == 0:
        delta_norm = {
            Difficulty.EASY: 0.95,
            Difficulty.MEDIUM: 0.78,
            Difficulty.HARD: 0.62,
        }[difficulty]
        label = "CHANGE"
    else:
        delta_norm = {
            Difficulty.EASY: 0.04,
            Difficulty.MEDIUM: 0.10,
            Difficulty.HARD: 0.18,
        }[difficulty]
        label = "NO_CHANGE"

    direction = [rng.gauss(0.0, 1.0) for _ in range(4)]
    norm = math.sqrt(sum(value * value for value in direction)) or 1.0
    direction = [value / norm for value in direction]
    latent_b = [
        value + delta_norm * direction[index] for index, value in enumerate(latent_a)
    ]

    sigma = _noise_scale(difficulty)
    missing = _missing_probability(difficulty)
    obs_a = _state_observations(
        rng=rng, latent=latent_a, count=8, sigma=sigma, missing_probability=missing
    )
    obs_b = _state_observations(
        rng=rng, latent=latent_b, count=8, sigma=sigma, missing_probability=missing
    )

    for offset, (values, reliability) in enumerate(obs_a):
        _set_row(features, mask, DATA_START + offset, [*values, reliability, 0.0])
    for offset, (values, reliability) in enumerate(obs_b):
        _set_row(features, mask, DATA_START + 8 + offset, [*values, reliability, 1.0])

    metadata = {
        "latent_a": latent_a,
        "latent_b": latent_b,
        "delta_norm": delta_norm,
        "no_change_max": 0.25,
        "change_min": 0.55,
    }
    return freeze_sample(
        task=TaskFamily.CHANGE,
        difficulty=difficulty,
        seed=seed,
        features=features,
        mask=mask,
        label=label,
        metadata=metadata,
    )


def _generate_conflict(difficulty: Difficulty, seed: int) -> BenchmarkSample:
    rng = _rng(TaskFamily.CONFLICT, difficulty, seed)
    features, mask = empty_canvas(TaskFamily.CONFLICT)

    latent_a = [rng.uniform(-1.0, 1.0) for _ in range(5)]
    if difficulty is Difficulty.AMBIGUOUS:
        delta_norm = rng.uniform(0.34, 0.58)
        label = "UNCERTAIN"
    elif seed % 2 == 0:
        delta_norm = {
            Difficulty.EASY: 1.25,
            Difficulty.MEDIUM: 1.00,
            Difficulty.HARD: 0.78,
        }[difficulty]
        label = "CONFLICT"
    else:
        delta_norm = {
            Difficulty.EASY: 0.04,
            Difficulty.MEDIUM: 0.10,
            Difficulty.HARD: 0.20,
        }[difficulty]
        label = "COMPATIBLE"

    direction = [rng.gauss(0.0, 1.0) for _ in range(5)]
    norm = math.sqrt(sum(value * value for value in direction)) or 1.0
    latent_b = [
        value + delta_norm * direction[index] / norm
        for index, value in enumerate(latent_a)
    ]

    sigma = _noise_scale(difficulty)
    missing = _missing_probability(difficulty)
    obs_a = _state_observations(
        rng=rng, latent=latent_a, count=6, sigma=sigma, missing_probability=missing
    )
    obs_b = _state_observations(
        rng=rng, latent=latent_b, count=6, sigma=sigma, missing_probability=missing
    )

    for offset, (values, reliability) in enumerate(obs_a):
        _set_row(features, mask, DATA_START + offset, [*values, reliability, 0.0])
    for offset, (values, reliability) in enumerate(obs_b):
        _set_row(features, mask, DATA_START + 6 + offset, [*values, reliability, 1.0])

    metadata = {
        "latent_a": latent_a,
        "latent_b": latent_b,
        "delta_norm": delta_norm,
        "compatible_max": 0.28,
        "conflict_min": 0.68,
    }
    return freeze_sample(
        task=TaskFamily.CONFLICT,
        difficulty=difficulty,
        seed=seed,
        features=features,
        mask=mask,
        label=label,
        metadata=metadata,
    )


def _generate_relation(difficulty: Difficulty, seed: int) -> BenchmarkSample:
    rng = _rng(TaskFamily.RELATION, difficulty, seed)
    features, mask = empty_canvas(TaskFamily.RELATION)

    time_a = rng.uniform(-2.0, 2.0)
    if difficulty is Difficulty.AMBIGUOUS:
        gap = rng.choice((-1.0, 1.0)) * rng.uniform(0.18, 0.44)
        label = "UNCERTAIN"
    else:
        relation = seed % 3
        if relation == 0:
            gap = {
                Difficulty.EASY: 1.40,
                Difficulty.MEDIUM: 1.00,
                Difficulty.HARD: 0.72,
            }[difficulty]
            label = "A_BEFORE_B"
        elif relation == 1:
            gap = -{
                Difficulty.EASY: 1.40,
                Difficulty.MEDIUM: 1.00,
                Difficulty.HARD: 0.72,
            }[difficulty]
            label = "B_BEFORE_A"
        else:
            gap = rng.uniform(-0.10, 0.10)
            label = "SAME_TIME"

    time_b = time_a + gap
    sigma = _noise_scale(difficulty) * 1.15
    missing = _missing_probability(difficulty)

    for event_index, true_time in enumerate((time_a, time_b)):
        for cue in range(7):
            local_sigma = sigma * rng.uniform(0.45, 1.8)
            observed = _add_noise(rng, true_time, local_sigma)
            reliability = 1.0 / max(local_sigma, 1e-6)
            if rng.random() < missing:
                observed = 0.0
                reliability = -reliability
            row = DATA_START + event_index * 7 + cue
            _set_row(
                features,
                mask,
                row,
                [observed, reliability, float(event_index), float(cue) / 6.0],
            )

    metadata = {
        "time_a": time_a,
        "time_b": time_b,
        "gap_b_minus_a": gap,
        "same_time_max": 0.15,
        "decisive_gap_min": 0.50,
    }
    return freeze_sample(
        task=TaskFamily.RELATION,
        difficulty=difficulty,
        seed=seed,
        features=features,
        mask=mask,
        label=label,
        metadata=metadata,
    )


def _inject_signature(
    *,
    rng: random.Random,
    features: list[list[float]],
    row: int,
    signature: tuple[float, ...],
    strength: float,
    sigma: float,
    occluded: bool,
) -> None:
    hidden_dim = rng.randrange(len(signature)) if occluded else -1
    for dim, value in enumerate(signature):
        if dim == hidden_dim:
            continue
        features[row][dim] += strength * value + rng.gauss(0.0, sigma)


def _generate_relevance(difficulty: Difficulty, seed: int) -> BenchmarkSample:
    rng = _rng(TaskFamily.RELEVANCE, difficulty, seed)
    features, mask = empty_canvas(TaskFamily.RELEVANCE)

    rows = 24
    sigma = _noise_scale(difficulty)
    for offset in range(rows):
        _set_row(
            features,
            mask,
            DATA_START + offset,
            [rng.gauss(0.0, sigma * 1.3) for _ in range(4)],
        )

    if difficulty is Difficulty.AMBIGUOUS:
        label = "UNCERTAIN"
        strength = rng.uniform(0.40, 0.62)
        pos_a = rng.randint(1, rows - 5)
        pos_b = pos_a + rng.randint(4, 7)
        occluded = True
    elif seed % 2 == 0:
        label = "RELEVANT"
        strength = {
            Difficulty.EASY: 1.30,
            Difficulty.MEDIUM: 1.00,
            Difficulty.HARD: 0.76,
        }[difficulty]
        pos_a = rng.randint(1, rows - 6)
        pos_b = pos_a + rng.randint(1, 4)
        occluded = difficulty is Difficulty.HARD and rng.random() < 0.30
    else:
        label = "NOT_RELEVANT"
        strength = {
            Difficulty.EASY: 1.20,
            Difficulty.MEDIUM: 0.95,
            Difficulty.HARD: 0.75,
        }[difficulty]
        mode = seed % 3
        if mode == 0:
            pos_a, pos_b = rng.randint(6, rows - 2), rng.randint(0, 4)
        elif mode == 1:
            pos_a = rng.randint(0, 4)
            pos_b = rng.randint(10, rows - 1)
        else:
            pos_a = rng.randint(0, rows - 1)
            pos_b = -1
        occluded = False

    _inject_signature(
        rng=rng,
        features=features,
        row=DATA_START + pos_a,
        signature=_SIGNATURE_A,
        strength=strength,
        sigma=sigma,
        occluded=occluded,
    )
    if pos_b >= 0:
        _inject_signature(
            rng=rng,
            features=features,
            row=DATA_START + pos_b,
            signature=_SIGNATURE_B,
            strength=strength,
            sigma=sigma,
            occluded=occluded,
        )

    metadata = {
        "pos_a": pos_a,
        "pos_b": pos_b,
        "strength": strength,
        "occluded": occluded,
        "max_relevant_gap": 4,
        "latent_ambiguous": label == "UNCERTAIN",
    }
    return freeze_sample(
        task=TaskFamily.RELEVANCE,
        difficulty=difficulty,
        seed=seed,
        features=features,
        mask=mask,
        label=label,
        metadata=metadata,
    )
