"""PyTorch adapters for the deterministic Step 1 benchmark."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch.utils.data import DataLoader, Dataset

from .generator import SPLIT_BASE_SEEDS, generate_sample
from .model import LABEL_TO_INDEX
from .schema import DIFFICULTIES, TASKS, BenchmarkSample


class Step01TorchDataset(Dataset[BenchmarkSample]):
    """Map-style view over the procedural benchmark without materializing it."""

    def __init__(self, split: str, count: int) -> None:
        if split not in SPLIT_BASE_SEEDS:
            raise ValueError(f"unknown split {split!r}")
        if count < 0:
            raise ValueError("count must be non-negative")
        self.split = split
        self.count = count

    def __len__(self) -> int:
        return self.count

    def __getitem__(self, index: int) -> BenchmarkSample:
        if index < 0:
            index += self.count
        if index < 0 or index >= self.count:
            raise IndexError(index)

        pair_count = len(TASKS) * len(DIFFICULTIES)
        task = TASKS[index % len(TASKS)]
        difficulty = DIFFICULTIES[(index // len(TASKS)) % len(DIFFICULTIES)]
        cycle = index // pair_count
        seed = SPLIT_BASE_SEEDS[self.split] + cycle * pair_count + index % pair_count
        return generate_sample(task, difficulty, seed)


def collate_samples(samples: Sequence[BenchmarkSample]) -> dict[str, object]:
    if not samples:
        raise ValueError("cannot collate an empty batch")

    features = torch.tensor(
        [sample.features for sample in samples],
        dtype=torch.float32,
    )
    mask = torch.tensor(
        [sample.mask for sample in samples],
        dtype=torch.bool,
    )

    # -100 is the standard CrossEntropy ignore index. Uncertain examples are
    # trained only through the dedicated uncertainty objective.
    label_targets = torch.tensor(
        [
            -100 if sample.label == "UNCERTAIN" else LABEL_TO_INDEX[sample.label]
            for sample in samples
        ],
        dtype=torch.long,
    )
    uncertainty_targets = torch.tensor(
        [1.0 if sample.label == "UNCERTAIN" else 0.0 for sample in samples],
        dtype=torch.float32,
    )

    return {
        "features": features,
        "mask": mask,
        "label_targets": label_targets,
        "uncertainty_targets": uncertainty_targets,
        "samples": tuple(samples),
    }


def make_loader(
    *,
    split: str,
    count: int,
    batch_size: int,
    shuffle: bool,
    seed: int,
    num_workers: int = 0,
) -> DataLoader:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if num_workers < 0:
        raise ValueError("num_workers must be non-negative")

    generator = torch.Generator()
    generator.manual_seed(seed)

    return DataLoader(
        Step01TorchDataset(split, count),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_samples,
        generator=generator,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
    )
