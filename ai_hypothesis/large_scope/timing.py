"""Opt-in learned-execution timing without changing runtime control semantics."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import torch


@dataclass(frozen=True, slots=True)
class SelectedWorkerTiming:
    call_count: int
    sample_count: int
    elapsed_seconds: float

    @property
    def samples_per_second(self) -> float | None:
        if self.elapsed_seconds <= 0.0:
            return None
        return self.sample_count / self.elapsed_seconds


class TimedSelectedWorkerBank:
    """Measure only ``forward_selected`` work while delegating the full bank API.

    CPU banks use ``perf_counter`` because execution is synchronous.

    CUDA banks record start/end CUDA events around every selected-worker call. Reading a
    snapshot requires the caller to synchronize the device first; this avoids inserting
    a synchronization into every neural batch and therefore avoids destroying the very
    batching behavior being measured.
    """

    def __init__(self, bank: Any) -> None:
        if not hasattr(bank, "forward_selected"):
            raise TypeError("timed selected-worker bank requires forward_selected")
        if not hasattr(bank, "population_width"):
            raise TypeError("timed selected-worker bank requires population_width")
        self.bank = bank
        self._call_count = 0
        self._sample_count = 0
        self._cpu_elapsed_seconds = 0.0
        self._cuda_events: list[tuple[torch.cuda.Event, torch.cuda.Event]] = []

    def __getattr__(self, name: str) -> Any:
        return getattr(self.bank, name)

    @property
    def population_width(self) -> int:
        return int(self.bank.population_width)

    def forward_selected(self, worker_indices, features, mask):
        batch_size = int(features.shape[0])
        self._call_count += 1
        self._sample_count += batch_size

        device = getattr(self.bank, "device", None)
        if isinstance(device, torch.device) and device.type == "cuda":
            with torch.cuda.device(device):
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                try:
                    return self.bank.forward_selected(worker_indices, features, mask)
                finally:
                    end.record()
                    self._cuda_events.append((start, end))

        started = time.perf_counter()
        try:
            return self.bank.forward_selected(worker_indices, features, mask)
        finally:
            self._cpu_elapsed_seconds += time.perf_counter() - started

    def reset_timing(self) -> None:
        """Reset counters after all previously recorded CUDA work has been synchronized."""

        self._call_count = 0
        self._sample_count = 0
        self._cpu_elapsed_seconds = 0.0
        self._cuda_events.clear()

    def snapshot_after_synchronize(self) -> SelectedWorkerTiming:
        """Return accumulated timing after the caller synchronized CUDA, when applicable."""

        elapsed_seconds = self._cpu_elapsed_seconds
        if self._cuda_events:
            elapsed_seconds += sum(
                start.elapsed_time(end) for start, end in self._cuda_events
            ) / 1000.0
        return SelectedWorkerTiming(
            call_count=self._call_count,
            sample_count=self._sample_count,
            elapsed_seconds=elapsed_seconds,
        )
