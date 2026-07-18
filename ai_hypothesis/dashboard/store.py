"""Thread-safe immutable dashboard index snapshot store."""

from __future__ import annotations

from pathlib import Path
from threading import Lock

from .indexer import DashboardIndexer, DashboardIndexSnapshot


class DashboardStore:
    def __init__(self, *, results_dir: Path, indexer: DashboardIndexer | None = None) -> None:
        self.results_dir = results_dir
        self.indexer = indexer or DashboardIndexer()
        self._lock = Lock()
        self._snapshot = self.indexer.build(results_dir)

    def snapshot(self) -> DashboardIndexSnapshot:
        with self._lock:
            return self._snapshot

    def reindex(self) -> DashboardIndexSnapshot:
        try:
            next_snapshot = self.indexer.build(self.results_dir)
        except Exception:
            return self.snapshot()
        with self._lock:
            self._snapshot = next_snapshot
            return self._snapshot
