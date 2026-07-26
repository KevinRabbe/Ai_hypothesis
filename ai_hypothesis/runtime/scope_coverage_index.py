"""Rebuildable incremental scope-coverage projection over the canonical Research Ledger.

The Research Ledger remains the only source of truth. This sidecar materializes only the
attempt/region counters needed by scope-aware scheduling so hot planning does not replay
all historical attempt events.
"""

from __future__ import annotations

import sqlite3
import threading
from pathlib import Path

from .contracts import LedgerEvent
from .ledger import SQLiteResearchLedger
from .projection_tail import LedgerProjectionTail, ProjectionCheckpoint
from .scope_coverage import ScopeRegionCoverage, ThreadScopeCoverage


_INDEX_SCHEMA_VERSION = "scope-coverage-index-v0"
_TERMINAL_ATTEMPT_EVENTS = {
    "ATTEMPT_COMPLETED": "completed",
    "ATTEMPT_PARTIAL": "partial",
    "ATTEMPT_FAILED": "failed",
    "ATTEMPT_CRASHED": "crashed",
    "ATTEMPT_INVALID_RESULT": "invalid",
}


class SQLiteIndexedScopeCoverage:
    """Incremental semantic equivalent of ``ScopeCoverageProjector`` for hot queries."""

    def __init__(
        self,
        ledger: SQLiteResearchLedger,
        index_path: str | Path = ":memory:",
    ) -> None:
        self.ledger = ledger
        self.path = str(index_path)
        self._tail = LedgerProjectionTail(ledger)
        self._lock = threading.RLock()
        self._validate_separate_storage()
        self._connection = sqlite3.connect(
            self.path,
            timeout=30.0,
            check_same_thread=False,
        )
        self._connection.row_factory = sqlite3.Row
        self._configure()
        self._create_schema()
        self._bind_source()

    def __enter__(self) -> "SQLiteIndexedScopeCoverage":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()

    @property
    def schema_version(self) -> str:
        return _INDEX_SCHEMA_VERSION

    @property
    def revision(self) -> int:
        with self._lock:
            raw = self._meta("revision")
            try:
                revision = int(raw)
            except (TypeError, ValueError) as error:
                raise RuntimeError("corrupt scope coverage index revision") from error
            if revision < 0:
                raise RuntimeError("corrupt scope coverage index revision")
            return revision

    def close(self) -> None:
        with self._lock:
            self._connection.close()

    def sync(self, *, page_size: int = 1000) -> int:
        return self._sync_to(target_sequence=None, page_size=page_size)

    def sync_through(self, sequence: int, *, page_size: int = 1000) -> int:
        if sequence < 0:
            raise ValueError("sequence must be non-negative")
        return self._sync_to(target_sequence=sequence, page_size=page_size)

    def rebuild(self, *, page_size: int = 1000) -> int:
        with self._lock, self._connection:
            self._connection.execute("DELETE FROM scope_region_worker")
            self._connection.execute("DELETE FROM scope_attempt_region")
            self._connection.execute("DELETE FROM scope_region")
            self._connection.execute("DELETE FROM scope_attempt")
            self._set_meta("revision", "0")
            self._set_meta("checkpoint_event_id", "")
        return self.sync(page_size=page_size)

    def for_thread(
        self,
        thread_id: str,
        *,
        sequence: int | None = None,
    ) -> ThreadScopeCoverage:
        if not thread_id or not thread_id.strip():
            raise ValueError("thread_id must be non-empty")
        with self._lock:
            if sequence is None:
                self.sync()
            else:
                self.sync_through(sequence)
            regions = self._connection.execute(
                """
                SELECT
                    region_id,
                    started_attempt_count,
                    completed_attempt_count,
                    partial_attempt_count,
                    failed_attempt_count,
                    crashed_attempt_count,
                    invalid_attempt_count,
                    first_sequence,
                    first_position,
                    last_sequence
                FROM scope_region
                WHERE thread_id = ?
                ORDER BY first_sequence, first_position, region_id
                """,
                (thread_id,),
            ).fetchall()
            snapshots = tuple(self._freeze_region(thread_id, row) for row in regions)
            return ThreadScopeCoverage(thread_id=thread_id, regions=snapshots)

    def _sync_to(self, *, target_sequence: int | None, page_size: int) -> int:
        if page_size <= 0:
            raise ValueError("page_size must be positive")
        with self._lock:
            checkpoint = self._checkpoint()
            if target_sequence is not None and target_sequence < checkpoint.sequence:
                raise ValueError("scope coverage index is ahead of requested ledger snapshot")
            for page in self._tail.iter_pages(
                checkpoint,
                target_sequence=target_sequence,
                page_size=page_size,
            ):
                with self._connection:
                    for event in page:
                        self._apply_event(event)
                    next_checkpoint = self._tail.checkpoint_after(page)
                    self._set_meta("revision", str(next_checkpoint.sequence))
                    self._set_meta("checkpoint_event_id", next_checkpoint.event_id)
                checkpoint = next_checkpoint
            return checkpoint.sequence

    def _apply_event(self, event: LedgerEvent) -> None:
        if event.event_type == "ATTEMPT_STARTED":
            self._apply_started(event)
            return
        terminal_kind = _TERMINAL_ATTEMPT_EVENTS.get(event.event_type)
        if terminal_kind is None or event.attempt_id is None:
            return
        self._apply_terminal(event, terminal_kind)

    def _apply_started(self, event: LedgerEvent) -> None:
        if event.attempt_id is None:
            raise ValueError("ATTEMPT_STARTED requires attempt_id")
        existing = self._connection.execute(
            "SELECT 1 FROM scope_attempt WHERE attempt_id = ?",
            (event.attempt_id,),
        ).fetchone()
        if existing is not None:
            raise ValueError(f"attempt {event.attempt_id!r} was started more than once")

        raw_region_ids = event.payload.get("scope_region_ids")
        if raw_region_ids is None or raw_region_ids == []:
            self._connection.execute(
                """
                INSERT INTO scope_attempt(
                    attempt_id, scoped, thread_id, worker_id, terminal_event_type
                ) VALUES (?, 0, NULL, NULL, NULL)
                """,
                (event.attempt_id,),
            )
            return
        if not isinstance(raw_region_ids, list):
            raise ValueError("ATTEMPT_STARTED scope_region_ids must be a list")
        if any(not isinstance(value, str) or not value.strip() for value in raw_region_ids):
            raise ValueError("scope_region_ids must contain non-empty strings")
        region_ids = tuple(raw_region_ids)
        if len(set(region_ids)) != len(region_ids):
            raise ValueError("scope_region_ids must be unique inside one Work Item")
        if event.thread_id is None:
            raise ValueError("scoped ATTEMPT_STARTED requires thread_id")
        worker_id = event.payload.get("worker_id")
        if not isinstance(worker_id, str) or not worker_id.strip():
            raise ValueError("scoped ATTEMPT_STARTED requires worker_id")

        self._connection.execute(
            """
            INSERT INTO scope_attempt(
                attempt_id, scoped, thread_id, worker_id, terminal_event_type
            ) VALUES (?, 1, ?, ?, NULL)
            """,
            (event.attempt_id, event.thread_id, worker_id),
        )
        for position, region_id in enumerate(region_ids):
            self._connection.execute(
                """
                INSERT INTO scope_attempt_region(attempt_id, region_id, region_position)
                VALUES (?, ?, ?)
                """,
                (event.attempt_id, region_id, position),
            )
            self._connection.execute(
                """
                INSERT INTO scope_region(
                    thread_id,
                    region_id,
                    started_attempt_count,
                    completed_attempt_count,
                    partial_attempt_count,
                    failed_attempt_count,
                    crashed_attempt_count,
                    invalid_attempt_count,
                    first_sequence,
                    first_position,
                    last_sequence
                ) VALUES (?, ?, 1, 0, 0, 0, 0, 0, ?, ?, ?)
                ON CONFLICT(thread_id, region_id) DO UPDATE SET
                    started_attempt_count = started_attempt_count + 1,
                    last_sequence = excluded.last_sequence
                """,
                (
                    event.thread_id,
                    region_id,
                    event.sequence,
                    position,
                    event.sequence,
                ),
            )
            self._connection.execute(
                """
                INSERT OR IGNORE INTO scope_region_worker(
                    thread_id, region_id, worker_id, first_sequence
                ) VALUES (?, ?, ?, ?)
                """,
                (event.thread_id, region_id, worker_id, event.sequence),
            )

    def _apply_terminal(self, event: LedgerEvent, terminal_kind: str) -> None:
        assert event.attempt_id is not None
        attempt = self._connection.execute(
            """
            SELECT scoped, thread_id, terminal_event_type
            FROM scope_attempt
            WHERE attempt_id = ?
            """,
            (event.attempt_id,),
        ).fetchone()
        if attempt is None or int(attempt["scoped"]) == 0:
            return
        if attempt["terminal_event_type"] is not None:
            raise ValueError(f"scoped attempt {event.attempt_id!r} has multiple terminal events")
        thread_id = str(attempt["thread_id"])
        if event.thread_id is not None and event.thread_id != thread_id:
            raise ValueError("scoped attempt terminal event changed thread identity")

        self._connection.execute(
            "UPDATE scope_attempt SET terminal_event_type = ? WHERE attempt_id = ?",
            (event.event_type, event.attempt_id),
        )
        column = f"{terminal_kind}_attempt_count"
        if column not in {
            "completed_attempt_count",
            "partial_attempt_count",
            "failed_attempt_count",
            "crashed_attempt_count",
            "invalid_attempt_count",
        }:
            raise RuntimeError("unsupported scope coverage terminal kind")
        rows = self._connection.execute(
            """
            SELECT region_id
            FROM scope_attempt_region
            WHERE attempt_id = ?
            ORDER BY region_position
            """,
            (event.attempt_id,),
        ).fetchall()
        for row in rows:
            self._connection.execute(
                f"""
                UPDATE scope_region
                SET {column} = {column} + 1,
                    last_sequence = ?
                WHERE thread_id = ? AND region_id = ?
                """,
                (event.sequence, thread_id, str(row["region_id"])),
            )

    def _freeze_region(self, thread_id: str, row: sqlite3.Row) -> ScopeRegionCoverage:
        worker_rows = self._connection.execute(
            """
            SELECT worker_id
            FROM scope_region_worker
            WHERE thread_id = ? AND region_id = ?
            ORDER BY first_sequence, rowid
            """,
            (thread_id, str(row["region_id"])),
        ).fetchall()
        return ScopeRegionCoverage(
            region_id=str(row["region_id"]),
            started_attempt_count=int(row["started_attempt_count"]),
            completed_attempt_count=int(row["completed_attempt_count"]),
            partial_attempt_count=int(row["partial_attempt_count"]),
            failed_attempt_count=int(row["failed_attempt_count"]),
            crashed_attempt_count=int(row["crashed_attempt_count"]),
            invalid_attempt_count=int(row["invalid_attempt_count"]),
            worker_ids=tuple(str(worker["worker_id"]) for worker in worker_rows),
            first_sequence=int(row["first_sequence"]),
            last_sequence=int(row["last_sequence"]),
        )

    def _checkpoint(self) -> ProjectionCheckpoint:
        raw_revision = self._meta("revision")
        raw_event_id = self._meta("checkpoint_event_id")
        try:
            revision = int(raw_revision)
        except (TypeError, ValueError) as error:
            raise RuntimeError("corrupt scope coverage index revision") from error
        checkpoint = ProjectionCheckpoint(revision, raw_event_id or "")
        checkpoint.validate()
        return checkpoint

    def _configure(self) -> None:
        self._connection.execute("PRAGMA foreign_keys = ON")
        if self.path != ":memory:":
            self._connection.execute("PRAGMA journal_mode = WAL")
            self._connection.execute("PRAGMA synchronous = NORMAL")

    def _create_schema(self) -> None:
        with self._connection:
            self._connection.execute(
                "CREATE TABLE IF NOT EXISTS scope_meta(key TEXT PRIMARY KEY, value TEXT NOT NULL)"
            )
            self._connection.execute(
                """
                CREATE TABLE IF NOT EXISTS scope_attempt(
                    attempt_id TEXT PRIMARY KEY,
                    scoped INTEGER NOT NULL CHECK(scoped IN (0, 1)),
                    thread_id TEXT,
                    worker_id TEXT,
                    terminal_event_type TEXT
                )
                """
            )
            self._connection.execute(
                """
                CREATE TABLE IF NOT EXISTS scope_attempt_region(
                    attempt_id TEXT NOT NULL,
                    region_id TEXT NOT NULL,
                    region_position INTEGER NOT NULL,
                    PRIMARY KEY(attempt_id, region_id),
                    FOREIGN KEY(attempt_id) REFERENCES scope_attempt(attempt_id) ON DELETE CASCADE
                )
                """
            )
            self._connection.execute(
                """
                CREATE TABLE IF NOT EXISTS scope_region(
                    thread_id TEXT NOT NULL,
                    region_id TEXT NOT NULL,
                    started_attempt_count INTEGER NOT NULL,
                    completed_attempt_count INTEGER NOT NULL,
                    partial_attempt_count INTEGER NOT NULL,
                    failed_attempt_count INTEGER NOT NULL,
                    crashed_attempt_count INTEGER NOT NULL,
                    invalid_attempt_count INTEGER NOT NULL,
                    first_sequence INTEGER NOT NULL,
                    first_position INTEGER NOT NULL,
                    last_sequence INTEGER NOT NULL,
                    PRIMARY KEY(thread_id, region_id)
                )
                """
            )
            self._connection.execute(
                """
                CREATE TABLE IF NOT EXISTS scope_region_worker(
                    thread_id TEXT NOT NULL,
                    region_id TEXT NOT NULL,
                    worker_id TEXT NOT NULL,
                    first_sequence INTEGER NOT NULL,
                    PRIMARY KEY(thread_id, region_id, worker_id)
                )
                """
            )
            self._connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_scope_region_thread ON scope_region(thread_id, first_sequence)"
            )
            self._connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_scope_attempt_thread ON scope_attempt(thread_id)"
            )

    def _bind_source(self) -> None:
        with self._connection:
            schema = self._meta("schema_version")
            if schema is None:
                self._set_meta("schema_version", _INDEX_SCHEMA_VERSION)
            elif schema != _INDEX_SCHEMA_VERSION:
                raise RuntimeError("scope coverage index schema version mismatch")

            source = self._meta("source_identity")
            if source is None:
                self._set_meta("source_identity", self._tail.source_identity)
            elif source != self._tail.source_identity:
                raise RuntimeError("scope coverage index is bound to a different Research Ledger")

            if self._meta("revision") is None:
                self._set_meta("revision", "0")
            if self._meta("checkpoint_event_id") is None:
                self._set_meta("checkpoint_event_id", "")
        self._tail.validate_checkpoint(self._checkpoint())

    def _validate_separate_storage(self) -> None:
        if self.path == ":memory:" or self.ledger.path == ":memory:":
            return
        ledger_path = Path(self.ledger.path).expanduser().resolve()
        index_path = Path(self.path).expanduser().resolve()
        if ledger_path == index_path:
            raise ValueError("scope coverage index must use separate storage from Research Ledger")

    def _meta(self, key: str) -> str | None:
        row = self._connection.execute(
            "SELECT value FROM scope_meta WHERE key = ?",
            (key,),
        ).fetchone()
        return None if row is None else str(row["value"])

    def _set_meta(self, key: str, value: str) -> None:
        self._connection.execute(
            """
            INSERT INTO scope_meta(key, value)
            VALUES (?, ?)
            ON CONFLICT(key) DO UPDATE SET value = excluded.value
            """,
            (key, value),
        )
