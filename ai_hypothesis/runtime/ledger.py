"""Minimal durable append-only Research Ledger.

SQLite is an implementation detail behind the runtime's LedgerEvent contract. The
semantic model remains append-only so a later storage engine can replace this
implementation without changing worker, projector, or scheduler contracts.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import uuid
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .contracts import LedgerEvent


_SCHEMA_VERSION = "research-ledger-v0"
_EVENT_PAYLOAD_SCHEMA = "runtime-event-v0"


class SQLiteResearchLedger:
    """Single-node durable ledger with ordered append and replay."""

    def __init__(self, path: str | Path) -> None:
        self.path = str(path)
        self._lock = threading.RLock()
        self._connection = sqlite3.connect(
            self.path,
            timeout=30.0,
            isolation_level=None,
            check_same_thread=False,
        )
        self._connection.row_factory = sqlite3.Row
        self._configure()
        self._create_schema()

    def _configure(self) -> None:
        with self._lock:
            self._connection.execute("PRAGMA foreign_keys = ON")
            self._connection.execute("PRAGMA busy_timeout = 30000")
            self._connection.execute("PRAGMA synchronous = NORMAL")
            self._connection.execute("PRAGMA journal_mode = WAL")

    def _create_schema(self) -> None:
        with self._lock, self._connection:
            self._connection.execute(
                """
                CREATE TABLE IF NOT EXISTS ledger_meta (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                )
                """
            )
            self._connection.execute(
                """
                CREATE TABLE IF NOT EXISTS ledger_events (
                    sequence INTEGER PRIMARY KEY AUTOINCREMENT,
                    event_id TEXT NOT NULL UNIQUE,
                    event_type TEXT NOT NULL,
                    payload_schema TEXT NOT NULL,
                    thread_id TEXT,
                    attempt_id TEXT,
                    reference_ids_json TEXT NOT NULL,
                    parent_event_ids_json TEXT NOT NULL,
                    payload_json TEXT NOT NULL
                )
                """
            )
            self._connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_ledger_events_thread ON ledger_events(thread_id, sequence)"
            )
            self._connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_ledger_events_attempt ON ledger_events(attempt_id, sequence)"
            )
            existing = self._connection.execute(
                "SELECT value FROM ledger_meta WHERE key = 'schema_version'"
            ).fetchone()
            if existing is None:
                self._connection.execute(
                    "INSERT INTO ledger_meta(key, value) VALUES('schema_version', ?)",
                    (_SCHEMA_VERSION,),
                )
            elif existing["value"] != _SCHEMA_VERSION:
                raise RuntimeError(
                    f"unsupported research ledger schema {existing['value']!r}; "
                    f"expected {_SCHEMA_VERSION!r}"
                )

    @property
    def schema_version(self) -> str:
        return _SCHEMA_VERSION

    def append_event(
        self,
        *,
        event_type: str,
        thread_id: str | None = None,
        attempt_id: str | None = None,
        reference_ids: Sequence[str] = (),
        parent_event_ids: Sequence[str] = (),
        payload: Mapping[str, Any] | None = None,
        payload_schema: str = _EVENT_PAYLOAD_SCHEMA,
        event_id: str | None = None,
    ) -> LedgerEvent:
        """Append one event and return it with its ledger-assigned sequence."""

        resolved_event_id = event_id or uuid.uuid4().hex
        references = tuple(reference_ids)
        parents = tuple(parent_event_ids)
        resolved_payload = dict(payload or {})

        # Serialization is a trust boundary. Reject non-JSON values before writing
        # so replay cannot later discover an unreadable durable event.
        references_json = _encode_json(references)
        parents_json = _encode_json(parents)
        payload_json = _encode_json(resolved_payload)

        probe = LedgerEvent(
            event_id=resolved_event_id,
            event_type=event_type,
            sequence=0,
            payload_schema=payload_schema,
            thread_id=thread_id,
            attempt_id=attempt_id,
            reference_ids=references,
            parent_event_ids=parents,
            payload=resolved_payload,
        )
        probe.validate()

        with self._lock, self._connection:
            cursor = self._connection.execute(
                """
                INSERT INTO ledger_events(
                    event_id,
                    event_type,
                    payload_schema,
                    thread_id,
                    attempt_id,
                    reference_ids_json,
                    parent_event_ids_json,
                    payload_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    resolved_event_id,
                    event_type,
                    payload_schema,
                    thread_id,
                    attempt_id,
                    references_json,
                    parents_json,
                    payload_json,
                ),
            )
            sequence = int(cursor.lastrowid)

        return LedgerEvent(
            event_id=resolved_event_id,
            event_type=event_type,
            sequence=sequence,
            payload_schema=payload_schema,
            thread_id=thread_id,
            attempt_id=attempt_id,
            reference_ids=references,
            parent_event_ids=parents,
            payload=resolved_payload,
        )

    def get_event(self, event_id: str) -> LedgerEvent | None:
        if not event_id:
            raise ValueError("event_id must be non-empty")
        with self._lock:
            row = self._connection.execute(
                "SELECT * FROM ledger_events WHERE event_id = ?",
                (event_id,),
            ).fetchone()
        return _row_to_event(row) if row is not None else None

    def read_events(
        self,
        *,
        after_sequence: int = 0,
        limit: int = 1000,
        thread_id: str | None = None,
    ) -> tuple[LedgerEvent, ...]:
        """Read ordered events after a checkpoint, optionally scoped to one thread."""

        if after_sequence < 0:
            raise ValueError("after_sequence must be non-negative")
        if limit <= 0:
            raise ValueError("limit must be positive")

        if thread_id is None:
            sql = "SELECT * FROM ledger_events WHERE sequence > ? ORDER BY sequence LIMIT ?"
            params: tuple[Any, ...] = (after_sequence, limit)
        else:
            if not thread_id:
                raise ValueError("thread_id must be non-empty when provided")
            sql = (
                "SELECT * FROM ledger_events "
                "WHERE sequence > ? AND thread_id = ? ORDER BY sequence LIMIT ?"
            )
            params = (after_sequence, thread_id, limit)

        with self._lock:
            rows = self._connection.execute(sql, params).fetchall()
        return tuple(_row_to_event(row) for row in rows)

    def latest_sequence(self) -> int:
        with self._lock:
            row = self._connection.execute(
                "SELECT COALESCE(MAX(sequence), 0) AS sequence FROM ledger_events"
            ).fetchone()
        return int(row["sequence"])

    def close(self) -> None:
        with self._lock:
            self._connection.close()

    def __enter__(self) -> "SQLiteResearchLedger":
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.close()


def _encode_json(value: object) -> str:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError) as error:
        raise ValueError("ledger values must be finite JSON-serializable data") from error


def _decode_tuple(payload: str) -> tuple[str, ...]:
    value = json.loads(payload)
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise RuntimeError("corrupt ledger reference list")
    return tuple(value)


def _decode_mapping(payload: str) -> dict[str, Any]:
    value = json.loads(payload)
    if not isinstance(value, dict):
        raise RuntimeError("corrupt ledger event payload")
    return value


def _row_to_event(row: sqlite3.Row) -> LedgerEvent:
    event = LedgerEvent(
        event_id=str(row["event_id"]),
        event_type=str(row["event_type"]),
        sequence=int(row["sequence"]),
        payload_schema=str(row["payload_schema"]),
        thread_id=row["thread_id"],
        attempt_id=row["attempt_id"],
        reference_ids=_decode_tuple(str(row["reference_ids_json"])),
        parent_event_ids=_decode_tuple(str(row["parent_event_ids_json"])),
        payload=_decode_mapping(str(row["payload_json"])),
    )
    event.validate()
    return event
