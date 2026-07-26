"""Rebuildable incremental materialized view for current Knowledge State.

Knowledge deltas and later assessments remain canonical append-only Research Ledger events.
This sidecar stores only the current rebuildable projection so verification/consolidation hot
paths can advance from the ledger tail rather than refolding every historical delta and
assessment on every query.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from pathlib import Path
from typing import Any, Iterable

from .contracts import KnowledgeAssessmentKind, LedgerEvent
from .knowledge import KnowledgeRecord, KnowledgeSnapshot, KnowledgeStatus
from .ledger import SQLiteResearchLedger
from .projection_tail import LedgerProjectionTail, ProjectionCheckpoint


_INDEX_SCHEMA_VERSION = "knowledge-state-index-v0"
_ASSESSMENT_TO_STATUS = {
    KnowledgeAssessmentKind.VERIFIED: KnowledgeStatus.VERIFIED,
    KnowledgeAssessmentKind.DISPUTED: KnowledgeStatus.DISPUTED,
    KnowledgeAssessmentKind.RETRACTED: KnowledgeStatus.RETRACTED,
}


class SQLiteIndexedKnowledgeState:
    """Incremental Knowledge State projector backed by disposable SQLite derived state."""

    def __init__(
        self,
        ledger: SQLiteResearchLedger,
        index_path: str | Path = ":memory:",
    ) -> None:
        self.ledger = ledger
        self.path = str(index_path)
        self.tail = LedgerProjectionTail(ledger)
        self._validate_separate_storage()
        self._lock = threading.RLock()
        self._connection = sqlite3.connect(
            self.path,
            timeout=30.0,
            check_same_thread=False,
        )
        self._connection.row_factory = sqlite3.Row
        self._configure()
        self._create_schema()
        self._bind_source()

    @property
    def schema_version(self) -> str:
        return _INDEX_SCHEMA_VERSION

    @property
    def revision(self) -> int:
        with self._lock:
            return self._checkpoint_unlocked().sequence

    def sync(self, *, page_size: int = 1000) -> int:
        return self._sync_to(target_sequence=None, page_size=page_size)

    def sync_through(self, sequence: int, *, page_size: int = 1000) -> int:
        if sequence < 0:
            raise ValueError("sequence must be non-negative")
        return self._sync_to(target_sequence=sequence, page_size=page_size)

    def rebuild(self, *, page_size: int = 1000) -> int:
        with self._lock:
            with self._connection:
                self._connection.execute("DELETE FROM knowledge_state_record")
                self._set_checkpoint(ProjectionCheckpoint())
            return self._sync_to(target_sequence=None, page_size=page_size)

    def snapshot(self, *, thread_id: str | None = None) -> KnowledgeSnapshot:
        with self._lock:
            self._sync_to(target_sequence=None, page_size=1000)
            return self._snapshot_unlocked(thread_id=thread_id)

    def project(
        self,
        events: Iterable[LedgerEvent],
        *,
        thread_id: str | None = None,
    ) -> KnowledgeSnapshot:
        """Duck-compatible incremental replacement for `KnowledgeStateProjector.project`."""

        history = tuple(events)
        target = history[-1].sequence if history else 0
        with self._lock:
            self._sync_to(target_sequence=target, page_size=1000)
            return self._snapshot_unlocked(thread_id=thread_id)

    def close(self) -> None:
        with self._lock:
            self._connection.close()

    def __enter__(self) -> "SQLiteIndexedKnowledgeState":
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.close()

    def _sync_to(self, *, target_sequence: int | None, page_size: int) -> int:
        if page_size <= 0:
            raise ValueError("page_size must be positive")
        with self._lock:
            checkpoint = self._checkpoint_unlocked()
            for page in self.tail.iter_pages(
                checkpoint,
                target_sequence=target_sequence,
                page_size=page_size,
            ):
                with self._connection:
                    for event in page:
                        self._apply_event(event)
                    checkpoint = self.tail.checkpoint_after(page)
                    self._set_checkpoint(checkpoint)
            return self._checkpoint_unlocked().sequence

    def _apply_event(self, event: LedgerEvent) -> None:
        if event.event_type == "KNOWLEDGE_DELTA_RECORDED":
            self._apply_delta(event)
        elif event.event_type == "KNOWLEDGE_ASSESSMENT_RECORDED":
            self._apply_assessment(event)

    def _apply_delta(self, event: LedgerEvent) -> None:
        delta_id = self._payload_text(event, "delta_id")
        kind = self._payload_text(event, "kind")
        summary = self._payload_text(event, "summary")
        source_reference_ids = self._payload_string_tuple(
            event,
            "source_reference_ids",
            fallback=tuple(
                reference_id
                for reference_id in event.reference_ids
                if reference_id != delta_id
            ),
        )
        causal_event_ids = self._payload_string_tuple(
            event,
            "causal_event_ids",
            fallback=(),
        )
        try:
            self._connection.execute(
                """
                INSERT INTO knowledge_state_record(
                    delta_id,
                    kind,
                    summary,
                    source_reference_ids_json,
                    causal_event_ids_json,
                    thread_id,
                    created_event_id,
                    created_sequence,
                    status,
                    assessment_reason,
                    assessment_event_id,
                    assessment_sequence
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, NULL, NULL, NULL)
                """,
                (
                    delta_id,
                    kind,
                    summary,
                    self._encode_string_tuple(source_reference_ids),
                    self._encode_string_tuple(causal_event_ids),
                    event.thread_id,
                    event.event_id,
                    event.sequence,
                    KnowledgeStatus.PROVISIONAL.value,
                ),
            )
        except sqlite3.IntegrityError as error:
            raise ValueError(
                f"knowledge delta {delta_id!r} was recorded more than once"
            ) from error

    def _apply_assessment(self, event: LedgerEvent) -> None:
        raw = self._payload_text(event, "assessment")
        try:
            assessment = KnowledgeAssessmentKind(raw)
        except ValueError as error:
            raise ValueError(f"invalid knowledge assessment {raw!r}") from error
        reason = self._optional_reason(event)
        if not event.reference_ids:
            raise ValueError("knowledge assessment requires at least one delta reference")

        for delta_id in event.reference_ids:
            row = self._connection.execute(
                """
                SELECT created_sequence
                FROM knowledge_state_record
                WHERE delta_id = ?
                """,
                (delta_id,),
            ).fetchone()
            if row is None:
                raise ValueError(
                    f"knowledge assessment references unknown delta {delta_id!r}"
                )
            if int(row["created_sequence"]) >= event.sequence:
                raise ValueError("knowledge assessment precedes knowledge delta creation")
            self._connection.execute(
                """
                UPDATE knowledge_state_record
                SET status = ?,
                    assessment_reason = ?,
                    assessment_event_id = ?,
                    assessment_sequence = ?
                WHERE delta_id = ?
                """,
                (
                    _ASSESSMENT_TO_STATUS[assessment].value,
                    reason,
                    event.event_id,
                    event.sequence,
                    delta_id,
                ),
            )

    def _snapshot_unlocked(self, *, thread_id: str | None) -> KnowledgeSnapshot:
        revision = self._checkpoint_unlocked().sequence
        if thread_id is None:
            rows = self._connection.execute(
                """
                SELECT * FROM knowledge_state_record
                ORDER BY created_sequence, delta_id
                """
            ).fetchall()
        else:
            if not thread_id or not thread_id.strip():
                raise ValueError("thread_id must be non-empty when provided")
            rows = self._connection.execute(
                """
                SELECT * FROM knowledge_state_record
                WHERE thread_id = ?
                ORDER BY created_sequence, delta_id
                """,
                (thread_id,),
            ).fetchall()
        return KnowledgeSnapshot(
            revision=revision,
            records=tuple(self._row_to_record(row) for row in rows),
        )

    @staticmethod
    def _row_to_record(row: sqlite3.Row) -> KnowledgeRecord:
        try:
            status = KnowledgeStatus(str(row["status"]))
        except ValueError as error:
            raise RuntimeError("corrupt indexed knowledge status") from error
        return KnowledgeRecord(
            delta_id=str(row["delta_id"]),
            kind=str(row["kind"]),
            summary=str(row["summary"]),
            source_reference_ids=SQLiteIndexedKnowledgeState._decode_string_tuple(
                str(row["source_reference_ids_json"])
            ),
            causal_event_ids=SQLiteIndexedKnowledgeState._decode_string_tuple(
                str(row["causal_event_ids_json"])
            ),
            thread_id=row["thread_id"],
            created_event_id=str(row["created_event_id"]),
            created_sequence=int(row["created_sequence"]),
            status=status,
            assessment_reason=row["assessment_reason"],
            assessment_event_id=row["assessment_event_id"],
            assessment_sequence=(
                int(row["assessment_sequence"])
                if row["assessment_sequence"] is not None
                else None
            ),
        )

    def _configure(self) -> None:
        with self._connection:
            self._connection.execute("PRAGMA foreign_keys = ON")
            self._connection.execute("PRAGMA busy_timeout = 30000")
            self._connection.execute("PRAGMA synchronous = NORMAL")
            self._connection.execute("PRAGMA journal_mode = WAL")

    def _create_schema(self) -> None:
        with self._connection:
            self._connection.execute(
                """
                CREATE TABLE IF NOT EXISTS knowledge_index_meta(
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                )
                """
            )
            self._connection.execute(
                """
                CREATE TABLE IF NOT EXISTS knowledge_state_record(
                    delta_id TEXT PRIMARY KEY,
                    kind TEXT NOT NULL,
                    summary TEXT NOT NULL,
                    source_reference_ids_json TEXT NOT NULL,
                    causal_event_ids_json TEXT NOT NULL,
                    thread_id TEXT,
                    created_event_id TEXT NOT NULL UNIQUE,
                    created_sequence INTEGER NOT NULL,
                    status TEXT NOT NULL,
                    assessment_reason TEXT,
                    assessment_event_id TEXT,
                    assessment_sequence INTEGER
                )
                """
            )
            self._connection.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_knowledge_state_thread_sequence
                ON knowledge_state_record(thread_id, created_sequence, delta_id)
                """
            )
            self._connection.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_knowledge_state_status_thread
                ON knowledge_state_record(status, thread_id, created_sequence)
                """
            )

    def _bind_source(self) -> None:
        expected = {
            "schema_version": _INDEX_SCHEMA_VERSION,
            "ledger_schema_version": self.ledger.schema_version,
            "ledger_source": self.tail.source_identity,
        }
        with self._connection:
            for key, value in expected.items():
                current = self._meta_optional(key)
                if current is None:
                    self._set_meta(key, value)
                elif current != value:
                    raise RuntimeError(
                        f"knowledge index {key} mismatch: {current!r} != {value!r}"
                    )
            if self._meta_optional("revision") is None:
                self._set_meta("revision", "0")
            if self._meta_optional("checkpoint_event_id") is None:
                self._set_meta("checkpoint_event_id", "")

    def _checkpoint_unlocked(self) -> ProjectionCheckpoint:
        revision_raw = self._meta("revision")
        event_id = self._meta("checkpoint_event_id")
        try:
            revision = int(revision_raw)
        except ValueError as error:
            raise RuntimeError("corrupt knowledge index revision") from error
        checkpoint = ProjectionCheckpoint(sequence=revision, event_id=event_id)
        try:
            checkpoint.validate()
        except ValueError as error:
            raise RuntimeError("corrupt knowledge index checkpoint") from error
        return checkpoint

    def _set_checkpoint(self, checkpoint: ProjectionCheckpoint) -> None:
        checkpoint.validate()
        self._set_meta("revision", str(checkpoint.sequence))
        self._set_meta("checkpoint_event_id", checkpoint.event_id)

    def _meta_optional(self, key: str) -> str | None:
        row = self._connection.execute(
            "SELECT value FROM knowledge_index_meta WHERE key = ?",
            (key,),
        ).fetchone()
        return str(row["value"]) if row is not None else None

    def _meta(self, key: str) -> str:
        value = self._meta_optional(key)
        if value is None:
            raise RuntimeError(f"knowledge index metadata {key!r} is missing")
        return value

    def _set_meta(self, key: str, value: str) -> None:
        self._connection.execute(
            """
            INSERT INTO knowledge_index_meta(key, value)
            VALUES (?, ?)
            ON CONFLICT(key) DO UPDATE SET value = excluded.value
            """,
            (key, value),
        )

    @staticmethod
    def _payload_text(event: LedgerEvent, key: str) -> str:
        value = event.payload.get(key)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(
                f"{event.event_type} requires non-empty string payload field {key!r}"
            )
        return value

    @staticmethod
    def _payload_string_tuple(
        event: LedgerEvent,
        key: str,
        *,
        fallback: tuple[str, ...],
    ) -> tuple[str, ...]:
        if key not in event.payload:
            return fallback
        value = event.payload[key]
        if not isinstance(value, list) or any(
            not isinstance(item, str) or not item for item in value
        ):
            raise ValueError(f"{event.event_type} payload {key!r} must be a string list")
        return tuple(value)

    @staticmethod
    def _optional_reason(event: LedgerEvent) -> str | None:
        value = event.payload.get("reason")
        if value is None:
            return None
        if not isinstance(value, str) or not value.strip():
            raise ValueError("knowledge assessment reason must be a non-empty string")
        return value

    @staticmethod
    def _encode_string_tuple(values: tuple[str, ...]) -> str:
        return json.dumps(values, ensure_ascii=False, separators=(",", ":"))

    @staticmethod
    def _decode_string_tuple(payload: str) -> tuple[str, ...]:
        value = json.loads(payload)
        if not isinstance(value, list) or any(
            not isinstance(item, str) or not item for item in value
        ):
            raise RuntimeError("corrupt indexed knowledge reference list")
        return tuple(value)

    def _validate_separate_storage(self) -> None:
        if self.path == ":memory:" or self.ledger.path == ":memory:":
            return
        if (
            Path(self.path).expanduser().resolve()
            == Path(self.ledger.path).expanduser().resolve()
        ):
            raise ValueError(
                "knowledge index must use rebuildable storage separate from the canonical ledger"
            )
