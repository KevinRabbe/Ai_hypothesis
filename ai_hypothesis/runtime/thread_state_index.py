"""Rebuildable incremental materialized view for Work Thread and Work Graph state.

The canonical Research Ledger remains the only source of truth. This sidecar stores compact
current thread state, graph edges, and last-worker continuity so scheduler cycles can advance
from the ledger tail rather than replaying all historical events.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from .contracts import LedgerEvent, ProjectedState, WorkPurpose
from .ledger import SQLiteResearchLedger
from .projection_tail import LedgerProjectionTail, ProjectionCheckpoint


_INDEX_SCHEMA_VERSION = "thread-state-index-v0"


class SQLiteIndexedThreadState:
    """Incremental Work Thread / Work Graph projector backed by disposable SQLite state."""

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
                self._connection.execute("DELETE FROM thread_dependency")
                self._connection.execute("DELETE FROM thread_fork")
                self._connection.execute("DELETE FROM thread_merge")
                self._connection.execute("DELETE FROM thread_state_record")
                self._set_checkpoint(ProjectionCheckpoint())
            return self._sync_to(target_sequence=None, page_size=page_size)

    def snapshot_all(self) -> tuple[ProjectedState, ...]:
        with self._lock:
            self._sync_to(target_sequence=None, page_size=1000)
            return self._snapshot_all_unlocked()

    def snapshot(self, *, thread_id: str) -> ProjectedState:
        if not thread_id or not thread_id.strip():
            raise ValueError("thread_id must be non-empty")
        with self._lock:
            self._sync_to(target_sequence=None, page_size=1000)
            return self._snapshot_one_unlocked(thread_id)

    def project_all(self, events: Iterable[LedgerEvent]) -> tuple[ProjectedState, ...]:
        """Duck-compatible replacement for ``ThreadStateProjector.project_all``."""

        history = tuple(events)
        target = history[-1].sequence if history else 0
        with self._lock:
            self._sync_to(target_sequence=target, page_size=1000)
            return self._snapshot_all_unlocked()

    def project(self, events: Iterable[LedgerEvent], *, thread_id: str) -> ProjectedState:
        """Duck-compatible replacement for ``ThreadStateProjector.project``."""

        if not thread_id or not thread_id.strip():
            raise ValueError("thread_id must be non-empty")
        history = tuple(events)
        target = history[-1].sequence if history else 0
        with self._lock:
            self._sync_to(target_sequence=target, page_size=1000)
            return self._snapshot_one_unlocked(thread_id)

    def last_worker_id(self, thread_id: str) -> str | None:
        if not thread_id or not thread_id.strip():
            raise ValueError("thread_id must be non-empty")
        with self._lock:
            self._sync_to(target_sequence=None, page_size=1000)
            row = self._connection.execute(
                "SELECT created, last_worker_id FROM thread_state_record WHERE thread_id = ?",
                (thread_id,),
            ).fetchone()
            if row is None or not bool(row["created"]):
                raise ValueError(f"thread {thread_id!r} has no THREAD_CREATED event")
            return str(row["last_worker_id"]) if row["last_worker_id"] is not None else None

    def close(self) -> None:
        with self._lock:
            self._connection.close()

    def __enter__(self) -> "SQLiteIndexedThreadState":
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.close()

    def _sync_to(self, *, target_sequence: int | None, page_size: int) -> int:
        if page_size <= 0:
            raise ValueError("page_size must be positive")
        with self._lock:
            checkpoint = self._checkpoint_unlocked()
            pages = self.tail.iter_pages(
                checkpoint,
                target_sequence=target_sequence,
                page_size=page_size,
            )
            with self._connection:
                for page in pages:
                    for event in page:
                        self._apply_event(event)
                    checkpoint = self.tail.checkpoint_after(page)
                self._validate_graph_unlocked()
                self._set_checkpoint(checkpoint)
            return self._checkpoint_unlocked().sequence

    def _apply_event(self, event: LedgerEvent) -> None:
        if event.event_type == "THREAD_FORKED":
            self._apply_fork(event)
        elif event.event_type == "THREAD_MERGED":
            self._apply_merge(event)

        if event.thread_id is None:
            return

        self._ensure_thread_row(event.thread_id)
        self._update_revision(event.thread_id, event.sequence)
        self._extend_thread_list(event.thread_id, "reference_ids_json", event.reference_ids)

        if event.event_type == "THREAD_CREATED":
            self._apply_thread_created(event)
        elif event.event_type == "THREAD_PURPOSE_SET":
            purpose = WorkPurpose(self._payload_text(event, "purpose"))
            self._connection.execute(
                "UPDATE thread_state_record SET purpose = ? WHERE thread_id = ?",
                (purpose.value, event.thread_id),
            )
        elif event.event_type == "THREAD_STATUS_SET":
            self._connection.execute(
                "UPDATE thread_state_record SET status = ? WHERE thread_id = ?",
                (self._payload_text(event, "status"), event.thread_id),
            )
        elif event.event_type == "THREAD_PAUSED":
            self._connection.execute(
                "UPDATE thread_state_record SET status = 'PAUSED' WHERE thread_id = ?",
                (event.thread_id,),
            )
        elif event.event_type == "THREAD_COMPLETED":
            self._connection.execute(
                "UPDATE thread_state_record SET status = 'COMPLETE' WHERE thread_id = ?",
                (event.thread_id,),
            )
        elif event.event_type == "HYPOTHESIS_PROPOSED":
            self._extend_thread_list(event.thread_id, "hypothesis_ids_json", event.reference_ids)
        elif event.event_type == "HYPOTHESIS_REJECTED":
            self._remove_thread_list(event.thread_id, "hypothesis_ids_json", event.reference_ids)
        elif event.event_type == "CONTRADICTION_FOUND":
            self._extend_thread_list(event.thread_id, "contradiction_ids_json", event.reference_ids)
        elif event.event_type == "CONTRADICTION_RESOLVED":
            self._remove_thread_list(event.thread_id, "contradiction_ids_json", event.reference_ids)
        elif event.event_type == "OPEN_QUESTION_ADDED":
            self._extend_thread_list(
                event.thread_id,
                "open_questions_json",
                (self._payload_text(event, "question"),),
            )
        elif event.event_type == "OPEN_QUESTION_RESOLVED":
            self._remove_thread_list(
                event.thread_id,
                "open_questions_json",
                (self._payload_text(event, "question"),),
            )
        elif event.event_type == "DEPENDENCY_ADDED":
            self._apply_dependency(event, add=True)
        elif event.event_type == "DEPENDENCY_REMOVED":
            self._apply_dependency(event, add=False)
        elif event.event_type == "THREAD_METADATA_UPDATED":
            self._update_metadata(event.thread_id, event.payload)
        elif event.event_type == "ATTEMPT_STARTED":
            worker_id = event.payload.get("worker_id")
            if not isinstance(worker_id, str) or not worker_id.strip():
                raise ValueError("ATTEMPT_STARTED is missing worker_id")
            self._connection.execute(
                """
                UPDATE thread_state_record
                SET last_worker_id = ?, last_attempt_sequence = ?
                WHERE thread_id = ?
                """,
                (worker_id, event.sequence, event.thread_id),
            )

    def _apply_thread_created(self, event: LedgerEvent) -> None:
        assert event.thread_id is not None
        row = self._thread_row(event.thread_id)
        if bool(row["created"]):
            raise ValueError(f"thread {event.thread_id!r} was created more than once")
        objective = self._payload_text(event, "objective")
        raw_purpose = event.payload.get("purpose", WorkPurpose.EXPLORE.value)
        try:
            purpose = WorkPurpose(str(raw_purpose))
        except ValueError as error:
            raise ValueError(f"invalid thread purpose {raw_purpose!r}") from error
        status = str(event.payload.get("status", "ACTIVE"))
        if not status:
            raise ValueError("THREAD_CREATED status must be non-empty")
        self._connection.execute(
            """
            UPDATE thread_state_record
            SET created = 1,
                created_sequence = ?,
                objective = ?,
                status = ?,
                purpose = ?
            WHERE thread_id = ?
            """,
            (event.sequence, objective, status, purpose.value, event.thread_id),
        )

    def _apply_dependency(self, event: LedgerEvent, *, add: bool) -> None:
        assert event.thread_id is not None
        for dependency_id in event.reference_ids:
            if not dependency_id:
                raise ValueError("dependency target IDs must be non-empty")
            if add:
                self._connection.execute(
                    """
                    INSERT INTO thread_dependency(thread_id, dependency_thread_id, sequence)
                    VALUES (?, ?, ?)
                    ON CONFLICT(thread_id, dependency_thread_id)
                    DO UPDATE SET sequence = MAX(sequence, excluded.sequence)
                    """,
                    (event.thread_id, dependency_id, event.sequence),
                )
            else:
                self._connection.execute(
                    "DELETE FROM thread_dependency WHERE thread_id = ? AND dependency_thread_id = ?",
                    (event.thread_id, dependency_id),
                )

    def _apply_fork(self, event: LedgerEvent) -> None:
        parent_id = self._require_thread_id(event)
        if not event.reference_ids:
            raise ValueError("THREAD_FORKED requires at least one target thread ID")
        self._ensure_thread_row(parent_id)
        for child_id in event.reference_ids:
            if not child_id:
                raise ValueError("THREAD_FORKED target IDs must be non-empty")
            self._ensure_thread_row(child_id)
            self._connection.execute(
                """
                INSERT INTO thread_fork(parent_thread_id, child_thread_id, sequence)
                VALUES (?, ?, ?)
                ON CONFLICT(parent_thread_id, child_thread_id)
                DO UPDATE SET sequence = MAX(sequence, excluded.sequence)
                """,
                (parent_id, child_id, event.sequence),
            )
            self._update_revision(parent_id, event.sequence)
            self._update_revision(child_id, event.sequence)

    def _apply_merge(self, event: LedgerEvent) -> None:
        target_id = self._require_thread_id(event)
        if not event.reference_ids:
            raise ValueError("THREAD_MERGED requires at least one target thread ID")
        self._ensure_thread_row(target_id)
        for source_id in event.reference_ids:
            if not source_id:
                raise ValueError("THREAD_MERGED target IDs must be non-empty")
            self._ensure_thread_row(source_id)
            existing = self._connection.execute(
                "SELECT target_thread_id FROM thread_merge WHERE source_thread_id = ?",
                (source_id,),
            ).fetchone()
            if existing is not None and str(existing["target_thread_id"]) != target_id:
                raise ValueError(f"thread {source_id!r} was merged into multiple targets")
            self._connection.execute(
                """
                INSERT INTO thread_merge(source_thread_id, target_thread_id, sequence)
                VALUES (?, ?, ?)
                ON CONFLICT(source_thread_id)
                DO UPDATE SET sequence = MAX(sequence, excluded.sequence)
                """,
                (source_id, target_id, event.sequence),
            )
            self._connection.execute(
                """
                UPDATE thread_state_record
                SET merged_into_thread_id = ?, status = 'COMPLETE'
                WHERE thread_id = ?
                """,
                (target_id, source_id),
            )
            self._update_revision(source_id, event.sequence)
            self._update_revision(target_id, event.sequence)

    def _validate_graph_unlocked(self) -> None:
        created_rows = self._connection.execute(
            "SELECT thread_id, created_sequence FROM thread_state_record WHERE created = 1"
        ).fetchall()
        created = {str(row["thread_id"]): int(row["created_sequence"]) for row in created_rows}

        dependencies: dict[str, list[str]] = {thread_id: [] for thread_id in created}
        for row in self._connection.execute(
            "SELECT thread_id, dependency_thread_id FROM thread_dependency"
        ).fetchall():
            thread_id = str(row["thread_id"])
            dependency_id = str(row["dependency_thread_id"])
            if thread_id not in created:
                continue
            if dependency_id not in created:
                raise ValueError(
                    f"dependency from {thread_id!r} references missing thread {dependency_id!r}"
                )
            if dependency_id == thread_id:
                raise ValueError("thread cannot depend on itself")
            dependencies[thread_id].append(dependency_id)

        forks: dict[str, list[str]] = {thread_id: [] for thread_id in created}
        for row in self._connection.execute(
            "SELECT parent_thread_id, child_thread_id, sequence FROM thread_fork"
        ).fetchall():
            parent_id = str(row["parent_thread_id"])
            child_id = str(row["child_thread_id"])
            sequence = int(row["sequence"])
            self._validate_relation_creation(
                created, parent_id, child_id, sequence, "THREAD_FORKED"
            )
            if parent_id == child_id:
                raise ValueError("thread cannot fork itself")
            forks[parent_id].append(child_id)

        for row in self._connection.execute(
            "SELECT source_thread_id, target_thread_id, sequence FROM thread_merge"
        ).fetchall():
            source_id = str(row["source_thread_id"])
            target_id = str(row["target_thread_id"])
            sequence = int(row["sequence"])
            self._validate_relation_creation(
                created, source_id, target_id, sequence, "THREAD_MERGED"
            )
            if source_id == target_id:
                raise ValueError("thread cannot merge into itself")

        self._assert_acyclic(forks, relation_name="fork ancestry")
        self._assert_acyclic(dependencies, relation_name="dependency")

    @staticmethod
    def _validate_relation_creation(
        created: Mapping[str, int],
        source_id: str,
        target_id: str,
        sequence: int,
        event_type: str,
    ) -> None:
        if source_id not in created:
            raise ValueError(f"{event_type} references missing thread {source_id!r}")
        if target_id not in created:
            raise ValueError(f"{event_type} references missing thread {target_id!r}")
        if created[source_id] > sequence:
            raise ValueError(f"{event_type} references source thread {source_id!r} before creation")
        if created[target_id] > sequence:
            raise ValueError(f"{event_type} references target thread {target_id!r} before creation")

    @staticmethod
    def _assert_acyclic(adjacency: Mapping[str, list[str]], *, relation_name: str) -> None:
        visiting: set[str] = set()
        visited: set[str] = set()

        def visit(thread_id: str) -> None:
            if thread_id in visited:
                return
            if thread_id in visiting:
                raise ValueError(f"{relation_name} cycle detected at {thread_id!r}")
            visiting.add(thread_id)
            for target_id in adjacency.get(thread_id, []):
                visit(target_id)
            visiting.remove(thread_id)
            visited.add(thread_id)

        for thread_id in adjacency:
            visit(thread_id)

    def _snapshot_all_unlocked(self) -> tuple[ProjectedState, ...]:
        rows = self._connection.execute(
            """
            SELECT * FROM thread_state_record
            WHERE created = 1
            ORDER BY created_sequence, thread_id
            """
        ).fetchall()
        return tuple(self._row_to_state(row) for row in rows)

    def _snapshot_one_unlocked(self, thread_id: str) -> ProjectedState:
        row = self._connection.execute(
            "SELECT * FROM thread_state_record WHERE thread_id = ? AND created = 1",
            (thread_id,),
        ).fetchone()
        if row is None:
            raise ValueError(f"thread {thread_id!r} has no THREAD_CREATED event")
        return self._row_to_state(row)

    def _row_to_state(self, row: sqlite3.Row) -> ProjectedState:
        thread_id = str(row["thread_id"])
        try:
            purpose = WorkPurpose(str(row["purpose"]))
        except ValueError as error:
            raise RuntimeError("corrupt indexed thread purpose") from error

        parents = tuple(
            str(item["parent_thread_id"])
            for item in self._connection.execute(
                """
                SELECT parent_thread_id FROM thread_fork
                WHERE child_thread_id = ?
                ORDER BY sequence, parent_thread_id
                """,
                (thread_id,),
            ).fetchall()
        )
        children = tuple(
            str(item["child_thread_id"])
            for item in self._connection.execute(
                """
                SELECT child_thread_id FROM thread_fork
                WHERE parent_thread_id = ?
                ORDER BY sequence, child_thread_id
                """,
                (thread_id,),
            ).fetchall()
        )
        merged_from = tuple(
            str(item["source_thread_id"])
            for item in self._connection.execute(
                """
                SELECT source_thread_id FROM thread_merge
                WHERE target_thread_id = ?
                ORDER BY sequence, source_thread_id
                """,
                (thread_id,),
            ).fetchall()
        )
        dependencies = tuple(
            str(item["dependency_thread_id"])
            for item in self._connection.execute(
                """
                SELECT dependency_thread_id FROM thread_dependency
                WHERE thread_id = ?
                ORDER BY sequence, dependency_thread_id
                """,
                (thread_id,),
            ).fetchall()
        )

        state = ProjectedState(
            revision=int(row["revision"]),
            thread_id=thread_id,
            objective=str(row["objective"]),
            status=str(row["status"]),
            purpose=purpose,
            reference_ids=self._decode_string_tuple(str(row["reference_ids_json"])),
            hypothesis_ids=self._decode_string_tuple(str(row["hypothesis_ids_json"])),
            contradiction_ids=self._decode_string_tuple(str(row["contradiction_ids_json"])),
            open_questions=self._decode_string_tuple(str(row["open_questions_json"])),
            dependency_thread_ids=dependencies,
            parent_thread_ids=parents,
            child_thread_ids=children,
            merged_from_thread_ids=merged_from,
            merged_into_thread_id=row["merged_into_thread_id"],
            metadata=self._decode_mapping(str(row["metadata_json"])),
        )
        state.validate()
        return state

    def _ensure_thread_row(self, thread_id: str) -> None:
        if not thread_id or not thread_id.strip():
            raise ValueError("thread_id must be non-empty")
        empty = self._encode_string_tuple(())
        self._connection.execute(
            """
            INSERT OR IGNORE INTO thread_state_record(
                thread_id,
                created,
                created_sequence,
                objective,
                status,
                purpose,
                revision,
                reference_ids_json,
                hypothesis_ids_json,
                contradiction_ids_json,
                open_questions_json,
                metadata_json,
                merged_into_thread_id,
                last_worker_id,
                last_attempt_sequence
            ) VALUES (?, 0, 0, '', 'ACTIVE', ?, 0, ?, ?, ?, ?, '{}', NULL, NULL, NULL)
            """,
            (
                thread_id,
                WorkPurpose.EXPLORE.value,
                empty,
                empty,
                empty,
                empty,
            ),
        )

    def _thread_row(self, thread_id: str) -> sqlite3.Row:
        row = self._connection.execute(
            "SELECT * FROM thread_state_record WHERE thread_id = ?",
            (thread_id,),
        ).fetchone()
        if row is None:
            raise RuntimeError("indexed thread row disappeared")
        return row

    def _update_revision(self, thread_id: str, sequence: int) -> None:
        self._connection.execute(
            "UPDATE thread_state_record SET revision = MAX(revision, ?) WHERE thread_id = ?",
            (sequence, thread_id),
        )

    def _extend_thread_list(
        self,
        thread_id: str,
        column: str,
        values: Iterable[str],
    ) -> None:
        row = self._thread_row(thread_id)
        current = list(self._decode_string_tuple(str(row[column])))
        present = set(current)
        changed = False
        for value in values:
            if value and value not in present:
                current.append(value)
                present.add(value)
                changed = True
        if changed:
            self._connection.execute(
                f"UPDATE thread_state_record SET {column} = ? WHERE thread_id = ?",
                (self._encode_string_tuple(tuple(current)), thread_id),
            )

    def _remove_thread_list(
        self,
        thread_id: str,
        column: str,
        values: Iterable[str],
    ) -> None:
        removals = set(values)
        if not removals:
            return
        row = self._thread_row(thread_id)
        current = self._decode_string_tuple(str(row[column]))
        reduced = tuple(value for value in current if value not in removals)
        if reduced != current:
            self._connection.execute(
                f"UPDATE thread_state_record SET {column} = ? WHERE thread_id = ?",
                (self._encode_string_tuple(reduced), thread_id),
            )

    def _update_metadata(self, thread_id: str, patch: Mapping[str, Any]) -> None:
        row = self._thread_row(thread_id)
        metadata = self._decode_mapping(str(row["metadata_json"]))
        metadata.update(patch)
        self._connection.execute(
            "UPDATE thread_state_record SET metadata_json = ? WHERE thread_id = ?",
            (self._encode_mapping(metadata), thread_id),
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
                CREATE TABLE IF NOT EXISTS thread_index_meta(
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                )
                """
            )
            self._connection.execute(
                """
                CREATE TABLE IF NOT EXISTS thread_state_record(
                    thread_id TEXT PRIMARY KEY,
                    created INTEGER NOT NULL,
                    created_sequence INTEGER NOT NULL,
                    objective TEXT NOT NULL,
                    status TEXT NOT NULL,
                    purpose TEXT NOT NULL,
                    revision INTEGER NOT NULL,
                    reference_ids_json TEXT NOT NULL,
                    hypothesis_ids_json TEXT NOT NULL,
                    contradiction_ids_json TEXT NOT NULL,
                    open_questions_json TEXT NOT NULL,
                    metadata_json TEXT NOT NULL,
                    merged_into_thread_id TEXT,
                    last_worker_id TEXT,
                    last_attempt_sequence INTEGER
                )
                """
            )
            self._connection.execute(
                """
                CREATE TABLE IF NOT EXISTS thread_dependency(
                    thread_id TEXT NOT NULL,
                    dependency_thread_id TEXT NOT NULL,
                    sequence INTEGER NOT NULL,
                    PRIMARY KEY(thread_id, dependency_thread_id)
                )
                """
            )
            self._connection.execute(
                """
                CREATE TABLE IF NOT EXISTS thread_fork(
                    parent_thread_id TEXT NOT NULL,
                    child_thread_id TEXT NOT NULL,
                    sequence INTEGER NOT NULL,
                    PRIMARY KEY(parent_thread_id, child_thread_id)
                )
                """
            )
            self._connection.execute(
                """
                CREATE TABLE IF NOT EXISTS thread_merge(
                    source_thread_id TEXT PRIMARY KEY,
                    target_thread_id TEXT NOT NULL,
                    sequence INTEGER NOT NULL
                )
                """
            )
            self._connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_thread_state_created ON thread_state_record(created, created_sequence)"
            )
            self._connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_thread_dependency_source ON thread_dependency(thread_id, sequence)"
            )
            self._connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_thread_fork_parent ON thread_fork(parent_thread_id, sequence)"
            )
            self._connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_thread_fork_child ON thread_fork(child_thread_id, sequence)"
            )
            self._connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_thread_merge_target ON thread_merge(target_thread_id, sequence)"
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
                        f"thread index {key} mismatch: {current!r} != {value!r}"
                    )
            if self._meta_optional("revision") is None:
                self._set_meta("revision", "0")
            if self._meta_optional("checkpoint_event_id") is None:
                self._set_meta("checkpoint_event_id", "")

    def _checkpoint_unlocked(self) -> ProjectionCheckpoint:
        try:
            revision = int(self._meta("revision"))
        except ValueError as error:
            raise RuntimeError("corrupt thread index revision") from error
        checkpoint = ProjectionCheckpoint(
            sequence=revision,
            event_id=self._meta("checkpoint_event_id"),
        )
        try:
            checkpoint.validate()
        except ValueError as error:
            raise RuntimeError("corrupt thread index checkpoint") from error
        return checkpoint

    def _set_checkpoint(self, checkpoint: ProjectionCheckpoint) -> None:
        checkpoint.validate()
        self._set_meta("revision", str(checkpoint.sequence))
        self._set_meta("checkpoint_event_id", checkpoint.event_id)

    def _meta_optional(self, key: str) -> str | None:
        row = self._connection.execute(
            "SELECT value FROM thread_index_meta WHERE key = ?",
            (key,),
        ).fetchone()
        return str(row["value"]) if row is not None else None

    def _meta(self, key: str) -> str:
        value = self._meta_optional(key)
        if value is None:
            raise RuntimeError(f"thread index metadata {key!r} is missing")
        return value

    def _set_meta(self, key: str, value: str) -> None:
        self._connection.execute(
            """
            INSERT INTO thread_index_meta(key, value)
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
    def _require_thread_id(event: LedgerEvent) -> str:
        if event.thread_id is None:
            raise ValueError(f"{event.event_type} requires thread_id")
        return event.thread_id

    @staticmethod
    def _encode_string_tuple(values: tuple[str, ...]) -> str:
        return json.dumps(values, ensure_ascii=False, separators=(",", ":"))

    @staticmethod
    def _decode_string_tuple(payload: str) -> tuple[str, ...]:
        value = json.loads(payload)
        if not isinstance(value, list) or any(
            not isinstance(item, str) or not item for item in value
        ):
            raise RuntimeError("corrupt indexed thread string list")
        return tuple(value)

    @staticmethod
    def _encode_mapping(value: Mapping[str, Any]) -> str:
        try:
            return json.dumps(
                dict(value),
                ensure_ascii=False,
                allow_nan=False,
                separators=(",", ":"),
                sort_keys=True,
            )
        except (TypeError, ValueError) as error:
            raise ValueError("thread metadata must be finite JSON-serializable data") from error

    @staticmethod
    def _decode_mapping(payload: str) -> dict[str, Any]:
        value = json.loads(payload)
        if not isinstance(value, dict):
            raise RuntimeError("corrupt indexed thread metadata")
        return value

    def _validate_separate_storage(self) -> None:
        if self.path == ":memory:" or self.ledger.path == ":memory:":
            return
        if (
            Path(self.path).expanduser().resolve()
            == Path(self.ledger.path).expanduser().resolve()
        ):
            raise ValueError(
                "thread index must use rebuildable storage separate from the canonical ledger"
            )
