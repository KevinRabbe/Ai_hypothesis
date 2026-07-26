"""Incremental rebuildable Work Thread / Work Graph materialization.

The Research Ledger remains canonical. This sidecar stores only compact current thread state,
ordered graph edges, and last-worker continuity so scheduler cycles can advance from the ledger
tail rather than refolding historical events.
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
    """Forward-only materialized replacement for ``ThreadStateProjector`` hot queries."""

    def __init__(self, ledger: SQLiteResearchLedger, index_path: str | Path = ":memory:") -> None:
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

    def snapshot_all_through(self, sequence: int) -> tuple[ProjectedState, ...]:
        with self._lock:
            self._sync_to(target_sequence=sequence, page_size=1000)
            return self._snapshot_all_unlocked()

    def snapshot(self, *, thread_id: str) -> ProjectedState:
        self._require_text_id("thread_id", thread_id)
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
        self._require_text_id("thread_id", thread_id)
        history = tuple(events)
        target = history[-1].sequence if history else 0
        with self._lock:
            self._sync_to(target_sequence=target, page_size=1000)
            return self._snapshot_one_unlocked(thread_id)

    def last_worker_id(self, thread_id: str) -> str | None:
        self._require_text_id("thread_id", thread_id)
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
            # One transaction spans the requested tail. This matters because replay allows some
            # relations to become valid only after later events in the same requested snapshot.
            with self._connection:
                for page in pages:
                    for event in page:
                        self._apply_event(event)
                    checkpoint = self.tail.checkpoint_after(page)
                self._validate_graph_unlocked()
                self._set_checkpoint(checkpoint)
            return checkpoint.sequence

    def _apply_event(self, event: LedgerEvent) -> None:
        if event.event_type == "THREAD_FORKED":
            self._apply_fork(event)
        elif event.event_type == "THREAD_MERGED":
            self._apply_merge(event)

        if event.thread_id is None:
            return

        thread_id = event.thread_id
        self._ensure_thread(thread_id)
        state = self._load_state(thread_id)
        state["revision"] = max(int(state["revision"]), event.sequence)
        self._extend_unique(state["references"], event.reference_ids)

        if event.event_type == "THREAD_CREATED":
            row = self._thread_row(thread_id)
            if bool(row["created"]):
                raise ValueError(f"thread {thread_id!r} was created more than once")
            state["objective"] = self._payload_text(event, "objective")
            raw_purpose = event.payload.get("purpose", WorkPurpose.EXPLORE.value)
            try:
                state["purpose"] = WorkPurpose(str(raw_purpose)).value
            except ValueError as error:
                raise ValueError(f"invalid thread purpose {raw_purpose!r}") from error
            status = str(event.payload.get("status", "ACTIVE"))
            if not status:
                raise ValueError("THREAD_CREATED status must be non-empty")
            state["status"] = status
            self._connection.execute(
                "UPDATE thread_state_record SET created = 1, created_sequence = ? WHERE thread_id = ?",
                (event.sequence, thread_id),
            )
        elif event.event_type == "THREAD_PURPOSE_SET":
            state["purpose"] = WorkPurpose(self._payload_text(event, "purpose")).value
        elif event.event_type == "THREAD_STATUS_SET":
            state["status"] = self._payload_text(event, "status")
        elif event.event_type == "THREAD_PAUSED":
            state["status"] = "PAUSED"
        elif event.event_type == "THREAD_COMPLETED":
            state["status"] = "COMPLETE"
        elif event.event_type == "HYPOTHESIS_PROPOSED":
            self._extend_unique(state["hypotheses"], event.reference_ids)
        elif event.event_type == "HYPOTHESIS_REJECTED":
            self._remove_values(state["hypotheses"], event.reference_ids)
        elif event.event_type == "CONTRADICTION_FOUND":
            self._extend_unique(state["contradictions"], event.reference_ids)
        elif event.event_type == "CONTRADICTION_RESOLVED":
            self._remove_values(state["contradictions"], event.reference_ids)
        elif event.event_type == "OPEN_QUESTION_ADDED":
            self._extend_unique(state["open_questions"], (self._payload_text(event, "question"),))
        elif event.event_type == "OPEN_QUESTION_RESOLVED":
            self._remove_values(state["open_questions"], (self._payload_text(event, "question"),))
        elif event.event_type == "DEPENDENCY_ADDED":
            self._apply_dependency(event, add=True)
        elif event.event_type == "DEPENDENCY_REMOVED":
            self._apply_dependency(event, add=False)
        elif event.event_type == "THREAD_METADATA_UPDATED":
            state["metadata"].update(event.payload)
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
                (worker_id, event.sequence, thread_id),
            )

        self._save_state(thread_id, state)

    def _apply_dependency(self, event: LedgerEvent, *, add: bool) -> None:
        assert event.thread_id is not None
        for dependency_id in event.reference_ids:
            self._require_text_id("dependency_thread_id", dependency_id)
            if add:
                # Duplicate ADD is idempotent and keeps first insertion order, matching replay.
                self._connection.execute(
                    """
                    INSERT OR IGNORE INTO thread_dependency(
                        thread_id, dependency_thread_id, sequence
                    ) VALUES (?, ?, ?)
                    """,
                    (event.thread_id, dependency_id, event.sequence),
                )
            else:
                self._connection.execute(
                    "DELETE FROM thread_dependency WHERE thread_id = ? AND dependency_thread_id = ?",
                    (event.thread_id, dependency_id),
                )

    def _apply_fork(self, event: LedgerEvent) -> None:
        parent_id = self._relation_source(event)
        if not event.reference_ids:
            raise ValueError("THREAD_FORKED requires at least one target thread ID")
        self._ensure_thread(parent_id)
        for child_id in event.reference_ids:
            self._require_text_id("child_thread_id", child_id)
            self._ensure_thread(child_id)
            self._connection.execute(
                """
                INSERT OR IGNORE INTO thread_fork(parent_thread_id, child_thread_id, sequence)
                VALUES (?, ?, ?)
                """,
                (parent_id, child_id, event.sequence),
            )
            self._advance_revision(parent_id, event.sequence)
            self._advance_revision(child_id, event.sequence)

    def _apply_merge(self, event: LedgerEvent) -> None:
        target_id = self._relation_source(event)
        if not event.reference_ids:
            raise ValueError("THREAD_MERGED requires at least one source thread ID")
        self._ensure_thread(target_id)
        for source_id in event.reference_ids:
            self._require_text_id("source_thread_id", source_id)
            self._ensure_thread(source_id)
            existing = self._connection.execute(
                "SELECT target_thread_id FROM thread_merge WHERE source_thread_id = ?",
                (source_id,),
            ).fetchone()
            if existing is not None and str(existing["target_thread_id"]) != target_id:
                raise ValueError(f"thread {source_id!r} was merged into multiple targets")
            self._connection.execute(
                """
                INSERT OR IGNORE INTO thread_merge(source_thread_id, target_thread_id, sequence)
                VALUES (?, ?, ?)
                """,
                (source_id, target_id, event.sequence),
            )
            source = self._load_state(source_id)
            source["merged_into"] = target_id
            source["status"] = "COMPLETE"
            source["revision"] = max(int(source["revision"]), event.sequence)
            self._save_state(source_id, source)
            self._advance_revision(target_id, event.sequence)

    def _validate_graph_unlocked(self) -> None:
        created_rows = self._connection.execute(
            "SELECT thread_id, created_sequence FROM thread_state_record WHERE created = 1"
        ).fetchall()
        created = {str(row["thread_id"]): int(row["created_sequence"]) for row in created_rows}

        dependencies: dict[str, list[str]] = {thread_id: [] for thread_id in created}
        for row in self._connection.execute(
            "SELECT thread_id, dependency_thread_id FROM thread_dependency ORDER BY sequence"
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
            "SELECT parent_thread_id, child_thread_id, sequence FROM thread_fork ORDER BY sequence"
        ).fetchall():
            parent_id = str(row["parent_thread_id"])
            child_id = str(row["child_thread_id"])
            sequence = int(row["sequence"])
            self._validate_relation_creation(created, parent_id, child_id, sequence, "THREAD_FORKED")
            if parent_id == child_id:
                raise ValueError("thread cannot fork itself")
            forks[parent_id].append(child_id)

        for row in self._connection.execute(
            "SELECT source_thread_id, target_thread_id, sequence FROM thread_merge ORDER BY sequence"
        ).fetchall():
            source_id = str(row["source_thread_id"])
            target_id = str(row["target_thread_id"])
            sequence = int(row["sequence"])
            self._validate_relation_creation(created, source_id, target_id, sequence, "THREAD_MERGED")
            if source_id == target_id:
                raise ValueError("thread cannot merge into itself")

        self._assert_acyclic(forks, relation_name="fork ancestry")
        self._assert_acyclic(dependencies, relation_name="dependency")

    def _snapshot_all_unlocked(self) -> tuple[ProjectedState, ...]:
        rows = self._connection.execute(
            """
            SELECT * FROM thread_state_record
            WHERE created = 1
            ORDER BY created_sequence, thread_id
            """
        ).fetchall()
        return tuple(self._row_to_projected(row) for row in rows)

    def _snapshot_one_unlocked(self, thread_id: str) -> ProjectedState:
        row = self._connection.execute(
            "SELECT * FROM thread_state_record WHERE thread_id = ? AND created = 1",
            (thread_id,),
        ).fetchone()
        if row is None:
            raise ValueError(f"thread {thread_id!r} has no THREAD_CREATED event")
        return self._row_to_projected(row)

    def _row_to_projected(self, row: sqlite3.Row) -> ProjectedState:
        thread_id = str(row["thread_id"])
        state = self._decode_state(str(row["state_json"]))
        try:
            purpose = WorkPurpose(str(state["purpose"]))
        except ValueError as error:
            raise RuntimeError("corrupt indexed thread purpose") from error

        projected = ProjectedState(
            revision=int(state["revision"]),
            thread_id=thread_id,
            objective=str(state["objective"]),
            status=str(state["status"]),
            purpose=purpose,
            reference_ids=tuple(state["references"]),
            hypothesis_ids=tuple(state["hypotheses"]),
            contradiction_ids=tuple(state["contradictions"]),
            open_questions=tuple(state["open_questions"]),
            dependency_thread_ids=self._edge_targets(
                "thread_dependency", "dependency_thread_id", "thread_id", thread_id
            ),
            parent_thread_ids=self._edge_targets(
                "thread_fork", "parent_thread_id", "child_thread_id", thread_id
            ),
            child_thread_ids=self._edge_targets(
                "thread_fork", "child_thread_id", "parent_thread_id", thread_id
            ),
            merged_from_thread_ids=self._edge_targets(
                "thread_merge", "source_thread_id", "target_thread_id", thread_id
            ),
            merged_into_thread_id=state["merged_into"],
            metadata=dict(state["metadata"]),
        )
        projected.validate()
        return projected

    def _edge_targets(
        self,
        table: str,
        target_column: str,
        source_column: str,
        source_id: str,
    ) -> tuple[str, ...]:
        rows = self._connection.execute(
            f"SELECT {target_column} FROM {table} WHERE {source_column} = ? "
            f"ORDER BY sequence, {target_column}",
            (source_id,),
        ).fetchall()
        return tuple(str(row[target_column]) for row in rows)

    def _ensure_thread(self, thread_id: str) -> None:
        self._require_text_id("thread_id", thread_id)
        self._connection.execute(
            """
            INSERT OR IGNORE INTO thread_state_record(
                thread_id, created, created_sequence, state_json, last_worker_id, last_attempt_sequence
            ) VALUES (?, 0, 0, ?, NULL, NULL)
            """,
            (thread_id, self._encode_state(self._empty_state())),
        )

    def _thread_row(self, thread_id: str) -> sqlite3.Row:
        row = self._connection.execute(
            "SELECT * FROM thread_state_record WHERE thread_id = ?",
            (thread_id,),
        ).fetchone()
        if row is None:
            raise RuntimeError("indexed thread row disappeared")
        return row

    def _load_state(self, thread_id: str) -> dict[str, Any]:
        return self._decode_state(str(self._thread_row(thread_id)["state_json"]))

    def _save_state(self, thread_id: str, state: Mapping[str, Any]) -> None:
        self._connection.execute(
            "UPDATE thread_state_record SET state_json = ? WHERE thread_id = ?",
            (self._encode_state(state), thread_id),
        )

    def _advance_revision(self, thread_id: str, sequence: int) -> None:
        state = self._load_state(thread_id)
        state["revision"] = max(int(state["revision"]), sequence)
        self._save_state(thread_id, state)

    @staticmethod
    def _empty_state() -> dict[str, Any]:
        return {
            "revision": 0,
            "objective": "",
            "status": "ACTIVE",
            "purpose": WorkPurpose.EXPLORE.value,
            "references": [],
            "hypotheses": [],
            "contradictions": [],
            "open_questions": [],
            "metadata": {},
            "merged_into": None,
        }

    @staticmethod
    def _extend_unique(target: list[str], values: Iterable[str]) -> None:
        present = set(target)
        for value in values:
            if value and value not in present:
                target.append(value)
                present.add(value)

    @staticmethod
    def _remove_values(target: list[str], values: Iterable[str]) -> None:
        removals = set(values)
        if removals:
            target[:] = [value for value in target if value not in removals]

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

    def _configure(self) -> None:
        with self._connection:
            self._connection.execute("PRAGMA foreign_keys = ON")
            self._connection.execute("PRAGMA busy_timeout = 30000")
            self._connection.execute("PRAGMA synchronous = NORMAL")
            self._connection.execute("PRAGMA journal_mode = WAL")

    def _create_schema(self) -> None:
        with self._connection:
            self._connection.execute(
                "CREATE TABLE IF NOT EXISTS thread_index_meta(key TEXT PRIMARY KEY, value TEXT NOT NULL)"
            )
            self._connection.execute(
                """
                CREATE TABLE IF NOT EXISTS thread_state_record(
                    thread_id TEXT PRIMARY KEY,
                    created INTEGER NOT NULL,
                    created_sequence INTEGER NOT NULL,
                    state_json TEXT NOT NULL,
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
                "CREATE INDEX IF NOT EXISTS idx_thread_created ON thread_state_record(created, created_sequence)"
            )
            self._connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_thread_dependency ON thread_dependency(thread_id, sequence)"
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
                    raise RuntimeError(f"thread index {key} mismatch: {current!r} != {value!r}")
            if self._meta_optional("revision") is None:
                self._set_meta("revision", "0")
            if self._meta_optional("checkpoint_event_id") is None:
                self._set_meta("checkpoint_event_id", "")

    def _checkpoint_unlocked(self) -> ProjectionCheckpoint:
        try:
            revision = int(self._meta("revision"))
        except ValueError as error:
            raise RuntimeError("corrupt thread index revision") from error
        checkpoint = ProjectionCheckpoint(revision, self._meta("checkpoint_event_id"))
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
    def _relation_source(event: LedgerEvent) -> str:
        if event.thread_id is None:
            raise ValueError(f"{event.event_type} requires thread_id")
        return event.thread_id

    @staticmethod
    def _require_text_id(name: str, value: str) -> None:
        if not value or not value.strip():
            raise ValueError(f"{name} must be non-empty")

    @staticmethod
    def _encode_state(state: Mapping[str, Any]) -> str:
        try:
            return json.dumps(
                dict(state),
                ensure_ascii=False,
                allow_nan=False,
                separators=(",", ":"),
                sort_keys=True,
            )
        except (TypeError, ValueError) as error:
            raise ValueError("indexed thread state must be finite JSON-serializable data") from error

    @classmethod
    def _decode_state(cls, payload: str) -> dict[str, Any]:
        value = json.loads(payload)
        if not isinstance(value, dict):
            raise RuntimeError("corrupt indexed thread state")
        required_lists = ("references", "hypotheses", "contradictions", "open_questions")
        for key in required_lists:
            items = value.get(key)
            if not isinstance(items, list) or any(not isinstance(item, str) or not item for item in items):
                raise RuntimeError(f"corrupt indexed thread state list {key!r}")
        metadata = value.get("metadata")
        if not isinstance(metadata, dict):
            raise RuntimeError("corrupt indexed thread metadata")
        for key in ("objective", "status", "purpose"):
            if not isinstance(value.get(key), str):
                raise RuntimeError(f"corrupt indexed thread field {key!r}")
        revision = value.get("revision")
        if isinstance(revision, bool) or not isinstance(revision, int) or revision < 0:
            raise RuntimeError("corrupt indexed thread revision")
        merged_into = value.get("merged_into")
        if merged_into is not None and (not isinstance(merged_into, str) or not merged_into):
            raise RuntimeError("corrupt indexed merged target")
        return value

    def _validate_separate_storage(self) -> None:
        if self.path == ":memory:" or self.ledger.path == ":memory:":
            return
        if Path(self.path).expanduser().resolve() == Path(self.ledger.path).expanduser().resolve():
            raise ValueError(
                "thread index must use rebuildable storage separate from the canonical ledger"
            )
