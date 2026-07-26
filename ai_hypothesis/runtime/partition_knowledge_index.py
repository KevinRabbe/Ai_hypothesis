"""Incremental partition -> attempt -> knowledge lineage over the canonical Research Ledger.

This materialized view stores only durable routing relationships needed by higher-level
consolidation. Knowledge status remains owned by Knowledge State; evidence payloads remain owned
by their original ledger events. The index is disposable and fully rebuildable.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from .contracts import LedgerEvent
from .ledger import SQLiteResearchLedger
from .projection_tail import LedgerProjectionTail, ProjectionCheckpoint


_INDEX_SCHEMA_VERSION = "partition-knowledge-lineage-index-v0"
_ALLOCATION_EVENT = "INTEGRATION_PARTITION_ALLOCATION_RECORDED"
_ALLOCATION_SCHEMA = "integration-partition-allocation-v0"


@dataclass(frozen=True, slots=True)
class IndexedHistoricalPartition:
    decision_id: str
    partition_id: str
    ordinal: int
    shard_index: int
    backlog_count: int
    oldest_pending_sequence: int
    evidence_ids: tuple[str, ...]
    attempt_id: str | None


@dataclass(frozen=True, slots=True)
class IndexedPartitionKnowledgeSource:
    thread_id: str
    decision_id: str
    partition_id: str
    attempt_id: str
    delta_id: str
    created_sequence: int


@dataclass(frozen=True, slots=True)
class IndexedPartitionDecisionLineage:
    decision_id: str
    thread_id: str
    decision_sequence: int
    projection_revision: int
    width: int
    provenance_event_id: str | None
    provenance_sequence: int | None
    partition_plan_revision: int | None
    shard_count: int | None
    batch_limit: int | None
    partitions: tuple[IndexedHistoricalPartition, ...]
    sources: tuple[IndexedPartitionKnowledgeSource, ...]

    @property
    def provenance_complete(self) -> bool:
        return self.provenance_event_id is not None

    @property
    def unstarted_partition_ids(self) -> tuple[str, ...]:
        return tuple(
            partition.partition_id
            for partition in self.partitions
            if partition.attempt_id is None
        )


@dataclass(frozen=True, slots=True)
class IndexedPartitionKnowledgeSnapshot:
    revision: int
    decisions: tuple[IndexedPartitionDecisionLineage, ...]
    missing_provenance_decision_ids: tuple[str, ...]

    @property
    def provenance_complete(self) -> bool:
        return not self.missing_provenance_decision_ids

    def require_complete(self) -> "IndexedPartitionKnowledgeSnapshot":
        if self.missing_provenance_decision_ids:
            raise ValueError(
                "partitioned integration history is missing durable allocation provenance"
            )
        return self

    def sources_for_thread(self, thread_id: str) -> tuple[IndexedPartitionKnowledgeSource, ...]:
        if not thread_id or not thread_id.strip():
            raise ValueError("thread_id must be non-empty")
        return tuple(
            source
            for decision in self.decisions
            if decision.thread_id == thread_id
            for source in decision.sources
        )

    def missing_provenance_for_thread(self, thread_id: str) -> tuple[str, ...]:
        if not thread_id or not thread_id.strip():
            raise ValueError("thread_id must be non-empty")
        missing = set(self.missing_provenance_decision_ids)
        return tuple(
            decision.decision_id
            for decision in self.decisions
            if decision.thread_id == thread_id and decision.decision_id in missing
        )


class SQLiteIndexedPartitionKnowledgeLineage:
    """Forward-only materialized partition lineage for consolidation and telemetry."""

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
                self._connection.execute("DELETE FROM partition_knowledge_source")
                self._connection.execute("DELETE FROM partition_attempt")
                self._connection.execute("DELETE FROM partition_assignment")
                self._connection.execute("DELETE FROM partition_decision")
                self._set_checkpoint(ProjectionCheckpoint())
            return self._sync_to(target_sequence=None, page_size=page_size)

    def snapshot(self) -> IndexedPartitionKnowledgeSnapshot:
        with self._lock:
            self.sync()
            return self._snapshot_unlocked()

    def snapshot_through(self, sequence: int) -> IndexedPartitionKnowledgeSnapshot:
        with self._lock:
            self.sync_through(sequence)
            return self._snapshot_unlocked()

    def close(self) -> None:
        with self._lock:
            self._connection.close()

    def __enter__(self) -> "SQLiteIndexedPartitionKnowledgeLineage":
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
                self._set_checkpoint(checkpoint)
            return checkpoint.sequence

    def _apply_event(self, event: LedgerEvent) -> None:
        if event.event_type == "SCHEDULER_DECISION_RECORDED":
            self._apply_decision(event)
        elif event.event_type == _ALLOCATION_EVENT:
            self._apply_provenance(event)
        elif event.event_type == "ATTEMPT_STARTED":
            self._apply_attempt_started(event)
        elif event.event_type == "KNOWLEDGE_DELTA_RECORDED":
            self._apply_knowledge_delta(event)

    def _apply_decision(self, event: LedgerEvent) -> None:
        action = event.payload.get("action")
        raw_reasons = event.payload.get("reason_codes")
        if action != "SYNTHESIZE" or not isinstance(raw_reasons, list):
            return
        if any(not isinstance(reason, str) for reason in raw_reasons):
            raise ValueError("scheduler reason_codes must be a string list")
        reasons = tuple(raw_reasons)
        if "BACKPRESSURE" not in reasons or "PARTITIONED_INTEGRATION" not in reasons:
            return

        decision_id = self._text(event.payload, "decision_id", "partitioned scheduler decision")
        if event.thread_id is None:
            raise ValueError("partitioned scheduler decision is missing thread_id")
        width = self._positive_int(event.payload, "width", "partitioned scheduler decision")
        projection_revision = self._non_negative_int(
            event.payload,
            "projection_revision",
            "partitioned scheduler decision",
        )
        try:
            self._connection.execute(
                """
                INSERT INTO partition_decision(
                    decision_id, thread_id, decision_sequence, projection_revision, width,
                    provenance_event_id, provenance_sequence, partition_plan_revision,
                    shard_count, batch_limit
                ) VALUES (?, ?, ?, ?, ?, NULL, NULL, NULL, NULL, NULL)
                """,
                (
                    decision_id,
                    event.thread_id,
                    event.sequence,
                    projection_revision,
                    width,
                ),
            )
        except sqlite3.IntegrityError as error:
            raise ValueError(
                "partitioned scheduler decision was recorded more than once"
            ) from error

    def _apply_provenance(self, event: LedgerEvent) -> None:
        payload = event.payload
        if payload.get("schema") != _ALLOCATION_SCHEMA:
            raise ValueError("invalid partition allocation provenance schema")
        decision_id = self._text(payload, "decision_id", "partition allocation provenance")
        decision = self._decision_row(decision_id)
        if decision is None:
            raise ValueError(
                "partition allocation provenance references a non-partitioned decision"
            )
        if event.thread_id != str(decision["thread_id"]):
            raise ValueError("partition allocation thread does not match scheduler decision")
        if event.sequence <= int(decision["decision_sequence"]):
            raise ValueError("partition allocation provenance precedes scheduler decision")
        if decision["provenance_event_id"] is not None:
            raise ValueError("partition allocation provenance was recorded more than once")
        existing_start = self._connection.execute(
            """
            SELECT MIN(started_sequence) AS first_start
            FROM partition_attempt
            WHERE decision_id = ?
            """,
            (decision_id,),
        ).fetchone()["first_start"]
        if existing_start is not None:
            raise ValueError(
                "partition allocation provenance was recorded after ATTEMPT_STARTED"
            )

        plan_revision = self._non_negative_int(
            payload,
            "partition_plan_revision",
            "partition allocation provenance",
        )
        shard_count = self._positive_int(
            payload,
            "shard_count",
            "partition allocation provenance",
        )
        batch_limit = self._positive_int(
            payload,
            "batch_limit",
            "partition allocation provenance",
        )
        width = self._positive_int(payload, "width", "partition allocation provenance")
        if width != int(decision["width"]):
            raise ValueError("partition allocation width does not match scheduler decision")
        raw_partitions = payload.get("partitions")
        if not isinstance(raw_partitions, list) or len(raw_partitions) != width:
            raise ValueError("partition allocation provenance has invalid partition list")

        seen_partition_ids: set[str] = set()
        seen_evidence_ids: set[str] = set()
        partition_ids: list[str] = []
        for ordinal, raw in enumerate(raw_partitions):
            if not isinstance(raw, Mapping):
                raise ValueError("partition allocation entry must be an object")
            partition_id = self._text(raw, "partition_id", "partition allocation entry")
            shard_index = self._non_negative_int(
                raw,
                "shard_index",
                "partition allocation entry",
            )
            backlog_count = self._positive_int(
                raw,
                "backlog_count",
                "partition allocation entry",
            )
            oldest = self._non_negative_int(
                raw,
                "oldest_pending_sequence",
                "partition allocation entry",
            )
            evidence_ids = self._string_tuple(raw, "evidence_ids", "partition allocation entry")
            if not evidence_ids:
                raise ValueError("partition allocation entry must assign evidence")
            if len(evidence_ids) > batch_limit:
                raise ValueError("partition allocation entry exceeds recorded batch limit")
            if shard_index >= shard_count:
                raise ValueError("partition allocation shard index exceeds shard count")
            if partition_id in seen_partition_ids:
                raise ValueError("partition allocation contains duplicate partition IDs")
            if seen_evidence_ids.intersection(evidence_ids):
                raise ValueError("partition allocation contains overlapping evidence authority")
            seen_partition_ids.add(partition_id)
            seen_evidence_ids.update(evidence_ids)
            partition_ids.append(partition_id)
            self._connection.execute(
                """
                INSERT INTO partition_assignment(
                    decision_id, partition_id, ordinal, shard_index, backlog_count,
                    oldest_pending_sequence, evidence_ids_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    decision_id,
                    partition_id,
                    ordinal,
                    shard_index,
                    backlog_count,
                    oldest,
                    self._encode_tuple(evidence_ids),
                ),
            )

        if tuple(event.reference_ids) != tuple(partition_ids):
            raise ValueError(
                "partition allocation event references do not match payload partition order"
            )
        self._connection.execute(
            """
            UPDATE partition_decision
            SET provenance_event_id = ?, provenance_sequence = ?,
                partition_plan_revision = ?, shard_count = ?, batch_limit = ?
            WHERE decision_id = ?
            """,
            (
                event.event_id,
                event.sequence,
                plan_revision,
                shard_count,
                batch_limit,
                decision_id,
            ),
        )

    def _apply_attempt_started(self, event: LedgerEvent) -> None:
        decision_id = event.payload.get("scheduler_decision_id")
        if not isinstance(decision_id, str):
            return
        decision = self._decision_row(decision_id)
        if decision is None:
            return
        if event.attempt_id is None:
            raise ValueError("partitioned ATTEMPT_STARTED is missing attempt_id")
        if event.sequence <= int(decision["decision_sequence"]):
            raise ValueError("partitioned ATTEMPT_STARTED precedes scheduler decision")
        if event.thread_id != str(decision["thread_id"]):
            raise ValueError("partitioned attempt thread does not match scheduler decision")
        worker_id = event.payload.get("worker_id")
        if not isinstance(worker_id, str) or not worker_id:
            raise ValueError("partitioned ATTEMPT_STARTED is missing worker_id")
        started_count = int(
            self._connection.execute(
                "SELECT COUNT(*) AS count FROM partition_attempt WHERE decision_id = ?",
                (decision_id,),
            ).fetchone()["count"]
        )
        if started_count >= int(decision["width"]):
            raise ValueError(
                "partitioned scheduler decision started more attempts than allocated width"
            )

        partition_id: str | None = None
        if decision["provenance_event_id"] is not None:
            matches = []
            for row in self._connection.execute(
                """
                SELECT partition_id, evidence_ids_json
                FROM partition_assignment
                WHERE decision_id = ?
                ORDER BY ordinal
                """,
                (decision_id,),
            ).fetchall():
                if self._decode_tuple(str(row["evidence_ids_json"])) == tuple(event.reference_ids):
                    matches.append(str(row["partition_id"]))
            if len(matches) != 1:
                raise ValueError(
                    "partitioned attempt input does not match durable partition allocation"
                )
            partition_id = matches[0]
            already_started = self._connection.execute(
                """
                SELECT attempt_id FROM partition_attempt
                WHERE decision_id = ? AND partition_id = ?
                """,
                (decision_id, partition_id),
            ).fetchone()
            if already_started is not None:
                raise ValueError(
                    "one durable integration partition was started more than once"
                )

        try:
            self._connection.execute(
                """
                INSERT INTO partition_attempt(
                    attempt_id, decision_id, partition_id, worker_id, started_sequence,
                    input_reference_ids_json
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    event.attempt_id,
                    decision_id,
                    partition_id,
                    worker_id,
                    event.sequence,
                    self._encode_tuple(tuple(event.reference_ids)),
                ),
            )
        except sqlite3.IntegrityError as error:
            raise ValueError("partitioned attempt ID was started more than once") from error

    def _apply_knowledge_delta(self, event: LedgerEvent) -> None:
        if event.attempt_id is None:
            return
        attempt = self._connection.execute(
            """
            SELECT attempt_id, decision_id, partition_id
            FROM partition_attempt
            WHERE attempt_id = ?
            """,
            (event.attempt_id,),
        ).fetchone()
        if attempt is None:
            return
        delta_id = self._text(event.payload, "delta_id", "KNOWLEDGE_DELTA_RECORDED")
        if attempt["partition_id"] is None:
            # Legacy partitioned attempts without durable allocation provenance remain unmapped.
            return
        decision = self._decision_row(str(attempt["decision_id"]))
        assert decision is not None
        if event.thread_id != str(decision["thread_id"]):
            raise ValueError("partition-produced knowledge targets another Work Thread")
        started_sequence = int(
            self._connection.execute(
                "SELECT started_sequence FROM partition_attempt WHERE attempt_id = ?",
                (event.attempt_id,),
            ).fetchone()["started_sequence"]
        )
        if event.sequence <= started_sequence:
            raise ValueError("partition-produced knowledge precedes ATTEMPT_STARTED")
        try:
            self._connection.execute(
                """
                INSERT INTO partition_knowledge_source(
                    delta_id, attempt_id, decision_id, partition_id, thread_id, created_sequence
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    delta_id,
                    event.attempt_id,
                    str(attempt["decision_id"]),
                    str(attempt["partition_id"]),
                    event.thread_id,
                    event.sequence,
                ),
            )
        except sqlite3.IntegrityError as error:
            raise ValueError(
                "one knowledge delta is attributed to multiple integration partitions"
            ) from error

    def _snapshot_unlocked(self) -> IndexedPartitionKnowledgeSnapshot:
        decision_rows = self._connection.execute(
            "SELECT * FROM partition_decision ORDER BY decision_sequence, decision_id"
        ).fetchall()
        decisions: list[IndexedPartitionDecisionLineage] = []
        missing: list[str] = []
        for decision in decision_rows:
            decision_id = str(decision["decision_id"])
            complete = decision["provenance_event_id"] is not None
            if not complete:
                missing.append(decision_id)
            partitions: list[IndexedHistoricalPartition] = []
            if complete:
                for row in self._connection.execute(
                    """
                    SELECT a.*,
                           p.attempt_id
                    FROM partition_assignment AS a
                    LEFT JOIN partition_attempt AS p
                      ON p.decision_id = a.decision_id
                     AND p.partition_id = a.partition_id
                    WHERE a.decision_id = ?
                    ORDER BY a.ordinal
                    """,
                    (decision_id,),
                ).fetchall():
                    partitions.append(
                        IndexedHistoricalPartition(
                            decision_id=decision_id,
                            partition_id=str(row["partition_id"]),
                            ordinal=int(row["ordinal"]),
                            shard_index=int(row["shard_index"]),
                            backlog_count=int(row["backlog_count"]),
                            oldest_pending_sequence=int(row["oldest_pending_sequence"]),
                            evidence_ids=self._decode_tuple(str(row["evidence_ids_json"])),
                            attempt_id=(
                                str(row["attempt_id"])
                                if row["attempt_id"] is not None
                                else None
                            ),
                        )
                    )
            source_rows = self._connection.execute(
                """
                SELECT * FROM partition_knowledge_source
                WHERE decision_id = ?
                ORDER BY created_sequence, delta_id
                """,
                (decision_id,),
            ).fetchall()
            sources = tuple(
                IndexedPartitionKnowledgeSource(
                    thread_id=str(row["thread_id"]),
                    decision_id=decision_id,
                    partition_id=str(row["partition_id"]),
                    attempt_id=str(row["attempt_id"]),
                    delta_id=str(row["delta_id"]),
                    created_sequence=int(row["created_sequence"]),
                )
                for row in source_rows
            )
            decisions.append(
                IndexedPartitionDecisionLineage(
                    decision_id=decision_id,
                    thread_id=str(decision["thread_id"]),
                    decision_sequence=int(decision["decision_sequence"]),
                    projection_revision=int(decision["projection_revision"]),
                    width=int(decision["width"]),
                    provenance_event_id=(
                        str(decision["provenance_event_id"])
                        if decision["provenance_event_id"] is not None
                        else None
                    ),
                    provenance_sequence=(
                        int(decision["provenance_sequence"])
                        if decision["provenance_sequence"] is not None
                        else None
                    ),
                    partition_plan_revision=(
                        int(decision["partition_plan_revision"])
                        if decision["partition_plan_revision"] is not None
                        else None
                    ),
                    shard_count=(
                        int(decision["shard_count"])
                        if decision["shard_count"] is not None
                        else None
                    ),
                    batch_limit=(
                        int(decision["batch_limit"])
                        if decision["batch_limit"] is not None
                        else None
                    ),
                    partitions=tuple(partitions),
                    sources=sources,
                )
            )
        return IndexedPartitionKnowledgeSnapshot(
            revision=self.revision,
            decisions=tuple(decisions),
            missing_provenance_decision_ids=tuple(missing),
        )

    def _decision_row(self, decision_id: str) -> sqlite3.Row | None:
        return self._connection.execute(
            "SELECT * FROM partition_decision WHERE decision_id = ?",
            (decision_id,),
        ).fetchone()

    def _configure(self) -> None:
        with self._connection:
            self._connection.execute("PRAGMA foreign_keys = ON")
            self._connection.execute("PRAGMA busy_timeout = 30000")
            self._connection.execute("PRAGMA synchronous = NORMAL")
            self._connection.execute("PRAGMA journal_mode = WAL")

    def _create_schema(self) -> None:
        with self._connection:
            self._connection.execute(
                "CREATE TABLE IF NOT EXISTS partition_lineage_meta(key TEXT PRIMARY KEY, value TEXT NOT NULL)"
            )
            self._connection.execute(
                """
                CREATE TABLE IF NOT EXISTS partition_decision(
                    decision_id TEXT PRIMARY KEY,
                    thread_id TEXT NOT NULL,
                    decision_sequence INTEGER NOT NULL,
                    projection_revision INTEGER NOT NULL,
                    width INTEGER NOT NULL,
                    provenance_event_id TEXT,
                    provenance_sequence INTEGER,
                    partition_plan_revision INTEGER,
                    shard_count INTEGER,
                    batch_limit INTEGER
                )
                """
            )
            self._connection.execute(
                """
                CREATE TABLE IF NOT EXISTS partition_assignment(
                    decision_id TEXT NOT NULL,
                    partition_id TEXT NOT NULL,
                    ordinal INTEGER NOT NULL,
                    shard_index INTEGER NOT NULL,
                    backlog_count INTEGER NOT NULL,
                    oldest_pending_sequence INTEGER NOT NULL,
                    evidence_ids_json TEXT NOT NULL,
                    PRIMARY KEY(decision_id, partition_id),
                    FOREIGN KEY(decision_id) REFERENCES partition_decision(decision_id) ON DELETE CASCADE
                )
                """
            )
            self._connection.execute(
                """
                CREATE TABLE IF NOT EXISTS partition_attempt(
                    attempt_id TEXT PRIMARY KEY,
                    decision_id TEXT NOT NULL,
                    partition_id TEXT,
                    worker_id TEXT NOT NULL,
                    started_sequence INTEGER NOT NULL,
                    input_reference_ids_json TEXT NOT NULL,
                    FOREIGN KEY(decision_id) REFERENCES partition_decision(decision_id) ON DELETE CASCADE
                )
                """
            )
            self._connection.execute(
                """
                CREATE UNIQUE INDEX IF NOT EXISTS idx_partition_attempt_once
                ON partition_attempt(decision_id, partition_id)
                WHERE partition_id IS NOT NULL
                """
            )
            self._connection.execute(
                """
                CREATE TABLE IF NOT EXISTS partition_knowledge_source(
                    delta_id TEXT PRIMARY KEY,
                    attempt_id TEXT NOT NULL,
                    decision_id TEXT NOT NULL,
                    partition_id TEXT NOT NULL,
                    thread_id TEXT NOT NULL,
                    created_sequence INTEGER NOT NULL,
                    FOREIGN KEY(attempt_id) REFERENCES partition_attempt(attempt_id) ON DELETE CASCADE
                )
                """
            )
            self._connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_partition_decision_thread ON partition_decision(thread_id, decision_sequence)"
            )
            self._connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_partition_source_thread ON partition_knowledge_source(thread_id, created_sequence, delta_id)"
            )
            self._connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_partition_source_partition ON partition_knowledge_source(decision_id, partition_id, created_sequence)"
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
                        f"partition lineage index {key} mismatch: {current!r} != {value!r}"
                    )
            if self._meta_optional("revision") is None:
                self._set_meta("revision", "0")
            if self._meta_optional("checkpoint_event_id") is None:
                self._set_meta("checkpoint_event_id", "")

    def _checkpoint_unlocked(self) -> ProjectionCheckpoint:
        try:
            revision = int(self._meta("revision"))
        except ValueError as error:
            raise RuntimeError("corrupt partition lineage index revision") from error
        checkpoint = ProjectionCheckpoint(
            sequence=revision,
            event_id=self._meta("checkpoint_event_id"),
        )
        try:
            checkpoint.validate()
        except ValueError as error:
            raise RuntimeError("corrupt partition lineage index checkpoint") from error
        return checkpoint

    def _set_checkpoint(self, checkpoint: ProjectionCheckpoint) -> None:
        checkpoint.validate()
        self._set_meta("revision", str(checkpoint.sequence))
        self._set_meta("checkpoint_event_id", checkpoint.event_id)

    def _meta_optional(self, key: str) -> str | None:
        row = self._connection.execute(
            "SELECT value FROM partition_lineage_meta WHERE key = ?",
            (key,),
        ).fetchone()
        return str(row["value"]) if row is not None else None

    def _meta(self, key: str) -> str:
        value = self._meta_optional(key)
        if value is None:
            raise RuntimeError(f"partition lineage metadata {key!r} is missing")
        return value

    def _set_meta(self, key: str, value: str) -> None:
        self._connection.execute(
            """
            INSERT INTO partition_lineage_meta(key, value)
            VALUES (?, ?)
            ON CONFLICT(key) DO UPDATE SET value = excluded.value
            """,
            (key, value),
        )

    @staticmethod
    def _text(payload: Mapping[str, object], key: str, label: str) -> str:
        value = payload.get(key)
        if not isinstance(value, str) or not value:
            raise ValueError(f"{label} has invalid {key}")
        return value

    @staticmethod
    def _positive_int(payload: Mapping[str, object], key: str, label: str) -> int:
        value = payload.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{label} has invalid {key}")
        return value

    @staticmethod
    def _non_negative_int(payload: Mapping[str, object], key: str, label: str) -> int:
        value = payload.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{label} has invalid {key}")
        return value

    @staticmethod
    def _string_tuple(payload: Mapping[str, object], key: str, label: str) -> tuple[str, ...]:
        value = payload.get(key)
        if not isinstance(value, list) or any(
            not isinstance(item, str) or not item for item in value
        ):
            raise ValueError(f"{label} has invalid {key}")
        return tuple(value)

    @staticmethod
    def _encode_tuple(values: tuple[str, ...]) -> str:
        return json.dumps(values, ensure_ascii=False, separators=(",", ":"))

    @staticmethod
    def _decode_tuple(payload: str) -> tuple[str, ...]:
        value = json.loads(payload)
        if not isinstance(value, list) or any(
            not isinstance(item, str) or not item for item in value
        ):
            raise RuntimeError("corrupt partition lineage string tuple")
        return tuple(value)

    def _validate_separate_storage(self) -> None:
        if self.path == ":memory:" or self.ledger.path == ":memory:":
            return
        if Path(self.path).expanduser().resolve() == Path(self.ledger.path).expanduser().resolve():
            raise ValueError(
                "partition lineage index must use rebuildable storage separate from the canonical ledger"
            )
