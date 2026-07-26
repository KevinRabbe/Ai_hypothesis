"""Rebuildable incremental integration projection over the canonical Research Ledger.

The Research Ledger remains the only source of truth. This sidecar stores compact derived
integration state so hot backlog/pressure queries advance from the ledger tail rather than
replaying all historical events. It is disposable and fully rebuildable.
"""

from __future__ import annotations

import sqlite3
import threading
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .contracts import EvidenceDispositionKind, KnowledgeDelta, LedgerEvent
from .integration import IntegrationBackpressureConfig, IntegrationBatch, PendingEvidence
from .ledger import SQLiteResearchLedger

_INDEX_SCHEMA_VERSION = "integration-index-v0"

@dataclass(frozen=True, slots=True)
class IndexedIntegrationSnapshot:
    revision: int
    evidence_count: int
    dispositioned_evidence_count: int
    backlog_count: int
    knowledge_delta_count: int
    oldest_backlog_age_sequences: int

@dataclass(frozen=True, slots=True)
class IndexedIntegrationOverview:
    global_snapshot: IndexedIntegrationSnapshot
    thread_snapshots: Mapping[str, IndexedIntegrationSnapshot]
    thread_pressure: Mapping[str, float]
    global_backpressured: bool
    def pressure_for(self, thread_id: str) -> float:
        return float(self.thread_pressure.get(thread_id, 0.0))

class SQLiteIndexedIntegrationTracker:
    """Incremental, rebuildable substitute for replay-heavy integration status queries."""
    def __init__(self, ledger: SQLiteResearchLedger, index_path: str | Path = ":memory:", config: IntegrationBackpressureConfig | None = None) -> None:
        self.ledger = ledger
        self.path = str(index_path)
        self.config = config or IntegrationBackpressureConfig()
        self.config.validate()
        self._validate_separate_storage()
        self._lock = threading.RLock()
        self._connection = sqlite3.connect(self.path, timeout=30.0, check_same_thread=False)
        self._connection.row_factory = sqlite3.Row
        self._configure(); self._create_schema(); self._bind_source()
    @property
    def schema_version(self) -> str: return _INDEX_SCHEMA_VERSION
    @property
    def revision(self) -> int:
        with self._lock:
            try: revision = int(self._meta("revision"))
            except (TypeError, ValueError) as error: raise RuntimeError("corrupt integration index revision") from error
            if revision < 0: raise RuntimeError("corrupt integration index revision")
            return revision
    def sync(self, *, page_size: int = 1000) -> int: return self._sync_to(target_sequence=None, page_size=page_size)
    def sync_through(self, sequence: int, *, page_size: int = 1000) -> int:
        if sequence < 0: raise ValueError("sequence must be non-negative")
        return self._sync_to(target_sequence=sequence, page_size=page_size)
    def _sync_to(self, *, target_sequence: int | None, page_size: int) -> int:
        if page_size <= 0: raise ValueError("page_size must be positive")
        with self._lock:
            self._validate_checkpoint(); checkpoint = self.revision
            if target_sequence is not None and target_sequence < checkpoint: raise ValueError("integration index is ahead of the requested ledger snapshot")
            if target_sequence == checkpoint: return checkpoint
            while True:
                page = self.ledger.read_events(after_sequence=checkpoint, limit=page_size)
                if not page: break
                eligible = page if target_sequence is None else tuple(event for event in page if event.sequence <= target_sequence)
                if not eligible: break
                previous = checkpoint
                with self._connection:
                    for event in eligible:
                        event.validate()
                        if event.sequence <= previous: raise ValueError("ledger tail must be in strictly increasing sequence order")
                        self._apply_event(event); previous = event.sequence
                    self._set_meta("revision", str(eligible[-1].sequence)); self._set_meta("checkpoint_event_id", eligible[-1].event_id)
                checkpoint = eligible[-1].sequence
                if target_sequence is not None and checkpoint >= target_sequence: break
                if len(eligible) < len(page) or len(page) < page_size: break
            if target_sequence is not None and checkpoint != target_sequence: raise RuntimeError("requested integration snapshot sequence is not available in canonical ledger")
            return checkpoint
    def rebuild(self, *, page_size: int = 1000) -> int:
        with self._lock, self._connection:
            self._connection.execute("DELETE FROM integration_evidence"); self._connection.execute("DELETE FROM integration_knowledge_delta"); self._connection.execute("DELETE FROM integration_unknown_disposition"); self._set_meta("revision", "0"); self._set_meta("checkpoint_event_id", "")
        return self.sync(page_size=page_size)
    def snapshot(self, *, thread_id: str | None = None) -> IndexedIntegrationSnapshot:
        with self._lock: self.sync(); return self._snapshot(thread_id=thread_id)
    def overview(self, events: Sequence[LedgerEvent] | None = None) -> IndexedIntegrationOverview:
        if events is None: target = None
        else:
            history = tuple(events); previous = -1
            for event in history:
                event.validate()
                if event.sequence <= previous: raise ValueError("events must be in strictly increasing sequence order")
                previous = event.sequence
            target = history[-1].sequence if history else 0
        with self._lock:
            self.sync() if target is None else self.sync_through(target)
            revision = self.revision; global_snapshot = self._snapshot(thread_id=None, revision=revision)
            evidence_rows = self._connection.execute("SELECT thread_id, COUNT(*) AS evidence_count, SUM(CASE WHEN first_disposition_sequence IS NOT NULL THEN 1 ELSE 0 END) AS dispositioned_count, SUM(CASE WHEN first_disposition_sequence IS NULL THEN 1 ELSE 0 END) AS backlog_count, MIN(CASE WHEN first_disposition_sequence IS NULL THEN created_sequence END) AS oldest_pending_sequence FROM integration_evidence WHERE thread_id IS NOT NULL GROUP BY thread_id").fetchall()
            delta_rows = self._connection.execute("SELECT thread_id, COUNT(*) AS delta_count FROM integration_knowledge_delta WHERE thread_id IS NOT NULL GROUP BY thread_id").fetchall()
            evidence_by_thread = {str(row["thread_id"]): row for row in evidence_rows}; delta_by_thread = {str(row["thread_id"]): int(row["delta_count"]) for row in delta_rows}
            thread_snapshots = {}; thread_pressure = {}
            for thread_id in set(evidence_by_thread) | set(delta_by_thread):
                row = evidence_by_thread.get(thread_id)
                snapshot = IndexedIntegrationSnapshot(revision, 0, 0, 0, delta_by_thread.get(thread_id, 0), 0) if row is None else self._snapshot_from_row(row, revision=revision, knowledge_delta_count=delta_by_thread.get(thread_id, 0))
                thread_snapshots[thread_id] = snapshot; thread_pressure[thread_id] = self._pressure_for_snapshot(snapshot)
            return IndexedIntegrationOverview(global_snapshot, thread_snapshots, thread_pressure, self._is_snapshot_backpressured(global_snapshot))
    def pending_batch(self, *, limit: int, thread_id: str | None = None) -> IntegrationBatch:
        if limit <= 0: raise ValueError("limit must be positive")
        with self._lock:
            self.sync()
            if thread_id is None: rows = self._connection.execute("SELECT evidence_id,event_id,thread_id,created_sequence FROM integration_evidence WHERE first_disposition_sequence IS NULL ORDER BY created_sequence,evidence_id LIMIT ?", (limit,)).fetchall()
            else:
                if not thread_id or not thread_id.strip(): raise ValueError("thread_id must be non-empty when provided")
                rows = self._connection.execute("SELECT evidence_id,event_id,thread_id,created_sequence FROM integration_evidence WHERE first_disposition_sequence IS NULL AND thread_id=? ORDER BY created_sequence,evidence_id LIMIT ?", (thread_id, limit)).fetchall()
            records=[]
            for row in rows:
                event=self.ledger.get_event(str(row["event_id"])); evidence_id=str(row["evidence_id"])
                if event is None or event.event_type != "EVIDENCE_ADDED": raise RuntimeError("integration index references a missing evidence event")
                if event.payload.get("evidence_id") != evidence_id: raise RuntimeError("integration index evidence identity no longer matches ledger")
                data=event.payload.get("data")
                records.append(PendingEvidence(evidence_id,evidence_id and event.event_id,event.sequence,event.thread_id,str(event.payload.get("kind","UNKNOWN")),str(event.payload.get("summary","")),tuple(r for r in event.reference_ids if r!=evidence_id),self._optional_float(event.payload.get("strength")),self._optional_float(event.payload.get("uncertainty")),dict(data) if isinstance(data,Mapping) else {}))
            return IntegrationBatch(revision=self.revision, records=tuple(records))
    def record_disposition(self, evidence_ids: Sequence[str], disposition: EvidenceDispositionKind, *, reason: str | None = None, thread_id: str | None = None) -> LedgerEvent:
        resolved=tuple(evidence_ids)
        if not resolved: raise ValueError("at least one evidence ID is required")
        if any(not x or not x.strip() for x in resolved): raise ValueError("evidence IDs must be non-empty")
        payload={"disposition":disposition.value}
        if reason is not None:
            if not reason.strip(): raise ValueError("reason must be non-empty when provided")
            payload["reason"]=reason
        event=self.ledger.append_event(event_type="INTEGRATION_DISPOSITION_RECORDED",thread_id=thread_id,reference_ids=resolved,payload=payload); self.sync(); return event
    def record_knowledge_delta(self, delta: KnowledgeDelta) -> LedgerEvent:
        delta.validate(); event=self.ledger.append_event(event_type="KNOWLEDGE_DELTA_RECORDED",thread_id=delta.thread_id,reference_ids=(delta.delta_id,*delta.reference_ids),parent_event_ids=delta.causal_event_ids,payload={"delta_id":delta.delta_id,"kind":delta.kind,"summary":delta.summary}); self.sync(); return event
    def pressure(self, *, thread_id: str | None = None) -> float: return self._pressure_for_snapshot(self.snapshot(thread_id=thread_id))
    def is_backpressured(self) -> bool: return self._is_snapshot_backpressured(self.snapshot())
    def unknown_disposition_reference_count(self) -> int:
        with self._lock: self.sync(); return int(self._connection.execute("SELECT COUNT(*) AS count FROM integration_unknown_disposition").fetchone()["count"])
    def redisposition_reference_count(self) -> int:
        with self._lock: self.sync(); return int(self._connection.execute("SELECT COALESCE(SUM(CASE WHEN disposition_count>1 THEN disposition_count-1 ELSE 0 END),0) AS count FROM integration_evidence").fetchone()["count"])
    def close(self) -> None:
        with self._lock: self._connection.close()
    def __enter__(self): return self
    def __exit__(self, exc_type, exc, traceback): self.close()
    def _apply_event(self,event):
        if event.event_type=="EVIDENCE_ADDED": self._apply_evidence(event)
        elif event.event_type=="INTEGRATION_DISPOSITION_RECORDED": self._apply_disposition(event)
        elif event.event_type=="KNOWLEDGE_DELTA_RECORDED": self._apply_knowledge_delta(event)
    def _apply_evidence(self,event):
        evidence_id=event.payload.get("evidence_id")
        if not isinstance(evidence_id,str) or not evidence_id: raise ValueError("EVIDENCE_ADDED is missing evidence_id")
        if self._connection.execute("SELECT sequence FROM integration_unknown_disposition WHERE evidence_id=? ORDER BY sequence LIMIT 1",(evidence_id,)).fetchone() is not None: raise ValueError("evidence disposition precedes evidence creation")
        try: self._connection.execute("INSERT INTO integration_evidence(evidence_id,event_id,thread_id,created_sequence,first_disposition_sequence,first_disposition_kind,disposition_count) VALUES(?,?,?,?,NULL,NULL,0)",(evidence_id,event.event_id,event.thread_id,event.sequence))
        except sqlite3.IntegrityError as error: raise ValueError(f"duplicate durable evidence ID {evidence_id!r}") from error
    def _apply_disposition(self,event):
        try: disposition=EvidenceDispositionKind(event.payload.get("disposition"))
        except (TypeError,ValueError) as error: raise ValueError("invalid durable evidence disposition") from error
        for evidence_id in event.reference_ids:
            row=self._connection.execute("SELECT created_sequence,first_disposition_sequence,disposition_count FROM integration_evidence WHERE evidence_id=?",(evidence_id,)).fetchone()
            if row is None: self._connection.execute("INSERT INTO integration_unknown_disposition(event_id,evidence_id,sequence) VALUES(?,?,?)",(event.event_id,evidence_id,event.sequence)); continue
            if event.sequence<int(row["created_sequence"]): raise ValueError("evidence disposition precedes evidence creation")
            if row["first_disposition_sequence"] is None: self._connection.execute("UPDATE integration_evidence SET first_disposition_sequence=?,first_disposition_kind=?,disposition_count=disposition_count+1 WHERE evidence_id=?",(event.sequence,disposition.value,evidence_id))
            else: self._connection.execute("UPDATE integration_evidence SET disposition_count=disposition_count+1 WHERE evidence_id=?",(evidence_id,))
    def _apply_knowledge_delta(self,event):
        delta_id=event.payload.get("delta_id")
        if not isinstance(delta_id,str) or not delta_id: raise ValueError("KNOWLEDGE_DELTA_RECORDED is missing delta_id")
        try: self._connection.execute("INSERT INTO integration_knowledge_delta(delta_id,thread_id,created_sequence) VALUES(?,?,?)",(delta_id,event.thread_id,event.sequence))
        except sqlite3.IntegrityError as error: raise ValueError(f"duplicate durable knowledge delta ID {delta_id!r}") from error
    def _snapshot(self, *, thread_id: str | None, revision: int | None = None) -> IndexedIntegrationSnapshot:
        rev=self.revision if revision is None else revision
        where="" if thread_id is None else " WHERE thread_id=?"; params=() if thread_id is None else (thread_id,)
        if thread_id is not None and (not thread_id or not thread_id.strip()): raise ValueError("thread_id must be non-empty when provided")
        row=self._connection.execute("SELECT COUNT(*) AS evidence_count,SUM(CASE WHEN first_disposition_sequence IS NOT NULL THEN 1 ELSE 0 END) AS dispositioned_count,SUM(CASE WHEN first_disposition_sequence IS NULL THEN 1 ELSE 0 END) AS backlog_count,MIN(CASE WHEN first_disposition_sequence IS NULL THEN created_sequence END) AS oldest_pending_sequence FROM integration_evidence"+where,params).fetchone()
        delta=self._connection.execute("SELECT COUNT(*) AS count FROM integration_knowledge_delta"+where,params).fetchone()
        return self._snapshot_from_row(row,revision=rev,knowledge_delta_count=int(delta["count"]))
    @staticmethod
    def _snapshot_from_row(row, *, revision, knowledge_delta_count):
        oldest=row["oldest_pending_sequence"]
        return IndexedIntegrationSnapshot(revision,int(row["evidence_count"] or 0),int(row["dispositioned_count"] or 0),int(row["backlog_count"] or 0),knowledge_delta_count,max(0,revision-int(oldest)) if oldest is not None else 0)
    def _pressure_for_snapshot(self,s): return max(self._ratio(s.backlog_count,self.config.max_backlog_count),self._ratio(s.oldest_backlog_age_sequences,self.config.max_backlog_age_sequences))
    def _is_snapshot_backpressured(self,s): return s.backlog_count>self.config.max_backlog_count or s.oldest_backlog_age_sequences>self.config.max_backlog_age_sequences
    @staticmethod
    def _ratio(value,limit): return 0.0 if value<=0 else 1.0 if limit<=0 else min(1.0,value/float(limit))
    @staticmethod
    def _optional_float(value):
        if value is None:return None
        if isinstance(value,bool) or not isinstance(value,(int,float)): raise ValueError("durable evidence scalar must be numeric or null")
        return float(value)
    def _configure(self):
        with self._connection:
            self._connection.execute("PRAGMA foreign_keys=ON"); self._connection.execute("PRAGMA busy_timeout=30000"); self._connection.execute("PRAGMA synchronous=NORMAL"); self._connection.execute("PRAGMA journal_mode=WAL")
    def _create_schema(self):
        with self._connection:
            self._connection.execute("CREATE TABLE IF NOT EXISTS integration_index_meta(key TEXT PRIMARY KEY,value TEXT NOT NULL)")
            self._connection.execute("CREATE TABLE IF NOT EXISTS integration_evidence(evidence_id TEXT PRIMARY KEY,event_id TEXT NOT NULL UNIQUE,thread_id TEXT,created_sequence INTEGER NOT NULL,first_disposition_sequence INTEGER,first_disposition_kind TEXT,disposition_count INTEGER NOT NULL DEFAULT 0)")
            self._connection.execute("CREATE TABLE IF NOT EXISTS integration_knowledge_delta(delta_id TEXT PRIMARY KEY,thread_id TEXT,created_sequence INTEGER NOT NULL)")
            self._connection.execute("CREATE TABLE IF NOT EXISTS integration_unknown_disposition(id INTEGER PRIMARY KEY AUTOINCREMENT,event_id TEXT NOT NULL,evidence_id TEXT NOT NULL,sequence INTEGER NOT NULL)")
            self._connection.execute("CREATE INDEX IF NOT EXISTS idx_integration_pending_global ON integration_evidence(first_disposition_sequence,created_sequence,evidence_id)")
            self._connection.execute("CREATE INDEX IF NOT EXISTS idx_integration_pending_thread ON integration_evidence(thread_id,first_disposition_sequence,created_sequence,evidence_id)")
            self._connection.execute("CREATE INDEX IF NOT EXISTS idx_integration_delta_thread ON integration_knowledge_delta(thread_id,created_sequence)")
            self._connection.execute("CREATE INDEX IF NOT EXISTS idx_integration_unknown_evidence ON integration_unknown_disposition(evidence_id,sequence)")
    def _bind_source(self):
        expected={"schema_version":_INDEX_SCHEMA_VERSION,"ledger_schema_version":self.ledger.schema_version,"ledger_source":self._ledger_source()}
        with self._connection:
            for key,value in expected.items():
                existing=self._meta_optional(key)
                if existing is None:self._set_meta(key,value)
                elif existing!=value:raise RuntimeError(f"integration index {key} mismatch: {existing!r} != {value!r}")
            if self._meta_optional("revision") is None:self._set_meta("revision","0")
            if self._meta_optional("checkpoint_event_id") is None:self._set_meta("checkpoint_event_id","")
    def _validate_checkpoint(self):
        revision=self.revision; event_id=self._meta("checkpoint_event_id")
        if revision==0:
            if event_id:raise RuntimeError("integration index has event ID at revision zero")
            return
        if not event_id:raise RuntimeError("integration index checkpoint is missing event ID")
        event=self.ledger.get_event(event_id)
        if event is None or event.sequence!=revision:raise RuntimeError("integration index checkpoint no longer matches canonical ledger")
    def _meta_optional(self,key):
        row=self._connection.execute("SELECT value FROM integration_index_meta WHERE key=?",(key,)).fetchone(); return str(row["value"]) if row is not None else None
    def _meta(self,key):
        value=self._meta_optional(key)
        if value is None:raise RuntimeError(f"integration index metadata {key!r} is missing")
        return value
    def _set_meta(self,key,value): self._connection.execute("INSERT INTO integration_index_meta(key,value) VALUES(?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",(key,value))
    def _ledger_source(self): return ":memory:" if self.ledger.path==":memory:" else str(Path(self.ledger.path).expanduser().resolve())
    def _validate_separate_storage(self):
        if self.path==":memory:" or self.ledger.path==":memory:":return
        if Path(self.ledger.path).expanduser().resolve()==Path(self.path).expanduser().resolve():raise ValueError("integration index must use rebuildable storage separate from the canonical ledger")
