# Incremental Work Thread / Work Graph State v0

## Purpose

`ThreadStateProjector` remains the semantic reference for rebuilding current Work Thread and Work Graph state from the append-only Research Ledger.

At large ledger histories, replaying every historical event on every scheduler cycle is an avoidable systems cost. `SQLiteIndexedThreadState` changes only that cost shape.

The Research Ledger remains canonical.

## Materialized state

The sidecar keeps compact rebuildable current state for each Work Thread:

- objective;
- purpose;
- status;
- thread revision;
- durable references;
- active hypotheses;
- active contradictions;
- open questions;
- metadata;
- merged-into target;
- last worker ID / last attempt sequence.

Work Graph relations are stored separately as ordered derived edges:

- dependencies;
- fork parent/child relations;
- merge source/target relations.

No worker hidden state, prompt/context body, evidence payload, model tensor, or full event history is copied into this index.

## Shared tail primitive

The index reuses `LedgerProjectionTail` and `ProjectionCheckpoint` from the incremental Knowledge State slice.

Normal synchronization is therefore:

```text
materialized checkpoint
        ↓
validate checkpoint against canonical ledger
        ↓
read only events after checkpoint
        ↓
reduce into current thread/graph state
        ↓
validate current graph
        ↓
advance checkpoint
```

The normal update cost depends on new events plus current graph validation, not total historical event count.

## Exact snapshot isolation

`sync_through(sequence)` and `snapshot_all_through(sequence)` can materialize exactly through one known Research Ledger sequence.

The index is forward-only. Once an instance has advanced beyond an older snapshot, it refuses to rewind. This matches the other materialized views and prevents later events from being silently mixed into an older scheduler decision.

## Replay-equivalent ordered graph semantics

Ordering is part of the existing `ProjectedState` contract.

The materialized graph therefore retains:

- first relation event sequence;
- within-event ordinal.

This matters because one event may contain multiple targets. For example:

```text
THREAD_FORKED refs = [child-2, child-1]
```

must remain `[child-2, child-1]`; lexical sorting is incorrect.

Duplicate relation additions are idempotent and retain their original position, matching the replay projector's `_extend_unique` behavior.

Dependency removal followed by re-addition creates a new position at the later event, also matching replay.

## Graph integrity

At the requested final snapshot the index enforces the same graph invariants as replay:

- dependencies target created Work Threads;
- no self dependency;
- dependency graph is acyclic;
- fork source/target exist;
- fork relations cannot predate either thread's creation;
- no self fork;
- fork ancestry is acyclic;
- merge source/target exist;
- merge relations cannot predate either thread's creation;
- no self merge;
- one source cannot merge into multiple targets.

Graph validation occurs after the entire requested tail is reduced, not after each page. This is required because a relation may become valid later inside the same requested snapshot.

If validation fails, the SQLite transaction rolls back and the projection checkpoint does not advance.

## Last-worker continuity

The replay-based `RuntimeControlLoop` currently scans history to recover the most recent worker for a Work Thread so `CONTINUE` can retain the same worker when appropriate.

`SQLiteIndexedThreadState` derives this during `ATTEMPT_STARTED` reduction and exposes:

```text
last_worker_id(thread_id)
```

This lets the future indexed control-cycle path remove that historical scan without introducing a separate attempt-history cache.

## Storage ownership

Persistent index storage must be separate from the canonical Research Ledger SQLite file.

The sidecar binds itself to:

- projection schema version;
- canonical ledger schema version;
- canonical ledger source identity;
- checkpoint sequence;
- checkpoint event ID.

A replaced, truncated, or rebound canonical ledger is rejected rather than silently continued.

## Rebuildability

The sidecar is disposable.

`rebuild()` deletes all derived thread/edge state, resets the checkpoint, and reconstructs the same `ProjectedState` objects from canonical history through bounded pages.

## Compatibility surface

The indexed view implements:

```text
project_all(events)
project(events, thread_id=...)
```

with the same output types as `ThreadStateProjector`.

It also exposes direct current-state methods for the upcoming no-full-history runtime path:

```text
snapshot_all()
snapshot_all_through(sequence)
snapshot(thread_id=...)
last_worker_id(thread_id)
```

## What this does not change

This slice does not change:

- Work Thread semantics;
- Work Graph semantics;
- scheduler policy;
- worker selection policy;
- evidence/knowledge semantics;
- hierarchy depth;
- the Research Ledger contract.

It changes only how current Work Thread / graph state is reconstructed.

## Next boundary

After Integration, Knowledge State, and Work Thread State all have incremental materializations, the remaining major scheduler-cycle waste is the composition layer's unconditional:

```text
ledger.read_all_events()
```

The next slice should add an explicit indexed runtime snapshot provider so a control cycle can consume:

```text
canonical revision
+ current ProjectedState tuple
+ integration overview
+ verification overview
+ last-worker IDs
```

without materializing the full historical event tuple.
