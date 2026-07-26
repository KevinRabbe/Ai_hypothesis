# Incremental Knowledge State v0

## Purpose

The Knowledge State projector is logically simple:

```text
KNOWLEDGE_DELTA_RECORDED
      ↓
PROVISIONAL record
      ↓
later KNOWLEDGE_ASSESSMENT_RECORDED
      ↓
VERIFIED / DISPUTED / RETRACTED current status
```

But repeatedly refolding every historical knowledge delta and assessment becomes increasingly expensive as the Research Ledger grows.

That cost now sits beneath several hot paths:

- verification pressure;
- purpose-aware VERIFY context;
- thread-level consolidation;
- automatic consolidation pressure;
- hierarchy telemetry;
- future branch/topic integration.

v0 therefore materializes the current Knowledge State as a **rebuildable incremental projection** while leaving the Research Ledger fully canonical.

## Shared ledger-tail primitive

This slice also introduces `LedgerProjectionTail` and `ProjectionCheckpoint`.

They contain the reusable part of incremental projections:

```text
projection checkpoint
      ↓ validate against canonical event ID
Research Ledger
      ↓ bounded pages after checkpoint
optional exact target sequence
      ↓
projection-specific reducer
```

The tail helper owns no derived state itself.

Each materialized view remains responsible for:

- its own schema;
- applying relevant event types;
- writing its derived checkpoint transactionally;
- rebuilding when required.

This lets multiple projections reuse one correctness model without forcing them into one giant database schema.

## Canonical checkpoint

A `ProjectionCheckpoint` contains:

```text
sequence
event_id
```

Sequence zero has no event ID.

Every nonzero checkpoint is validated by fetching that exact event from the canonical ledger and requiring the sequence to match.

This protects derived state from continuing after:

- ledger replacement;
- truncation;
- accidental rebinding;
- corrupted checkpoint metadata.

## Exact snapshot boundaries

`LedgerProjectionTail.iter_pages(...)` can advance either:

### To latest

Consume all currently available events after the checkpoint.

### Through an exact sequence

Consume only events up to a caller-supplied canonical snapshot boundary.

This is important because the control architecture already treats one scheduler cycle as snapshot-isolated.

A materialized projection must not accidentally include events appended after that scheduler snapshot merely because it is faster than canonical replay.

## Knowledge index

`SQLiteIndexedKnowledgeState` stores the current derived state of each knowledge delta:

- delta ID;
- kind;
- summary;
- source-reference IDs;
- causal event IDs;
- target Work Thread;
- creation event/sequence;
- current status;
- latest assessment reason/event/sequence.

It returns the existing public objects:

- `KnowledgeRecord`;
- `KnowledgeSnapshot`.

It also implements a `project(events, thread_id=...)` surface compatible with `KnowledgeStateProjector` consumers.

Therefore verification and consolidation code do not need a new knowledge contract merely because state is materialized.

## Event reducers

### Knowledge delta

A new `KNOWLEDGE_DELTA_RECORDED` event inserts one `PROVISIONAL` record.

Duplicate durable delta identity is rejected.

### Assessment

A `KNOWLEDGE_ASSESSMENT_RECORDED` event updates current derived status to:

- `VERIFIED`;
- `DISPUTED`;
- `RETRACTED`.

Assessment reason/event/sequence are retained.

Assessment of a delta that does not yet exist is invalid history and fails the projection transaction.

The canonical assessment event itself is never mutated.

## Same semantics as full replay

The incremental projection must produce the same `KnowledgeSnapshot` as `KnowledgeStateProjector` for the same ledger boundary.

Regression construction compares both globally and per Work Thread across:

- provisional creation;
- verification;
- dispute;
- retraction;
- knowledge-to-knowledge references;
- thread-level consolidation records.

The materialization changes only execution cost, not knowledge semantics.

## Verification compatibility

`KnowledgeVerificationTracker` already accepts a knowledge projector object.

Regression coverage passes `SQLiteIndexedKnowledgeState` directly as that projector and requires the same unresolved-knowledge behavior.

This is important because verification pressure is one of the scheduler hot paths that otherwise refolds all historical knowledge repeatedly.

## Forward-only behavior

Like the integration index, one materialized Knowledge State instance is forward-only.

If it has advanced to revision 1,000, asking it to project revision 900 is rejected rather than mutating current state backward.

Historical analysis can still use:

- canonical replay;
- another projection instance;
- a future versioned snapshot facility.

The normal runtime current-state path is monotonic.

## Transactional pages

Each tail page is reduced inside one SQLite transaction.

The materialized checkpoint advances only after every relevant event in that page succeeds.

If an assessment references an unknown delta, for example:

```text
page starts at revision 500
assessment fails at 507
```

then the page is rolled back and the derived checkpoint remains 500.

The canonical ledger remains unchanged and can be diagnosed/repaired at the source.

## Rebuildability

`rebuild()` deletes derived knowledge rows and replays the canonical ledger through bounded pages.

A rebuild must produce the same Knowledge Snapshot as the incrementally maintained view.

This keeps schema evolution simple:

> derived indexes may be thrown away; published/canonical history is never rewritten to satisfy a cache.

## Separate storage

A persistent materialized Knowledge State may not share the same SQLite file as the canonical Research Ledger.

The sidecar binds to:

- its own projection schema version;
- canonical ledger schema version;
- canonical ledger source identity;
- checkpoint sequence + event ID.

This keeps failure ownership explicit.

## Scaling shape

Full replay has a cost that grows with accumulated knowledge history:

```text
O(total knowledge + assessment events)
```

for each replay-heavy consumer.

Incremental materialization changes normal update cost toward:

```text
O(new ledger events since checkpoint)
```

plus indexed reads of current knowledge.

This matters because higher integration levels deliberately create more durable knowledge relationships over time. Without incremental projection, successful information organization would eventually make the control plane slower simply because it had accumulated useful history.

## Why keep the full ledger anyway

Current-state materialization is not a substitute for history.

The full append-only ledger remains necessary for:

- provenance;
- scientific reproducibility;
- retractions/disputes;
- causal lineage;
- rare evidence drill-down;
- rebuilding projections;
- auditing scheduler/worker behavior.

The architecture is therefore:

```text
large immutable history
      ↓ incremental tail
small current materialized views
      ↓ bounded context
workers / scheduler
```

rather than trying to keep the entire history in every active context or recomputing every view from scratch.

## What this proves architecturally

The same pattern now serves at least two distinct views:

1. raw integration backlog/status;
2. current Knowledge State.

That justifies `LedgerProjectionTail` as a shared primitive rather than an abstraction introduced speculatively.

The likely next target is Work Thread / Work Graph current state, because scheduler candidate construction still depends on replaying that projection.

Before creating another sidecar, however, a shared coordinator may be useful so several incremental views can consume one captured canonical tail rather than each issuing its own ledger reads.

## Non-goals

v0 does not add:

- a second source of truth;
- semantic knowledge compression;
- a new knowledge status;
- truth promotion;
- distributed state;
- historical rewind;
- a measured speedup claim;
- replacement of the Research Ledger.

It only materializes the current knowledge projection behind the same final architecture contract.
