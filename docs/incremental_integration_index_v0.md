# Incremental integration index v0

## Purpose

The canonical Research Ledger is append-only and replayable, which makes correctness simple. But a full replay on every scheduler/control query scales with total historical information rather than with new information.

That is the wrong hot-path shape once the system contains millions of evidence and knowledge events.

v0 therefore adds a **rebuildable incremental integration index**:

```text
canonical Research Ledger
        ↓ new tail only
SQLiteIndexedIntegrationTracker
        ↓
backlog / pressure / pending-batch queries
```

The Research Ledger remains the only source of truth.

## Architecture rule

> Durable truth remains append-only; expensive current-state projections may be materialized as disposable, rebuildable indexes.

The index may be deleted at any time and reconstructed from sequence zero.

No worker, scheduler, or knowledge contract depends on the sidecar database being authoritative.

## What the index stores

The sidecar stores only derived integration state needed for hot queries:

### Evidence identity/status

- evidence ID;
- original evidence event ID;
- source Work Thread;
- creation sequence;
- first disposition sequence/kind;
- total disposition-reference count.

### Knowledge-delta identity

- delta ID;
- target Work Thread;
- creation sequence.

### Integrity diagnostics

Disposition references to evidence IDs not yet known to the projection remain recorded separately rather than disappearing.

### Projection checkpoint

- index schema version;
- canonical ledger schema version;
- canonical ledger source identity;
- last applied ledger sequence;
- event ID at that sequence.

## What the index does not store

It deliberately does not duplicate:

- evidence summaries/payloads;
- tensors;
- Work Item contexts;
- worker outputs beyond derived status;
- full knowledge payloads;
- hidden state.

When a pending evidence batch is requested, the index finds the oldest pending evidence IDs/event IDs and fetches the original `EVIDENCE_ADDED` events from the canonical ledger.

This keeps the sidecar compact and avoids a second copy of the information mass we are trying to manage.

## Incremental tail processing

After initial construction, normal synchronization starts at the persisted projection checkpoint:

```text
index revision = R
        ↓
ledger.read_events(after_sequence=R)
        ↓
apply only new events
        ↓
new revision
```

The low-level ledger page size remains bounded.

A 1,000-event read limit therefore cannot silently truncate canonical replay: synchronization continues page by page until the requested boundary is reached.

## Snapshot-isolated synchronization

The control loop already evaluates scheduling against an immutable ledger-event snapshot.

The index supports two synchronization modes:

### `sync()`

Advance through all events currently available in the canonical ledger.

Useful for ordinary latest-state queries.

### `sync_through(sequence)`

Advance exactly through one known ledger snapshot boundary.

This is used by `overview(events)` when the caller supplies its already-captured event tuple.

If the control loop captured revision 50 and another writer appends revisions 51–60 before integration pressure is computed, the indexed projection still stops at 50.

The scheduler therefore cannot accidentally make a revision-50 decision using revision-60 integration state.

## Monotonic projection

The sidecar is an incremental forward-only materialized view.

If it has already advanced to revision 60, a caller may not ask the same instance to project revision 50.

That request is rejected instead of attempting to rewind mutable derived state.

Callers that require simultaneous historical snapshots should use:

- an independent projection instance;
- canonical replay;
- or a future versioned snapshot/index mechanism.

The normal runtime path is monotonic, so this restriction keeps v0 simple.

## Checkpoint integrity

The index does not trust sequence number alone.

At every synchronization it verifies that:

```text
stored checkpoint event ID
        ↓
canonical ledger lookup
        ↓
exists at stored checkpoint sequence
```

This detects a replaced, truncated, or otherwise rebound canonical ledger instead of continuing from an unrelated history that happens to have the same row count.

For persistent files, the index also stores the resolved canonical ledger source path and ledger schema version.

## Separate storage

A persistent index file may not be the canonical Research Ledger file.

Keeping derived state physically separate reinforces the ownership boundary:

```text
Research Ledger = canonical truth
integration index = rebuildable acceleration
```

Both may use SQLite/WAL locally, but they have different failure/recovery semantics.

## Transactional tail application

Each synchronized ledger page is applied inside one SQLite transaction.

The checkpoint advances only after the page's derived changes have been written successfully.

If an event violates projection integrity, the page rolls back and the stored revision remains at the prior safe checkpoint.

## Causal integrity

The index preserves the same important integration invariants as full replay.

It rejects:

- duplicate durable evidence IDs;
- duplicate durable knowledge-delta IDs;
- malformed disposition kinds;
- a disposition that is later revealed to precede evidence creation;
- non-increasing canonical tail order;
- checkpoint/event-ID mismatch.

Unknown disposition references remain visible until/unless a later evidence creation proves the history causally invalid.

Repeated dispositions remain countable separately from unique resolved evidence.

## Scheduler-facing overview

`overview(...)` exposes the same structural information needed by the scheduler/control path:

### Global

- evidence count;
- unique dispositioned evidence count;
- backlog count;
- knowledge-delta count;
- oldest backlog age;
- global backpressure state.

### Per Work Thread

- evidence count;
- dispositioned count;
- backlog count;
- knowledge-delta count;
- oldest backlog age;
- normalized integration pressure.

The projection computes these with indexed SQL rather than replaying every historical event into Python structures on every scheduler cycle.

## Bounded pending work

`pending_batch(limit=N, thread_id=...)` uses indexed pending-evidence ordering to find only the next N evidence identities.

It then retrieves exactly those canonical evidence events and returns the same `IntegrationBatch`/`PendingEvidence` shape already consumed by the purpose-aware context router.

This means the final worker contract does not change when the runtime switches from replay-based integration tracking to indexed integration tracking.

## Compatibility path

The sidecar intentionally exposes the existing integration tracker surface used by the runtime:

- `overview(...)`;
- `pending_batch(...)`;
- `snapshot(...)`;
- `pressure(...)`;
- `is_backpressured()`;
- `record_disposition(...)`;
- `record_knowledge_delta(...)`.

A regression passes `SQLiteIndexedIntegrationTracker` directly into `PurposeContextRouter` and requires the normal `BACKPRESSURE` synthesis view to receive the same bounded evidence records.

No new worker/context protocol is introduced.

## Rebuild

`rebuild()` deletes all derived rows, resets the projection checkpoint, and replays the canonical ledger through bounded pages.

The rebuilt result must match an incrementally maintained result.

Therefore index corruption or implementation changes do not require repairing canonical history.

A future schema migration may simply invalidate and rebuild this derived store when practical.

## Scaling shape

Without a materialized view, a hot integration query tends toward:

```text
cost ≈ O(total historical ledger events)
```

per replay-heavy scheduler/control cycle.

With the incremental index, normal synchronization tends toward:

```text
cost ≈ O(new ledger events since checkpoint)
```

plus indexed SQL for the requested current-state query.

Pending worker context remains bounded by the requested batch limit rather than total backlog size.

No measured speedup is claimed until real workloads are run.

## Relationship to information integration bandwidth

The index does not solve the semantic integration problem.

It removes one avoidable systems cost around that problem: repeatedly reconstructing already-known integration state from the entire durable history.

This matters because at very large population sizes the system may produce enormous numbers of evidence events even when each worker emits only a small packet.

The architecture should spend compute on:

- new exploration;
- integration;
- verification;
- useful higher-level synthesis;

not on repeatedly replaying unchanged old bookkeeping.

## What remains replay-heavy

v0 indexes only the raw integration projection.

Other current views still rebuild substantially from history, including:

- Work Thread / Work Graph state;
- full Knowledge State + assessments;
- verification pressure;
- partition lineage;
- thread-consolidation pressure;
- hierarchy telemetry.

Those should not automatically receive separate bespoke databases.

The high-leverage next step is to reuse the same **checkpointed tail → rebuildable derived view** primitive wherever profiling or architecture shows repeated full replay on a hot path.

A shared projection/tail coordinator may eventually feed several views from one canonical event tail.

## Non-goals

v0 does not add:

- a second source of truth;
- distributed storage;
- event streaming infrastructure;
- Kafka/Postgres;
- semantic evidence compression;
- another scheduler policy;
- a worker-visible cache;
- arbitrary historical rewind;
- a claimed performance improvement.

It changes only the cost shape of reconstructing current raw-integration state while preserving the final runtime contracts.
