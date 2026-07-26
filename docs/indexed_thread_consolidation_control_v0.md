# Pinned Indexed Thread-Consolidation Control v0

## Purpose

#50 removed raw-ledger replay from the question:

> What thread-consolidation work exists at revision N?

The remaining automatic-control problem was snapshot continuity.

The original `ThreadConsolidationControlAdapter` solved this by retaining the **entire event tuple** used when signals were computed, then replaying the planner against that tuple during context preparation.

That is semantically safe but scales with ledger history.

This slice replaces the full event cache with:

```text
captured runtime revision N
        ↓
indexed pressure
        ↓
indexed bounded plan
        ↓
cache one bounded WorkPreparation
        ↓
scheduler decision / trace append
        ↓
context returns cached WorkPreparation
```

No historical event tuple is retained.

## Generic captured-revision primitive

`PinnedIndexedRuntimeSnapshotProvider` extends the indexed runtime snapshot provider from #48 with one small generic contract:

```text
current_revision
```

The property is set only after a complete indexed runtime snapshot has been captured successfully.

This is deliberately not consolidation-specific. Any future adapter that must stay aligned with a scheduler snapshot can depend on the same captured-revision token.

## Why the global revision cannot be inferred from ProjectedState

A Work Thread's `ProjectedState.revision` is the most recent event affecting that thread.

It is **not** necessarily the global canonical Research Ledger revision.

Therefore an adapter must not independently call `ledger.latest_sequence()` after scheduler snapshot capture and assume it is looking at the same world.

The pinned provider gives signal/context adapters the exact global revision used by candidate construction.

## Indexed signal path

`IndexedThreadConsolidationControlAdapter.signals(state)`:

1. reads `revision_provider.current_revision`;
2. computes indexed consolidation pressure at exactly that revision;
3. asks the caller's existing SignalProvider for domain signals;
4. raises `synthesis_need` only when consolidation pressure is greater;
5. only in that case, computes the exact indexed bounded consolidation plan;
6. materializes the existing bounded `WorkPreparation` at the same revision;
7. caches that preparation under the thread's projected revision;
8. records route ownership.

The adapter therefore owns the route only when it actually supplied the winning synthesis demand.

A stronger domain-owned `synthesis_need` remains domain-owned and is not relabelled as thread consolidation.

## Bounded cache instead of event cache

The cached object is the already-finalized `WorkPreparation`:

- selected lower knowledge delta IDs;
- compact current Knowledge records;
- synthesis constraints;
- causal/provenance context;
- `synthesis_mode = THREAD_CONSOLIDATION`;
- `synthesis_route = THREAD_CONSOLIDATION`;
- `consolidation_pressure_revision = N`.

Its size is bounded by the existing consolidation selection limit.

The cache does **not** retain:

- Research Ledger history;
- partition history;
- complete Knowledge State;
- evidence payloads;
- worker hidden state.

## Scheduler wrapper remains unchanged

The existing `ThreadConsolidationScheduler` from #43 is reused.

It still wraps a generic scheduler and appends reason code:

```text
THREAD_CONSOLIDATION
```

only when:

- the delegate chose `SYNTHESIZE / SYNTHESIS_NEEDED`;
- the matching control adapter owns that route.

With `TracingScheduler` outside it, the final routed reason remains durable provenance.

No new hierarchy-specific scheduler implementation is introduced.

## Context after scheduler tracing

The key runtime sequence is:

```text
PinnedIndexedRuntimeSnapshotProvider.capture() -> revision N
        ↓
adapter.signals(...) -> bounded preparation cached at N
        ↓
SchedulerV0 -> ThreadConsolidationScheduler
        ↓
TracingScheduler appends SCHEDULER_DECISION_RECORDED at N+1
        ↓
adapter.context(...)
        ↓
return cached preparation from N
```

The context phase performs no lineage/Knowledge projection.

Therefore the N+1 scheduler trace cannot force a materialized view to rewind to N, and it cannot change which lower knowledge the decision consumes.

This is the indexed replacement for retaining the original full event tuple.

## Route integrity

A routed consolidation decision is accepted only when:

- its thread matches the ProjectedState;
- the adapter's cached revision still equals the runtime provider's captured revision;
- the adapter owned the matching thread/projected-revision signal;
- the bounded preparation still exists.

A caller cannot forge `THREAD_CONSOLIDATION` by manually adding the reason code.

A new runtime snapshot automatically invalidates the previous adapter cache.

## Configuration integrity

The indexed pressure projector's:

```text
minimum_source_deltas
```

must equal the indexed planner's readiness minimum.

This preserves #43's rule that pressure cannot say “ready” under a different source minimum than the planner itself.

## End-to-end regression

A hostile Research Ledger whose `read_all_events()` raises immediately is seeded with:

- one active Work Thread;
- two evidence records;
- one partitioned integration allocation;
- two distinct historical partitions;
- one partition-produced lower knowledge delta from each partition.

The runtime uses:

```text
PinnedIndexedRuntimeSnapshotProvider
IndexedThreadConsolidationControlAdapter
SchedulerV0
ThreadConsolidationScheduler
TracingScheduler
IndexedRuntimeControlLoop
IndexedWorkerRuntime
```

Required behavior:

1. runtime snapshot captures revision N;
2. consolidation pressure reaches the synthesis threshold;
3. Scheduler v0 chooses generic SYNTHESIZE;
4. route is tagged `THREAD_CONSOLIDATION`;
5. scheduler trace durably records the routed reason;
6. worker receives only the bounded two-delta consolidation context from N;
7. worker emits a higher provisional `THREAD_CONSOLIDATION` delta;
8. indexed Worker Runtime persists it without full history replay;
9. next indexed pressure projection sees both lower sources consumed and returns zero.

## Context projection-count regression

A counting Knowledge State is used to make the cache boundary observable.

During one routed cycle the expected Knowledge State projections are:

1. one global projection for pressure;
2. one thread projection for planner;
3. one thread projection to construct the bounded preparation.

After Scheduler tracing appends the durable decision, `context(...)` must not perform a fourth projection.

This guards against accidentally reintroducing an old-snapshot materialization read during context preparation.

## Remaining scaling boundary

At this point the automatic thread-consolidation control path no longer requires `read_all_events()`.

The current pressure/planner implementation still scans relevant **materialized** lineage/current-knowledge rows. That is intentionally not hidden behind another cache yet.

The next optimization should be driven by traces:

- if materialized-row pressure projection is cheap enough, stop;
- if it becomes hot at large knowledge populations, colocate derived indexes and turn the same logic into indexed SQL joins/materialized counts.

Do not add another hierarchy/cache layer merely because it is possible.

## Research status

No scheduler heuristic, hierarchy depth, consolidation selection rule, knowledge semantics, verification policy, worker architecture, or truth-promotion rule changes here.

This slice only replaces the old full-event snapshot handoff with a pinned revision + bounded preparation handoff.
