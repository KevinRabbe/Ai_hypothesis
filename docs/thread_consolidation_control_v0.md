# Automatic thread-consolidation control v0

## Purpose

PR #41 made thread-level consolidation executable, but a caller still had to manually decide when to run it.

PR #42 added generic `synthesis_need` to Scheduler v0 without teaching the scheduler any hierarchy-specific concept.

This slice connects the two:

```text
partition-produced knowledge backlog
      ↓
one-pass consolidation-pressure projection
      ↓
SchedulerSignals.synthesis_need
      ↓
Scheduler v0 → SYNTHESIS_NEEDED
      ↓
THREAD_CONSOLIDATION route tag
      ↓
bounded thread-consolidation context
      ↓
ordinary Worker Runtime
      ↓
higher provisional knowledge
```

The scheduler remains hierarchy-agnostic.

## One-pass pressure projection

`ThreadConsolidationPressureProjector` projects all Work Threads from one supplied ledger history.

It does **not** call the full per-thread planner once for every Work Thread.

The projection performs the expensive global work once:

1. rebuild exact partition lineage;
2. rebuild current Knowledge State;
3. index partition-produced knowledge by source Work Thread;
4. remove retracted lower knowledge;
5. derive lower knowledge already consumed by active `THREAD_CONSOLIDATION` deltas;
6. count remaining source deltas and partitions per Work Thread.

It exposes:

- pending lower-source count per thread;
- pending source-partition count per thread;
- normalized synthesis pressure per thread;
- Work Threads whose exact partition provenance is incomplete.

## Pressure rule

Default provisional configuration:

```text
minimum_source_deltas = 2
full_pressure_count   = 8
```

A Work Thread has zero consolidation pressure until it contains enough pending lower deltas to form a useful consolidation attempt.

Then:

```text
pressure = min(1, pending_source_count / full_pressure_count)
```

The exact mapping is provisional.

The readiness threshold must match the actual `ThreadConsolidationPlanner.minimum_source_deltas`; the control adapter rejects mismatched configuration.

## Incomplete provenance

A Work Thread with missing historical partition-allocation provenance gets **zero automatic consolidation pressure**.

The system does not guess which partition produced its knowledge.

This does not block unrelated Work Threads whose lineage is complete.

## Signal adapter

`ThreadConsolidationControlAdapter.signals(state)` wraps an ordinary domain `SignalProvider`.

It:

1. gets the domain's existing `SchedulerSignals`;
2. reads consolidation pressure for the Work Thread;
3. replaces `synthesis_need` only when consolidation pressure is greater than the existing domain value.

Therefore thread consolidation does not silently steal a synthesis request already owned by another domain policy.

The adapter records that it owns the synthesis route only when **it actually raised the signal**.

## Decision routing

`ThreadConsolidationScheduler` wraps the generic scheduler.

It does not compute pressure or inspect knowledge.

When the delegate returns:

```text
action = SYNTHESIZE
reason contains SYNTHESIS_NEEDED
```

and the matching signal snapshot says thread consolidation supplied the winning synthesis demand, the wrapper appends:

```text
THREAD_CONSOLIDATION
```

The final decision therefore carries both:

- the generic scheduler reason: `SYNTHESIS_NEEDED`;
- the concrete context route: `THREAD_CONSOLIDATION`.

If `TracingScheduler` wraps this scheduler, the durable scheduler trace records the final routed reason.

Recommended order:

```text
SchedulerV0
      ↓
PartitionedBackpressureScheduler   (optional raw-evidence width)
      ↓
ThreadConsolidationScheduler
      ↓
TracingScheduler
```

The thread-consolidation wrapper only acts on generic `SYNTHESIS_NEEDED`, so raw `BACKPRESSURE` synthesis is left to the partitioned/raw-evidence path.

## Context routing

`ThreadConsolidationControlAdapter.context(state, decision)` intercepts only a decision explicitly tagged `THREAD_CONSOLIDATION`.

It requires that the matching signal snapshot actually owned that route.

A caller cannot forge the context merely by constructing a decision with the reason code.

The adapter then uses the **same ledger-event snapshot used for its synthesis-pressure decision** to:

1. run the exact thread-consolidation planner;
2. verify that the work is still ready in that snapshot;
3. rebuild compact Knowledge State;
4. prepare the bounded `THREAD_CONSOLIDATION` Work Item.

The context includes:

- lower knowledge records;
- exact lower delta authority IDs;
- source partition IDs;
- pending source/partition counts;
- the consolidation pressure revision;
- explicit `synthesis_route = THREAD_CONSOLIDATION`.

Everything else delegates to the caller's context provider.

## Snapshot cache

The control adapter caches the ledger history and pressure projection for the current observed durable revision.

All Work Threads evaluated while the ledger remains at that revision reuse the same global consolidation projection rather than replaying history per thread.

For the selected thread, the exact event tuple used when its signal was produced is retained under:

```text
(thread_id, thread_state_revision)
```

The later context step reuses that tuple even if scheduler tracing appends a new event between signal evaluation and Work Item preparation.

This prevents the scheduler trace itself from changing which knowledge was selected for the decision it records.

The cache is cleared when a new durable revision is observed.

## Automatic runtime loop

The normal `RuntimeControlLoop` needs no new hierarchy API.

Use:

```text
signal_provider  = control.signals
context_provider = control.context
scheduler        = TracingScheduler(
    ledger,
    ThreadConsolidationScheduler(SchedulerV0(...), control=control),
)
```

Then a ready Work Thread follows the ordinary path:

```text
RuntimeControlLoop
      ↓
control.signals(state)
      ↓
synthesis_need
      ↓
Scheduler v0
      ↓
SYNTHESIS_NEEDED + THREAD_CONSOLIDATION
      ↓
control.context(...)
      ↓
WorkItem(purpose=SYNTHESIZE)
      ↓
WorkerRuntime / homogeneous WorkerBank
      ↓
KnowledgeDelta(kind=THREAD_CONSOLIDATION)
```

The new higher knowledge delta is provisional and its lower references automatically remove those sources from future consolidation pressure while the higher delta remains active.

If it is later retracted, the lower sources become pending again and pressure can return.

## End-to-end regression contract

The construction regression starts with two partition-local knowledge deltas from two historical partitions.

It requires the normal control loop to:

1. observe nonzero consolidation pressure;
2. choose generic synthesis;
3. tag the decision `THREAD_CONSOLIDATION`;
4. persist that final reason through scheduler tracing;
5. give the worker a bounded thread-consolidation context;
6. execute through ordinary Worker Runtime;
7. persist a higher `THREAD_CONSOLIDATION` delta referencing the two lower deltas;
8. project the higher delta as `PROVISIONAL`;
9. reduce the next consolidation pressure to zero.

No manual consolidation trigger is used in that path.

## Relationship to raw integration backpressure

There are now two distinct information-organization loops:

```text
raw evidence overload
      ↓
BACKPRESSURE synthesis
      ↓
partition-local knowledge
```

and:

```text
accumulated partition knowledge
      ↓
SYNTHESIS_NEEDED / THREAD_CONSOLIDATION
      ↓
thread-level knowledge
```

They share workers and persistence but consume different bounded views.

Scheduler v0 only sees generic numerical pressure and purpose; the route/context adapters preserve the semantic distinction.

## Non-goals

v0 does not add:

- learned consolidation scheduling;
- semantic retrieval;
- adaptive pressure thresholds;
- thread-consolidation width > 1;
- branch/topic consolidation;
- truth promotion;
- another scheduler implementation;
- another persistence store.

It only closes the control loop for the first higher integration level using the existing final architecture boundaries.
