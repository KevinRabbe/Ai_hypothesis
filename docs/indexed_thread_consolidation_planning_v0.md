# Indexed Thread-Consolidation Planning / Pressure v0

## Purpose

The existing thread-consolidation planner and pressure projector are semantically correct but rebuild their inputs from the complete Research Ledger:

```text
full ledger
  ↓
PartitionedIntegrationLineageProjector
  +
KnowledgeStateProjector
  ↓
pending partition knowledge / pressure
```

After #46 and #49, both required inputs already have rebuildable materialized views.

This slice changes the composition to:

```text
incremental partition→knowledge lineage
        +
incremental Knowledge State
        ↓
indexed consolidation planner / pressure
```

No raw ledger replay is required.

## Existing contracts remain canonical

`IndexedThreadConsolidationPlanner` returns the existing:

```text
ThreadConsolidationPlan
```

`IndexedThreadConsolidationPressureProjector` returns the existing:

```text
ThreadConsolidationPressureOverview
```

Therefore these remain unchanged:

- `prepare_thread_consolidation_work(...)`;
- `ThreadConsolidationSource`;
- generic scheduler `synthesis_need`;
- Scheduler v0;
- worker/runtime contracts.

## Exact revision input

Both indexed projections require an explicit canonical Research Ledger sequence.

They advance:

- partition lineage through that sequence;
- Knowledge State through that sequence.

The only canonical-ledger read needed by the composition layer is the single boundary event used to pin the Knowledge State projector to the exact revision.

This prevents later assessments or consolidations from leaking into an older scheduler decision.

The materialized views are forward-only. If another consumer has already advanced the same instance beyond the requested revision, it refuses rewind rather than silently returning future state.

## Indexed planner semantics

For one selected Work Thread, the planner:

1. loads exact materialized partition lineage at revision N;
2. rejects missing durable partition provenance for that Work Thread;
3. loads current Knowledge State for the thread at N;
4. maps partition-produced delta IDs back to their historical partition IDs;
5. excludes retracted lower knowledge;
6. derives lower deltas already consumed by active `THREAD_CONSOLIDATION` knowledge;
7. rejects causal inversion where a higher consolidation references lower knowledge created later;
8. groups remaining lower deltas by partition;
9. applies the existing bounded round-robin selection;
10. returns the existing `ThreadConsolidationPlan`.

### Selection policy remains unchanged

Default planner configuration remains owned by `ThreadConsolidationConfig`:

```text
selection_limit = 32
minimum_source_deltas = 2
```

This slice does not claim those numbers are optimal.

## Indexed pressure semantics

The pressure projector reproduces the existing one-pass pressure contract across all Work Threads:

- pending partition-produced knowledge count;
- pending source-partition count;
- normalized synthesis pressure;
- incomplete-provenance thread IDs.

The existing pressure mapping remains:

```text
pending < minimum_source_deltas
    -> 0

otherwise
    -> min(1, pending / full_pressure_count)
```

No scheduler weight or threshold changes here.

## Retraction behavior

An active higher `THREAD_CONSOLIDATION` delta consumes only lower partition deltas it explicitly references.

If that higher delta is retracted in Knowledge State:

```text
active consolidation
    ↓ RETRACTED
lower partition sources become pending again
    ↓
consolidation pressure rises again
```

Because consumption is derived from current Knowledge State, no mutable consumed flag is required.

Retracted lower partition knowledge remains excluded.

## Missing provenance

The planner preserves the existing thread-local strictness:

- missing provenance on the selected Work Thread → planning rejected;
- missing provenance on another Work Thread → does not block the selected thread.

The all-thread pressure projector marks incomplete threads and assigns them zero automatic consolidation pressure rather than guessing historical routing.

## Regression equivalence

The same canonical histories are fed into:

```text
ThreadConsolidationPlanner
vs
IndexedThreadConsolidationPlanner
```

and:

```text
ThreadConsolidationPressureProjector
vs
IndexedThreadConsolidationPressureProjector
```

The returned dataclasses must compare equal.

Covered transitions include:

- three pending partition deltas across two partitions;
- identical bounded round-robin selection;
- active higher consolidation reducing pending sources;
- higher consolidation retraction reopening sources;
- exact historical revision before vs after consolidation;
- selected-thread missing provenance;
- incomplete-thread pressure suppression.

A hostile ledger whose `read_all_events()` raises immediately is used to require the indexed path to operate entirely through bounded canonical reads + materialized views.

## Cost shape

This slice changes:

```text
O(total raw ledger history replay)
```

into approximately:

```text
O(new events required to advance materialized views)
+
O(relevant materialized lineage/current-knowledge rows inspected)
```

It intentionally does **not** introduce another pressure cache yet.

The derived-row scan is substantially smaller than replaying every event/payload, but it can still grow with accumulated partition knowledge. Real traces should determine whether another materialized current-pressure projection is justified.

## Why not require one shared derived SQLite file yet?

The current materialized indexes can already coexist in the same derived SQLite database file because their table/meta namespaces are separate.

That could enable direct SQL joins later.

This slice does not make colocation mandatory because:

- deployment flexibility is useful;
- semantic equivalence can be proven first;
- a Python join over compact derived rows may already be sufficient;
- forcing a storage topology before profiling would be premature.

If profiling shows the derived-row join is hot, the same contracts can be implemented as SQL joins without changing scheduler/consolidation semantics.

## Next boundary

The current `ThreadConsolidationControlAdapter` still caches a complete event tuple to preserve signal→context snapshot identity.

The next slice should replace that with a much smaller cache:

```text
signal time at revision N
    ↓
indexed pressure + indexed bounded plan
    ↓
cache only route ownership + bounded WorkPreparation
    ↓
scheduler decision
    ↓
context returns the already-pinned bounded preparation
```

That removes the final full-ledger dependency from automatic thread-consolidation routing without requiring materialized-view rewind after scheduler tracing appends new events.

## Research status

No hierarchy level, scheduler heuristic, consolidation selection rule, knowledge semantics, verification policy, or worker architecture changes.

This slice changes only how current consolidation demand is reconstructed.
