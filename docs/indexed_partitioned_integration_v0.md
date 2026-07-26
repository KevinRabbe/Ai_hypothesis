# Indexed Partitioned Raw Integration v0

## Purpose

After #51, automatic higher-level thread consolidation can run from pinned materialized state without replaying the Research Ledger.

The main remaining scheduler→context hot path was the **raw evidence side**:

```text
PartitionedBackpressureScheduler
    ↓
IntegrationPartitionAllocator.plan(ledger)
    ↓
IntegrationPartitionProjector.project(ledger.read_all_events())
```

The context router then repeated that partition projection again after scheduler tracing.

That is the wrong cost shape for a large evidence population.

This slice replaces it with:

```text
incremental integration identity/backlog index
        ↓
classify each newly pending evidence into its #36 shard once
        ↓
indexed per-shard backlog/count/oldest queries
        ↓
one immutable partition plan at scheduler revision N
        ↓
cache selected partitions under scheduler decision ID
        ↓
trace append / provenance append / worker execution
```

No full ledger replay is required.

## Existing shard algorithm remains authoritative

This slice deliberately does **not** implement another hash function.

The deterministic shard assignment frozen by #36 remains owned by:

```text
IntegrationPartitionProjector
```

For each newly seen pending evidence identity, `IndexedIntegrationPartitionPlanner` loads that one canonical `EVIDENCE_ADDED` event and asks the existing projector to classify the single event.

The resulting:

```text
shard_index
partition_id
```

are persisted in a tiny auxiliary table beside the existing incremental integration index.

Therefore:

- no shortened SHA digest is introduced;
- no alternate modulo rule is introduced;
- no semantic clustering is introduced;
- no partition identity rule changes.

Each evidence item is classified once under the configured shard count, rather than reclassified on every scheduler cycle.

## Reuse of the integration sidecar

No second evidence-status database is created.

The planner reuses #45/#48 tables for:

- evidence ID;
- canonical evidence event ID;
- Work Thread ownership;
- creation sequence;
- current pending/disposition state.

It adds only:

```text
integration_partition_assignment
--------------------------------
evidence_id
shard_count
shard_index
partition_id
```

and a tiny partition-config metadata table.

Evidence payloads remain in the canonical Research Ledger.

## Configuration changes

Shard assignment depends on `shard_count`.

The auxiliary assignment table binds to the active shard count. If a planner with a different shard count opens the same derived integration database, the auxiliary assignments are discarded and lazily rebuilt from the already-materialized evidence identities.

This rebuild does not require replaying all ledger events.

`batch_limit` does not affect shard identity, so it is not part of assignment identity; it only bounds the worker-facing record buffer.

## Indexed partition plan

For one Work Thread at exact revision N, the planner:

1. advances the incremental integration tracker exactly through N;
2. classifies only currently pending evidence identities missing an assignment;
3. queries pending backlog count and oldest sequence per shard;
4. queries only the next bounded `batch_limit` evidence rows per non-empty shard;
5. fetches canonical evidence payloads only for those bounded rows;
6. builds an immutable `IndexedIntegrationPartitionPlan`.

The public plan preserves the #36 information required by scheduling and provenance:

- revision;
- Work Thread;
- shard count;
- batch limit;
- stable partition ID;
- shard index;
- total backlog count;
- oldest pending sequence;
- bounded evidence records / IDs / causal event IDs.

It validates non-overlapping evidence authority across partitions.

## Execution ordering remains unchanged

Non-empty partitions are ordered exactly by the existing #37 mechanical rule:

1. larger backlog;
2. older pending evidence;
3. shard index;
4. stable partition ID.

No semantic relevance score is introduced.

## WorkPreparation contract remains unchanged

The indexed planner emits the same bounded partition WorkPreparation shape:

```text
context_view = SYNTHESIZE
synthesis_mode = INTEGRATION_PARTITION
integration_revision = N
integration_partition = {
    partition_id,
    thread_id,
    shard_index,
    shard_count,
    backlog_count,
    oldest_pending_sequence,
}
pending_evidence = bounded compact evidence records
causal_event_ids = selected canonical evidence events
```

with the existing constraints:

```text
max_pending_evidence = batch_limit
emit_structured_knowledge_deltas = True
disposition_consumed_evidence = True
preserve_source_thread_ownership = True
```

Regression construction compares the resulting `WorkPreparationBatch` directly with the existing replay `IntegrationPartitionAllocator` on the same canonical history.

## Pinned scheduler allocation

`IndexedPartitionedBackpressureScheduler` wraps the same stable scheduler `choose(...)` contract.

It widens only when:

- integration backpressure is active;
- delegate action is `SYNTHESIZE`;
- delegate reason contains `BACKPRESSURE`.

At the `PinnedIndexedRuntimeSnapshotProvider.current_revision`, it builds **one** indexed plan for the selected Work Thread.

Actual width remains:

```text
min(
    configured max integration width,
    caller remaining neural-attempt budget,
    current non-empty partition count,
)
```

Widened decisions retain reason:

```text
PARTITIONED_INTEGRATION
```

The exact selected plan/partitions are cached under the scheduler decision ID.

## Context does not re-plan after tracing

Wrapper sequence:

```text
Pinned runtime snapshot at N
        ↓
SchedulerV0
        ↓
IndexedPartitionedBackpressureScheduler
    builds/caches plan at N
        ↓
TracingScheduler
    appends SCHEDULER_DECISION_RECORDED at N+1
        ↓
IndexedPartitionedIntegrationContextRouter
    returns cached partition batch from N
    persists exact allocation provenance
        ↓
ATTEMPT_STARTED × width
```

The context router performs no partition projection.

Therefore scheduler tracing cannot change:

- width;
- partition IDs;
- evidence authority;
- backlog metadata used by the decision;
- worker context.

## Durable allocation provenance

The context router persists the existing:

```text
INTEGRATION_PARTITION_ALLOCATION_RECORDED
```

contract with:

- scheduler decision ID;
- decision projection revision;
- partition-plan revision;
- shard count;
- batch limit;
- width;
- ordered partition descriptors;
- exact bounded evidence IDs.

Event identity remains:

```text
integration-partition-allocation-v0:<full SHA-256(decision_id)>
```

Repeated context preparation is idempotent when the actual assignment identity is unchanged.

Assignment identity compares:

- schema;
- decision ID;
- shard count;
- batch limit;
- width;
- ordered partition IDs;
- shard indices;
- exact evidence IDs.

Plan revision/backlog age/count are historical diagnostics and do not by themselves turn an identical retry into a conflicting allocation.

A genuinely different assignment under the same scheduler decision ID is rejected.

## No-replay runtime regression

A hostile Research Ledger whose:

```text
read_all_events(...)
```

raises immediately is used with:

```text
PinnedIndexedRuntimeSnapshotProvider
IndexedRuntimeIntegrationTracker
IndexedIntegrationPartitionPlanner
IndexedPartitionedBackpressureScheduler
TracingScheduler
IndexedPartitionedIntegrationContextRouter
IndexedRuntimeControlLoop
IndexedWorkerRuntime
```

With 32 pending evidence records, the regression requires:

- backpressure synthesis;
- width > 1 when multiple shards are non-empty;
- width bounded by configured maximum;
- one WorkerBank batch;
- distinct partition IDs;
- pairwise-disjoint evidence authority;
- exact pinned integration revision in every worker context;
- durable scheduler trace containing final widened width/reason;
- one durable partition-allocation provenance event matching worker authority.

No full ledger replay is permitted anywhere in that path.

## Backlog updates

The auxiliary shard table is only routing metadata.

Current pending state remains authoritative in the integration index.

When evidence is dispositioned, the next exact-revision partition query automatically excludes it without mutating or deleting its shard assignment.

Thus:

```text
stable evidence→shard identity
+
mutable integration pending status
```

remain separate concerns.

## Replay equivalence regression

On an ordinary Research Ledger, the same history is sent through:

```text
IntegrationPartitionAllocator
vs
IndexedIntegrationPartitionPlanner
```

The regression compares:

- ordered partition IDs;
- shard indices/count;
- backlog count;
- oldest pending sequence;
- bounded evidence IDs;
- compact evidence context records;
- final `WorkPreparationBatch`.

This keeps the replay projector as the semantic reference while the indexed planner changes only the systems cost.

## Cost shape

Normal repeated planning becomes approximately:

```text
O(new pending evidence needing first shard classification)
+
O(non-empty shard aggregate queries)
+
O(worker batch_limit × selected width canonical payload fetches)
```

instead of:

```text
O(total Research Ledger history)
```

or repeatedly transporting every pending evidence payload through Python.

## Remaining hot-path policy

After this slice, the core indexed scheduler paths are replay-free for:

- Work Thread / graph state;
- integration backlog/pressure;
- raw evidence partitioning/widening;
- worker continuity;
- durable generated-ID validation;
- Knowledge State / verification;
- partition→knowledge lineage;
- thread-consolidation planning/pressure;
- automatic thread-consolidation routing.

Remaining `read_all_events()` usages should now be classified before optimization:

- semantic replay/reference implementations: keep;
- scientific diagnostics/telemetry: usually keep unless measured hot;
- any still-reachable normal runtime path: optimize only when identified.

Do not materialize every projection merely because the pattern exists.

## Research status

No shard algorithm, scheduler heuristic, worker architecture, evidence semantics, knowledge semantics, hierarchy level, or truth-promotion rule changes here.

This slice changes only the cost and snapshot stability of partitioned raw evidence integration.
