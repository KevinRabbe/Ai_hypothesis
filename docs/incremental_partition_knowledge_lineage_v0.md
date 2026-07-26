# Incremental Partition → Knowledge Lineage v0

## Purpose

Thread-level consolidation needs one durable relationship:

```text
partitioned scheduler decision
        ↓
historical partition allocation
        ↓
ATTEMPT_STARTED
        ↓
partition-produced KnowledgeDelta
```

The existing `PartitionedIntegrationLineageProjector` reconstructs this correctly from the full Research Ledger. That remains the semantic/reference implementation.

At large history, replaying all evidence, scheduler decisions, attempts, outputs and provenance on every consolidation-pressure query is unnecessary.

`SQLiteIndexedPartitionKnowledgeLineage` materializes only the immutable routing relationship needed by higher integration levels.

The Research Ledger remains canonical.

## Deliberately narrow ownership

The index owns derived historical routing/provenance only:

- partitioned scheduler decision identity;
- decision thread / sequence / projection revision / width;
- durable partition-allocation event identity and sequence;
- historical shard count / batch limit / plan revision;
- ordered partition descriptors;
- exact bounded partition evidence authority;
- attempt → partition mapping;
- partition-produced knowledge-delta identity;
- unstarted allocated partitions;
- missing durable partition provenance.

It does **not** own:

- evidence payloads;
- evidence dispositions as state;
- full attempt outcome telemetry;
- knowledge status;
- verification state;
- consolidation consumption state;
- worker hidden state;
- prompts/context;
- scheduler pressure.

Knowledge status remains owned by Knowledge State. Integration throughput/outcome telemetry remains owned by its existing projections.

## Why evidence identities appear in this index

The replay allocation projector distinguishes:

- evidence that existed before an attempt;
- non-evidence Work Item authority;
- evidence that only appears later in history.

To preserve that causal validation without copying evidence payloads, the sidecar stores only:

```text
evidence_id
creation_sequence
```

It also stores ordered attempt references.

This permits two important checks:

1. a provenance-complete partition attempt must refer exactly to already-existing evidence;
2. a legacy attempt may contain unknown/non-evidence references, but if one of those IDs is later created as evidence the history is rejected as future-evidence causal inversion.

No evidence body is duplicated.

## Durable decision classification

Only scheduler decisions with all of:

```text
action = SYNTHESIZE
BACKPRESSURE
PARTITIONED_INTEGRATION
```

enter this materialization.

Ordinary synthesis and width-1 non-partitioned backpressure work remain outside it.

## Allocation provenance

`INTEGRATION_PARTITION_ALLOCATION_RECORDED` remains the historical routing authority.

The index validates the same important invariants as the replay lineage projector:

- schema marker;
- referenced decision exists and is partitioned;
- one provenance event per decision;
- matching Work Thread;
- matching allocated width;
- provenance occurs after scheduler decision;
- provenance occurs before any attempt start;
- positive shard count / batch limit / backlog counts;
- non-negative shard indices / plan revision / oldest pending sequence;
- partition count equals width;
- unique partition IDs within one allocation;
- no evidence authority overlap across partitions;
- partition evidence count within batch limit;
- shard index below shard count;
- event reference order equals payload partition order.

Partition IDs are **not** global identities. The same logical partition ID may legitimately recur in later scheduler decisions. Historical partition identity is therefore:

```text
(decision_id, partition_id)
```

## Attempt mapping

When durable provenance exists, one `ATTEMPT_STARTED` maps to a partition only when its ordered evidence-authority tuple exactly equals one recorded partition tuple.

Approximate overlap is forbidden.

A partition may be started at most once within one allocation.

Starts may not exceed scheduler width.

The attempt must target the same Work Thread as the scheduler decision.

## Legacy missing provenance

Partitioned integration attempts created before durable allocation provenance existed remain representable.

The index does not infer their partition.

Their decision remains in:

```text
missing_provenance_decision_ids
```

and the public decision record has:

```text
provenance_complete = False
partitions = ()
sources = ()
```

`require_complete()` provides the strict reproducibility/consolidation boundary.

If a provenance event appears only **after** an attempt already started, the history is rejected rather than retroactively repaired.

## Attempt causal integrity

The public index does not expose terminal/disposition telemetry, but it observes those events as causal guards.

For indexed partition attempts:

```text
ATTEMPT_STARTED
    ↓
zero or more integration/knowledge outputs
    ↓
exactly zero or one terminal event
```

The sidecar rejects:

- output before `ATTEMPT_STARTED`;
- knowledge/disposition output after terminal;
- more than one terminal event;
- terminal before attempt start.

Terminal state is kept in a tiny private derived table only to enforce this ordering.

## Partition-produced knowledge

A `KNOWLEDGE_DELTA_RECORDED` becomes a partition knowledge source only when:

- its attempt belongs to a partitioned scheduler decision;
- the decision has durable allocation provenance;
- the attempt maps exactly to one historical partition;
- the knowledge event targets the same Work Thread;
- output causality is valid.

The index stores only:

```text
delta_id
attempt_id
decision_id
partition_id
thread_id
creation_sequence
```

Whether that delta is currently PROVISIONAL / VERIFIED / DISPUTED / RETRACTED is intentionally not copied here.

## Public snapshot

`IndexedPartitionKnowledgeSnapshot` exposes:

- exact materialized revision;
- ordered partitioned decisions;
- historical partitions;
- partition-produced knowledge sources;
- unstarted partition IDs;
- missing-provenance decision IDs.

It is intentionally smaller than `PartitionedIntegrationLineageSnapshot` because higher consolidation does not need disposition traffic, terminal status, progress flags, or other allocation-outcome telemetry.

## Exact snapshot boundaries

The index reuses `LedgerProjectionTail` and `ProjectionCheckpoint`.

It supports:

```text
sync()
sync_through(sequence)
snapshot()
snapshot_through(sequence)
rebuild()
```

The materialization is forward-only. An already-advanced instance refuses to rewind to an older requested snapshot.

This allows future consolidation control to pin lineage and Knowledge State to the same canonical scheduler revision.

## Transaction / rebuild semantics

A requested ledger tail is reduced transactionally. Any causal/provenance failure rolls back the derived changes and leaves the checkpoint at its previous safe revision.

Persistent sidecar storage binds to:

- lineage-index schema version;
- Research Ledger schema version;
- Research Ledger source identity;
- checkpoint sequence;
- checkpoint event ID.

The sidecar must use a separate SQLite file from the canonical Research Ledger.

`rebuild()` deletes all derived lineage rows and reconstructs them from canonical history.

## Regression strategy

The primary tests construct one canonical history and compare the new materialized view with `PartitionedIntegrationLineageProjector` for the overlapping semantic surface:

- decision identity / thread / width;
- provenance event and configuration;
- exact historical partitions;
- unstarted partition IDs;
- produced knowledge-delta IDs;
- missing-provenance decisions.

Adversarial coverage includes:

- provenance after attempt start;
- exact attempt/partition evidence mismatch;
- future evidence referenced by an earlier legacy attempt;
- knowledge output after terminal;
- repeated logical partition IDs across different decisions;
- exact historical snapshots;
- >1,000-event tail pagination;
- persistent restart;
- rebuild equality;
- rewind rejection;
- same-file canonical/sidecar storage rejection.

## Scaling role

This slice removes one full-history join, but it deliberately does **not** yet make thread-consolidation pressure itself incremental.

The intended next composition is:

```text
partition lineage index
        +
indexed Knowledge State
        ↓
thread-consolidation pressure/current pending-source projection
        ↓
generic synthesis_need
```

That next layer can derive mutable pending/consumed state from two current materializations without replaying the raw Research Ledger.

## Research status

No scheduler policy, consolidation selection policy, knowledge status semantics, hierarchy depth, worker architecture, or truth-promotion rule changes here.

This slice changes only the cost of reconstructing immutable partition-to-knowledge routing history.
