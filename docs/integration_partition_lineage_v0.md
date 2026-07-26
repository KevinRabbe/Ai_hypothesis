# Integration partition lineage v0

## Purpose

Partitioned integration now has three durable layers:

```text
scheduler decision
      ↓
partition allocation provenance
      ↓
worker attempts
      ↓
knowledge deltas / evidence dispositions
```

This slice joins those layers into one rebuildable lineage view.

The key architecture rule is:

> Higher integration levels must consume the knowledge actually produced by historical partitions, not infer old partition membership from today's routing configuration.

## Why durable lineage is necessary

A partition ID depends on routing configuration such as shard count.

If the system later changes from 8 to 16 shards, replaying current routing rules over old evidence is not a trustworthy way to answer:

- which partition was actually assigned;
- which worker attempted it;
- which knowledge delta came from it;
- which raw evidence fed that delta.

PR #39 records the concrete bounded allocation before neural execution.

`PartitionedIntegrationLineageProjector` combines that event with #38's allocation-outcome projection.

## Historical partition record

`HistoricalIntegrationPartition` preserves:

- partition ID;
- shard index;
- backlog count at allocation time;
- oldest pending sequence at allocation time;
- exact bounded evidence IDs assigned.

It is historical provenance, not a current queue object.

## Partition-attempt lineage

`PartitionAttemptLineage` binds one historical partition to the exact attempt that consumed its evidence authority.

The attempt side already includes:

- attempt ID;
- worker ID;
- input evidence IDs;
- terminal status;
- dispositions;
- knowledge delta IDs;
- knowledge-referenced evidence.

Therefore one lineage record can answer:

```text
partition P
    ↓
worker W / attempt A
    ↓
knowledge deltas K1, K2
    ↓
raw evidence E...
```

without reconstructing worker hidden state.

## Exact matching rule

An attempt maps to a partition only when its assigned evidence tuple exactly matches that partition's recorded evidence tuple.

Approximate matching is forbidden.

This prevents a corrupted or stale Work Item from being attributed to the wrong historical partition merely because the two assignments overlap.

One partition may map to at most one started attempt for one scheduler allocation.

## Unstarted partitions

The allocation provenance may contain more partitions than eventually reached `ATTEMPT_STARTED`.

For example, the process might fail after planning width 4 but before all four starts were durably appended.

The lineage therefore exposes:

`unstarted_partition_ids`

rather than silently treating the missing attempts as completed or deleting the allocation intent.

This preserves the distinction between:

- allocated work;
- started work;
- terminal work.

## Legacy histories

Partitioned integration allocations created before durable partition provenance exists cannot be reconstructed exactly.

The projector does **not** guess.

It returns the allocation with:

- no provenance event;
- no partition records;
- its decision ID listed in `missing_provenance_decision_ids`.

`PartitionedIntegrationLineageSnapshot.require_complete()` rejects such a history when exact lineage is required.

This gives two modes:

### Analysis mode

Inspect old histories while explicitly seeing missing provenance.

### Consolidation/reproducibility mode

Require complete provenance and refuse to build higher-level knowledge on inferred historical partition identity.

## Causal validation

Partition provenance must satisfy:

```text
scheduler decision
      ↓
partition allocation provenance
      ↓
ATTEMPT_STARTED
```

The lineage projector rejects:

- partition provenance for a non-partitioned decision;
- duplicate provenance for one decision;
- thread mismatch;
- width mismatch;
- provenance preceding the scheduler decision;
- provenance recorded after an attempt already started;
- malformed schema/config;
- duplicate partition IDs;
- shard index outside shard count;
- partition evidence larger than recorded batch limit;
- evidence authority overlapping between partitions;
- event reference order differing from partition payload order;
- attempts whose evidence authority does not exactly match any recorded partition;
- one partition mapped to multiple attempts.

The underlying allocation-outcome projector separately validates the full evidence/decision/attempt/output/terminal causal chain.

## Relationship to the next hierarchy level

Raw evidence should not be repeatedly reprocessed at every integration level.

The intended hierarchy is now executable in provenance terms:

```text
raw evidence
      ↓
partitioned synthesis
      ↓
partition-local provisional knowledge deltas
      ↓
thread-level consolidation
      ↓
branch/topic consolidation
      ↓
global knowledge state
```

Thread-level consolidation can now select the `knowledge_delta_ids` produced by completed partition attempts for one source Work Thread.

It does not need:

- raw worker thought traces;
- current shard configuration;
- all raw evidence payloads.

If deeper provenance is needed, every knowledge delta still references its source evidence and causal ledger events.

## Important quality constraint

Hash partitioning is mechanical, not semantic.

Related evidence may land in different raw partitions.

Therefore partition-produced knowledge should be treated as an intermediate compression layer, not necessarily the final thread conclusion.

A later consolidation pass exists specifically to reconnect information across partitions before promoting stronger knowledge.

This preserves the project's rare-evidence rule:

> local compression may reduce active context, but original evidence and provenance remain recoverable.

## Non-goals

v0 does not add:

- thread-level consolidation policy;
- semantic clustering;
- a learned hierarchy router;
- confidence aggregation;
- knowledge promotion;
- another persistence store;
- a new worker architecture.

It supplies the exact historical lineage that those future operations can safely consume.
