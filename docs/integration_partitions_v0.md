# Integration partitions v0

## Purpose

Knowledge integration may become the primary scaling bottleneck long before neural worker compute is exhausted.

A single hot Work Thread can eventually own more unresolved evidence than one bounded synthesis attempt can process efficiently.

v0 adds a mechanical scale-out primitive:

```text
pending evidence
      ↓
source Work Thread
      ↓
deterministic hash shard
      ↓
bounded integration partition
      ↓
ordinary SYNTHESIZE Work Item
```

This is **not** semantic clustering and does not introduce a new integration service.

## Architecture rule

> Partition integration load without changing evidence ownership, worker architecture, or the Research Ledger contract.

Every evidence item remains durably owned by the Work Thread where it was generated.

Inside that Work Thread, its stable `evidence_id` determines one shard:

```text
sha256(evidence_id) → shard index
```

The same evidence ID, shard count, and Work Thread therefore produce the same partition across replay/restart.

## Default configuration

```text
shard_count = 8
batch_limit = 32
```

Both are provisional systems parameters.

They answer different questions:

- `shard_count`: how many mechanically independent integration lanes may exist per Work Thread;
- `batch_limit`: how much pending evidence one bounded synthesis attempt may receive.

No claim is made that 8 or 32 is optimal.

## Source-thread ownership

The partition key is:

```text
(source thread ID, shard index)
```

Evidence from different Work Threads is never mixed merely because the hashes collide onto the same numeric shard.

That preserves:

- Work Thread continuity;
- authority boundaries;
- per-thread integration pressure;
- provenance;
- later graph/fork/merge reasoning.

Cross-thread synthesis remains a higher-level knowledge operation, not an accidental consequence of storage sharding.

## Stable partition identity

Partition identity contains:

- a full SHA-256 digest of the source-thread identity;
- shard index;
- shard count;
- partition schema/version marker.

The raw thread ID remains present separately in the projection.

Changing the shard count intentionally creates a different partition identity space.

## Bounded-memory projection

The projection does **not** create another in-memory copy of the evidence backlog.

For each non-empty partition it retains only:

- full backlog count;
- oldest pending sequence;
- source thread;
- shard identity;
- at most `batch_limit` compact `PendingEvidence` records.

Therefore a partition containing one million unresolved evidence records can still expose a 32-record next batch while retaining the true backlog count.

The Research Ledger remains the canonical complete history.

## Two-pass replay

The projector uses two bounded logical passes over the supplied event history:

1. resolve final evidence dispositions;
2. count still-pending evidence and retain only the next bounded records per partition.

This prevents already-resolved early evidence from occupying the partition's limited active buffer.

No new durable index or database is introduced in v0.

If replay cost later becomes measurable, the same projection can be incrementally indexed behind this contract.

## Causal integrity

The projector rejects:

- non-increasing ledger sequence order;
- duplicate durable evidence IDs;
- a disposition recorded before the referenced evidence exists.

Unknown disposition references that never resolve to evidence do not remove real backlog; the separate integration telemetry surface already exposes that integrity noise.

## Immutable plan boundary

`IntegrationPartitionPlan` freezes together:

- ledger revision;
- shard count;
- batch limit;
- exact non-empty partitions;
- total unresolved backlog count.

Work preparation consumes:

```text
(plan, partition_id)
```

rather than accepting revision and limits separately.

This prevents callers from accidentally combining one partition with policy/provenance from another projection snapshot.

## Partition Work Item

`prepare_partition_integration_work(...)` converts one bounded partition into an ordinary `WorkPreparation`.

The worker receives:

- exact evidence IDs as its authority references;
- compact evidence records;
- evidence event IDs for causal provenance;
- integration revision;
- partition ID;
- source thread;
- shard index/count;
- full partition backlog count;
- oldest pending sequence.

The context is marked:

```text
context_view   = SYNTHESIZE
synthesis_mode = INTEGRATION_PARTITION
```

Constraints request:

- structured knowledge deltas;
- explicit dispositions for consumed evidence;
- preservation of source-thread ownership.

No special integration model is required.

## Relationship to purpose-aware context routing

PR #34 establishes:

```text
BACKPRESSURE SYNTHESIZE → bounded evidence context
```

Partitions are the scale-out form of that same context.

At low volume, one thread-level pending batch is enough.

At higher volume, several non-overlapping partitions of the same Work Thread can be prepared independently and executed through the same homogeneous Worker Bank.

The semantic worker contract does not change.

## Relationship to permanent exploration

PR #35 preserves a nonzero exploration lane even while integration is overloaded.

Integration partitions solve a different problem:

> increase the amount of integration work that can be processed in parallel when the integration lane itself needs more width.

Together the intended future shape is:

```text
large worker population
      ↓
massive evidence production
      ↓
thread-owned evidence queues
      ↓
mechanical integration partitions
      ↓
parallel bounded synthesis
      ↓
compact provisional knowledge
      ↓
verification / challenge
```

## Why no semantic clustering yet

Hash partitioning does not try to decide which evidence records are related.

That is deliberate.

Semantic clustering would introduce new architecture-specific questions:

- embedding choice;
- clustering threshold;
- reassignment behavior;
- rare-evidence preservation;
- cluster drift;
- retrieval cost.

None is required simply to prove that integration load can be divided mechanically behind stable contracts.

Semantic grouping can later operate *inside* or *above* these partitions if measured traces show value.

## Future hierarchy

This primitive is intentionally compatible with later levels:

```text
raw evidence partitions
      ↓
thread-local knowledge deltas
      ↓
branch/topic synthesis Work Threads
      ↓
cross-topic knowledge deltas
      ↓
global knowledge state
```

Higher levels should consume compact knowledge/deltas rather than replaying all raw lower-level evidence.

That future hierarchy does not require a new worker architecture or a new communication protocol.

## Non-goals

v0 does not add:

- semantic evidence clustering;
- a learned integration router;
- cross-thread evidence mixing;
- distributed storage;
- another canonical queue database;
- specialized integration weights;
- integration-width scheduler policy;
- automatic topic hierarchy creation;
- claims about optimal shard count or batch size.

Those remain later policy/measurement questions behind the same final runtime boundaries.
