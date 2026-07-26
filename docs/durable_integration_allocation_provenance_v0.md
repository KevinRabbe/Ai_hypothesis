# Durable integration allocation provenance v0

## Problem

Partitioned integration work previously carried exact partition identity only in transient worker context.

The Research Ledger preserved:

- scheduler decision ID and width;
- attempt ID / worker ID;
- exact evidence references;
- knowledge deltas and dispositions.

But after replay it could not prove the exact historical partition configuration that produced those attempts:

- partition IDs;
- shard count;
- batch limit;
- partition ordering;
- partition-plan revision.

That becomes important once integration configuration changes or higher integration levels consume partition-produced knowledge.

## Design

The bounded context router now records one append-only event when it materializes a partitioned integration decision:

```text
INTEGRATION_PARTITION_ALLOCATION_RECORDED
```

The event is part of the same Research Ledger. No new store or queue is introduced.

## What is persisted

The event contains only compact allocation identity/provenance:

- schema/version marker;
- scheduler decision ID;
- scheduler decision projection revision;
- integration partition-plan revision;
- shard count;
- partition batch limit;
- allocated width;
- ordered selected partition descriptors.

Each selected partition descriptor contains:

- partition ID;
- shard index;
- full partition backlog count at preparation time;
- oldest pending sequence at preparation time;
- exact bounded evidence IDs assigned to that partition attempt.

The event references the selected partition IDs directly.

## What is deliberately not persisted

The provenance event does not copy:

- evidence payloads;
- neural inputs/tensors;
- full Work Item context;
- full pending backlog;
- worker hidden state;
- prompts or generated reasoning.

The evidence payload remains canonical in its original `EVIDENCE_ADDED` event.

This preserves the architecture rule:

> persist identities, provenance and knowledge-changing results; keep active context bounded and reconstructable.

## Bounded size

The event is bounded by the already-bounded integration allocation:

```text
allocated width × partition batch limit
```

With current provisional defaults:

```text
4 workers × 32 evidence IDs = at most 128 evidence IDs
```

This is allocation provenance, not a duplicate backlog.

## Deterministic event identity

The provenance event ID is deterministically derived from the scheduler decision ID:

```text
integration-partition-allocation-v0:
    sha256(decision_id)
```

There can therefore be at most one canonical partition-allocation provenance event per scheduler decision.

## Retry semantics

Retry identity is based on the actual worker assignment, not incidental ledger revision changes.

Two preparations are considered the same allocation when they preserve:

- decision ID;
- shard count;
- batch limit;
- width;
- ordered partition IDs;
- shard indices;
- exact evidence IDs assigned to every partition.

Fields such as:

- partition-plan revision;
- backlog count;
- oldest pending sequence;

remain historical diagnostics, but do not by themselves make an otherwise identical retry a new allocation.

This matters because recording the provenance event itself advances the Research Ledger sequence without changing pending evidence.

### Conflict

If the same scheduler decision ID is later reused for a different evidence assignment, preparation fails with a provenance conflict.

A scheduler decision is therefore an immutable allocation identity.

New evidence or changed backlog may be handled by a **new** scheduler decision, never by silently mutating an old one.

## Why a ledger event instead of persisting worker context

The runtime does not need the whole active context for durable allocation provenance.

A dedicated typed event has several advantages:

- append-only history remains explicit;
- allocation identity is versioned;
- no giant context serialization path is introduced;
- no core WorkItem ABI change is required;
- replay can query integration allocation provenance directly;
- future storage/indexing can optimize this event type independently.

This follows the project's existing pattern: durable facts are events; current views are projections.

## Causal position

A normal partitioned integration lifecycle becomes:

```text
SCHEDULER_DECISION_RECORDED
        ↓
partition plan projected
        ↓
INTEGRATION_PARTITION_ALLOCATION_RECORDED
        ↓
ATTEMPT_STARTED × width
        ↓
knowledge deltas / evidence dispositions
        ↓
terminal attempt events
```

The allocation event therefore captures what the scheduler decision was concretely turned into before neural execution begins.

## Higher-level integration

This provenance is the bridge required for later hierarchical integration.

A higher-level projector can now answer exactly:

- which partition produced a knowledge delta;
- which raw evidence IDs fed that partition;
- which shard configuration was active;
- which scheduler decision spent the compute;
- how much parallel width was allocated.

That supports a future chain such as:

```text
raw evidence
    ↓
partitioned integration attempts
    ↓
partition-local knowledge deltas
    ↓
thread-level consolidation
    ↓
branch/topic consolidation
```

without inferring historical partition identity from current configuration.

## Non-goals

v0 does not persist:

- arbitrary Work Item context;
- every scheduler signal;
- neural hidden state;
- a semantic cluster assignment;
- distributed-storage location;
- a new integration hierarchy policy.

It records only the compact allocation identity necessary for trustworthy replay and later consolidation.
