# Partitioned integration width v0

## Purpose

PR #36 makes a large evidence backlog mechanically divisible into bounded thread-owned partitions.

This slice makes that partitioning executable as **parallel integration width** without changing Scheduler v0 itself.

The intended path is:

```text
BACKPRESSURE + SYNTHESIZE
        ↓
base scheduler decision
        ↓
partition-aware width wrapper
        ↓
N distinct integration partitions
        ↓
N WorkPreparations
        ↓
one homogeneous WorkerBank batch
        ↓
N independent AttemptResults
        ↓
one Research Ledger
```

## Why this is a wrapper

The scheduler's stable responsibility remains choosing:

- Work Thread;
- action;
- purpose;
- base allocation.

The amount of **mechanically available integration parallelism** is derived from the current partition plan, not learned reasoning.

Instead of putting partition knowledge into Scheduler v0, `PartitionedBackpressureScheduler` wraps any scheduler implementing the normal `choose(...)` contract.

This preserves:

- Scheduler v0 as a small inspectable baseline;
- future scheduler replacement;
- scheduler tracing wrappers;
- the same `SchedulerDecision` boundary.

## Activation boundary

Width is modified only when all are true:

1. the caller reports integration backpressure;
2. the delegate decision action is `SYNTHESIZE`;
3. the decision carries the `BACKPRESSURE` reason.

Exploration, verification, challenge, progression, pause, completion, and ordinary/final synthesis are not modified by this wrapper.

This matters because PR #35 reserves a permanent exploration lane under backpressure. A `BACKPRESSURE_EXPLORATION` decision remains exploration and is never reinterpreted as integration width.

## Width rule

Default provisional configuration:

```text
max_integration_width = 4
```

For a selected Work Thread:

```text
width = min(
    configured integration width,
    caller neural-attempt capacity,
    number of current non-empty integration partitions
)
```

Therefore the decision cannot allocate more integration workers than there are distinct partition contexts to give them.

No empty partition and no duplicate evidence work is created to satisfy a requested width.

The widened decision receives reason code:

`PARTITIONED_INTEGRATION`

## Partition ordering

When a Work Thread has more partitions than the current width, v0 orders partitions by:

1. larger backlog first;
2. older pending evidence first;
3. shard index;
4. stable partition ID.

This is an inspectable mechanical policy, not a semantic relevance score.

It can later be replaced without changing partition or worker contracts.

## Context routing

`PartitionedIntegrationContextRouter` intercepts only decisions marked:

- `SYNTHESIZE`;
- `BACKPRESSURE`;
- `PARTITIONED_INTEGRATION`.

It reprojects the current partition plan and creates one `WorkPreparation` per allocated partition.

Every preparation contains:

- a distinct partition ID;
- exact evidence IDs for worker authority;
- compact bounded evidence records;
- causal evidence-event IDs;
- source-thread ownership;
- partition backlog metadata.

The router verifies that evidence authority does not overlap across the batch.

All other decisions delegate unchanged to the supplied context provider, normally `PurposeContextRouter`.

## One neural batch

`RuntimeControlLoop` already supports `WorkPreparationBatch`.

Therefore no new execution mechanism is needed:

```text
one SchedulerDecision(width=N)
      ↓
WorkPreparationBatch(N)
      ↓
N WorkerAssignments
      ↓
WorkerRuntime.run_batch(...)
      ↓
WorkerBank.execute_batch(N requests)
```

The Worker Bank can execute the selected homogeneous workers vectorized on the device.

Each attempt still gets:

- its own attempt ID;
- its own worker ID;
- its own Work Item ID;
- its own evidence authority;
- its own result events.

Only neural execution is fused.

## Durable outputs

An integration worker still returns the ordinary `AttemptResult`:

- knowledge deltas;
- evidence dispositions;
- observations/failures if relevant.

Worker Runtime commits each result independently.

A malformed result from one partition therefore does not conceptually require a different persistence architecture.

## Relationship to knowledge-integration bandwidth

This width primitive gives the system one direct response when telemetry shows:

```text
evidence generation rate > unique evidence absorption rate
```

Before inventing semantic clustering or a learned integration controller, the runtime can simply spend more homogeneous workers on disjoint pending partitions.

Future traces can measure whether increasing integration width:

- raises unique evidence absorption / second;
- reduces backlog growth;
- reduces backlog age;
- increases duplicate processing;
- saturates storage/control overhead;
- changes knowledge-delta production.

No such performance result is claimed by this construction slice.

## Wrapper composition

A typical composition is:

```text
SchedulerV0
    ↓
PartitionedBackpressureScheduler
    ↓
TracingScheduler (optional)
```

and:

```text
domain context provider
    ↓
PurposeContextRouter
    ↓
PartitionedIntegrationContextRouter
```

Tracing should observe the **final widened SchedulerDecision** so allocation provenance records the actual integration width that consumed neural compute.

## Failure/inconsistency behavior

If the base scheduler requests backpressure synthesis but no current non-empty partition exists, the width wrapper leaves the base decision untouched.

The ordinary bounded context path can then reject the impossible empty synthesis state rather than silently fabricating work.

If the ledger changes between width selection and context preparation such that fewer partitions remain, the context router rejects the stale allocation instead of overlapping or inventing evidence authority.

## Non-goals

v0 does not add:

- semantic partition ranking;
- learned integration width;
- adaptive width tuning from telemetry;
- specialized integrator weights;
- cross-thread raw-evidence mixing;
- distributed execution;
- a new scheduler implementation;
- a claim that width 4 is optimal.

It only makes mechanically available integration parallelism executable behind existing final architecture contracts.
