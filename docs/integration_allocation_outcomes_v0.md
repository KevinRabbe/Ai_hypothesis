# Integration allocation outcomes v0

## Purpose

PR #37 makes backpressure integration width executable across deterministic non-overlapping evidence partitions.

This slice makes the result of each integration allocation observable without adding a reward function or new runtime logging.

The central question is:

> When integration width increases, does the system absorb more unique evidence and produce more useful compact knowledge, or merely execute more attempts and disposition traffic?

The projection is entirely read-only over existing Research Ledger events.

## Existing provenance is sufficient

No new write path is required.

The ledger already contains:

```text
SCHEDULER_DECISION_RECORDED
    decision_id
    width
    reason_codes
    thread_id
        ↓
ATTEMPT_STARTED
    scheduler_decision_id
    attempt_id
    worker_id
    exact Work Item reference IDs
        ↓
INTEGRATION_DISPOSITION_RECORDED
KNOWLEDGE_DELTA_RECORDED
ATTEMPT_COMPLETED / PARTIAL / FAILED / CRASHED / INVALID_RESULT
```

The projector joins those events by stable decision and attempt identity.

## Which decisions count

An allocation is classified as integration work only when its durable scheduler trace has:

```text
action = SYNTHESIZE
reason contains BACKPRESSURE
```

This includes both:

- ordinary width-1 backpressure synthesis;
- partitioned integration width carrying `PARTITIONED_INTEGRATION`.

Ordinary/final synthesis is excluded.

Therefore width summaries compare the same operational purpose rather than mixing final answer synthesis with backlog absorption.

## Attempt-level outcome

`IntegrationAttemptOutcome` records, for one traced attempt:

- attempt ID;
- worker ID;
- evidence IDs present in the Work Item authority;
- count of non-evidence input references;
- terminal event type, if any;
- `progress_made`, when durably present;
- input evidence dispositioned by the attempt;
- raw disposition-reference count;
- disposition references outside the original input authority;
- knowledge delta IDs produced;
- assigned input evidence referenced by those knowledge deltas.

It does not assign a usefulness score.

## Allocation-level outcome

`IntegrationAllocationOutcome` groups all attempts belonging to one scheduler decision.

It exposes:

### Width execution

- allocated width;
- started attempt count;
- terminal attempt count;
- width utilization.

An allocation with width 4 and only three starts is observable as 0.75 utilization instead of being silently treated as a successful width-4 run.

### Input authority

- total evidence references assigned across attempts;
- unique evidence assigned;
- duplicate input authority.

For properly partitioned integration:

```text
duplicate_input_authority_count = 0
```

If the same evidence is accidentally assigned to two workers, that duplication remains visible.

### Unique absorption versus raw traffic

The projection separates:

- raw disposition references;
- unique assigned evidence dispositioned;
- duplicate disposition references.

Example:

```text
worker A dispositions E17
worker B dispositions E17

raw disposition references       = 2
unique evidence absorbed         = 1
duplicate disposition references = 1
```

Width therefore cannot look better merely because it repeats the same integration work.

### Input absorption fraction

For one allocation:

```text
unique assigned evidence dispositioned
--------------------------------------
unique assigned evidence
```

This answers how much of the bounded input authority was actually resolved by the allocation.

It is not a semantic correctness metric.

A worker may disposition evidence as duplicate, irrelevant, invalid, local-only, or integrated; the separate integration telemetry surface retains those semantic categories globally.

### Knowledge production

The allocation records:

- number of knowledge deltas produced;
- how many unique assigned evidence IDs are referenced by those deltas.

This helps distinguish:

```text
high disposition throughput + little knowledge production
```

from:

```text
high disposition throughput + compact knowledge creation
```

It still does not prove that the resulting knowledge is correct. Knowledge remains provisional until verification/challenge changes its derived status.

## Width summaries

`summarize_integration_allocations_by_width(...)` groups observed allocations by their actual traced width.

For each width it reports:

- allocation count;
- partitioned allocation count;
- started attempts;
- terminal attempts;
- total unique assigned evidence across allocations;
- duplicate input authority;
- total unique assigned evidence dispositioned;
- raw disposition-reference traffic;
- duplicate disposition traffic;
- knowledge-delta production;
- assigned evidence referenced by knowledge deltas;
- mean per-allocation input absorption fraction.

These are observational summaries, not a recommendation for a width.

## Why totals are per allocation

The same evidence may legitimately appear in different allocations at different times—for example after a failed or partial attempt.

Width summaries therefore sum each allocation's internally unique counts rather than globally deduplicating an entire historical run.

This preserves the actual work performed at each allocation width.

Cross-time retry/reprocessing behavior can be studied separately from per-allocation overlap.

## Causal replay contract

The projector enforces the durable causal chain:

```text
evidence exists
      ↓
scheduler decision recorded
      ↓
ATTEMPT_STARTED
      ↓
output events
      ↓
exactly one terminal event
```

It rejects:

- non-increasing ledger sequences;
- duplicate durable evidence IDs;
- duplicate scheduler decision IDs;
- `ATTEMPT_STARTED` before its scheduler decision;
- attempt input referencing evidence created only later;
- duplicate attempt starts;
- attempt thread differing from its scheduler decision;
- more attempt starts than allocated width;
- output events preceding `ATTEMPT_STARTED`;
- more than one terminal event;
- integration/knowledge output after a terminal event;
- malformed integration trace fields.

Corrupt history must never be converted into apparently meaningful scaling statistics.

## Non-evidence references

A Work Item may contain references that are not evidence IDs, such as documents or other durable knowledge objects.

The projector counts those separately as non-evidence input references.

It does not reinterpret them as evidence.

Likewise, a disposition referencing something outside the attempt's original evidence authority remains visible as out-of-input traffic rather than inflating unique absorption.

Worker Runtime normally prevents unauthorized dispositions before persistence; the projector keeps this diagnostic because historical/imported/corrupt ledger data should still be auditable.

## Relationship to global integration bandwidth

PR #33 answers the system-level question:

> Is evidence entering the system faster than it is being absorbed?

This projector answers the allocation-level question:

> What did the worker width assigned to integration actually accomplish?

Together they support later analysis such as:

```text
width 1 → 2 → 4
      ↓
unique evidence absorbed / allocation
unique evidence absorbed / second
backlog growth / second
knowledge deltas / second
duplicate authority / traffic
terminal attempt fraction
```

Wall-clock rates still require external timing windows. This projection itself intentionally contains no timing assumptions.

## Interpreting integration width

A larger width is promising when, under comparable workloads, it increases:

- unique evidence absorption;
- knowledge production;
- backlog drain;

without unacceptable increases in:

- duplicate input authority;
- duplicate disposition traffic;
- failed/crashed attempts;
- runtime/storage overhead.

A larger width is not beneficial merely because more workers were active.

This directly follows the project's optimization target:

> useful uncertainty removed per unit compute, not worker activity.

## No reward function

v0 deliberately does not combine these fields into one scalar reward.

A scalar would force premature assumptions about the relative value of:

- dispositioning evidence;
- producing knowledge;
- preserving rare evidence;
- verification quality;
- compute cost;
- latency;
- novelty.

The raw causal measurements should exist first. A later scheduler-learning or allocation-value model can use them only after real workload evidence shows which outcomes matter.

## Non-goals

v0 does not add:

- adaptive integration width;
- a learned scheduler;
- a usefulness reward;
- semantic correctness scoring;
- wall-clock timing;
- new ledger events;
- new persistence schema;
- semantic evidence clustering;
- a claim that width 2 or 4 is better.

It only makes the outcome of existing integration allocations inspectable and causally trustworthy.
