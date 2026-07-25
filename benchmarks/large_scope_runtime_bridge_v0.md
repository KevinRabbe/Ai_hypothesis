# Large-Scope Runtime Bridge v0

## Purpose

This construction layer connects the frozen large-scope relevance benchmark to the persistent population runtime without changing the benchmark question or Worker v1.

It is not a new research gate and it does not claim an adaptive-allocation result.

The bridge exists so the same controlled large-scope workload can later exercise:

- persistent Work Threads;
- scheduler allocation provenance;
- per-worker scope allocation;
- durable evidence;
- scope coverage projection;
- worker rotation / repeated attempts;
- later adaptive width, depth, scope, and purpose.

## Equivalence invariant

For one fixed benchmark world, width, worker mode, checkpoint bank, and evidence configuration:

> Direct benchmark execution and persistent-runtime execution must inspect the same windows, use the same worker indices, and produce the same local continuous evidence.

The runtime path may add durable metadata and provenance, but it must not silently change the neural experiment.

## Runtime mapping

One inspected benchmark window becomes one `WorkItem`.

Each Work Item carries:

- one stable opaque `scope_region_id`;
- the frozen Step 1 window features and mask as bounded execution context;
- the source window index and deterministic seed metadata;
- the benchmark split and worker-mode identity;
- the region ID as a source reference.

The generic runtime remains unaware of relevance labels or tensor layout.

## Stable region identity

Large-scope region identity is derived from:

- benchmark version;
- split;
- world seed;
- window index;
- frozen local-window seed.

Those values are hashed into an opaque `scope-*` ID.

The ID is stable for the same logical source window while avoiding use of display coordinates or storage paths as durable identity.

## Worker execution adapter

`LargeScopeRuntimeWorkerBank` adapts the existing selected-worker execution path to the generic `WorkerBank.execute_batch(...)` contract.

It:

1. validates that every Work Item owns exactly one scope region;
2. maps persistent worker IDs back to frozen checkpoint indices;
3. batches all prepared windows through `forward_selected(...)`;
4. reuses Step 2 `build_evidence_matrix(...)`;
5. returns one ordinary `EvidenceContribution` per inspected region.

The contribution preserves:

- RELEVANT evidence;
- NOT_RELEVANT evidence;
- uncertainty;
- invalid-label mass;
- local top margin;
- local decoded label;
- source region;
- source window;
- worker index;
- benchmark split/mode/version.

No world-level acceptance threshold is added.

## Worker-mode controls

The direct benchmark has two causal controls and the runtime bridge must preserve both.

### `same_worker`

The same frozen checkpoint is reused across different inspected regions.

This isolates scope expansion from weight diversity.

A benchmark-specific planned worker selector is used because the generic runtime selector intentionally prefers distinct workers for ordinary width expansion.

### `diverse_workers`

Distinct independently weighted checkpoints inspect the same deterministic window prefix used by the direct benchmark.

This measures scope + weight diversity.

## Scheduler control

`FixedScopeScheduler` exists only to reproduce an exact benchmark width inside the persistent runtime.

It is not Scheduler v0 and is not an adaptive policy.

The fixed scheduler is wrapped by the generic `TracingScheduler`, so every allocation is recorded as `SCHEDULER_DECISION_RECORDED` and remains joinable to its attempts and outcomes.

## Durable causal chain

The expected runtime history is:

```text
scheduler decision
    -> Work Items with exact scope_region_ids
    -> ATTEMPT_STARTED
    -> selected Worker v1 execution
    -> EVIDENCE_ADDED
    -> ATTEMPT_COMPLETED
```

From the same ledger history the runtime can independently derive:

- allocation outcomes;
- per-region coverage;
- worker redundancy/diversity on a region;
- resource usage;
- original evidence provenance.

No separate benchmark-state database is introduced.

## Required regression

The bridge is considered structurally correct only when a deterministic fake selected-worker bank demonstrates that direct and persistent execution have equal:

- inspected window indices;
- worker indices;
- local decoded labels;
- RELEVANT evidence;
- NOT_RELEVANT evidence;
- uncertainty;
- invalid-label mass;
- top margin.

The same regression must also show:

- all scoped attempts point to the recorded scheduler decision;
- scope coverage contains the exact inspected regions;
- allocation-outcome projection sees the expected width and evidence count;
- the same-worker control genuinely reuses one checkpoint across distinct regions.

## Non-goals

This bridge does not establish:

- that persistent execution improves benchmark quality;
- that adaptive routing helps;
- the best exploration policy;
- a learned scheduler;
- a world-level relevance threshold;
- a large-population advantage.

Those remain empirical questions. The bridge only ensures that future runtime experiments can answer them without changing the underlying workload while the experiment is in progress.
