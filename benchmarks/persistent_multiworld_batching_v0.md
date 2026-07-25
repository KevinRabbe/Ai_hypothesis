# Persistent Multi-World Batching v0

## Purpose

The direct large-scope runner already amortizes neural launch overhead across many generated worlds. The persistent runtime must be able to do the same before its end-to-end cost can be compared fairly.

This protocol composes:

- persistent Work Threads;
- per-thread scope coverage;
- restart-safe checkpoint sequences;
- cross-thread control batching;
- one homogeneous WorkerBank call per round.

It does not add a new neural model or routing policy.

## Round model

One benchmark world owns one persistent Work Thread.

For `world_count = W` and fixed per-world `step_width = K`, one round creates:

```text
W scheduler decisions
W × K scoped Work Items
1 WorkerRuntime batch
1 WorkerBank call
```

Example:

```text
3 worlds × width 2
    -> 3 independent scheduler decisions
    -> 6 local window attempts
    -> 1 neural call
```

Each world's scheduler and attempt provenance remains independent even though execution is fused.

## Scientific equivalence

Before any adaptive deviation or failure, after `R` rounds the persistent result for each world must equal a direct benchmark run at:

```text
direct width = R × step_width
```

while that width remains within the world size and before redundancy begins.

Equality means the same:

- window order;
- worker-index order;
- local labels;
- RELEVANT evidence;
- NOT_RELEVANT evidence;
- uncertainty;
- invalid-label mass;
- top margin;
- candidate/rank diagnostics.

Cross-world batching is therefore allowed to change launch count and runtime overhead, not the experiment's learned observations.

## Thread-aware worker selection

Different worlds have different deterministic checkpoint orders because worker planning is seeded by each world.

The generic control loop now supports an optional:

```text
choose_many_for_thread(thread_id, ...)
```

worker-selector hook.

`WorkerSelectorV0` implements the hook by delegating to its previous behavior, so existing generic semantics remain unchanged.

Selectors that do not implement the hook still use the legacy `choose_many(...)` path.

The multi-world benchmark selector delegates each thread to its own `PersistentScopeWorkerSelector`.

Therefore:

- each world continues its own deterministic worker sequence;
- restart continuation still comes from the previous persisted worker ID for that thread;
- the same-worker control may choose a different single checkpoint for different worlds, while reusing that checkpoint within each world;
- diverse-worker mode cycles independently per world.

## Thread identity

Each world/mode pair gets one stable opaque thread ID derived from:

- benchmark version;
- split;
- world seed;
- worker mode;
- window count.

Format:

```text
scope-world-<digest>
```

Display coordinates or storage paths are not identity.

## Resume contract

Every world thread retains the same frozen metadata used by the single-world persistent baseline:

- benchmark version;
- split;
- world seed;
- worker mode;
- window count;
- step width;
- population width;
- ordered worker-bank identity;
- evidence configuration.

A reconstructed multi-world experiment can therefore continue rounds from the same ledger without restarting scope or worker sequences.

An unrelated active Work Thread causes construction to fail. v0 intentionally expects a dedicated experiment ledger/chunk so the round can guarantee that every active benchmark world receives exactly one decision.

## Batch invariants

For a batch containing `W` worlds and step width `K`:

1. `run_many(max_threads=W, max_attempts=W×K)` is used.
2. Every world must receive exactly one decision in the round.
3. The returned neural-attempt count must equal `W×K`.
4. All assignments execute through one WorkerBank call.
5. Results are partitioned back to the originating Work Threads.
6. Coverage and worker continuation are derived independently per thread.

The experiment raises if a round violates the expected thread or neural budget.

## Chunking strategy

A future full runner can process large datasets in bounded chunks such as 64 worlds:

```text
64 worlds × width 2 = 128 local attempts / WorkerBank call
```

Then the next persistent round reuses the same 64 Work Threads and performs the next 128 local attempts in one call.

This mirrors the direct benchmark's bounded world batching while preserving long-lived state.

No unbounded in-memory world set is required.

## Required regression

A deterministic fake selected-worker bank must show that:

- three worlds × width 2 use one six-window WorkerBank call per round;
- two rounds use two WorkerBank calls total, not six;
- each world after two rounds equals its direct width-4 control;
- process reconstruction between rounds preserves every world's source progression and worker sequence;
- same-worker mode uses one checkpoint per world while still batching all worlds together;
- duplicate world identity is rejected before creating ledger history.

## What this does not establish

This protocol does not establish:

- the best world chunk size;
- the best step width;
- an adaptive routing advantage;
- asynchronous scheduling;
- compiler/CUDA-graph gains;
- real Worker v1 persistent throughput.

Its purpose is to remove an artificial per-Work-Thread launch penalty before those questions are measured.
