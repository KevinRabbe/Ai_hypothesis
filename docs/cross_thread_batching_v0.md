# Cross-Thread Batching v0

## Purpose

The persistent runtime already batches multiple attempts belonging to one scheduler decision. Real workloads can also contain many independent Work Threads that are simultaneously ready for learned processing.

Without a cross-thread batching boundary, those threads would produce one WorkerBank/GPU launch per scheduler decision even when their workers share one homogeneous architecture.

Cross-thread batching v0 removes that avoidable launch boundary while keeping scheduling and provenance logically separate.

## Principle

> **Fuse homogeneous neural execution, not scheduler meaning.**

One control call may now produce several independent scheduler decisions:

```text
Work Thread A -> decision A -> Work Items A
Work Thread B -> decision B -> Work Items B
Work Thread C -> decision C -> Work Items C
                           |
                           v
                 one WorkerRuntime batch
                           |
                           v
                    one WorkerBank call
```

Every attempt still retains:

- its Work Thread;
- scheduler decision ID;
- Work Item ID;
- worker ID;
- scope/reference provenance;
- resource usage;
- evidence/knowledge outputs.

## Snapshot isolation

`RuntimeControlLoop.run_many(...)` projects Work Threads, dependency state, integration pressure, verification pressure, and scheduler signals once at the start of the call.

Every thread selected in that call comes from that same candidate snapshot.

A Work Thread can receive at most one scheduler decision per batch.

This intentionally means:

- completing a dependency during the batch does not unlock its dependents inside that same batch;
- pausing/completing one thread does not cause another eligibility re-projection mid-batch;
- new follow-up threads created by attempt results are not schedulable until the next control call.

The next batch sees the newly committed ledger state.

This keeps cross-thread batching deterministic and avoids order-dependent intra-batch control feedback.

## Two independent bounds

The caller supplies:

- `max_threads`: maximum number of distinct Work Threads receiving a decision;
- `max_attempts`: maximum total neural Work Items flattened into the WorkerBank call.

Per-thread scheduler width still applies.

Example:

```text
max_threads = 3
max_attempts = 3
scheduler prefers width = 2
```

may produce:

```text
Thread A -> width 2
Thread B -> width 1
Thread C -> deferred
```

The second scheduler call receives `max_width=1`, so a cooperative scheduler can consume the exact remaining attempt budget.

A scheduler that returns a width greater than the supplied remaining bound is rejected.

## Control-only actions

`PAUSE` and `COMPLETE` remain scheduler decisions but create no neural assignments.

They therefore do not consume `max_attempts`.

They do consume one `max_threads` slot because that thread has received its one decision for the snapshot.

If every selected decision is control-only, WorkerRuntime receives an empty assignment batch and WorkerBank is not invoked.

## Result partitioning

Assignments are flattened in decision order, executed once, then partitioned back into `ControlStep` objects according to the exact assignment count of each decision.

`ControlBatch` exposes:

- `steps`: separate scheduler/control outcomes;
- flattened `assignments`;
- flattened `results`;
- `neural_attempt_count`.

The returned per-thread `ControlStep` surface is the same semantic object used by `run_once(...)`.

`run_once(...)` is implemented as a one-thread `run_many(...)` call so the single-thread and cross-thread paths cannot silently diverge.

## Worker identity

Cross-thread batching does not require one globally unique worker checkpoint per simultaneous Work Item.

The WorkerSelector controls which persistent worker IDs are assigned. A stateless homogeneous neural checkpoint may therefore appear in more than one Work Item when a benchmark/control explicitly allows that behavior.

Every neural transformation still counts as one attempt and remains separately attributable.

## Failure semantics

WorkerRuntime remains the only owner of attempt execution semantics.

Therefore:

- every flattened assignment records `ATTEMPT_STARTED` before execution;
- a batch-level WorkerBank exception records each affected attempt as crashed;
- invalid result identity/count is rejected at the existing WorkerRuntime boundary;
- valid results already preserve their own thread/decision provenance.

Cross-thread batching does not introduce a second commit path.

## Required regressions

The v0 boundary is considered structurally correct when tests establish:

1. three independent threads produce one WorkerBank call rather than three;
2. each decision still projects as a separate allocation outcome;
3. `max_attempts` caps the sum of per-thread widths;
4. each Work Thread is selected at most once per batch;
5. dependency completion does not unlock a dependent until the next snapshot;
6. `run_once(...)` retains the original one-decision result surface.

## What this does not establish

Cross-thread batching does not establish:

- asynchronous execution;
- optimal batch size;
- dynamic latency/throughput scheduling;
- fair scheduling across very large thread populations;
- compiler/CUDA-graph gains;
- multi-device or distributed execution.

Those are separate systems questions.

This v0 slice only removes a known unnecessary launch boundary while preserving the persistent runtime's causal model.
