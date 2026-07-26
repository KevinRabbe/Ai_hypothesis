# Indexed Runtime Control v0

## Purpose

The baseline `RuntimeControlLoop` remains the semantic/reference composition path.

Its original implementation intentionally favored simplicity and reconstructs scheduler state from:

```text
ledger.read_all_events()
```

That is acceptable during early architecture construction but becomes the wrong cost shape once the Research Ledger contains millions of durable events.

`IndexedRuntimeControlLoop` proves the same core scheduler/worker composition can instead operate from rebuildable current-state materializations.

## Current indexed control snapshot

One control cycle now freezes:

```text
canonical Research Ledger revision N
        │
        ├─ Work Thread / Work Graph state through N
        ├─ last-worker continuity through N
        ├─ integration pressure through N
        └─ verification pressure through N
```

The scheduler receives only current derived state plus the exact canonical revision boundary.

It does not need the historical event tuple.

## Exact revision isolation

`IndexedRuntimeSnapshotProvider.capture()` first records the canonical ledger sequence.

It then advances every scheduler-facing materialized view only through that exact sequence.

A regression deliberately appends event `N+1` immediately after revision `N` is captured. The resulting snapshot must still:

- report revision `N`;
- exclude `N+1` from Work Thread state;
- keep integration overview at `N`;
- keep verification overview at `N`.

A later capture may then consume `N+1`.

This is the concrete consumer that required the exact `sync_through(...)` repair on the incremental integration index.

## No full-history Work Thread replay

`IndexedThreadRuntimeState` extends the Work Thread materialization from #47 with one atomic scheduler-facing capture:

```text
capture_through(sequence)
    -> ProjectedState tuple
    -> last worker ID per thread
```

The same sidecar therefore replaces both:

- `ThreadStateProjector.project_all(full_history)`;
- reverse scanning full history to find the previous worker.

## No scheduler-decision history scan

The optimized control loop writes the selected scheduler decision ID directly into every generated `WorkItem`.

The indexed Worker Runtime therefore requires:

```text
work_item.scheduler_decision_id != None
```

and never needs the baseline fallback that scans historical `SCHEDULER_DECISION_RECORDED` events.

## No generated-ID history scan

The baseline Worker Runtime protects durable object identity by collecting every historical evidence ID and knowledge-delta ID whenever a worker emits useful output.

At large history this becomes another hidden `O(total history)` cost.

`IndexedRuntimeIntegrationTracker` reuses the already-materialized integration identity tables and exposes bounded candidate queries:

```text
existing_generated_ids(current_batch_ids)
existing_delta_ids(current_assessment_ids)
```

Only IDs actually produced/referenced by the current bounded worker batch are queried.

The indexed Worker Runtime preserves:

- evidence/delta cross-type collision detection;
- existing durable ID reuse rejection;
- same-batch generated ID collision rejection;
- knowledge assessment requiring an existing durable delta.

It does not load all historical IDs into memory.

## Hostile replay regression

The primary regression uses a Research Ledger subclass whose:

```text
read_all_events(...)
```

immediately raises `AssertionError`.

Despite that, the runtime must successfully perform repeated productive cycles that:

1. create a Work Thread;
2. capture scheduler state;
3. select a worker;
4. execute a WorkerBank request;
5. emit a new evidence contribution;
6. emit a new knowledge delta referencing that evidence;
7. validate durable identity;
8. persist the result;
9. run another scheduler cycle;
10. preserve `CONTINUE` worker continuity.

Additional regressions require:

- an existing evidence-ID collision to be rejected without replay;
- a knowledge assessment of an existing delta to succeed without replay;
- graph mutations to use indexed Work Thread state rather than full history.

## Why the integration identity index is reused

No separate durable-object registry is introduced.

The integration sidecar already contains the two identities required by Worker Runtime validation:

```text
integration_evidence.evidence_id
integration_knowledge_delta.delta_id
```

The indexed runtime adds bounded lookup behavior over those existing derived tables.

This follows the project rule of preferring one primitive that solves multiple problems over creating another subsystem.

## Baseline remains available

The existing:

```text
RuntimeControlLoop
WorkerRuntime
ThreadStateProjector
IntegrationTracker
KnowledgeStateProjector
```

remain valid rebuild/reference implementations.

The indexed path is an optional optimized composition layer. This keeps semantic correctness separable from materialization/performance policy.

## Important remaining replay boundaries

This slice does **not** claim that every current signal/context adapter is full-history-free.

In particular, some higher-level adapters still own their own replay behavior. The most important known example is thread-consolidation control, whose pressure/lineage path currently rebuilds from the full ledger snapshot.

Likewise, a context provider that independently asks for latest state may choose a later revision than the scheduler snapshot unless that provider has its own snapshot-aware contract.

Therefore the guarantee of this slice is deliberately precise:

> Core RuntimeControlLoop scheduler-state composition and WorkerRuntime durable-ID/provenance validation can execute without `read_all_events()`.

It is not yet:

> every possible user-supplied signal/context provider is replay-free and revision-pinned.

## Next target

The next high-value slice is to remove full-history replay from the thread-consolidation pressure / lineage path using the materialized Knowledge State and durable partition-allocation data already available.

Only after those higher-level adapters are snapshot-aware should the indexed path become the default runtime composition.

## Research status

No scheduler heuristic, worker architecture, evidence semantics, knowledge semantics, hierarchy policy, or truth-promotion rule changes here.

This slice changes the systems cost of turning durable history into one current control decision.
