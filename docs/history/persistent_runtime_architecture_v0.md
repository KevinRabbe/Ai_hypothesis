# Persistent Runtime Architecture v0 — Historical / Deferred Summary

## Status

**Deferred reusable architecture. Not required by the current Gate-2 population-compute substrate.**

The complete implementation remains in Git history and the historical PR stack, primarily #2–#53.

Repository-hygiene CI later proved that the focused `ai_hypothesis.population_compute` package and canonical regression suite pass with `ai_hypothesis/runtime` physically absent from the checkout. Therefore this architecture can leave the canonical active tree without removing its historical value.

## Why this program existed

The earlier project direction assumed that a large population of neural workers would need a durable system to preserve useful work across attempts and prevent local computation from disappearing when a worker finished.

The runtime program explored a general architecture around:

```text
Worker Bank
    ↓
bounded Worker Runtime
    ↓
append-only Research Ledger
    ↓
rebuildable current-state projections
    ↓
Scheduler / bounded context
    ↓
more worker attempts
```

The central systems concern was information integration: a large population might produce evidence faster than a shared system could verify, organize and reuse it.

## Stable ideas worth preserving

### 1. Append-only canonical provenance

The Research Ledger treated durable events as canonical history rather than mutating current-state rows in place.

Important preserved principles:

- monotonic event order;
- stable event/attempt/evidence/knowledge identities;
- explicit causal references;
- failures remain durable information;
- derived current state is rebuildable from canonical history;
- provenance should survive retries/restarts.

This remains a useful design if later population workloads require long-lived work across many independent attempts.

### 2. Workers are disposable; work state is not

The runtime separated persistent Work Threads from temporary worker execution.

A worker could fail, rotate or be replaced while the logical work thread retained:

- objective;
- references;
- open questions;
- hypotheses/contradictions;
- dependencies;
- progress state;
- durable evidence and knowledge.

This is compatible with the current project's principle that weak neural runtime units should not become heavyweight autonomous agents.

### 3. Roles come from context, not specialized worker types

The scheduler/context model used generic purposes such as:

- `EXPLORE`;
- `PROGRESS`;
- `CHALLENGE`;
- `VERIFY`;
- `SYNTHESIZE`.

The same homogeneous worker machinery could receive different bounded contexts rather than requiring separately trained role-specific models.

That idea may remain useful if future Gate-4+ workloads introduce adaptive search, verification or evidence integration.

### 4. Bounded local context over unbounded durable history

The architecture explicitly rejected copying the full global knowledge/history into every active worker.

Instead:

- durable global history may grow;
- derived indexes may summarize current state;
- every individual Work Item receives an explicit bounded selection;
- scope/references form an authority boundary.

This matches the current large-population communication principle: most information should never need to move to every runtime state.

### 5. Evidence is not truth

The runtime distinguished:

```text
evidence contribution
        ↓
provisional knowledge
        ↓
verification/challenge
        ↓
verified / disputed / retracted current state
```

Claims were not silently promoted to facts merely because a worker emitted them.

Knowledge status remained derived from append-only assessments.

This is a strong reusable principle for any future open-ended research/search workload.

### 6. Preserve minority/contradictory information

The evidence/knowledge path deliberately kept independent records rather than immediately averaging/voting them away.

Rare, disputed or minority evidence could remain available for later challenge or synthesis.

That principle remains conceptually aligned with population computation even though the old independently weighted Worker-v1 path is no longer canonical.

## Scheduler lessons

### Deterministic baseline before learned control

Scheduler v0 used explicit/inspectable rules and bounded signals before considering a learned scheduler.

The general lesson survives:

> Do not add learned routing until deterministic allocation has produced traces demonstrating a concrete limitation.

### Permanent structured exploration

The scheduler included structured-random exploration and later corrected an important failure mode: integration backpressure must not reduce exploration probability to zero.

The provisional historical policy used a dominant integration lane plus a small permanent exploration lane.

The exact numeric share was never established as optimal. The reusable invariant is only that persistent housekeeping pressure should not permanently stop discovery.

### Allocation provenance

Later runtime work linked scheduler decisions to Work Items, attempts and outcomes so resource allocation could eventually be evaluated causally.

Useful future chain:

```text
scheduler decision
    → exact bounded assignment
    → attempt
    → resource usage
    → evidence / knowledge / terminal outcome
```

Do not optimize scheduling from aggregate outputs that cannot be traced back to the allocation that caused them.

## Information-integration architecture

### Raw evidence backlog

The runtime measured whether evidence generation exceeded evidence disposition/integration capacity.

Telemetry separated:

- evidence created;
- unique evidence dispositioned;
- raw disposition traffic;
- duplicate/redundant processing;
- unresolved backlog;
- backlog age;
- knowledge production;
- evidence references consumed by knowledge.

A useful future lesson is to distinguish **unique useful absorption** from raw processing traffic.

### Deterministic integration partitions

Pending evidence was mechanically sharded within its source Work Thread.

Important intent:

- avoid one giant integration queue;
- maintain thread ownership;
- bound each active integration batch;
- preserve full backlog only in canonical history;
- keep partitioning mechanical rather than pretending hash shards are semantic clusters.

### Parallel integration width

Partitioned backpressure work could allocate multiple workers only when they received disjoint evidence authority.

The program did **not** establish that a particular integration width was optimal. It built the execution/control surface needed for future measurements.

### Higher-level consolidation

The architecture added another level:

```text
raw evidence
    → partition-local provisional knowledge
    → thread-level consolidation
```

Higher-level knowledge referenced lower knowledge IDs; there was no separate mutable hierarchy database.

Retraction of a higher consolidation reopened lower sources because consumed state was derived from active references rather than stored as irreversible flags.

This is a useful append-only hierarchy pattern if future workloads need multi-level integration.

## Indexed/materialized runtime

The first versions reconstructed state by replaying complete ledger history.

Later PRs #45–#53 built rebuildable SQLite sidecars and pinned snapshots for:

- integration state;
- Knowledge State;
- Work Thread/graph state;
- partition-to-knowledge lineage;
- consolidation planning/pressure;
- indexed control execution.

The goal was to change repeated query cost from roughly total-history replay toward:

```text
new events since checkpoint
+ bounded indexed query
```

while retaining the Research Ledger as canonical truth.

### Exact snapshot boundary

A major correctness requirement was that one scheduler/control decision observe one pinned revision. Concurrent later appends must not leak into an older decision's projected state.

That principle is reusable for any future persistent concurrent runtime.

### Qualification status

Historical PR #53's indexed-runtime GitHub Actions lane executed the focused indexed architecture regressions and reported 71 tests passing at its qualified head.

This was correctness/mechanics qualification, not proof of practical performance advantage.

No claim should be made that the indexed architecture is currently faster or necessary merely because it is more scalable in asymptotic replay shape.

## Cross-thread batching

The runtime later allowed multiple independent scheduler decisions to share one homogeneous neural execution batch while retaining separate decision/attempt provenance.

Core rule:

> **Fuse homogeneous neural execution, not scheduler meaning.**

This is a strong reusable systems principle for future GPU population execution.

Each thread retained its own:

- scheduler decision;
- bounded context;
- worker assignment;
- attempt identity;
- evidence/knowledge outputs.

Only the compatible neural execution was fused.

## Why this architecture is deferred now

The current shared-weight population-compute program deliberately returned to a much smaller scientific substrate.

Gate 0 and Gate 1 could be answered without introducing:

- durable Work Threads;
- Research Ledger;
- evidence hierarchy;
- scheduler policy;
- large-scope persistent search.

Gate 2 now asks a cleaner architecture-specific question using persistent **neural state inside one model execution protocol**, not the earlier durable multi-attempt research-runtime abstraction.

Keeping the whole runtime active would therefore add conceptual and maintenance surface without contributing to the current causal experiment.

## Reactivation conditions

Reconsider this architecture only if a future measured workload requires one or more of:

- work that must survive across independent model invocations;
- many asynchronous exploration/verification attempts;
- durable evidence provenance;
- explicit contradiction/claim lifecycle;
- information production faster than immediate neural integration;
- adaptive scheduling over many persistent work threads;
- large external source scope where only bounded subsets can be processed at once.

Reactivation should begin from the simplest required contracts, not automatically restore every historical subsystem.

## What not to carry forward automatically

Do not assume future work needs:

- SQLite specifically;
- the historical scheduler weights/thresholds;
- the historical shard count/batch limits;
- the exact hierarchy depth;
- the old Worker-v1 adapter;
- replay/materialization complexity before profiling shows it matters.

Those were implementations/provisional policies, not permanent scientific truths.

## Historical PR lineage

The main open/qualified historical chain spans approximately #2–#53, with later large-scope work extending the runtime through #54/#55/#86.

The exact PR discussions/commits remain the authoritative archaeological record after this compact summary replaces the need to keep every implementation file in the canonical active checkout.
