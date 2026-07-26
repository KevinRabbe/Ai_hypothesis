# Purpose-aware context views v0

## Principle

Worker roles come from bounded context and mission, not from specialized worker architectures.

The homogeneous population remains:

- same architecture;
- independently learned weights;
- interchangeable runtime contract;
- different assigned Work Items and bounded context views.

`PurposeContextRouter` makes the first two generic context views executable without creating another subsystem.

## What the router owns

The router receives the same `(ProjectedState, SchedulerDecision)` pair as any other runtime context provider.

It only intercepts:

1. backpressure-driven `SYNTHESIZE` decisions;
2. `VERIFY` decisions.

Everything else is delegated unchanged to the caller's domain context provider.

This preserves domain ownership of:

- `EXPLORE`;
- `PROGRESS`;
- `CHALLENGE`;
- ordinary/final `SYNTHESIZE`.

The router does not change Scheduler v0, worker weights, Worker Runtime, ledger schema, evidence semantics, or knowledge status.

## Backpressure synthesis view

Scheduler v0 already emits:

`action = SYNTHESIZE`

with reason code:

`BACKPRESSURE`

when global integration backlog is beyond the configured limit.

Only that form of synthesis is automatically routed to pending evidence.

The view uses the existing `prepare_bounded_integration_work(...)` helper and gives the worker:

- a hard-capped oldest pending evidence batch;
- exact evidence IDs as Work Item references;
- compact evidence records;
- source/provenance references;
- evidence event IDs for causal provenance;
- the integration projection revision;
- an explicit `SYNTHESIZE` context-view marker.

The constraints tell the worker to emit structured knowledge deltas and disposition consumed evidence through the normal `AttemptResult` contract.

The worker does **not** receive the whole evidence backlog.

### Empty backlog

A backpressure synthesis decision for a thread with no pending evidence is rejected.

That indicates scheduler/context inconsistency and must not silently become arbitrary neural work.

## Ordinary synthesis remains separate

A `SYNTHESIZE` decision without the `BACKPRESSURE` reason is delegated to the domain context provider.

This is deliberate.

Backpressure synthesis means:

> convert unresolved evidence into compact provisional knowledge.

A later domain/final synthesis can instead mean:

> combine verified knowledge, unresolved conflicts, and the required final output.

Those are different information views even though both use the same worker purpose label.

## Verification view

`VERIFY` decisions use the current Knowledge State projection plus `KnowledgeVerificationTracker`.

The view contains only bounded unresolved knowledge for the selected Work Thread:

- `PROVISIONAL` deltas;
- `DISPUTED` deltas;
- exact delta IDs;
- source references;
- causal event references;
- current knowledge status;
- the knowledge projection revision.

The Work Item references are exactly the delta IDs the verifier is authorized to assess.

This aligns with Worker Runtime's existing authority boundary: the worker may not assess knowledge outside its supplied references.

### Independence constraints

The verification view explicitly marks:

- independent verification;
- worker identity hidden;
- vote/majority counts hidden.

The compact knowledge records themselves contain no worker identity or population vote count.

The verifier therefore evaluates the claim and its provenance rather than social/majority metadata.

## Same ledger requirement

The router, integration tracker, and verification tracker must use the exact same `SQLiteResearchLedger` instance.

Matching path strings are insufficient because two independent `:memory:` databases can share the same textual path while containing different histories.

This is enforced at construction.

## Bounded context

Default limits are:

- integration evidence: 32 records;
- verification knowledge: 32 records.

Both are configurable positive integers.

The limits constrain active worker context, not durable global state. The full evidence/knowledge history remains addressable through the Research Ledger.

## Runtime path

The full generic path is now:

```text
Research Ledger
      ↓
State / integration / verification projections
      ↓
Scheduler v0
      ↓
SchedulerDecision
      ↓
PurposeContextRouter
      ├─ BACKPRESSURE SYNTHESIZE → bounded pending evidence
      ├─ VERIFY                  → bounded unresolved knowledge
      └─ other                   → domain context provider
      ↓
ordinary WorkItem
      ↓
homogeneous Worker Bank
      ↓
AttemptResult
      ↓
Research Ledger
```

There is no worker-to-worker chat and no special integrator/verifier model.

## Why this matters for information scaling

As evidence volume grows, the system can redirect ordinary population compute toward integration without changing worker architecture.

The same population can therefore provide both:

- evidence production;
- evidence absorption / verification.

The scheduler controls the compute share; the context view controls what that compute sees.

This is the intended meaning of:

> global cooperation, local competition.

and:

> roles come from context, not worker types.

## Non-goals

v0 does not add:

- a learned context router;
- semantic relevance retrieval;
- hierarchical integration;
- specialized synthesis weights;
- specialized verifier weights;
- worker identity/vote information;
- direct worker communication;
- a new persistence store;
- a new scheduler policy.

Those remain replaceable implementation/policy questions behind the same final runtime boundaries.
