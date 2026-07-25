# Coverage-Aware Scope Planner v0

## Purpose

This is the first deterministic persistent-scope planning baseline for the large-scope relevance workload.

It uses the generic runtime's durable scope coverage projection to decide **which source regions should be inspected next**. It does not replace Scheduler v0, does not learn a routing policy, and does not claim that adaptive allocation is empirically beneficial.

The active empirical gate remains minority-evidence utilization.

## Input state

The planner receives:

- one frozen `LargeScopeRelevanceSample`;
- one Research Ledger;
- one worker-mode identity (`same_worker` or `diverse_workers`);
- a Work Thread ID.

The expected source universe is explicit: every window in the frozen benchmark world has one stable opaque scope-region ID.

Coverage is rebuilt from `ATTEMPT_STARTED` plus terminal attempt events through `ScopeCoverageProjector`.

No separate coverage store exists.

## Selection policy

For one requested width, regions are ranked in three classes.

### 1. Never attempted

Regions with no prior scoped attempt are selected first.

This maximizes new scope before spending compute on retries or redundancy.

### 2. Attempted but unresolved

Regions whose attempts only crashed or produced invalid results are selected after never-seen regions.

They remain missing coverage and are eventually retried, but one repeatedly failing region cannot prevent the rest of the source from being inspected.

### 3. Resolved

After missing scope is exhausted—or when requested width exceeds the remaining missing set—the planner selects already resolved regions with the fewest resolved inspections.

This creates deterministic balanced redundancy rather than repeatedly verifying the same easy region.

Ties in all three classes preserve the benchmark's deterministic inspection order.

## Missing-coverage signal

For an explicit expected region universe:

```text
missing_coverage = unresolved_region_count / expected_region_count
```

`CoverageAwareScopePlanner.augment_signals(...)` injects that observed value into Scheduler v0 metadata using `max(derived, caller_supplied)` semantics.

Therefore domain-specific logic may always demand stronger missing-coverage pressure, while the generic coverage projection never suppresses it.

The planner does not set importance, uncertainty, novelty, verification need, cost, or progress. Those remain separate scheduler signals.

## Width × scope behavior

The planner is also a RuntimeControlLoop context provider.

For width `N` it returns exactly `N` bounded Work Preparations, one per selected source region.

This means:

- width controls how many attempts execute now;
- coverage history controls which regions receive those attempts;
- WorkPreparationBatch carries differentiated scope to each worker;
- Worker Runtime persists exact scope before execution;
- the next planning round observes the committed result through the same ledger.

## Failure semantics

A valid `FAILED` worker result still counts as a resolved inspection because the bounded worker attempt executed and returned usable information.

`ATTEMPT_CRASHED` and `ATTEMPT_INVALID_RESULT` remain unresolved coverage.

This preserves the project principle that failed approaches can be useful information while infrastructure/execution failure must not masquerade as completed inspection.

## Required invariants

The baseline must preserve these properties:

1. With empty history, width prefixes match the frozen deterministic inspection order.
2. Resolved regions are skipped while unseen regions remain.
3. A crashed region does not monopolize exploration while unseen scope exists.
4. A crashed region is retried after never-seen scope is exhausted.
5. After complete coverage, redundancy goes to least-replicated regions.
6. Missing-coverage pressure equals the durable resolved-coverage fraction complement.
7. Stronger caller-supplied scheduler pressure is never lowered.
8. Multiple RuntimeControlLoop rounds advance to new scope without hidden planner memory.

The final invariant is especially important: planner continuity must come entirely from the Research Ledger, not from a mutable in-memory cursor.

## Why no learned routing yet

The benchmark does not yet provide evidence that a learned scope policy is needed.

A deterministic coverage baseline gives us:

- inspectable decisions;
- exact reproducibility;
- no training confound;
- a clean fixed-policy baseline for later adaptive experiments;
- traces that can later train or evaluate a learned value estimator if real results justify one.

## What this does not establish

This construction does not establish:

- that adaptive scope beats fixed prefixes;
- the optimal retry policy;
- the optimal exploration/exploitation ratio;
- the value of uncertainty-guided zoom-in;
- the value of dynamic width;
- a Gate 5 success.

Those require real Worker v1 execution under normalized end-to-end budgets.
