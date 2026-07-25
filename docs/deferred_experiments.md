# Deferred Experiments

This file records empirical questions that remain important but do **not** block construction.

The rule is:

> **A deferred experiment must preserve enough information to answer a future decision, while construction continues behind stable contracts.**

Each entry records the question, frozen setup, provisional construction assumption, decision trigger, and implementation that may change when evidence arrives.

## D1 — Step 2A minority evidence rescue

**Status:** deferred; implementation prepared; real 16-worker run not yet executed.

### Question

Can inference-visible population evidence distinguish a genuinely useful strong minority candidate from a noisy minority outlier well enough to improve final class selection under a small harm budget?

### Frozen setup

- Worker architecture: frozen ~50K Worker v1.
- Population: 16 independently trained checkpoints.
- Data: validation only; frozen test set remains unopened.
- Count: 20,000 validation samples.
- Development/confirmation: seeded random half-split of answerable validation rows.
- Candidate proposal: strongest protected valid non-primary alternative by maximum worker evidence.
- Gate: tiny class-weighted logistic decision score.
- Threshold: selected on development under `max_harm_rate = 0.001`, then applied once to confirmation.
- Implementation: `ai_hypothesis.step02.run_minority_rescue` and `ai_hypothesis.step02.rescue`.

Expected local command:

```powershell
$checkpoints = 1..16 | ForEach-Object {
    "results\step01\checkpoint_50k_extended_15k\seed_$($_)\best.pt"
}

python -m ai_hypothesis.step02.run_minority_rescue `
    --checkpoints $checkpoints `
    --device cuda `
    --backend vmap `
    --count 20000 `
    --batch-size 256 `
    --max-harm-rate 0.001
```

### Provisional construction assumption

The final evidence-selection/reducer policy is replaceable. Runtime contracts persist evidence, hypotheses, contradictions, provenance, and knowledge changes without depending on this specific rescue gate.

Construction therefore proceeds with no assumption that rescue-v0 is the final reducer.

### Decision trigger

Run this experiment before a decision actually requires choosing or freezing a reducer-v1/evidence-utilization policy, or before a downstream comparison would be invalid without a fixed final decision rule.

### What may change afterward

- candidate proposal policy;
- rescue/reducer scoring;
- evidence features;
- acceptance threshold;
- final class-selection implementation.

The Research Ledger, Projected State, Scheduler Decision, Worker Bank, and Worker Runtime contracts should not change because of this result.

## D2 — Useful population width

**Status:** deferred until a downstream decision requires a useful-width estimate.

### Question

At what population width do marginal workers stop adding enough unique useful information to justify their end-to-end execution and integration cost?

### Provisional construction assumption

Population width is dynamic and bounded by configuration/resource policy. No fixed optimal width is encoded into runtime contracts.

### Decision trigger

Run when choosing production/default width, comparing population organization against dense baselines, or allocating hardware budgets where the useful-width frontier matters.

### What may change afterward

- default width;
- width caps;
- adaptive-allocation policy;
- worker-size/population organization choices.

## D3 — Scheduler v0 tuning

**Status:** intentionally deferred until real runtime traces exist.

### Question

Which scheduler scoring weights, thresholds, and exploration share produce the best useful-information/compute frontier on real Work Threads?

### Provisional construction assumption

Scheduler v0 uses simple inspectable weights and permanent structured exploration. Numeric values are policy, not architecture.

### Decision trigger

Run after enough Work Thread traces exist to compare allocation policies meaningfully.

### What may change afterward

- priority weights;
- challenge/verification/stagnation thresholds;
- exploration probability;
- eventually the internal scheduler policy or learned value estimator.

The `SchedulerDecision` contract remains stable.

## D4 — Knowledge integration bandwidth

**Status:** deferred until evidence volume actually stresses integration.

### Question

At what useful-evidence production rate does preservation, deduplication, verification, connection, routing, and synthesis become the dominant population-scaling bottleneck?

### Provisional construction assumption

The local append-only ledger, rebuildable projections, compact knowledge deltas, bounded context, and scheduler backpressure are sufficient at current scale.

### Decision trigger

Activate when measurements show growing integration backlog, verification backlog, excessive duplicate evidence, rising integration latency, or loss of rare decisive evidence.

### What may change afterward

- ledger partitioning/indexing/storage engine;
- integration fan-out;
- number/depth of hierarchical integration Work Threads;
- batching and routing strategy;
- backpressure thresholds.

The semantic Work Item, Attempt Result, Ledger Event, Projected State, Scheduler Decision, and Knowledge Delta contracts remain stable.

## Construction consequence

None of these deferred experiments creates a stop point for unrelated work.

When one becomes decision-relevant, run it, update the affected policy/implementation, and continue. Until then, construction proceeds through the final architecture using provisional replaceable implementations.
