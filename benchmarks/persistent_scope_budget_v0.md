# Persistent Scope Budget v0

## Purpose

This protocol creates the normalized persistent-runtime baseline required before any adaptive-scope policy is allowed to claim value.

The key rule is:

> When the scope planner has no reason to deviate, the same total local neural-evaluation budget must produce the same inspected windows, worker sequence, and local evidence as the direct fixed-prefix benchmark.

Persistent execution may cost more because it writes ledger history, projects state, and schedules multiple bounded steps. That overhead is part of the system cost and must not be hidden.

## Why this baseline is necessary

Without an equality baseline, a future adaptive result could accidentally mix together:

- different source windows;
- different worker checkpoints;
- different evidence configuration;
- different neural evaluation count;
- different batching;
- persistent-runtime overhead;
- actual routing policy value.

This protocol removes those confounds one at a time.

## Fixed step width

The persistent baseline uses a fixed `step_width` and a fixed number of steps.

```text
total local attempts = step_width × step_count
```

The coverage-aware planner chooses source regions for each step.

Before any failures or evidence-guided policy changes, the planner consumes the same deterministic inspection order used by the direct benchmark. Therefore, for total budget `N <= window_count`, the accumulated persistent evidence should equal a direct width-`N` run.

## Worker sequence

### Same-worker control

One deterministic checkpoint is reused for every source region and every step.

This remains the scope-only control.

### Diverse-worker control

All checkpoint identities are placed in the same deterministic cyclic order used by the direct benchmark.

The next persistent batch starts immediately after the last worker ID recorded by the previous `ATTEMPT_STARTED` batch.

Therefore worker-sequence continuation is reconstructed from durable ledger history rather than an in-memory step counter.

A process restart must not restart checkpoint assignment at worker zero.

## Stable worker identity

A checkpoint path is not worker identity.

`HomogeneousWorkerBank` now derives a stable worker identity from:

- the structural `UnitConfig`;
- exact model-state tensor names;
- tensor dtypes;
- tensor shapes;
- exact tensor bytes.

Identity format:

```text
weights-sha256-<digest>
```

Two copied checkpoint files with identical architecture and weights therefore have the same worker identity.

Different learned weights have different identities.

Duplicate weight identities are rejected inside one homogeneous population. Repeating one worker for a control is expressed at selection/execution time instead of pretending the repeated weights are distinct population members.

## Worker-bank identity

Persistent experiments also derive one ordered worker-bank identity from the ordered stable worker IDs:

```text
worker-bank-sha256-<digest>
```

This changes when:

- any worker's weights change;
- the checkpoint set changes;
- checkpoint order changes.

The worker-bank ID is persisted in thread metadata and every large-scope evidence contribution.

## Resume contract

A persistent experiment thread may be resumed only when all of these remain identical:

- benchmark version;
- split;
- world seed;
- worker mode;
- window count;
- `step_width`;
- population width;
- ordered worker-bank identity;
- complete evidence configuration.

A mismatch fails before scheduling new neural work.

This prevents a long-lived ledger from silently mixing logically different experiments.

## Persistent evaluation

The evaluator reconstructs threshold-free results from ledger history.

It reports:

- attempted local windows;
- evidence-producing attempts;
- scheduler-decision count;
- distinct workers used;
- resolved unique regions;
- coverage fraction;
- duplicate evidence count;
- all per-attempt `WindowEvidence` records;
- strongest current candidate;
- target evidence/rank for offline scoring;
- strongest distractor evidence;
- total thread ledger-event count;
- worker-bank identity.

Repeated inspections of one region remain separate evidence records. For candidate/rank diagnostics, the current v0 evaluator preserves the strongest observed RELEVANT evidence per region. No world-level acceptance threshold is introduced.

## Required direct-equivalence regression

For a deterministic selected-worker bank, compare:

```text
direct width 8
```

against:

```text
persistent step_width 2 × 4 steps
```

under both worker modes.

Before any adaptive deviation, they must match on:

- window order;
- worker-index order;
- local decoded labels;
- RELEVANT evidence;
- NOT_RELEVANT evidence;
- uncertainty;
- invalid-label mass;
- local margin;
- strongest candidate;
- target rank/evidence when inspected.

The persistent run additionally proves durable scheduler, attempt, evidence, coverage, and worker-bank provenance.

## Restart regression

A two-step run followed by process-level reconstruction of the experiment object and two more steps must match an uninterrupted four-step run.

Both source-region progression and worker sequence must continue from ledger history.

## Budget beyond full coverage

When total persistent budget exceeds the number of source regions, the current coverage baseline converts surplus attempts into balanced redundancy over the least-replicated regions.

These repeats are counted explicitly as duplicate evidence rather than being hidden.

## What this does not establish

This baseline does not establish:

- that persistent execution is faster;
- that persistent execution improves quality;
- that adaptive scope beats fixed scope;
- an optimal exploration/verification rule;
- an optimal dynamic width policy;
- a Gate 5 success.

Its job is stricter: make future adaptive comparisons scientifically attributable.
