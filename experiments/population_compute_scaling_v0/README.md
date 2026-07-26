# Population Compute Scaling Gate v0

## Purpose

This experiment replaces the previous immediate objective of finding the smallest independently useful worker.

The new primary question is:

> **With learned parameters held fixed, does increasing population computation produce increasing system-level capability?**

The experiment is intentionally falsifiable. If additional runtime workers and recurrent population updates do not produce a reproducible capability curve after the two preregistered communication variants, this path is stopped rather than expanded indefinitely.

Ant colonies are an inspiration for the weak-local-unit / strong-population possibility. They are not an implementation blueprint.

## Frozen scientific boundary

The learned model is one shared parameter set. Runtime worker count does **not** create additional learned weights.

A runtime worker is temporary state plus a local observation/context. Many runtime workers may reuse the same learned update machinery.

Across a population-size curve, all of the following must remain identical:

- learned parameter count;
- exact parameter fingerprint/checkpoint;
- worker/update architecture;
- training data and training procedure;
- benchmark examples;
- output decoder;
- hardware;
- compiler/execution mode.

The experimental variables are:

- available/active runtime-worker count;
- recurrent population-update depth;
- communication budget/topology;
- total worker updates.

Compiler optimization remains a separate systems variable and is not mixed into the neural result.

## First benchmark family — collective relay

The first benchmark should require information to move across weak local processors rather than reward independent majority voting.

Each world contains many local key/value relationships. A query begins at one key. The answer is reached only after following a multi-hop chain whose required relationships are distributed across local worker contexts among distractors.

A single local worker never receives the complete chain.

The task separates three quantities:

1. **scope** — which local relationships become inspected;
2. **communication** — whether a discovered intermediate key reaches the worker/context containing the next relationship;
3. **recurrent depth** — how many population updates are available to continue the chain.

Difficulty is controlled independently by world size, distractor count, and required hop count.

This is a synthetic architectural benchmark, not a claim about language intelligence. Its job is to test whether the population substrate can turn additional runtime states/updates into additional capability before larger learned workloads are attempted.

## Population counts

Development curve:

- 1;
- 4;
- 16;
- 64;
- 256 runtime workers.

Only after the curve is positive and implementation profiling shows that the experiment remains meaningful:

- 1,024;
- 4,096;
- 16,384;
- larger counts up to the local-machine practical limit.

The project does not jump directly to 10K-100K workers merely because those counts are technically representable.

## Required controls

### A — no communication

Workers receive local context and run the same shared neural update, but cannot exchange population state.

Purpose: determine how much of any gain comes only from additional local coverage/independent attempts.

### B — sparse shared communication

Workers exchange only the bounded signal/state defined by the population architecture.

This is the primary architecture condition.

### C — serial compute control

Spend the same number of worker updates through a small number of recurrent states rather than a wide population.

Purpose: distinguish a population-state benefit from the trivial fact that more FLOPs can help.

The serial control is interpretive for Gate v0. Population-vs-dense efficiency becomes a later gate only if population scaling itself exists.

## Communication variants allowed before a kill decision

Only two population communication designs may be tried before reassessing the hypothesis:

1. `sparse_shared_v0` — one bounded shared signal field / summary accessible to relevant workers;
2. `hierarchical_summary_v0` — local groups with bounded group summaries and sparse cross-group promotion.

Do not iterate through unlimited routing designs until one produces a desirable graph.

## Measurements

For every condition record:

- task count and solved count;
- solve rate by difficulty;
- learned parameter count and exact parameter fingerprint;
- active workers;
- available workers;
- recurrent rounds;
- total worker updates;
- messages/signals emitted;
- communicated scalar values or bytes;
- peak worker-state bytes;
- wall time;
- device execution time when measurable;
- compiler/execution mode as provenance only.

The primary graph is:

> **capability vs population compute at fixed learned parameters**

Secondary graphs are capability vs worker count, capability vs worker updates, and capability vs communication volume.

## Preregistered Gate-v0 interpretation

The exact numeric thresholds below are provisional until the first neural implementation is runnable. They must be frozen before looking at a trained development scaling curve.

A communication variant is considered to show a useful scaling signal only when all of the following hold on frozen confirmation worlds across at least three independent training seeds:

1. the 256-worker endpoint improves solve rate over the 1-worker condition by at least **5 percentage points** on at least two nontrivial difficulty tiers;
2. the curve is not a single isolated spike: at least three of the four adjacent population steps are non-decreasing within a 1-point tolerance;
3. the communicating condition beats its matched no-communication endpoint by at least **5 percentage points** on at least one multi-hop tier;
4. learned parameter count and parameter fingerprint are identical for every point in the compared curve;
5. no result is accepted if malformed accounting makes worker updates, communication volume, or benchmark scope incomparable.

These thresholds are not claims of optimality. They define the minimum effect worth continuing to investigate.

## Kill criterion

Stop or redirect the weak-unit population-computation hypothesis for this benchmark family when both preregistered communication variants fail the scaling criteria after ordinary training/debugging correctness has been established.

In particular, a flat curve where 1, 4, 16, 64, and 256 workers have essentially the same capability while worker updates and communication keep growing is a negative result, not a prompt to keep adding architecture.

A positive population curve does **not** yet establish superiority over a 1B dense model. It only earns the next experiment.

## Sequential plan

### Gate 0A — executable benchmark contract

Build deterministic world generation, population conditions, exact accounting, and result validation without claiming neural performance.

### Gate 0B — minimal shared-weight population

Implement one shared neural update cell with many runtime states. No specialized workers, learned router, external memory system, or compiler optimization.

### Gate 0C — sparse communication v0

Add the smallest bounded shared signal path required for multi-hop relay. Run the development curve.

### Gate 0D — frozen confirmation

Freeze training/configuration/criteria and run untouched confirmation worlds across at least three training seeds.

### Gate 0E — one allowed rescue variant

Only if `sparse_shared_v0` fails despite correct training and mechanics, test `hierarchical_summary_v0` without changing the benchmark objective or learned-parameter budget.

### Decision

- positive scaling -> continue to larger counts, adaptive activation, richer tasks, and equal-budget dense/serial comparisons;
- no scaling after both variants -> stop/redirect the architecture;
- scaling exists but communication/runtime cost explodes -> investigate the information-transport bottleneck rather than worker intelligence.

## Current implementation slice

The first code slice freezes the experiment schema and accounting rules in `ai_hypothesis.population_compute`. It deliberately does not train a neural model yet. That keeps scientific bookkeeping independent from model tuning and gives the later runner a contract that cannot silently change after results are visible.
