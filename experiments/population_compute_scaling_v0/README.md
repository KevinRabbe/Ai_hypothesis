# Population Compute Scaling Gate v0

## Purpose

This experiment replaces the previous immediate objective of finding the smallest independently useful worker.

The primary question is:

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

The first benchmark requires information to move across weak local processors rather than reward independent majority voting.

Each world contains many local key/value relationships. A query begins at one key. The answer is reached only after following a multi-hop chain whose required relationships are distributed across local worker contexts among distractors.

A single local worker never receives the complete chain.

The task separates three quantities:

1. **scope availability** — whether every required chain relationship is present inside the active worker prefix;
2. **communication/utilization** — whether the population can turn the available distributed relationships into the answer;
3. **recurrent depth** — how many population updates are available to continue the chain.

Difficulty is controlled independently by world size, distractor count, and required hop count.

This is a synthetic architectural benchmark, not a claim about language intelligence. Its job is to test whether the population substrate can turn additional runtime states/updates into additional capability before larger learned workloads are attempted.

## Controlled scope thresholds

The original uniform shuffle made the 1/4/16/64/256 curve scientifically weak: for multi-hop chains, almost every smaller prefix was information-incomplete and the full chain appeared almost entirely at 256 workers.

`collective-relay-v0` therefore freezes a controlled first-complete population threshold for every world.

For the 256-slot benchmark:

- relay-2 and relay-4 cycle through thresholds `4, 16, 64, 256`;
- relay-8 cycles through thresholds `16, 64, 256` because eight distinct chain records cannot fit below eight active slots;
- consecutive world seeds cycle deterministically through the admissible thresholds;
- all required chain edges are inside the declared threshold;
- at least one required edge lies beyond the previous frozen population point.

Therefore a world is guaranteed to be information-incomplete at the previous population point and information-complete at its declared threshold and every larger point.

This creates a graded scope curve instead of an accidental endpoint cliff while keeping the placement rule deterministic and model-independent.

The threshold is benchmark metadata only. It is never supplied as neural input.

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

Workers receive local context and run the same shared neural update, but population-produced state cannot move between workers during recurrent updates.

A final pooled readout is still required to produce one system answer. This condition therefore measures what can be achieved from additional scope and a one-shot set-style aggregation without recurrent inter-worker information flow.

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

## Mandatory scope/capability decomposition

Raw solve rate is not sufficient evidence because increasing worker count also exposes more source records.

Every run must record:

- `task_count`;
- `information_complete_count`;
- `solved_information_complete_count`;
- `solved_count`.

Derived measurements are:

- **information-complete rate** = fraction of worlds whose full required chain is inside the active prefix;
- **solve rate given complete information** = how often the system solves worlds it actually has enough information to solve;
- **solve rate given incomplete information** = diagnostic only; exact success here should be near accidental guessing and must not be interpreted as genuine relay capability;
- **raw solve rate** = the end-to-end result after both scope availability and neural utilization are applied.

Communication and no-communication conditions at the same population point must have exactly the same information-complete count. A curve is invalid if this scope identity does not hold.

The primary interpretation is therefore:

```text
more population
    -> more required information becomes available
    -> shared population either does or does not exploit that information
    -> raw system capability changes
```

A positive raw curve with flat/poor conditional solve rate means **scope scaling exists but the neural population is not becoming better at using available distributed information**.

A communication advantage on the same complete-information worlds is stronger evidence that the shared population computation itself matters.

## Measurements

For every condition record:

- task count and solved count;
- information-complete count/rate;
- solved count/rate conditional on information being complete;
- solved count/rate conditional on information being incomplete;
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

The primary graph remains:

> **capability vs population compute at fixed learned parameters**

It must be shown together with the information-complete curve so scope exposure is visible rather than hidden.

Secondary graphs are conditional capability vs worker count, capability vs worker updates, and capability vs communication volume.

## Frozen Gate-v0 interpretation

The criteria below are now **frozen before any trained development scaling curve is inspected**.

A communication variant is considered to show a useful scaling signal only when all of the following hold on frozen confirmation worlds across at least three independent training seeds:

1. the 256-worker endpoint improves raw solve rate over the 1-worker condition by at least **5 percentage points** on at least two nontrivial difficulty tiers;
2. the curve is not a single isolated spike: at least three of the four adjacent population steps are non-decreasing within a 1-point tolerance;
3. the communicating condition beats its matched no-communication endpoint by at least **5 percentage points** on at least one multi-hop tier;
4. learned parameter count and parameter fingerprint are identical for every point in the compared curve;
5. communication and no-communication points use identical benchmark worlds and identical information-complete counts;
6. no result is accepted if malformed accounting makes worker updates, communication volume, benchmark scope, or the scope/capability decomposition incomparable.

These thresholds are not claims of optimality. They define the minimum effect worth continuing to investigate.

Conditional solve rate is reported and interpreted alongside the frozen pass/fail criteria. It is not given a post-hoc success threshold after results are visible.

## Kill criterion

Stop or redirect the weak-unit population-computation hypothesis for this benchmark family when both preregistered communication variants fail the scaling criteria after ordinary training/debugging correctness has been established.

In particular, a flat curve where 1, 4, 16, 64, and 256 workers have essentially the same capability while worker updates and communication keep growing is a negative result, not a prompt to keep adding architecture.

Likewise, a raw curve that rises only because the information-complete rate rises, while conditional utilization remains weak and communication provides no advantage, does not establish the intended population-computation effect.

A positive population curve does **not** yet establish superiority over a 1B dense model. It only earns the next experiment.

## Sequential plan

### Gate 0A — executable benchmark contract

Build deterministic world generation, controlled scope thresholds, population conditions, exact accounting, and result validation without claiming neural performance.

### Gate 0B — minimal shared-weight population

Implement one shared neural update cell with many runtime states. No specialized workers, learned router, external memory system, or compiler optimization.

### Gate 0C — sparse communication v0

Connect the relay encoder/training runner to the shared population cell and run the development curve with mandatory scope/capability decomposition.

### Gate 0D — frozen confirmation

Freeze training/configuration and run untouched confirmation worlds across at least three training seeds. Gate criteria are already frozen before development results.

### Gate 0E — one allowed rescue variant

Only if `sparse_shared_v0` fails despite correct training and mechanics, test `hierarchical_summary_v0` without changing the benchmark objective or learned-parameter budget.

### Decision

- positive scaling -> continue to larger counts, adaptive activation, richer tasks, and equal-budget dense/serial comparisons;
- no scaling after both variants -> stop/redirect the architecture;
- scaling exists but communication/runtime cost explodes -> investigate the information-transport bottleneck rather than worker intelligence.

## Current implementation slice

Gate 0A and the minimal Gate 0B substrate are now constructed:

- deterministic relay worlds;
- controlled first-complete scope thresholds;
- exact scope/capability accounting contract;
- fixed bit encoding without a learned identity table;
- one shared recurrent population cell;
- bounded `sparse_shared_v0` communication;
- fixed-parameter fingerprint/accounting invariants.

No trained development curve has been inspected yet.

The next implementation boundary is the training/evaluation runner that uses this frozen benchmark and emits the first development curves without changing the contract above.
