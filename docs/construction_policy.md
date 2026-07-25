# Construction Policy

## Core rule

> **Construction does not wait for empirical research results.**

Experiments answer empirical questions. They do not stop independent implementation work.

When a result is unavailable:

1. record the unresolved empirical question;
2. state the provisional assumption currently used by construction;
3. keep that assumption behind a stable semantic contract;
4. continue every implementation path that does not require the answer to be physically measured now;
5. run the deferred experiment when its result becomes decision-relevant;
6. replace or tune the affected implementation behind the existing contract if the assumption was wrong.

The preferred failure mode is **replacing a policy or implementation**, not redesigning the architecture or stopping development.

## What may remain provisional

Examples include:

- reducer thresholds and evidence-selection rules;
- useful population width;
- scheduler scoring weights and exploration fractions;
- worker-size sweet spots;
- integration fan-out and hierarchy depth;
- storage/indexing strategies;
- compiler and batching modes;
- quantization choices;
- verification budgets.

These values and mechanisms should remain replaceable behind the frozen architecture contracts.

## What construction should preserve

Construction should preserve the stable semantics defined in [`architecture_contracts.md`](architecture_contracts.md):

- bounded immutable Work Items;
- structured Attempt Results;
- append-only Ledger Events;
- rebuildable Projected State;
- bounded Scheduler Decisions;
- provenance-preserving Knowledge Deltas;
- recursive integration through the same contracts;
- backpressure when information production exceeds useful absorption.

## Deferred-test ledger

A deferred experiment should record at least:

- the exact question;
- the frozen inputs/configuration needed to reproduce it;
- the provisional assumption used meanwhile;
- what future decision would make the result relevant;
- which implementation may change if the assumption is falsified.

The minority-rescue diagnostic is one example: its implementation remains available and reproducible, but its GPU run does not block construction of the population runtime because the runtime contracts do not depend on whether that specific rescue gate succeeds.

## No speculative overbuilding

Non-blocking construction does **not** mean implementing every future subsystem immediately.

The rule is:

> **Always keep building useful independent slices; never build scale machinery merely because it might be needed later.**

So construction continues continuously, while each slice should still be the smallest final-shape implementation that provides real forward progress.
