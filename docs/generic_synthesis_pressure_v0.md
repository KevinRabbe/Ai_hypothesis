# Generic scheduler synthesis pressure v0

## Purpose

Scheduler v0 already understands several reasons to spend neural compute:

- exploration / missing coverage;
- progression;
- contradiction challenge;
- verification;
- raw evidence integration backpressure.

The runtime now also has higher-level knowledge-organization work such as thread-level consolidation. Scheduler v0 should be able to allocate compute to such work without learning any hierarchy-specific concept.

v0 therefore adds one generic scheduler signal:

```text
synthesis_need ∈ [0, 1]
```

and one generic threshold:

```text
synthesis_threshold = 0.65
```

The scheduler does not know why synthesis is needed.

## Stable boundary

When a selected thread has:

```text
synthesis_need >= synthesis_threshold
```

Scheduler v0 emits:

```text
action  = SYNTHESIZE
purpose = SYNTHESIZE
reason  = SYNTHESIS_NEEDED
width   = 1
```

A later context router decides what bounded synthesis view corresponds to that need.

This keeps the scheduler responsible for **when to allocate compute**, while projections/context providers remain responsible for **what information the worker should see**.

## No hierarchy knowledge in Scheduler v0

Scheduler v0 does not import or inspect:

- partition lineage;
- thread consolidation plans;
- knowledge hierarchy levels;
- semantic clusters;
- topic identities.

A thread-consolidation projector may raise `synthesis_need`, but so could a future domain-specific final-synthesis projector or another knowledge-organization layer.

The scheduler contract remains generic.

## Priority contribution

`synthesis_need` participates in the ordinary thread priority score with weight 1.0, alongside uncertainty, contradiction, coverage, novelty, verification need, and other provisional signals.

This lets a thread with substantial accumulated information compete for compute rather than waiting indefinitely behind threads that are still generating more evidence.

The numeric weight is provisional and remains replaceable behind `SchedulerDecision`.

## Action precedence

Outside raw integration backpressure, v0 action precedence is:

```text
structured exploration draw
      ↓ otherwise
verification threshold
      ↓
contradiction challenge threshold
      ↓
generic synthesis threshold
      ↓
resume / stagnation rotation / progress
```

This means:

- verification remains more urgent than another organizational pass;
- an explicit contradiction challenge is not hidden by synthesis demand;
- synthesis can preempt ordinary progress/stagnation work when accumulated information needs organization.

## Raw integration backpressure remains separate

When global raw-evidence integration backpressure is being serviced, the existing backpressure path remains authoritative:

```text
VERIFY if verification pressure is high
otherwise SYNTHESIZE with BACKPRESSURE
```

Generic `SYNTHESIS_NEEDED` is not added to that decision.

This distinction is important because the context differs:

- `BACKPRESSURE` synthesis consumes pending raw evidence;
- generic synthesis may consume compact knowledge from a higher integration level.

The permanent backpressure exploration lane from PR #35 also remains unchanged.

## Width

Generic synthesis is width 1 in Scheduler v0.

Width is not assumed to help every synthesis task. Specialized mechanical wrappers may widen a synthesis decision when the task exposes genuinely independent bounded partitions, as PR #37 does for raw evidence integration.

A future higher-level consolidation width policy can use the same pattern without modifying the generic scheduler action.

## Compatibility

Both new fields are appended to their existing dataclasses:

```text
SchedulerSignals.synthesis_need
SchedulerConfig.synthesis_threshold
```

Existing positional construction therefore retains its previous meaning.

Defaults preserve prior behavior:

```text
synthesis_need = 0.0
```

so no existing thread begins synthesizing merely because this capability exists.

## Why not a thread-consolidation flag

A field such as:

```text
thread_consolidation_needed
```

would couple Scheduler v0 to one current hierarchy implementation.

The final architecture already treats purpose as a generic control dimension. `synthesis_need` follows that rule and leaves domain/hierarchy meaning in context projections.

## Future use

A synthesis-pressure provider can derive the signal from any bounded projection, for example:

```text
partition-local knowledge backlog
      ↓
thread consolidation pressure
      ↓
SchedulerSignals.synthesis_need
      ↓
SYNTHESIS_NEEDED
      ↓
thread-consolidation context router
```

Later hierarchy levels can reuse the exact scheduler primitive.

## Non-goals

v0 does not add:

- thread-consolidation scheduling itself;
- semantic synthesis routing;
- learned scheduling;
- adaptive synthesis thresholds;
- synthesis width > 1;
- truth promotion;
- a new persistence event.

It adds only the generic scheduler vocabulary required for bounded knowledge-organization work to compete for compute.
