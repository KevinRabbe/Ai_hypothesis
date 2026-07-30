# Gate-7 preparation — execution-engine equivalence protocol

**STATUS: PREPARATION ONLY — NO GATE-7 SCIENTIFIC DATA.**

Gate-7 high-scale execution will require a tensorized engine. Before that engine may produce any scientific Gate-7 result, it must demonstrate semantic equivalence to the qualified eager reference on small synthetic/bridge cases.

Gate-6 has now shown that the next scientific variable is routing bandwidth K as population N grows. This equivalence protocol therefore protects both population and K treatments from implementation drift.

## Reference implementation

The reference semantics are the qualified eager Gates 5–6 behavior for:

- deterministic answer-blind candidate sampling;
- bounded score visibility;
- matched hash-control selection;
- score quantization/tie ordering;
- answer-blind capacity thinning/pruning;
- recurrent child refinement;
- work accounting;
- exact terminal-path generation.

The reference is not a performance target.

## Required transcript equivalence

For deterministic synthetic/bridge worlds, checkpoint/model state and scheduler configuration, compare reference and tensor engines slot-by-slot.

The tensor engine must reproduce exactly:

1. incoming live candidate path/ID set;
2. bounded visible candidate IDs for every tested K;
3. selected parent ID;
4. branch child IDs/paths;
5. answer-blind overflow retention/prune identities;
6. terminal path transcript;
7. productive/sink slot counts;
8. learned recurrent-update accounting;
9. score-observation accounting;
10. final exact-search coverage.

For global mode, selected parent identity must match the reference quantized-score + deterministic-tie ordering.

For bounded-score mode, parent selection must be identical using only the sampled K scores.

For matched hash mode, selection must remain score-blind and use the same visible candidate IDs as its paired score-K condition whenever incoming live banks are identical.

## State/score numerical equivalence

Preferred: exact FP32 equality when operation order is preserved.

If vectorization changes floating operation ordering enough to preclude bitwise equality, freeze one tight numerical tolerance **before any Gate-7 scientific exposure**, and prove that all discrete routing/pruning/terminal transcripts remain exactly identical under that tolerance.

No tolerance may be widened after seeing a scientific result.

## K ladder coverage

Equivalence testing must include at least:

- K16;
- K32;
- K64;
- global reference;
- matched answer-blind controls for tested K values used in the admitted Gate-7 primary bandwidth analysis.

Larger K values used by the admitted campaign must receive the same deterministic sampler/selection regression coverage before scientific execution.

## Population bridge coverage

Before high-scale admission, equivalence must be demonstrated at multiple small/medium populations that exercise:

- no capacity overflow;
- active overflow pruning;
- sparse live masks/holes after parent removal;
- terminal generation;
- divergent score/hash treatment histories.

N64/N128/N256 synthetic cases are preferred because they overlap the already-understood Gate-6 regime without reusing Gate-6 scientific worlds.

## Hot-path synchronization boundary

The tensor engine is specifically intended to eliminate accidental host/device synchronization.

Scientific hot path must not call CUDA-dependent:

- `.item()`;
- `.cpu()`;
- `.numpy()`;
- `.tolist()`;
- Python `float()`/`int()` on CUDA values;
- Python branch conditions driven by unsynchronized CUDA scalars.

Aggregate telemetry may cross to CPU at explicit, auditable synchronization points after the relevant routing decisions are complete.

## Bounded-computation requirement

For any bounded `score_K` condition before parent selection:

- sampler work may depend on N only through deterministic integer index generation / live-mask access;
- exactly `min(K, live_count)` neural scores may be gathered/read for the causal selection;
- no full score sort/rank/reduction may execute before the parent is selected;
- evaluation-only diagnostics must be structurally separated from the selection function.

This is stronger than the information-channel rule: it prevents an implementation from remaining computationally global while merely hiding score values from the selector.

## Answer-blind pruning requirement

Capacity retention may depend on:

- public runtime seed;
- slot index;
- candidate path/slot identity;

and must not depend on:

- neural score/state value;
- hidden answer;
- scheduler outcome labels.

The tensorized retention set must match the frozen reference retention set for equivalence fixtures unless a separately versioned Gate-7 scientific protocol deliberately freezes a new answer-blind retention primitive before exposure.

## Physical batching equivalence

The tensor engine may gather ready neural updates from multiple independent worlds into larger physical CUDA batches.

This is admitted only if the transcript proves that each world's logical scheduler observes exactly the same sequence it would observe under separate execution.

Physical batch size may not alter:

- per-world selected parent;
- candidate state ownership;
- recurrent update count;
- sampling namespace;
- score visibility;
- pruning decisions.

## Stage-A reuse equivalence

A common Stage-A frontier may be constructed once and copied into multiple scheduler conditions for evaluation efficiency only after a regression proves:

- copied states/scores/IDs are identical at treatment start;
- no storage aliasing permits one scheduler to mutate another;
- every condition retains the same logical learned-work accounting;
- capability results match separately recomputed Stage-A execution.

Cached Stage-A execution cannot be used for a historical wall-clock speedup claim without explicit schedule disclosure.

## Profiling isolation

Performance profiling uses synthetic worlds or explicit engineering namespaces only. Profiles are not capability evidence.

Measure separately:

- current eager reference;
- eager tensorized engine;
- compiled tensorized engine;
- CUDA-graph mode if later enabled.

The compiler remains an independent variable and cannot be baked into the semantic baseline.

## Admission sequence

```text
qualified eager reference
        ↓
implement tensor-bank engine
        ↓
exact small-scale semantic equivalence across K ladder
        ↓
profile eager-vs-eager
        ↓
freeze Gate-7 bandwidth-scaling scientific runner
        ↓
ONLY THEN expose Gate-7 worlds
```

Any transcript mismatch blocks scientific admission until resolved. A performance optimization that changes semantics is rejected or becomes a separately versioned scientific treatment; it cannot be silently substituted into the baseline.
