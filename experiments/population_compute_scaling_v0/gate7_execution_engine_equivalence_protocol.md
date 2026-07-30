# Gate-7 preparation — execution-engine equivalence boundary

**STATUS: PREPARATION ONLY.**

The high-scale campaign needs a tensorized runtime, but performance engineering must not silently change scheduler semantics. Gate-6 remains untouched. This document freezes how a future Gate-7 tensor-bank engine is qualified against the current eager object-based reference before any high-scale scientific exposure.

## Reference semantics

Use the already-qualified eager Gate-6/Gate-5 primitives as the semantic reference for operations that remain shared:

- score quantization;
- deterministic score tie ordering;
- K16/K8 deterministic sample membership;
- K16 score-vs-hash shared sampler condition;
- answer-blind population thinning;
- answer-blind capacity pruning;
- child input semantics;
- exactly two child branches;
- exactly eight recurrent updates/generated child;
- persistent candidate-specific recurrent state;
- terminal generation semantics.

Gate-6 scientific code itself is not modified.

## Qualification requirement for a tensorized engine

Before the tensor engine can be admitted for Gate-7 scientific use, run both engines on synthetic/public test worlds and small populations where the reference engine is practical.

For every compared world/checkpoint surrogate/scheduler/slot require exact equality of:

1. initial frontier path identities;
2. post-thinning live candidate path set;
3. deterministic bounded visible candidate path set;
4. selected parent path;
5. child path identities;
6. answer-blind overflow-retained path set;
7. generated terminal path multiset/transcript;
8. productive/sink learned-work accounting;
9. visible score-observation count.

For recurrent tensors/scores, use the frozen numerical tolerance appropriate to the execution mode. Eager FP32 tensorized admission must not loosen numerical precision merely for speed.

## Hot-path synchronization prohibition

The admitted tensor engine must have no per-candidate CUDA-to-Python scalar extraction in Stage A or Stage B.

Source/regression guards must reject hot-path use of:

```text
.item()
.cpu()
.tolist()
Python float(cuda_tensor)
CUDA-dependent Python branching
```

except at explicit post-batch telemetry/provenance synchronization boundaries.

## Bounded-computation requirement

For `bounded_score_k16` before parent selection:

- sampler work may depend on N only through deterministic integer index generation / live-mask access;
- exactly `min(16, live_count)` neural scores may be gathered/read for the causal selection;
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
current eager reference
        ↓
implement tensor-bank engine
        ↓
exact small-scale semantic equivalence
        ↓
profile eager-vs-eager
        ↓
freeze Gate-7 scientific runner
        ↓
ONLY THEN expose high-scale worlds
```

A performance optimization that fails equivalence is rejected or becomes a separately versioned scientific treatment; it cannot be silently substituted into the baseline.
