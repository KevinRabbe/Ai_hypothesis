# Gate-7 preparation — execution-engine equivalence protocol

**STATUS: PREPARATION ONLY — NO GATE-7 SCIENTIFIC DATA.**

Gate-7 high-scale execution requires a tensorized runtime, but performance engineering must not silently change scheduler semantics. Gate-6 scientific code remains untouched.

## Current qualification state

The first generation-synchronous Stage-A tensorization slice is already qualified against the eager reference on synthetic Gate-6-shaped worlds. It reproduces path identities, recurrent states and scores exactly and has measured a 48.24x local Stage-A wall-speedup on the user's RTX 4060 Ti.

That measurement is engineering evidence only. **Full dynamic Stage-B equivalence is still required before any Gate-7 scientific world may be generated.**

## Reference semantics

Use the already-qualified eager Gate-5/Gate-6 behavior as the semantic reference for:

- score quantization;
- deterministic score tie ordering;
- deterministic bounded sample membership;
- score/hash shared sampling when incoming banks match;
- answer-blind thinning and capacity retention;
- child input semantics;
- exactly two child branches;
- exactly eight recurrent updates/generated child;
- persistent candidate-specific recurrent state;
- terminal generation semantics;
- logical learned-work accounting.

## Full Stage-B tensor-bank admission requirement

Before scientific use, run eager and tensor engines on deterministic synthetic/public worlds and small/medium populations. For every compared world/scheduler/slot require exact equality of:

1. incoming live candidate path/ID set;
2. bounded visible candidate IDs for every tested K;
3. selected parent path/ID;
4. branch child path/IDs;
5. answer-blind overflow-retained path/ID set;
6. terminal path multiset/transcript;
7. productive/sink slot accounting;
8. learned recurrent-update accounting;
9. score-observation accounting;
10. final exact-search coverage.

For recurrent tensors/scores, preserve exact eager FP32 equality where operation order permits. If vectorization causes harmless floating-order differences, any numerical tolerance must be frozen before Gate-7 science and all discrete routing/pruning transcripts must remain exact.

## K ladder equivalence coverage

At minimum qualify:

- K16;
- K32;
- K64;
- global reference;
- matched answer-blind hash controls for every K used in primary Gate-7 analysis.

Any larger admitted K must receive the same deterministic sampler/selection regressions before exposure.

## Population bridge coverage

Use synthetic N64/N128/N256 cases that exercise:

- no overflow;
- active overflow pruning;
- sparse live masks/holes after parent removal;
- child insertion into free slots;
- terminal generation;
- divergent score/hash histories.

No Gate-6 scientific world namespace is reused.

## Hot-path synchronization prohibition

The scientific tensor hot path must not perform CUDA-dependent:

```text
.item()
.cpu()
.numpy()
.tolist()
Python float()/int() on CUDA values
CUDA-dependent Python branching
```

except at explicit post-decision aggregate telemetry/provenance boundaries.

## Bounded-computation requirement

For bounded `score_K` before parent selection:

- sampler work may depend on N only through deterministic integer index generation / live-mask access;
- exactly `min(K, live_count)` neural scores may be gathered/read for causal selection;
- no full score sort/rank/reduction may execute before parent selection;
- evaluation-only diagnostics are structurally separated from causal selection.

This requirement is stronger than merely hiding non-sampled score values: the implementation itself must not remain computationally global before bounded selection.

## Answer-blind capacity handling

Capacity retention may depend on public runtime seed, slot index and candidate path/slot identity. It must not depend on neural score/state, hidden answer or outcome labels.

The tensorized retention set must match the frozen eager reference on equivalence fixtures unless a separately versioned Gate-7 protocol freezes a new answer-blind retention primitive before scientific exposure.

## Physical batching equivalence

Ready updates from multiple independent worlds may be gathered into larger CUDA batches only if each world's logical transcript remains identical to separate execution.

Physical batch size may not alter selected parents, state ownership, recurrent update count, sampling namespace, score visibility or pruning decisions.

## Stage-A reuse equivalence

The common Stage-A frontier may be built once and copied into scheduler treatments only after regressions prove:

- identical copied states/scores/IDs;
- no mutable storage aliasing;
- unchanged treatment-start state;
- unchanged capability outputs relative to separately recomputed Stage-A;
- unchanged logical work accounting.

## Profiling isolation

Performance profiles remain engineering-only. Compare independently:

```text
eager object reference
eager tensorized baseline
compiled tensorized mode
CUDA-graph mode
```

Compiler/runtime graph selection is a separate execution variable and cannot be baked into the semantic baseline.

## Admission sequence

```text
qualified eager reference
        ↓
qualified tensorized Stage-A (DONE)
        ↓
measured local Stage-A profile: 48.24x (DONE)
        ↓
implement dynamic Stage-B tensor bank
        ↓
exact Stage-B transcript equivalence across K ladder
        ↓
profile full eager tensorized scheduler
        ↓
freeze Gate-7 bandwidth-scaling runner
        ↓
ONLY THEN expose Gate-7 scientific worlds
```

Any transcript mismatch blocks scientific admission until resolved. An optimization that changes semantics is rejected or becomes a separately versioned scientific treatment.
