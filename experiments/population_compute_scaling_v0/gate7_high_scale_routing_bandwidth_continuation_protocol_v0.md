# Gate-7 high-scale routing-bandwidth post-confirmation continuation v0

## Status

**PREREGISTERED / DATA-FROZEN PROTOCOL — EXECUTION CLOSED.**

This protocol is bound to the valid confirmation result:

```text
confirmation execution head: 7afa6f204215bac7da4623e231ec34ef3b7fdc9f
confirmation result head:    ae8bd8544a03e48f4f397d2ca5ae933d9247e430
result SHA-256:               725e3749ba5fed7cdcbb6d61df81bcc77a7b69bacfdc82d553efb06f5ff888da
audit SHA-256:                27a46ba0feccf6b3322885334819e0e7a07bb02be930122eb1f063c65d69fb99
manifest SHA-256:             e7c1823dc59a50b58250cab0f7b18b95ca42b831e90182f07295680b6986b263
outcome:                      G7_ROUTING_BANDWIDTH_FRONTIER_NOT_CONFIRMED
N8192 passing K:              256, 512
N8192 smallest passing K:     256
```

A second confirmation is closed. This is a new continuation study with a fresh namespace, not a rerun or extension of the confirmation worlds.

## Scientific purpose

The completed confirmation established that the 64-world N8192 routing frontier was a finite-sample false frontier. It also demonstrated that a first missing K cannot safely be treated as a monotonic campaign stop.

The continuation therefore measures the complete remaining high-scale population ladder under the stronger 512-world evidence standard.

## Fixed remaining population ladder

```text
N16384
N32768
N65536
N131072
```

All four tiers are fixed before exposure.

## Fixed condition matrix at every N

Each checkpoint and population evaluates all fourteen conditions:

```text
global_score
global_hash
bounded_score_k16
bounded_hash_k16
bounded_score_k32
bounded_hash_k32
bounded_score_k64
bounded_hash_k64
bounded_score_k128
bounded_hash_k128
bounded_score_k256
bounded_hash_k256
bounded_score_k512
bounded_hash_k512
```

There is:

- no ascending first-pass exposure;
- no suppression of larger K after a pass;
- no rescue K selected after seeing results;
- no stopping after a tier with no K<=512;
- no stopping after a tier whose global reference is non-viable.

The complete fixed matrix is necessary because the observed finite-range K behavior is non-monotonic.

## Evidence standard

For every checkpoint/population:

- 512 fresh paired worlds;
- exactly eight physical batches of 64 worlds;
- one immutable B64 frontier per checkpoint/batch, reusable across all fixed conditions;
- exact checkpoints T0/T1/T2;
- 10,000 deterministic paired-bootstrap samples;
- deterministic checkpoint-stratified global-reference bootstrap;
- 0.70 hint reliability;
- unchanged 0.05 global non-inferiority margin;
- 128 terminal Stage-B parent activations;
- two terminal child lanes and eight recurrent updates per child;
- FP32;
- compiler, CUDA graphs and mixed precision disabled.

## Per-tier classification

The global learned reference is viable only when:

1. every checkpoint has a positive global-score minus global-hash point delta; and
2. the checkpoint-stratified pooled 95% CI lower bound is greater than zero.

A K passes only when all three checkpoints satisfy both:

1. learned score minus matched hash CI lower bound is greater than zero; and
2. learned score minus global learned CI lower bound is greater than -0.05.

Every population records the complete ordered passing-K set.

Tier outcomes are:

```text
G7_CONTINUATION_K_REQUIRED
G7_CONTINUATION_NO_K_LE_512
G7_CONTINUATION_REFERENCE_NOT_VIABLE
```

For a viable reference with at least one passing K:

```text
K_required(N) = smallest passing tested K
```

The complete passing set remains part of the result so non-monotonic K behavior is not discarded.

## Campaign continuation rule

Scientific tier outcomes do not stop the campaign.

```text
N16384  -> continue
N32768  -> continue
N65536  -> continue
N131072 -> complete
```

The only admitted early truncation is an actual execution resource frontier. A resource frontier must be recorded at the next uncompleted population, with all completed populations forming an exact contiguous prefix.

Campaign outcomes are:

```text
G7_POST_CONFIRMATION_LADDER_COMPLETE
G7_POST_CONFIRMATION_RESOURCE_FRONTIER_REACHED
```

## Interpretation boundary

This continuation can establish tested K_required values or no-pass/reference outcomes for N16384 through N131072 under K<=512 and the fixed 512-world design.

It cannot by itself establish:

- an asymptotic scaling law;
- monotonic K_required growth or decline;
- that no K>512 could pass;
- that a non-viable reference can never recover at a later N;
- a universal maximum useful population;
- confirmation beyond the one completed N4096/N8192 study.

## Pre-exposure boundary

This protocol branch contains only:

- pure-Python constants and classifiers;
- this preregistration document;
- synthetic structural tests;
- protocol-only CI.

It contains no continuation world generator, checkpoint loader, Torch dependency, execution runner, PowerShell wrapper or result artifact.
