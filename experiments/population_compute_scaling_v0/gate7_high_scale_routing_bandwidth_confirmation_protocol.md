# Gate-7 v0 — routing-bandwidth frontier confirmation protocol

## Status

**PREREGISTERED / DATA-FROZEN — CONFIRMATION EXECUTION CLOSED.**

This protocol is stacked on the exact screening-result record head:

`07b6397f2a9d4f71ed789d6c7011e12b4cbf90e0`

Bound screening evidence:

```text
result SHA-256: d76c8b0753a518b4c61b3ff42c1f3e85902e2e492342f23fa6706459ee13a9b5
audit SHA-256:  7352621ef5c5199cba98070e2f2511674bd2f4aba8b20b48c0ec87436c5204d5
outcome:          G7_ROUTING_BANDWIDTH_FRONTIER_REACHED
```

No confirmation hidden path, hint stream, runtime seed, checkpoint execution or result exists on this protocol-only branch.

## Screening result being confirmed

The 64-world screening found:

```text
N1024  K_required = 256
N2048  K_required = 128
N4096  K_required = 512
N8192  no K<=512 passed
```

At N8192 the global learned reference remained strongly viable, while the complete bounded K ladder failed the preregistered all-checkpoint non-inferiority rule. The screening therefore stopped at a routing-bandwidth frontier rather than a reference or resource frontier.

The observed K sequence is non-monotonic and must not be fitted as a scaling law. Confirmation targets the finite boundary only.

## Confirmation question

> With a new untouched 512-world namespace and the same exact checkpoint family, scheduler, work budget and primary criteria, does N4096/K512 replicate as a passing anchor while no K<=512 passes at N8192?

## Exact checkpoint family

All conditions use the same three qualified 19,649-parameter scale-neutral transition checkpoints:

```text
T0 be1b18e22a1f713b48de8934aca1c0302f1255342684158c6ae31a7c3618a719
T1 a9b712736f440168ccd86ee070b8d301be0eb834b9d17418d40c0175207cadfb
T2 cd7869511d07679cb1ad430743cad031299a575413e8a853ef9116d545c7480a
```

No checkpoint selection, retraining, fine-tuning, ensembling or replacement is permitted.

## Fixed confirmation matrix

### N4096 anchor

Every checkpoint executes exactly:

```text
global_score
global_hash
bounded_score_k512
bounded_hash_k512
```

This tests replication of the last screening tier with a valid K_required result.

### N8192 frontier

Every checkpoint executes exactly:

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

The full K ladder is frozen before confirmation exposure. There is no sequential first-pass stopping in confirmation and no post-result choice of a preferred K.

This prevents the screening's near-boundary K256 observation from becoming a selectively tested rescue condition. K256 is included because every preregistered N8192 K is included.

## Fresh confirmation worlds

For each population and world index:

- hidden path namespace: `gate7-high-scale-routing-bandwidth-confirmation-hidden-v0`;
- hint namespace: `gate7-high-scale-routing-bandwidth-confirmation-hints-v0`;
- runtime namespace: `gate7-high-scale-routing-bandwidth-confirmation-runtime-v0`;
- namespace inputs include population, world index and task depth;
- hint reliability remains 0.70;
- world indices are exactly 0 through 511;
- all score/hash/global conditions share the same 512 paired worlds.

The confirmation namespaces must be distinct from every screening and bridge namespace.

## Fixed evaluation geometry

```text
worlds/checkpoint/population = 512
physical evaluation batch    = 64
physical batches              = 8
Stage-A parent slots/world    = N - 1
Stage-B parent slots/world    = 128
active child lanes            = 2
recurrent updates/child       = 8
logical learned updates/world = (N - 1 + 128) * 16
```

The qualified B64 row-chunked frontier substrate remains mandatory. Compiler, CUDA graphs and mixed precision remain off.

An implementation may construct one B64 frontier and reuse it across all fixed conditions for that checkpoint/population/world batch. It may not reduce the logical world count, alter world pairing or change condition semantics.

## Statistical procedure

Each checkpoint/condition pair has 512 paired Boolean coverage outcomes.

The confirmation uses:

- 10,000 deterministic paired bootstrap samples per checkpoint comparison;
- deterministic stratified paired bootstrap for global-reference viability, independently resampling 512 pairs inside each checkpoint stratum and averaging the three stratum means;
- the same percentile interval construction and strict inequalities as screening;
- the same five-percentage-point non-inferiority margin.

### Global-reference viability

At each population:

1. `global_score - global_hash` point delta must be strictly positive on T0, T1 and T2;
2. the stratified pooled bootstrap CI low must be strictly greater than zero.

### K pass rule

A fixed K passes only when all six checkpoint criteria hold:

```text
for each checkpoint C:
  CI_low(score_K - hash_K) > 0
  CI_low(score_K - global_score) > -0.05
```

The strict threshold is unchanged. Equality to zero or -0.05 fails.

## Frozen outcome classifier

Outcomes are hierarchical:

1. `G7_CONFIRMATION_ANCHOR_REFERENCE_NOT_REPLICATED`
   - N4096 global reference is not viable.
2. `G7_CONFIRMATION_ANCHOR_K512_NOT_REPLICATED`
   - N4096 reference is viable but K512 does not pass all six criteria.
3. `G7_CONFIRMATION_N8192_REFERENCE_NOT_REPLICATED`
   - the N4096 anchor passes, but the N8192 global reference is not viable.
4. `G7_ROUTING_BANDWIDTH_FRONTIER_NOT_CONFIRMED`
   - the anchor passes, N8192 global reference is viable, and at least one fixed K passes at N8192.
   - report every passing K and the smallest passing K; do not suppress non-monotonic passing values.
5. `G7_ROUTING_BANDWIDTH_FRONTIER_CONFIRMED`
   - N4096 K512 passes;
   - N8192 global reference remains viable;
   - none of K16, K32, K64, K128, K256 or K512 passes all six criteria.

A failure to replicate the anchor is not reinterpreted as frontier confirmation. A loss of the N8192 global reference is classified separately from bounded routing failure.

## Required artifact

The eventual confirmation artifact must report:

- exact checkpoint SHA/fingerprint identities;
- all 512 world indices and runtime seeds per population;
- all condition coverage vectors and rates;
- score-observation accounting;
- logical learned-work identity;
- all paired bootstrap deltas and intervals;
- both stratified global-reference summaries;
- N4096 K512 pass/fail;
- every passing N8192 K, including non-monotonic sets;
- the exact frozen outcome;
- runtime and peak VRAM as engineering telemetry only;
- training/checkpoint-selection/confirmation provenance flags;
- independent audit and recursive manifest hashes.

## Interpretation boundary

`G7_ROUTING_BANDWIDTH_FRONTIER_CONFIRMED` would replicate the finite N4096→N8192 boundary under this exact task and K<=512 ladder. It would not establish an asymptotic law or prove that no larger K could work.

`G7_ROUTING_BANDWIDTH_FRONTIER_NOT_CONFIRMED` would mean the higher-powered untouched sample produced at least one N8192 passing K. It would not invalidate the screening artifact; it would supersede only the confidence placed in the frontier claim.

No outcome permits automatic expansion to N16384+, addition of K>512, retraining, or a second confirmation namespace.

## Pre-exposure lock

Before execution can be admitted, CI must prove:

- exact screening-result and checkpoint identities remain bound;
- only N4096 and N8192 are reachable;
- N4096 exposes only global/hash and K512/hash;
- N8192 exposes the complete fixed K ladder;
- 512 worlds and B64 batching are immutable;
- 10,000 bootstrap samples and the 0.05 margin are immutable;
- the outcome hierarchy cannot classify an anchor failure as frontier confirmation;
- no confirmation world generator, checkpoint loader, runner or result artifact exists on the protocol branch.