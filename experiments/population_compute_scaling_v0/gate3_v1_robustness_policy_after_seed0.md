# Gate-3 v1 — Robustness policy frozen after development seed 0

## Status

**FROZEN AFTER SEED 0, BEFORE ANY ADDITIONAL TRAINING SEED**

Seed 0 produced valid development Outcome C: latent capacity strongly improved coverage and separated from both controls, while S10 L256 exactly matched L64.

This document freezes what may happen next before seed 1 or seed 2 is trained.

## Purpose

Additional seeds are not an attempt to search for a Gate-3 v1 Outcome D checkpoint.

They test whether the observed mechanism is robust to training stochasticity:

> At fixed learned parameters, active neural width, recurrent refinement per evaluated child and total learned work, does maintaining a larger dormant population of distinct persistent hypotheses repeatedly improve no-replay search coverage relative to L1 and to the frozen population controls?

The L256-vs-L64 plateau is an observed seed-0 fact and must not be tuned away.

## Frozen additional seeds

Exactly two additional training seeds are authorized:

```text
seed 1
seed 2
```

No other training seed may be run under this robustness protocol.

Seeds 1 and 2 must use the exact same frozen recipe as seed 0 except for the training RNG seed:

```text
steps:                  1,200
batch size:             256
optimizer:              AdamW
learning rate:          3e-4
weight decay:           1e-4
gradient clip:          1.0
loss:                   SmoothL1Loss
learned parameters:     19,649
active child lanes:     2
updates/evaluated child:8
score quantization:     1e-3
```

The same held-out development worlds starting at `2^30`, the same 36-cell matrix, the same 256 worlds/depth, and the same 2,000 paired bootstrap samples are reused. Reusing the development worlds is deliberate: these runs test checkpoint/training-seed robustness while holding evaluation noise fixed.

Confirmation remains closed.

## Per-seed reconstruction

Each robustness seed must independently reconstruct the same five primary effects from raw paired coverage vectors:

1. S8 stable L64 vs L1;
2. S10 stable L256 vs L1;
3. S10 stable L256 vs L64;
4. S10 stable L256 vs collapsed L256;
5. S10 stable L256 vs reshuffled L256.

The original seed-0 analyzer remains immutable. Robustness artifacts use a separate analyzer that accepts only seeds 1 and 2 and otherwise enforces the same frozen mechanics and statistics.

## Aggregate robustness classification

After seeds 1 and 2 complete, classify the three-seed set using the following precommitted map.

### R1 — robust latent-population mechanism

Assign `R1_ROBUST_LATENT_POPULATION_MECHANISM` only if **all three seeds** satisfy all of:

- S8 L64-vs-L1 paired bootstrap CI low > 0;
- S10 L256-vs-L1 paired bootstrap CI low > 0;
- S10 stable-vs-collapsed paired bootstrap CI low > 0;
- S10 stable-vs-reshuffled paired bootstrap CI low > 0;
- S10 L256-vs-L64 coverage delta >= 0.

This supports a robust latent-population mechanism but does not require the useful frontier to extend beyond L64.

### R2 — mechanism robust but frontier location varies

Assign `R2_ROBUST_MECHANISM_VARIABLE_FRONTIER` if all three seeds have positive-CI separation for the four mechanism comparisons above, but at least one seed has S10 L256-vs-L64 delta < 0 or the incremental frontier comparison changes sign across seeds.

This supports the mechanism while showing that the useful reserve frontier is checkpoint-sensitive.

### R3 — mixed robustness

Assign `R3_MIXED_ROBUSTNESS` when the mechanism comparisons remain directionally positive in most seeds but one or more required CIs include zero.

### R4 — failed replication

Assign `R4_FAILED_REPLICATION` if any additional seed has either:

- S10 L256-vs-L1 CI high <= 0; or
- S10 stable-vs-collapsed CI high <= 0; or
- S10 stable-vs-reshuffled CI high <= 0.

This means the central seed-0 mechanism did not robustly replicate.

## Confirmation boundary

Gate-3 v1 confirmation remains closed under every robustness outcome.

Reason: the preregistered Gate-3 v1 intended positive pattern required S10 L256 > L64. Seed 0 did not satisfy that condition, so no later seed can retroactively convert this v1 development series into the originally intended confirmation path.

If robustness supports the latent-population mechanism, the next scientific step is a separately versioned frontier experiment designed **before results** to determine whether useful latent population can scale beyond the observed L64 plateau under a harder/broader search regime.

No v1 hyperparameter, workload, capacity ladder or search budget may be altered after seed 0 and still be called the same Gate-3 v1 protocol.
