# Gate-3 v1 — three-seed robustness result

## Status

**R1_ROBUST_LATENT_POPULATION_MECHANISM**

This classification is assigned mechanically from the robustness map frozen after seed 0 and before seeds 1 or 2 were trained.

Confirmation remains closed.

## Frozen policy applied

The preregistered R1 rule requires every one of seeds 0, 1 and 2 to satisfy:

1. S8 L64-vs-L1 paired bootstrap CI low > 0;
2. S10 L256-vs-L1 paired bootstrap CI low > 0;
3. S10 stable-vs-collapsed paired bootstrap CI low > 0;
4. S10 stable-vs-reshuffled paired bootstrap CI low > 0;
5. S10 L256-vs-L64 coverage delta >= 0.

All five conditions hold for all three independent learned checkpoints.

## Three-seed primary effects

| Primary comparison | Seed 0 | Seed 1 | Seed 2 |
|---|---:|---:|---:|
| S8 L64 vs L1 delta | +0.515625 | +0.51171875 | +0.515625 |
| CI low | +0.453125 | +0.453125 | +0.45703125 |
| S10 L256 vs L1 delta | +0.5859375 | +0.58203125 | +0.5703125 |
| CI low | +0.51953125 | +0.515625 | +0.50390625 |
| S10 L256 vs L64 delta | 0.0 | 0.0 | 0.0 |
| stable vs collapsed delta | +0.5859375 | +0.58203125 | +0.5703125 |
| CI low | +0.5234375 | +0.51953125 | +0.5078125 |
| stable vs reshuffled delta | +0.5000000 | +0.51171875 | +0.48828125 |
| CI low | +0.42578125 | +0.44140625 | +0.4140625 |

## Stable S10 coverage by reserve capacity

```text
seed 0: L1 .0273 -> L4 .1172 -> L16 .5039 -> L64 .6133 -> L256 .6133
seed 1: L1 .0273 -> L4 .1172 -> L16 .4766 -> L64 .6094 -> L256 .6094
seed 2: L1 .0273 -> L4 .1172 -> L16 .5273 -> L64 .5977 -> L256 .5977
```

Across all three independently initialized learned checkpoints:

- L1 is identical at 7/256 covered worlds;
- larger stable latent reserves produce a large capability increase;
- L64 and L256 are exactly identical within each checkpoint;
- the top stable coverage varies only modestly across checkpoints: 157/256, 156/256, 153/256;
- collapsed L256 remains identical to L1;
- reshuffling persistent neural continuity destroys most of the stable-reserve advantage.

## Scientific interpretation

The three-seed result supports the following narrow mechanism claim for this controlled no-replay search workload:

> At fixed learned parameters, fixed active neural width, fixed recurrent refinement per evaluated child, fixed search rounds and fixed total learned recurrent work, retaining a larger dormant population of distinct persistent hypotheses robustly improves exact search coverage across independent learned checkpoints.

The control pattern further supports that the benefit depends on both:

1. maintaining multiple distinct alternatives; and
2. preserving candidate-specific neural state/score continuity.

This is materially different from Gate-3 v0, where increasing population divided the same learned-work budget across more simultaneously processed hypotheses and performance deteriorated.

## Frontier result

The useful frontier does **not** extend from L64 to L256 in this workload:

```text
seed 0: L256 - L64 = 0.0
seed 1: L256 - L64 = 0.0
seed 2: L256 - L64 = 0.0
```

This is not treated as a failure of the latent-population mechanism. Under the frozen robustness policy it is an R1 result because the policy explicitly did not require the frontier to extend beyond L64.

It does mean Gate-3 v1 did not satisfy the original intended positive development pattern requiring S10 L256 > L64. Therefore Gate-3 v1 confirmation remains closed and no later robustness seed can retroactively open it.

## What this does not establish

This result does not establish:

- general intelligence;
- superiority on arbitrary workloads;
- superiority to every serial/search algorithm, including algorithms allowed to replay or recompute discarded branches;
- per-FLOP, per-joule or training-efficiency superiority;
- that useful capacity will continue increasing beyond L64;
- a positive Gate-3 confirmation verdict.

## Next scientific question

The next experiment must be separately versioned and frozen before data:

> Under a harder or broader search regime where L64 is no longer sufficient to retain the useful frontier, does the beneficial latent-population mechanism continue scaling to larger dormant populations without increasing active neural width or total learned work?

That is a frontier-scaling question, not a retry of Gate-3 v1.

## Evidence lineage

Seed 0:
- checkpoint: `e63748a1182472d03c540f9123c3442ce44e130519e2176681034568826db590`
- result: `26559f5c48ae2971dbb83507afbe9346c6575653e352202ad8b450b739423342`
- audit: `289be3ba58c22a9276220804daf7358d97ff5402533cb89e21e1b3c5f53ccf32`
- manifest: `d31fcd36bdaf3416a1926e390d934da926623f34a99497e04340bf1fbc2b773a`

Seed 1:
- checkpoint: `8ce384627f5543fa4fe78498f9990f8214cf2a4afed0c9b734e86761ad13c989`
- result: `8525272080af7121b3c427a94be77e00b714d2248ab58ddf9e6b228154604eae`
- audit: `94f06266ebbf5400e4a89878ec5b6adcc018b40cb32bd4417d692de15daa1271`
- manifest: `94b88791f716c6d469d1891ed5847141997e3fc0e096215874b1fbf9234d1967`

Seed 2:
- checkpoint: `103826bbd9451b965eced5134d1674cb8e893f5d3b378cb828312ffdb6fc9a37`
- result: `728dd91017c84f75084439a86e481a7f3bf2effcf3ca067f8c143ce65ef55217`
- audit: `f1a7bdf46fa33c3be715221292726cc869dac77b660c9a7a8821285d4ed61138`
- manifest: `59e3367c8c7311af76512e06c2074143c206177a3815a33d2186c79c771b2237`

Policy source:
- `experiments/population_compute_scaling_v0/gate3_v1_robustness_policy_after_seed0.md`

Per-seed records:
- `experiments/population_compute_scaling_v0/gate3_v1_development_seed0_result.md`
- `experiments/population_compute_scaling_v0/gate3_v1_robustness_seed1_result.md`
- `experiments/population_compute_scaling_v0/gate3_v1_robustness_seed2_result.md`
