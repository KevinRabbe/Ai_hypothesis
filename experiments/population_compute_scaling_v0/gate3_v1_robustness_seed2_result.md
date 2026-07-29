# Gate-3 v1 — robustness seed 2 result

## Status

**VALID ROBUSTNESS DEVELOPMENT RESULT — OUTCOME C: LATENT CAPACITY HELPS BUT SATURATES EARLY**

This is the final additional training seed authorized by the robustness policy frozen after seed 0.

- measured Git head: `f0afe4febb860dfbb559fafec0e486e3130358bc`
- training seed: `2`
- learned parameters: `19,649`
- confirmation opened: `false`
- independent artifact audit: `artifact_valid=true`, `errors=[]`
- directional outcome: `C_LATENT_CAPACITY_HELPS_BUT_SATURATES_EARLY`

No scientific tuning parameter or workload rule was changed from seed 0/1.

## Stable exact coverage

```text
S6:  L1 0.1367 -> L4 0.4023 -> L16 0.3789
S8:  L1 0.0859 -> L4 0.2656 -> L16 0.6016 -> L64 0.6016
S10: L1 0.0273 -> L4 0.1172 -> L16 0.5273 -> L64 0.5977 -> L256 0.5977
```

The S10 L64 and L256 stable conditions cover exactly the same 153/256 development worlds.

## Primary paired effects

| Comparison | Delta | 95% paired bootstrap CI |
|---|---:|---:|
| S8 stable L64 vs L1 | +0.515625 | [0.45703125, 0.58203125] |
| S10 stable L256 vs L1 | +0.5703125 | [0.50390625, 0.6328125] |
| S10 stable L256 vs L64 | 0.0 | [0.0, 0.0] |
| S10 stable L256 vs collapsed L256 | +0.5703125 | [0.5078125, 0.62890625] |
| S10 stable L256 vs reshuffled L256 | +0.48828125 | [0.4140625, 0.5625] |

## Controls

At S10/L256:

```text
stable reserve          0.5977
collapsed diversity     0.0273
reshuffled continuity   0.1094
```

The collapsed logical-one control remains identical to L1, while deterministic reassignment of persistent neural history removes most of the stable-reserve advantage.

## Interpretation

Seed 2 independently reproduces the mechanism observed in seeds 0 and 1:

> With learned parameters, active neural width, recurrent updates per evaluated child, search rounds, noisy evidence and total learned work fixed, a larger dormant population of distinct persistent hypotheses substantially improves no-replay exact search coverage.

The useful capacity frontier again saturates by L64 under this workload. L256 produces no incremental coverage over L64.

This is robustness-development evidence only. It does not assign a positive Gate-3 verdict and it does not open confirmation.

## Provenance

- checkpoint SHA-256: `103826bbd9451b965eced5134d1674cb8e893f5d3b378cb828312ffdb6fc9a37`
- result SHA-256: `728dd91017c84f75084439a86e481a7f3bf2effcf3ca067f8c143ce65ef55217`
- audit SHA-256: `f1a7bdf46fa33c3be715221292726cc869dac77b660c9a7a8821285d4ed61138`
- manifest SHA-256: `59e3367c8c7311af76512e06c2074143c206177a3815a33d2186c79c771b2237`
- parameter fingerprint: `8afaf956f200f41ea914eafdd1b5f151dd303cae1552165c74e12bb8c945af02`

Output root used locally:

`F:\gate3_v1_sparse_active_reserve_robustness_seed_2`
