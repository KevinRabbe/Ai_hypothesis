# Gate-7 high-scale routing-bandwidth confirmation v0 — result

## Status

**VALID FRESH CONFIRMATION EVIDENCE — THE SCREENING FRONTIER AT N8192 WAS NOT CONFIRMED.**

Exact admitted confirmation-execution head:

`7afa6f204215bac7da4623e231ec34ef3b7fdc9f`

Independent audit:

```text
artifact_valid = true
errors = []
confirmation_outcome = G7_ROUTING_BANDWIDTH_FRONTIER_NOT_CONFIRMED
anchor_k512_passed = true
passing_k_at_n8192 = [256, 512]
```

No training, checkpoint selection, adaptive exposure, second confirmation or post-result condition was performed.

## Frozen confirmation design

- exact transition checkpoints T0/T1/T2;
- N4096 anchor: global score/hash and K512 score/hash;
- N8192 frontier: global score/hash and the complete fixed K ladder 16, 32, 64, 128, 256, 512;
- 512 untouched paired worlds/checkpoint/population;
- eight physical batches of 64 worlds;
- 10,000 deterministic paired-bootstrap samples;
- unchanged five-percentage-point non-inferiority margin;
- 128 terminal Stage-B activations;
- FP32 with compiler, CUDA graphs and mixed precision disabled.

## N4096 anchor replicated

The global learned reference remained viable:

| Checkpoint | Global score | Global hash | Delta | 95% CI |
| ---: | ---: | ---: | ---: | ---: |
| T0 | 0.3164 | 0.0293 | +0.2871 | [+0.2461,+0.3301] |
| T1 | 0.2930 | 0.0293 | +0.2637 | [+0.2227,+0.3047] |
| T2 | 0.2891 | 0.0293 | +0.2598 | [+0.2188,+0.3027] |

Stratified pooled global-reference delta:

`+0.2702`, 95% CI `[+0.2467,+0.2943]`.

K512 passed every frozen checkpoint criterion:

| Checkpoint | K512 score | K512 hash | learned-vs-hash CI low | K512-vs-global CI low |
| ---: | ---: | ---: | ---: | ---: |
| T0 | 0.3105 | 0.0332 | +0.2363 | -0.0156 |
| T1 | 0.2969 | 0.0332 | +0.2227 | -0.0059 |
| T2 | 0.2891 | 0.0332 | +0.2148 | -0.0098 |

Therefore the preregistered N4096 anchor replicated.

## N8192 frontier was overturned

The N8192 global learned reference also remained viable:

| Checkpoint | Global score | Global hash | Delta | 95% CI |
| ---: | ---: | ---: | ---: | ---: |
| T0 | 0.2500 | 0.0176 | +0.2324 | [+0.1934,+0.2734] |
| T1 | 0.2461 | 0.0176 | +0.2285 | [+0.1895,+0.2676] |
| T2 | 0.2305 | 0.0176 | +0.2129 | [+0.1758,+0.2520] |

Stratified pooled global-reference delta:

`+0.2246`, 95% CI `[+0.2025,+0.2467]`.

Every tested K retained learned-over-hash value. The frozen non-inferiority criterion relative to global learned routing separated the ladder as follows:

| K | T0 score-vs-global CI low | T1 | T2 | All-checkpoint pass |
| ---: | ---: | ---: | ---: | :---: |
| 16 | -0.1934 | -0.1875 | -0.1777 | no |
| 32 | -0.1406 | -0.1445 | -0.1270 | no |
| 64 | -0.1172 | -0.1113 | -0.0996 | no |
| 128 | -0.0703 | -0.0645 | -0.0684 | no |
| 256 | -0.0234 | -0.0156 | -0.0234 | **yes** |
| 512 | -0.0156 | -0.0156 | -0.0195 | **yes** |

Under the frozen tested ladder, the smallest passing N8192 budget is therefore:

`K_required(8192) = 256`.

Its tested visibility ratio is:

`256 / 8192 = 0.03125 = 3.125%`.

K512 also passed, so the confirmation preserves the complete passing set `[256, 512]` rather than assuming monotonicity from one point.

## Scientific interpretation

The original 64-world screen classified N8192 as `G7_ROUTING_BANDWIDTH_FRONTIER_REACHED`. The higher-powered untouched 512-world confirmation rejected that classification.

The earlier stop was a finite-sample false frontier caused by uncertainty around the five-point global non-inferiority boundary. It was not:

- a collapse of learned routing signal;
- a failure of the global learned reference;
- a resource frontier;
- evidence that K greater than 512 was required.

The confirmed finite-range statement is instead:

> At N8192, observing scores for 128 candidates was insufficient under the frozen criterion, while observing 256 candidates was sufficient across all three checkpoints.

This does not establish an asymptotic scaling law. The confirmed points remain non-monotonic across population:

```text
N4096: K_required = 512
N8192: K_required = 256
```

Earlier 64-world exploratory estimates at N1024 and N2048 remain screening evidence and should not be pooled with the 512-world confirmation points as though they had equal precision.

## Provenance

- result SHA-256: `725e3749ba5fed7cdcbb6d61df81bcc77a7b69bacfdc82d553efb06f5ff888da`
- audit SHA-256: `27a46ba0feccf6b3322885334819e0e7a07bb02be930122eb1f063c65d69fb99`
- manifest SHA-256: `e7c1823dc59a50b58250cab0f7b18b95ca42b831e90182f07295680b6986b263`
- local output root: `F:\gate7_high_scale_routing_bandwidth_confirmation_v0`

## Next scientific boundary

A second confirmation is closed. The next permissible study is a separately frozen post-confirmation continuation of the population ladder, beginning at N16384, using the stronger 512-world evidence standard and a fresh namespace. This result branch adds no continuation protocol or execution path.
