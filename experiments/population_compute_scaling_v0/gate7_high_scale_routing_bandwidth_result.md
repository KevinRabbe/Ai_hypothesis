# Gate-7 v0 — high-scale routing-bandwidth screening result

## Status

**VALID FRESH SCIENTIFIC EVIDENCE — `G7_ROUTING_BANDWIDTH_FRONTIER_REACHED` AT N8192.**

Exact admitted execution head:

`39ed74ffb93ac4ee3824666351fecc1f83a85f97`

The independent artifact audit returned:

```text
artifact_valid = true
errors = []
campaign_outcome = G7_ROUTING_BANDWIDTH_FRONTIER_REACHED
confirmation_opened = false
training_performed = false
checkpoint_selection_performed = false
```

The campaign used all three exact qualified scale-neutral transition checkpoints, 64 fresh paired worlds per checkpoint/population, physical batch 64, the frozen N/K ladders, 128 terminal Stage-B activations, 2,000 deterministic paired bootstrap samples, and the preregistered strict first-pass rule.

No checkpoint was selected, retrained, fine-tuned or replaced. Confirmation remained closed.

## Finite-range result

| Population N | Global reference pooled delta | Pooled 95% CI | Reference viable | Smallest passing K | K/N | Tier outcome |
| ---: | ---: | ---: | :---: | ---: | ---: | --- |
| 1,024 | +0.2969 | [+0.2083, +0.3854] | yes | 256 | 0.2500 | `G7_K_REQUIRED_256` |
| 2,048 | +0.3177 | [+0.2448, +0.3906] | yes | 128 | 0.0625 | `G7_K_REQUIRED_128` |
| 4,096 | +0.3281 | [+0.2656, +0.3906] | yes | 512 | 0.1250 | `G7_K_REQUIRED_512` |
| 8,192 | +0.2448 | [+0.1823, +0.3073] | yes | none at K<=512 | — | `G7_ROUTING_BANDWIDTH_FRONTIER_REACHED` |

The campaign stopped at N8192 exactly as preregistered. N16384, N32768, N65536 and N131072 were not exposed after the routing-bandwidth frontier.

## Exact checkpoint-global reference results

| N | Checkpoint | Global score | Global hash | Delta | Paired 95% CI |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1,024 | T0 | 0.4062 | 0.1562 | +0.2500 | [+0.0938, +0.4062] |
| 1,024 | T1 | 0.4844 | 0.1562 | +0.3281 | [+0.1719, +0.4844] |
| 1,024 | T2 | 0.4688 | 0.1562 | +0.3125 | [+0.1562, +0.4688] |
| 2,048 | T0 | 0.3750 | 0.0312 | +0.3438 | [+0.2188, +0.4688] |
| 2,048 | T1 | 0.3438 | 0.0312 | +0.3125 | [+0.1875, +0.4375] |
| 2,048 | T2 | 0.3281 | 0.0312 | +0.2969 | [+0.1719, +0.4219] |
| 4,096 | T0 | 0.3438 | 0.0156 | +0.3281 | [+0.2188, +0.4375] |
| 4,096 | T1 | 0.3438 | 0.0156 | +0.3281 | [+0.2188, +0.4375] |
| 4,096 | T2 | 0.3438 | 0.0156 | +0.3281 | [+0.2188, +0.4531] |
| 8,192 | T0 | 0.2656 | 0.0156 | +0.2500 | [+0.1406, +0.3594] |
| 8,192 | T1 | 0.2500 | 0.0156 | +0.2344 | [+0.1250, +0.3438] |
| 8,192 | T2 | 0.2656 | 0.0156 | +0.2500 | [+0.1406, +0.3750] |

The global learned reference therefore remained clearly viable at every exposed population. The N8192 stop is not a collapse of learned routing value relative to the answer-blind global control.

## Passing K boundaries

### N1024 — K256

K256 was the first K for which all three checkpoints satisfied both preregistered primary criteria:

- `CI_low(score_K - hash_K) > 0`;
- `CI_low(score_K - global_score) > -0.05`.

| Checkpoint | Score K256 | Hash K256 | CI low vs hash | CI low vs global |
| ---: | ---: | ---: | ---: | ---: |
| T0 | 0.4531 | 0.1562 | +0.1406 | +0.0000 |
| T1 | 0.4844 | 0.1562 | +0.1719 | +0.0000 |
| T2 | 0.4688 | 0.1562 | +0.1562 | +0.0000 |

K512 was not exposed at N1024 by the first-pass rule.

### N2048 — K128

| Checkpoint | Score K128 | Hash K128 | CI low vs hash | CI low vs global |
| ---: | ---: | ---: | ---: | ---: |
| T0 | 0.3594 | 0.0469 | +0.1875 | -0.0469 |
| T1 | 0.3281 | 0.0469 | +0.1406 | -0.0469 |
| T2 | 0.3281 | 0.0469 | +0.1406 | +0.0000 |

K256 and K512 were not exposed at N2048.

### N4096 — K512

| Checkpoint | Score K512 | Hash K512 | CI low vs hash | CI low vs global |
| ---: | ---: | ---: | ---: | ---: |
| T0 | 0.3438 | 0.0156 | +0.2188 | +0.0000 |
| T1 | 0.3438 | 0.0156 | +0.2031 | +0.0000 |
| T2 | 0.3438 | 0.0156 | +0.2188 | +0.0000 |

This is the last population at which a preregistered K<=512 passed across all three checkpoints.

## N8192 routing-bandwidth frontier

All six K values were exposed in ascending order and none passed all six checkpoint-level primary criteria.

The learned-vs-hash criterion was positive for K32 through K512 on all checkpoints. The decisive failure was non-inferiority to the still-viable global learned reference.

### K256 — nearest observed boundary candidate

| Checkpoint | Score K256 | Hash K256 | CI low vs hash | CI low vs global | Pass? |
| ---: | ---: | ---: | ---: | ---: | :---: |
| T0 | 0.2812 | 0.0156 | +0.1406 | -0.0625 | no |
| T1 | 0.2500 | 0.0156 | +0.1250 | -0.0469 | yes |
| T2 | 0.2656 | 0.0156 | +0.1406 | +0.0000 | yes |

T0 missed the strict `>-0.05` non-inferiority boundary by 0.0125 in bootstrap-CI-low units. That does not permit reclassification; K256 failed under the frozen rule.

### K512 — maximum preregistered visibility

| Checkpoint | Score K512 | Hash K512 | CI low vs hash | CI low vs global | Pass? |
| ---: | ---: | ---: | ---: | ---: | :---: |
| T0 | 0.2344 | 0.0000 | +0.1250 | -0.0781 | no |
| T1 | 0.2344 | 0.0000 | +0.1250 | -0.0625 | no |
| T2 | 0.2344 | 0.0000 | +0.1406 | -0.0781 | no |

Increasing visibility from K256 to K512 did not restore global non-inferiority in this screening sample. The preregistered campaign outcome is therefore:

`G7_ROUTING_BANDWIDTH_FRONTIER_REACHED`

## Interpretation boundary

This result supports the following finite-range statements under this exact task, checkpoint family, scheduler and work budget:

1. Learned score-based routing remains strongly superior to answer-blind full-population routing through N8192.
2. The smallest passing bounded score-visibility budgets observed were K256 at N1024, K128 at N2048 and K512 at N4096.
3. At N8192, no preregistered K<=512 preserved both learned-over-hash value and five-point global non-inferiority on all three checkpoints.
4. The observed frontier is a routing-bandwidth frontier, not a GPU/resource frontier and not a global learned-reference frontier.

The non-monotonic sequence `256, 128, 512` must not be fitted as a scaling law from this screening alone. Each tier contains 64 paired worlds and the protocol explicitly classifies this as exploratory frontier localization rather than familywise-controlled confirmation.

This result does not establish:

- an asymptotic Big-O relation;
- that K_required is intrinsically non-monotonic;
- that the true N8192 requirement is above 512 under a higher-powered confirmation sample;
- a universal maximum useful population;
- AGI, general intelligence or superiority to arbitrary serial algorithms.

## Provenance

- result SHA-256: `d76c8b0753a518b4c61b3ff42c1f3e85902e2e492342f23fa6706459ee13a9b5`
- independent audit SHA-256: `7352621ef5c5199cba98070e2f2511674bd2f4aba8b20b48c0ec87436c5204d5`
- recursive manifest SHA-256: `b384cf1ba4ed6f338365320dc7134b9c3d4cc80ff01fc2bd70b18300faf87c3c`
- local output root: `F:\gate7_high_scale_routing_bandwidth_v0`

## Confirmation boundary

Confirmation remains closed. A separately versioned, data-frozen confirmation protocol may use a new untouched namespace and higher world count to test representative boundary points selected from this screening result. It must not reinterpret or overwrite this screening artifact.