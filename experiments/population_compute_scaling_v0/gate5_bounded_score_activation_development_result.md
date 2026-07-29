# Gate-5 v0 — bounded score-visibility development result

## Status

**DEVELOPMENT RESULT RECORDED — `G5_B2_ROBUST_BOUNDED_SCORE_ACTIVATION`.**

This record preserves the first admitted Gate-5 v0 development run executed under the frozen pre-result protocol.

- measured Git head: `1064ab6b20b10691503b71b806a7ec5d516f0a80`
- training performed: **no**
- confirmation opened: **no**
- development worlds: **256/checkpoint**
- checkpoints: exact three frozen Gate-3 v1 checkpoints
- reserve capacity: **L256**
- Stage A breadth warm-up: **63 parent slots**
- Stage B bounded/global adaptive phase: **96 parent slots**
- total scheduled parent slots/world: **159**
- active child lanes/productive slot: **2**
- recurrent updates/child: **8**
- learned recurrent updates/world: **2,544**
- strict bounded-visibility admitted runtime: **yes**
- independent artifact audit: **valid, errors=[]**

## Frozen checkpoint identities

| checkpoint | SHA-256 | parameter fingerprint | learned parameters |
|---|---|---|---:|
| C0 | `e63748a1182472d03c540f9123c3442ce44e130519e2176681034568826db590` | `e4f9990f08e85393a155637cfa50b5831d779770447ec3716fc9e67946992afc` | 19,649 |
| C1 | `8ce384627f5543fa4fe78498f9990f8214cf2a4afed0c9b734e86761ad13c989` | `2a57984f755ee1404fd828c2df36a5efff8f13a541b7d7c93a891014fbf4897c` | 19,649 |
| C2 | `103826bbd9451b965eced5134d1674cb8e893f5d3b378cb828312ffdb6fc9a37` | `8afaf956f200f41ea914eafdd1b5f151dd303cae1552165c74e12bb8c945af02` | 19,649 |

## Raw condition coverage

| checkpoint | global | K4 | K8 | K16 | K32 | K16 hash |
|---|---:|---:|---:|---:|---:|---:|
| C0 | 0.8515625 | 0.7890625 | 0.84765625 | 0.84765625 | 0.84765625 | 0.3984375 |
| C1 | 0.87109375 | 0.80078125 | 0.85546875 | 0.85546875 | 0.86328125 | 0.3984375 |
| C2 | 0.8671875 | 0.81640625 | 0.859375 | 0.8515625 | 0.8671875 | 0.3984375 |

## Candidate-score observation budget

Mean Stage-B score observations/world:

| checkpoint | global | K4 | K8 | K16 | K32 | K16 hash |
|---|---:|---:|---:|---:|---:|---:|
| C0 | 5068.21875 | 384 | 768 | 1536 | 3072 | 0 |
| C1 | 5092.3203125 | 384 | 768 | 1536 | 3072 | 0 |
| C2 | 5065.9609375 | 384 | 768 | 1536 | 3072 | 0 |

Thus K16 used only about 30% of the global scheduler's score observations while preserving near-global coverage, and K8 used only about 15%.

## Preregistered primary effects

### Learned bounded-routing effect: `bounded_score_k16 - bounded_hash_k16`

| checkpoint | delta | paired-bootstrap 95% CI |
|---|---:|---:|
| C0 | **+0.44921875** | **[+0.37890625, +0.5234375]** |
| C1 | **+0.45703125** | **[+0.37890625, +0.52734375]** |
| C2 | **+0.453125** | **[+0.37890625, +0.52734375]** |

Every CI low is strictly above zero.

### Primary non-inferiority effect: `bounded_score_k16 - global_score`

Frozen non-inferiority margin: `delta_NI = 0.05`.

| checkpoint | delta | paired-bootstrap 95% CI | NI passed? |
|---|---:|---:|---:|
| C0 | -0.00390625 | **[-0.0234375, +0.015625]** | yes |
| C1 | -0.015625 | **[-0.0390625, +0.00390625]** | yes |
| C2 | -0.015625 | **[-0.03125, -0.00390625]** | yes |

Every lower bound is strictly greater than `-0.05`, so K16 satisfies the preregistered non-inferiority criterion on all three checkpoints.

## Preregistered bounded-visibility frontier

Paired deltas vs global and lower CI bounds:

| K | C0 delta / low | C1 delta / low | C2 delta / low | all-checkpoint 5pp NI? |
|---|---|---|---|---|
| K4 | -0.0625 / -0.109375 | -0.0703125 / -0.109375 | -0.05078125 / -0.0859375 | no |
| K8 | -0.00390625 / -0.02734375 | -0.015625 / -0.046875 | -0.0078125 / -0.03125 | **yes** |
| K16 | -0.00390625 / -0.0234375 | -0.015625 / -0.0390625 | -0.015625 / -0.03125 | **yes** |
| K32 | -0.00390625 / -0.01171875 | -0.0078125 / -0.0234375 | 0.0 / -0.01171875 | **yes** |

The preregistered descriptive frontier therefore reports:

`smallest_noninferior_k = 8`

This is development evidence only. Gate-5's primary treatment remains K16 because that was frozen before data exposure.

## Frozen development classification

The independent auditor returned:

```text
artifact_valid = true
errors = []
directional_outcome = G5_B2_ROBUST_BOUNDED_SCORE_ACTIVATION
scientific_status = DEVELOPMENT_ONLY_NO_GATE_VERDICT
smallest_noninferior_k = 8
```

This mechanically satisfies development Outcome **B2** because all three checkpoints satisfy both preregistered K16 conditions:

1. `bounded_score_k16 - bounded_hash_k16` paired-bootstrap CI low `> 0`;
2. `bounded_score_k16 - global_score` paired-bootstrap CI low `> -0.05`.

## Supported development-stage conclusion

Within this controlled fixed-population hypothesis-search regime, useful learned activation routing survives a bounded candidate-score visibility channel. At K16, the scheduler observes at most 16 live candidate scores per activation and retains capability that is non-inferior to full-reserve score inspection under the frozen five-percentage-point margin, while strongly outperforming the matched answer-blind K16 routing control.

The preregistered frontier additionally indicates that K8 was already non-inferior to global on all three development checkpoints, but this is descriptive frontier evidence rather than a replacement for the frozen K16 primary treatment.

## Claims boundary

This result does not establish:

- a final Gate-5 verdict before independent confirmation;
- physical decentralization or distributed-machine execution;
- arbitrary graph/locality robustness;
- AGI or arbitrary-task generalization;
- optimal communication complexity;
- per-FLOP/per-joule superiority;
- scaling to 20K/100K runtime workers;
- that K8 or K16 is universally sufficient.

## Confirmation boundary

The frozen Gate-5 protocol permits a separately versioned confirmation only after `G5_B2_ROBUST_BOUNDED_SCORE_ACTIVATION`.

That condition is now met.

**No Gate-5 confirmation world was generated or inspected by this development run.** Confirmation remains closed until a separate confirmation protocol is frozen and qualified.

## Provenance

Local admitted artifact hashes reported by the runner:

- result SHA-256: `be362de078d9d20025ecc2983ae0fd65a0069548fc094cae992a2aa7754be7e2`
- independent audit SHA-256: `af4259c003d9a0bb467b5d6be85aab73b68d3835aed74ed2187f413b73e6de46`
- recursive manifest SHA-256: `b950b2075e95c470f15b9ba6d4d4f7c818a09945c3f6364badf6dd6498d9ba66`

Local evidence root reported by the runner:

`F:\gate5_bounded_score_activation_development_v0`
