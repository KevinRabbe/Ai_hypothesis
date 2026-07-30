# Gate-5 v0 — confirmed bounded score-visibility result

## Status

**FINAL CONFIRMATION EVIDENCE — CONFIRMED.**

Frozen confirmation protocol head before data exposure:

`d6541dc1e2f9c15c4c30408b2616ec04f10affe9`

Independent confirmation outcome:

`GATE5_CONFIRMED_BOUNDED_SCORE_ACTIVATION`

The artifact auditor returned `artifact_valid = true` with `errors = []`.

## Frozen confirmation design actually executed

- no training or fine-tuning;
- exact same three frozen 19,649-parameter checkpoints;
- 512 untouched confirmation worlds/checkpoint;
- depth 8, noisy-hint reliability 0.70;
- L256 fixed in every condition;
- Stage A: 63 common breadth warm-up parent slots;
- Stage B: 96 adaptive parent slots;
- exactly 159 productive parent expansions/world;
- two active child lanes;
- eight recurrent updates/generated child;
- exactly 2,544 learned recurrent updates/world;
- six frozen conditions: global, K4, K8, K16, K32, K16 hash;
- strict bounded-score visibility runtime unchanged from qualified development;
- 4,000 deterministic paired bootstrap samples.

## Confirmation coverage

| checkpoint | global | K4 | K8 | K16 | K32 | K16 hash |
|---|---:|---:|---:|---:|---:|---:|
| C0 | 0.8320 | 0.8027 | 0.8184 | 0.8242 | 0.8281 | 0.3516 |
| C1 | 0.8379 | 0.7969 | 0.8281 | 0.8223 | 0.8359 | 0.3516 |
| C2 | 0.8262 | 0.7930 | 0.8340 | 0.8203 | 0.8242 | 0.3516 |

## Frozen primary confirmation effects

### K16 learned-routing effect

`bounded_score_k16 - bounded_hash_k16`

| checkpoint | delta | paired-bootstrap 95% CI |
|---|---:|---:|
| C0 | +0.47265625 | [0.41796875, 0.52734375] |
| C1 | +0.470703125 | [0.4140625, 0.525390625] |
| C2 | +0.46875 | [0.416015625, 0.5234375] |

All three lower bounds are strictly greater than zero.

### K16 non-inferiority to global learned routing

`bounded_score_k16 - global_score`

Frozen non-inferiority margin: `-0.05`.

| checkpoint | delta | paired-bootstrap 95% CI |
|---|---:|---:|
| C0 | -0.0078125 | [-0.0234375, 0.0078125] |
| C1 | -0.015625 | [-0.03125, -0.001953125] |
| C2 | -0.005859375 | [-0.01953125, 0.0078125] |

All three lower bounds are strictly greater than `-0.05`.

Therefore every preregistered primary acceptance condition passed.

## Descriptive K frontier

Paired deltas versus global:

| checkpoint | K4 | K8 | K16 | K32 |
|---|---:|---:|---:|---:|
| C0 | -0.029296875 | -0.013671875 | -0.0078125 | -0.00390625 |
| C1 | -0.041015625 | -0.009765625 | -0.015625 | -0.001953125 |
| C2 | -0.033203125 | +0.0078125 | -0.005859375 | -0.001953125 |

Corresponding 95% CI lows:

| checkpoint | K4 | K8 | K16 | K32 |
|---|---:|---:|---:|---:|
| C0 | -0.0546875 | -0.033203125 | -0.0234375 | -0.009765625 |
| C1 | -0.06640625 | -0.025390625 | -0.03125 | -0.009765625 |
| C2 | -0.060546875 | -0.01171875 | -0.01953125 | -0.009765625 |

The preregistered descriptive analysis independently returned:

`smallest_noninferior_k = 8`

K4 does not meet the 5pp non-inferiority rule on all checkpoints; K8 does.

## Score-visibility cost

Mean Stage-B score observations/world:

| checkpoint | global | K4 | K8 | K16 | K32 | K16 hash |
|---|---:|---:|---:|---:|---:|---:|
| C0 | 5068.23046875 | 384 | 768 | 1536 | 3072 | 0 |
| C1 | 5091.30078125 | 384 | 768 | 1536 | 3072 | 0 |
| C2 | 5065.6015625 | 384 | 768 | 1536 | 3072 | 0 |

Thus the primary K16 scheduler used about 70% fewer score observations than global ranking while satisfying the frozen non-inferiority rule. The descriptive K8 frontier used about 85% fewer score observations and was also non-inferior on all three checkpoints under the same 5pp criterion.

## Provenance

- confirmation protocol/pre-data Git head: `d6541dc1e2f9c15c4c30408b2616ec04f10affe9`
- result SHA-256: `e49ad074080673b93bf8c22347a9ea34b16ccca4a7e6d12d934df844cfe7ef96`
- independent audit SHA-256: `282f45dc04c6dd8977796647d3f3094f20c0f931a85867dd35086b8bf5c76f63`
- recursive manifest SHA-256: `4e2f110be21a12c0a0d3f3e1526335f7fc35b524459f9dddb66876762073cc27`

Checkpoint identities remained exactly frozen:

- C0 checkpoint SHA-256 `e63748a1182472d03c540f9123c3442ce44e130519e2176681034568826db590`, fingerprint `e4f9990f08e85393a155637cfa50b5831d779770447ec3716fc9e67946992afc`;
- C1 checkpoint SHA-256 `8ce384627f5543fa4fe78498f9990f8214cf2a4afed0c9b734e86761ad13c989`, fingerprint `2a57984f755ee1404fd828c2df36a5efff8f13a541b7d7c93a891014fbf4897c`;
- C2 checkpoint SHA-256 `103826bbd9451b965eced5134d1674cb8e893f5d3b378cb828312ffdb6fc9a37`, fingerprint `8afaf956f200f41ea914eafdd1b5f151dd303cae1552165c74e12bb8c945af02`.

## Supported conclusion

Gate-5 v0 confirms the narrow claim that, in this controlled fixed-population hypothesis-search regime, learned adaptive work allocation can survive a bounded candidate-score visibility channel. K16 strongly outperforms a matched answer-blind K16 routing control and retains near-global exact-solution capability under the frozen five-percentage-point non-inferiority margin across all three independent frozen checkpoints.

The descriptive frontier additionally shows that K8 is the smallest preregistered tested K that satisfies that same non-inferiority rule on all three confirmation checkpoints.

## Claims boundary

This result does **not** establish physical decentralization, arbitrary communication graphs, asynchronous distributed execution, universal sufficiency of K8/K16, per-FLOP/per-joule superiority, AGI, arbitrary-task generalization, or scaling to 1K/10K/100K runtime workers.

Gate-5 v0 is closed. No alternate confirmation namespace, alternate K rescue, margin change, checkpoint selection or rerun is permitted.