# Gate-4 v0 — adaptive activation development result

## Status

**DEVELOPMENT RESULT RECORDED — `G4_A2_ROBUST_ADAPTIVE_ACTIVATION_BENEFIT`.**

This record preserves the first admitted Gate-4 v0 development run exactly as executed under the frozen pre-result protocol.

- measured Git head: `f28012b5f252ed5f0257b4e87e44e65015719703`
- training performed: **no**
- confirmation opened: **no**
- development worlds: **256/checkpoint**
- checkpoints: exact three frozen Gate-3 v1 checkpoints
- reserve capacity: **L256 in every condition**
- scheduled parent slots/world: **159**
- active child lanes/productive slot: **2**
- recurrent updates/child: **8**
- learned recurrent updates/world: **2,544**
- independent artifact audit: **valid, errors=[]**

## Frozen checkpoint identities

| checkpoint | SHA-256 | parameter fingerprint | learned parameters |
|---|---|---|---:|
| C0 | `e63748a1182472d03c540f9123c3442ce44e130519e2176681034568826db590` | `e4f9990f08e85393a155637cfa50b5831d779770447ec3716fc9e67946992afc` | 19,649 |
| C1 | `8ce384627f5543fa4fe78498f9990f8214cf2a4afed0c9b734e86761ad13c989` | `2a57984f755ee1404fd828c2df36a5efff8f13a541b7d7c93a891014fbf4897c` | 19,649 |
| C2 | `103826bbd9451b965eced5134d1674cb8e893f5d3b378cb828312ffdb6fc9a37` | `8afaf956f200f41ea914eafdd1b5f151dd303cae1552165c74e12bb8c945af02` | 19,649 |

## Raw condition coverage

| checkpoint | adaptive_score | static_generation | adaptive_hash |
|---|---:|---:|---:|
| C0 | 0.84375 | 0.62109375 | 0.484375 |
| C1 | 0.8515625 | 0.58984375 | 0.484375 |
| C2 | 0.85546875 | 0.62109375 | 0.484375 |

All conditions used the same 159 productive parent slots/world and zero sink slots in the admitted run, preserving the frozen 2,544 learned recurrent updates/world.

Representative scheduler telemetry:

- `adaptive_score`: mean live reserve approximately 8.5 / 8.9 / 9.9 candidates for C0/C1/C2; mean terminal count approximately 149.6 / 149.4 / 148.8;
- `static_generation`: mean live reserve 74.4; terminal count 64.0;
- `adaptive_hash`: mean live reserve 34.1; terminal count 122.8.

These telemetry differences are consequences of the scheduler policies, not differences in learned-work budget.

## Preregistered paired effects

### Primary: `adaptive_score - static_generation`

| checkpoint | delta | paired-bootstrap 95% CI |
|---|---:|---:|
| C0 | **+0.22265625** | **[+0.171875, +0.28125]** |
| C1 | **+0.26171875** | **[+0.203125, +0.3203125]** |
| C2 | **+0.234375** | **[+0.1796875, +0.29296875]** |

All three CI lows are strictly above zero.

### Learned-routing control: `adaptive_score - adaptive_hash`

| checkpoint | delta | paired-bootstrap 95% CI |
|---|---:|---:|
| C0 | **+0.359375** | **[+0.28515625, +0.43359375]** |
| C1 | **+0.3671875** | **[+0.296875, +0.4453125]** |
| C2 | **+0.37109375** | **[+0.30078125, +0.44921875]** |

All three CI lows are strictly above zero.

### Descriptive scheduler comparison: `static_generation - adaptive_hash`

| checkpoint | delta | paired-bootstrap 95% CI |
|---|---:|---:|
| C0 | +0.13671875 | [+0.0546875, +0.22265625] |
| C1 | +0.10546875 | [+0.0234375, +0.1875] |
| C2 | +0.13671875 | [+0.05859375, +0.21875] |

## Frozen outcome classification

The preregistered mutually-exclusive precedence was:

`A4 → A2 → A1 → A3 → A0 → mixed`.

The independent auditor returned:

```text
artifact_valid = true
errors = []
directional_outcome = G4_A2_ROBUST_ADAPTIVE_ACTIVATION_BENEFIT
scientific_status = DEVELOPMENT_ONLY_NO_GATE_VERDICT
```

This mechanically satisfies Gate-4 development Outcome **A2** because every checkpoint has paired-bootstrap CI low `> 0` for both:

1. `adaptive_score - static_generation`;
2. `adaptive_score - adaptive_hash`.

## What the development result supports

Within this controlled fixed-population hypothesis-search regime, the frozen learned neural score can be used to allocate a fixed active neural-work budget more effectively than both:

- a matched static generation-synchronous breadth schedule; and
- the same dynamic one-parent-at-a-time queue mechanics using answer-blind hash priority instead of the learned score.

The result therefore supports the narrow development-stage mechanism that **where active neural work is spent matters, and the learned hypothesis score contains useful routing information for that allocation**.

The causal controls held fixed:

- learned parameters;
- public information;
- hidden-answer separation;
- latent population capacity L256;
- two active child lanes;
- eight recurrent updates/child;
- 159 scheduled parent slots;
- 2,544 total learned recurrent updates/world.

Only activation scheduling differed.

## What this does not establish

This development result does not establish:

- a final Gate-4 verdict before independent confirmation;
- AGI or general intelligence;
- arbitrary-task generalization;
- globally optimal scheduling;
- per-FLOP or per-joule superiority;
- superiority over every serial, replay, tree-search, or hand-engineered scheduler;
- that score-driven activation will help in every workload.

## Confirmation boundary

The frozen protocol states that only `G4_A2_ROBUST_ADAPTIVE_ACTIVATION_BENEFIT` may permit a separately versioned confirmation protocol.

That condition is now met.

**No Gate-4 confirmation world was generated or inspected by this development run.** Confirmation remains closed until a separate confirmation protocol is frozen and qualified.

## Provenance

Local admitted artifact hashes reported by the runner:

- result SHA-256: `9d422d176d4840fc2608849aa9209d906c896f4cba701414211b628461c4616b`
- independent audit SHA-256: `90f3d201cdeebc075caa89d5f0235c1d86ad654ad8000405f663996cf96f659a`
- recursive manifest SHA-256: `ef20892e8a179598c4c0894a4391a2d013a70879849daaf05718ec76b508b336`

Local evidence root reported by the runner:

`F:\gate4_adaptive_activation_development_v0`
