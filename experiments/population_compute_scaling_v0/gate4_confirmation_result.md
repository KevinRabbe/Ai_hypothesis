# Gate-4 v0 — adaptive activation confirmation result

## Status

**FINAL CONFIRMATION EVIDENCE — `GATE4_CONFIRMED_ADAPTIVE_ACTIVATION_BENEFIT`.**

This record preserves the first admitted Gate-4 v0 confirmation run executed under the frozen confirmation protocol.

- measured Git head: `ffaa336ba5a71273da1d2739278523666d43353f`
- training performed: **no**
- confirmation opened: **yes**
- confirmation worlds: **512/checkpoint**
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

## Confirmation condition coverage

| checkpoint | adaptive_score | static_generation | adaptive_hash |
|---|---:|---:|---:|
| C0 | 0.8671875 | 0.673828125 | 0.478515625 |
| C1 | 0.865234375 | 0.673828125 | 0.478515625 |
| C2 | 0.865234375 | 0.66796875 | 0.478515625 |

All conditions used exactly 159 productive parent slots/world and zero sink slots in the admitted confirmation run, preserving the frozen 2,544 learned recurrent updates/world.

## Preregistered confirmation effects

### Primary: `adaptive_score - static_generation`

| checkpoint | delta | paired-bootstrap 95% CI |
|---|---:|---:|
| C0 | **+0.193359375** | **[+0.15625, +0.23046875]** |
| C1 | **+0.19140625** | **[+0.154296875, +0.23046875]** |
| C2 | **+0.197265625** | **[+0.16015625, +0.232421875]** |

All three CI lows are strictly above zero.

### Learned-routing confirmation control: `adaptive_score - adaptive_hash`

| checkpoint | delta | paired-bootstrap 95% CI |
|---|---:|---:|
| C0 | **+0.388671875** | **[+0.3359375, +0.443359375]** |
| C1 | **+0.38671875** | **[+0.333984375, +0.439453125]** |
| C2 | **+0.38671875** | **[+0.330078125, +0.439453125]** |

All three CI lows are strictly above zero.

### Secondary descriptive comparison: `static_generation - adaptive_hash`

| checkpoint | delta | paired-bootstrap 95% CI |
|---|---:|---:|
| C0 | +0.1953125 | [+0.1328125, +0.25390625] |
| C1 | +0.1953125 | [+0.13671875, +0.255859375] |
| C2 | +0.189453125 | [+0.130859375, +0.25] |

## Frozen confirmation verdict

The independent auditor returned:

```text
artifact_valid = true
errors = []
confirmation_outcome = GATE4_CONFIRMED_ADAPTIVE_ACTIVATION_BENEFIT
scientific_status = FINAL_GATE4_CONFIRMATION_EVIDENCE
```

This satisfies every frozen Gate-4 v0 confirmation requirement:

1. independent artifact audit valid with no errors;
2. exact checkpoint identities/fingerprints/parameter counts preserved;
3. information, L256 population and 159-slot / 2,544-update work invariants preserved;
4. `adaptive_score - static_generation` CI low > 0 on C0/C1/C2;
5. `adaptive_score - adaptive_hash` CI low > 0 on C0/C1/C2.

## Supported narrow conclusion

Within this controlled fixed-population hypothesis-search regime, a frozen learned hypothesis score **can route a fixed active neural-work budget more effectively** than both:

- a matched static generation-synchronous breadth schedule; and
- the same dynamic one-parent-at-a-time queue mechanics using answer-blind hash priority.

Together with confirmed Gate-3 v3, the evidence now supports two separable controlled mechanisms:

1. **persistent population capacity matters** when a smaller reserve is genuinely capacity-binding;
2. **learned dynamic activation matters** when deciding which live hypotheses receive scarce active neural processing.

## Claims boundary

This result does not establish AGI, arbitrary-task generalization, globally optimal scheduling, superiority over every serial/replay/search algorithm, per-FLOP/per-joule superiority, or that learned adaptive activation helps on every workload.

## Stop rule

Gate-4 v0 is closed. Do not alter checkpoints, L256, 159-slot / 2,544-update work, scheduler semantics, depth/reliability, or confirmation namespace to seek a different outcome.

## Provenance

Local admitted confirmation artifact hashes reported by the runner:

- result SHA-256: `cda2ba5741d64315fbec338981a8064eacbeb2aeb3849ed5436c26ac2fbabf8f`
- independent audit SHA-256: `b322c54ace0b5673e0476dd70fc15c07869e677731346ae725e4ca3bfdaf79be`
- recursive manifest SHA-256: `6caa2580de3206bd183232b895adc53b24c1fc6e891b3cc7666138d90d482c9b`

Local evidence root reported by the runner:

`F:\gate4_adaptive_activation_confirmation_v0`
