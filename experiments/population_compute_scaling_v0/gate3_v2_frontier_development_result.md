# Gate-3 v2 — ambiguity-frontier development result

## Status

**VALID DEVELOPMENT RESULT — `V2_F0_NO_BEYOND_L64_EXTENSION`**

Measured scientific head: `b760d8b417faf05dabce26da70fdcdca857e1df1`.

Training performed: **false**. The exact three frozen Gate-3 v1 checkpoints were reused and SHA-256 verified. Confirmation remained closed.

## Frozen question

Gate-3 v2 asked whether useful dormant persistent-hypothesis capacity extends beyond the Gate-3 v1 `L64` plateau when public hint evidence is made more ambiguous, while scorer, learned parameters, depth, active neural width, recurrent refinement per evaluated child, search rounds and total learned work remain fixed.

The preregistered ambiguity tiers were:

- `A60`: hint reliability `0.60`;
- `A55`: hint reliability `0.55`.

Stable capacities were `L1 / L16 / L64 / L256`, with collapsed and reshuffled controls at `L256`.

## Measured stable coverage

### Checkpoint 0

```text
A60: L1 0.0117 -> L16 0.3438 -> L64 0.3984 -> L256 0.3984
A55: L1 0.0078 -> L16 0.2812 -> L64 0.3438 -> L256 0.3438
```

### Checkpoint 1

```text
A60: L1 0.0117 -> L16 0.3477 -> L64 0.4102 -> L256 0.4102
A55: L1 0.0078 -> L16 0.2891 -> L64 0.3398 -> L256 0.3398
```

### Checkpoint 2

```text
A60: L1 0.0117 -> L16 0.3477 -> L64 0.4023 -> L256 0.4023
A55: L1 0.0078 -> L16 0.2773 -> L64 0.3359 -> L256 0.3359
```

## Primary frontier result

The preregistered central comparison was `stable L256 - stable L64`.

It is exactly zero for every checkpoint at both ambiguity tiers:

```text
C0 A60: 0.0, CI [0.0, 0.0]
C0 A55: 0.0, CI [0.0, 0.0]
C1 A60: 0.0, CI [0.0, 0.0]
C1 A55: 0.0, CI [0.0, 0.0]
C2 A60: 0.0, CI [0.0, 0.0]
C2 A55: 0.0, CI [0.0, 0.0]
```

Therefore the frozen classifier assigns:

`V2_F0_NO_BEYOND_L64_EXTENSION`

## Lower-frontier effect remains positive

`stable L64 - stable L16` remains positive with paired-bootstrap CI low above zero in all six checkpoint/tier cells:

```text
C0 A60: +0.0546875, CI [0.0234375, 0.0859375]
C0 A55: +0.0625000, CI [0.0312500, 0.09765625]
C1 A60: +0.0625000, CI [0.02734375, 0.1015625]
C1 A55: +0.05078125, CI [0.01171875, 0.08984375]
C2 A60: +0.0546875, CI [0.01953125, 0.09375]
C2 A55: +0.05859375, CI [0.01953125, 0.09765625]
```

Thus ambiguity does not erase the latent-population effect; it preserves a useful capacity increase through `L64` but does not extend usefulness to `L256`.

## Controls remain separated

At `L256`, stable reserve remains strongly above both frozen controls for every checkpoint and tier.

Stable-minus-collapsed deltas range from `+0.328125` to `+0.3984375`, with every paired-bootstrap CI low > 0.

Stable-minus-reshuffled deltas range from `+0.234375` to `+0.296875`, with every paired-bootstrap CI low > 0.

Therefore the Gate-3 v1 diversity/continuity mechanism survives the ambiguity increase.

## Capacity-pressure interpretation

The answer-blind pressure telemetry explains the plateau directly:

- `L1` reached nominal capacity in every world;
- `L16` reached nominal capacity in every world;
- `L64` reached nominal capacity in **0%** of worlds at both tiers for all three checkpoints;
- `L256` likewise never reached nominal capacity.

The preregistered ambiguity increase therefore did not generate enough simultaneously live evaluated hypotheses for `L64` itself to become the binding resource. Under this search topology and fixed 256-round budget, `L256` cannot improve capability because the runtime never needs the extra slots.

This is explanatory secondary evidence only; the primary F0 assignment comes from the paired exact-coverage vectors.

## Scientific interpretation

Gate-3 v2 supports the following narrow result:

> Lowering public-hint reliability from the v1 regime to preregistered `0.60` and `0.55` does not extend useful dormant hypothesis capacity beyond `L64` under the frozen depth-10 binary best-first search topology and 256-round learned-work budget.

At the same time, the robust population mechanism remains visible: `L64 > L16`, stable > collapsed, and stable > reshuffled across all three frozen checkpoints.

The failure mode is therefore not loss of the population mechanism. It is lack of **frontier pressure**: the live search reserve never grows enough for `L64` to bind.

## Boundaries

This result does not show that capacities above 64 are generally useless. It shows only that the preregistered ambiguity manipulation was insufficient to make capacities above 64 useful under this particular binary search topology, depth and search budget.

Per the frozen stop rule:

- no new v2 ambiguity tiers may be added;
- no new v2 capacities may be added;
- checkpoints may not be retrained;
- v2 search rounds or controls may not be changed;
- v2 confirmation remains closed.

A later experiment that changes topology or the way cheap latent possibilities are generated must be separately versioned and frozen before data.

## Provenance

- result SHA-256: `f9878f64c879adda729d02a01427cb5623bd9dc61047be45d78f940a851ad353`
- independent audit SHA-256: `347534bf1706528913a8cd938f33ea22a46c66cddaeadd8ccfa7d0792068c38bc`
- recursive manifest SHA-256: `6a1c71264c88a40a14f07a392a9065deb55c27c98043fbad4cb4639fcdffc8b3`
- independent audit: `artifact_valid=true`, `errors=[]`
- development outcome: `V2_F0_NO_BEYOND_L64_EXTENSION`
- confirmation opened: `false`
