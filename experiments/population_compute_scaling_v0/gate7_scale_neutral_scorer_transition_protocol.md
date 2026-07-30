# Gate-7 scale-neutral scorer transition — pre-science protocol

## Status

**FROZEN AS A PRE-SCIENCE TRANSITION CANDIDATE BEFORE ANY SCALE-NEUTRAL CHECKPOINT IS TRAINED OR ANY GATE-7 CAPABILITY WORLD IS OPENED.**

This transition exists because the exact frozen Gate-3 v1 scorer represents child depth with a ten-position one-hot and world depth with a three-position S6/S8/S10 one-hot. It therefore cannot honestly represent the Gate-7 high-scale population ladder beyond the existing depth-10 domain.

This is a model-representation transition, not a Gate-7 capability result.

## Goal

Create a scorer whose learned parameter count and recurrent machinery remain identical to the Gate-3 v1 scorer while removing the fixed maximum-depth representation.

The transition must preserve the intended causal boundary:

- no population N input;
- no routing K input;
- no reserve-capacity input;
- no slot identity input;
- no hidden-answer input;
- no population attention;
- candidate-specific recurrent state only;
- public noisy hint and branch action only;
- exactly eight recurrent updates/evaluated child.

## Frozen architecture

```text
input features:           19
input projection:         Linear(19 -> 32)
activation:               SiLU
recurrent state:          one-layer GRU(32 -> 64)
normalization:            LayerNorm(64)
priority head:            Linear(64 -> 1)
trainable parameters:     19,649
updates/evaluated child:  8
```

The architecture width is therefore unchanged from Gate-3 v1. Only the semantic encoding of the first thirteen inputs changes.

## Scale-neutral 19-input encoding

The final six inputs remain unchanged:

- noisy-hint token: `0 / 1 / sink` (3 one-hot features);
- branch-action token: `0 / 1 / sink` (3 one-hot features).

The first thirteen inputs are deterministic bounded functions of candidate child depth `d` and public world depth `D`, with `1 <= d <= D`:

1. `d / D`;
2. `(D - d) / D`;
3. `1 / D`;
4. `1 / d`;
5. `1 / (D - d + 1)`;
6. `d / (d + 1)`;
7. `D / (D + 1)`;
8. `(D - d) / (D - d + 1)` with exact zero when `D == d`;
9. `sin(pi * d / D)`;
10. `cos(pi * d / D)`;
11. `sin(2*pi * d / D)`;
12. `cos(2*pi * d / D)`;
13. `2*d/D - 1`.

No feature depends on a frozen maximum depth. All thirteen values remain finite and bounded for every positive integer depth.

Sink inputs use the same positional block for the scheduled child depth and the existing sink tokens for hint/action. They carry no world evidence or branch action.

## Frozen transition training surface

The training target remains exactly the Gate-3 v1 evidence target:

```text
+log(0.70 / 0.30) when candidate action matches the observed noisy hint
-log(0.70 / 0.30) when candidate action contradicts the observed noisy hint
```

The target after prefix position `d` remains cumulative signed evidence divided by full public world depth `D`.

Frozen optimization recipe:

```text
training seeds:      0 / 1 / 2
optimizer:           AdamW
steps/checkpoint:    1,200
batch size:          256
learning rate:       3e-4
weight decay:        1e-4
gradient clip norm:  1.0
loss:                SmoothL1Loss
updates/child:       8
precision:           eager FP32
```

Training depth schedule cycles deterministically over every integer world depth:

`6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18`.

Training candidate paths are sampled independently from the hidden answer exactly as in Gate-3 v1. Training-world and training-candidate seeds must use new `gate7-scale-neutral-transition-*` namespaces and may not reuse any Gate-7 bridge/development/confirmation namespace.

Controls, population N, reserve capacity and routing K are never used during training.

No compiler, CUDA graph, custom fusion or mixed precision is part of the checkpoint definition.

## Mandatory checkpoint qualification

Each of the three trained checkpoints must independently satisfy:

- exact trainable parameter count `19,649`;
- finite parameters;
- finite training losses;
- recorded full checkpoint SHA-256;
- recorded parameter fingerprint;
- architecture/encoding constants exactly matching this protocol.

Checkpoint training by itself does not admit Gate-7 high-scale science.

## Fresh low-scale mechanism bridge

Before any high-scale Gate-7 capability namespace may be generated, the three scale-neutral checkpoints must pass a separately versioned bridge on fresh transition-only depth-10 worlds.

The bridge is not allowed to reuse Gate-6 development worlds or any future Gate-7 development/confirmation world.

Bridge geometry deliberately reuses the already-understood Gate-6 mechanical regime:

- hidden/public world depth: 10;
- common complete Stage-A frontier depth: 8;
- Stage-A width: 256;
- Stage-A learned updates/world: 4,080;
- Stage-B capacities: N128 and N256;
- Stage-B slots: 128;
- Stage-B learned updates/world: 2,048;
- total learned recurrent updates/world: 6,128;
- K16 score and matched K16 hash use identical sampled candidate sets when incoming reserves are identical;
- global score is the full-reserve reference;
- exact coverage remains evaluation-only.

Use exactly 256 fresh bridge worlds, evaluation batch 64, and 2,000 deterministic paired bootstrap samples.

### Bridge acceptance

For each transition checkpoint T0/T1/T2:

1. **learned-routing mechanism at N128:** paired 95% CI low for `K16_score - K16_hash` is strictly `> 0`;
2. **learned-routing mechanism at N256:** paired 95% CI low for `K16_score - K16_hash` is strictly `> 0`;
3. **near-global routing at N128:** paired 95% CI low for `K16_score - global_score` is strictly `> -0.05`;
4. **global low-scale competence:** on the same fresh bridge worlds, transition `global_score` at N256 must be non-inferior by 5 percentage points to the seed-matched original Gate-3 v1 checkpoint's `global_score` at N256: paired 95% CI low for `transition_global - original_global` is strictly `> -0.05`.

All four criteria must pass on all three checkpoint pairs.

N256 `K16_score - global_score` is recorded descriptively only. Gate-6 already showed that fixed K16 becomes checkpoint-sensitive at N256, so forcing a uniformly positive N256 near-global bridge would incorrectly turn this transition into a rescue of Gate-6.

### Bridge failure

Any bridge failure means:

`GATE7_SCALE_NEUTRAL_TRANSITION_NOT_QUALIFIED`

High-scale Gate-7 development remains CLOSED. No post-result change to encoding, training depths, optimizer, K16 bridge criteria or bridge namespace is permitted under this transition version.

### Bridge success

Only if all twelve primary checkpoint criteria pass may the transition be labeled:

`GATE7_SCALE_NEUTRAL_TRANSITION_QUALIFIED`

That label only qualifies the replacement scorer as a mechanism-preserving substrate. It is not evidence for 1K/10K/100K population capability scaling.

## Gate-7 high-scale boundary

Even after transition qualification, the actual Gate-7 `K_required(N)` development protocol must be frozen separately before any high-scale capability world is opened.

The prepared population ladder remains:

`512 -> 1K -> 2K -> 4K -> 8K -> 16K -> 32K -> 64K -> 128K`.

The prepared K ladder remains:

`16 -> 32 -> 64 -> 128 -> 256 -> 512`, excluding `K >= N`.

The transition protocol does not preassign which K must succeed at any high-scale N.