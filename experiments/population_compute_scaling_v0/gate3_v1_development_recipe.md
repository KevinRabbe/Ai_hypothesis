# Gate-3 v1 — Development recipe

## Status

**FROZEN BEFORE FIRST GATE-3 v1 DEVELOPMENT RESULT**

This record freezes the first Gate-3 v1 learned scorer, training recipe and development evaluation surface before any v1 development evidence is inspected.

## Shared scorer

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

The 19 inputs contain only:

- child-depth one-hot over the maximum depth 10;
- world-depth one-hot for S6/S8/S10;
- noisy-hint token `0/1/sink`;
- branch-action token `0/1/sink`.

There is no reserve-capacity feature, slot identity, hidden-answer input or population attention.

## Training target

Training uses candidate prefix trajectories sampled independently from the hidden answer.

For each prefix bit:

```text
+log(0.70 / 0.30)   when candidate action matches the observed noisy hint
-log(0.70 / 0.30)   when candidate action contradicts the observed noisy hint
```

The target after prefix position `d` is the cumulative signed evidence divided by the world's full hidden depth `D`.

This produces a bounded, depth-aware priority target that rewards a recurrent state for accumulating evidence along a candidate trajectory without exposing the hidden answer to the runtime scorer.

Frozen loss:

`SmoothL1Loss`

Loss is computed after every candidate-prefix step and averaged across the trajectory.

## First development training

```text
training seed:       0
optimizer:           AdamW
steps:               1,200
batch size:          256
learning rate:       3e-4
weight decay:        1e-4
gradient clip norm:  1.0
```

Depth cycles deterministically:

`S6 -> S8 -> S10 -> repeat`

Every candidate child in training receives exactly eight recurrent updates, matching the frozen runtime primitive.

Training world seeds are deterministic functions of training seed, optimizer step, sample index and depth and remain strictly below `2^30`.

Controls and reserve capacities are never used for training.

## Development evaluation

```text
world-domain start:       2^30
worlds per depth:         256
batched worlds/call:      64
paired bootstrap samples: 2,000
confirmation opened:      false
```

The same 256 world seeds at a depth are reused across every reserve capacity and control mode.

Full development matrix:

```text
S6:  3 capacities x 3 modes =  9
S8:  4 capacities x 3 modes = 12
S10: 5 capacities x 3 modes = 15
Total                        36
```

Primary outcome: exact search coverage — whether the hidden terminal path occurs anywhere in the answer-blind runtime's generated-terminal transcript.

Each condition must record at minimum:

- paired world seeds;
- exact-coverage vector by world;
- coverage rate;
- generated terminal count by world;
- unique generated terminal count by world;
- productive search rounds by world;
- sink rounds by world;
- productive-work fraction by world;
- exact total learned updates/world;
- learned parameter count;
- checkpoint fingerprint;
- reserve capacity and control mode.

## Paired statistics

Use deterministic 2,000-sample paired bootstrap intervals over exact-coverage differences.

Record:

- stable capacity versus L1;
- stable adjacent-capacity comparisons;
- stable versus collapsed at matched nominal L;
- stable versus reshuffled at matched L.

The five preregistered primary comparisons remain:

1. S8 stable L64 > L1;
2. S10 stable L256 > L1;
3. S10 stable L256 > L64;
4. S10 stable L256 > collapsed L256;
5. S10 stable L256 > reshuffled L256.

## First-run boundary

Only training seed `0` is admitted initially.

Its result must be labeled:

`DEVELOPMENT_ONLY_NOT_ASSIGNED`

No Gate-3 v1 confirmation world may be opened by the development runner.

After seed 0:

- a strongly negative pattern may stop v1;
- an ambiguous or positive pattern requires a robustness rule frozen before any additional training seed is run;
- even a clean development pattern cannot assign a positive Gate-3 verdict.

## Execution boundary

The admitted capability baseline is eager PyTorch FP32.

No `torch.compile`, CUDA graphs, custom fusion or mixed precision may be enabled in this development result. Compiler/runtime optimization remains a separate experimental variable.
