# Gate-3 development recipe v0

## Status

**FROZEN BEFORE FIRST ADMITTED DEVELOPMENT RESULT**

This document freezes the exact Gate-3 scorer architecture, training recipe and development evaluation surface before any Gate-3 development result is inspected.

It is subordinate to `gate3_hypothesis_population_protocol_v0.md`; changing any item below after the first admitted development result requires a new development-recipe version.

## Shared learned scorer

One learned parameter set is reused across every runtime hypothesis state and every population width.

Frozen architecture:

```text
input features:              26
learned input projection:    Linear(26 -> 32)
projection activation:       SiLU
recurrent update:            one-layer GRU(32 -> 64)
persistent state width:      64
output normalization:        LayerNorm(64)
score head:                  Linear(64 -> 1)
trainable parameters:        19,873
```

Input features contain only:

- phase one-hot over the maximum 16 Gate-3 phases;
- depth one-hot for H4/H6/H8;
- observation-kind one-hot;
- observed-bit one-hot;
- branch-action `0/1/none` one-hot.

There is:

- no population-width embedding;
- no slot ID;
- no learned per-hypothesis parameter block;
- no attention across states;
- no learned parameter that scales with runtime population.

Repeated recurrent updates use standard eager PyTorch `nn.GRU` sequence execution. This is the baseline neural primitive, not a compiler experiment: `torch.compile`, CUDA graphs, custom fusion and mixed precision are disabled for the admitted v0 development run.

## Training objective

Training teaches the shared scorer to assign a scalar evidence score to one candidate hypothesis trajectory.

It does **not** train a separate policy for each population width.

Candidate trajectories are sampled independently from the world hidden path.

### Before delayed reveals

For each candidate bit already proposed, the target uses the known noisy-hint generative likelihood:

```text
log(0.70) when candidate_bit == noisy_hint
log(0.30) otherwise
```

The accumulated hint score is divided by hidden depth `D`.

### After delayed reveals begin

The full noisy-hint score remains, and every candidate/reveal mismatch adds the frozen penalty:

```text
16.0
```

before division by `D`.

This makes exact delayed evidence dominate noisy-hint preference while keeping a finite supervised regression target.

Frozen loss:

`SmoothL1Loss`

The loss is computed after every phase and averaged over the `2D` phases.

## Recurrence exposure during training

Training cycles the exact 12 stable `(depth,width)` schedules:

```text
H4: W1, W4, W16
H6: W1, W4, W16, W64
H8: W1, W4, W16, W64, W256
```

At each optimizer step:

```text
condition = conditions[step % 12]
```

The candidate trajectory receives the exact recurrent repeat count specified by that condition's frozen Gate-3 work schedule.

This exposes the same shared scorer to both narrow/deep and wide/shallow recurrence patterns without giving it a width feature.

Control modes are never used for training.

## Optimizer recipe

Frozen development training recipe:

```text
training seed:          0 for the first admitted development run
optimizer:              AdamW
steps:                  1,200
batch size:             128
learning rate:          3e-4
weight decay:           1e-4
gradient clip norm:     1.0
model parameters:       19,873
```

Torch CPU/CUDA seeds are set from the training seed before model initialization.

Training-world and candidate-path sampling are domain-separated deterministic functions of:

- training seed;
- optimizer step;
- sample index;
- hidden depth;
- runtime width condition.

Training world seeds remain strictly below `2^30`.

## Development evaluation

Frozen development evaluation:

```text
world domain start:     2^30
worlds per depth:       256
evaluation batch size:  64
bootstrap samples:      2,000
confirmation opened:    false
```

The same 256 world seeds for a depth are reused across every width and control, enabling paired per-world comparisons.

Evaluation covers the complete 36-cell matrix:

```text
H4: 3 widths x 3 modes = 9
H6: 4 widths x 3 modes = 12
H8: 5 widths x 3 modes = 15
Total                   = 36
```

Modes:

- stable diverse;
- collapsed diversity;
- reshuffled continuity.

The batched eager evaluator must remain mechanically equivalent to the single-world reference runtime. Qualification compares the two on predictions, per-world solve outcomes, candidate-survival diagnostics and learned-work accounting.

## Recorded condition evidence

Every development condition records at minimum:

- exact solve rate;
- bit accuracy;
- solved-by-world vector;
- bit-accuracy-by-world vector;
- exact world seeds;
- correct-hypothesis survival rate by phase;
- mean unique live hypotheses by phase;
- learned updates per world;
- unique world observations per world;
- learned parameter count;
- checkpoint parameter fingerprint;
- depth, width and control mode.

## Paired comparisons

Paired bootstrap intervals are reconstructed from identical-world exact-solve vectors.

The result contains:

- stable width versus W1 comparisons;
- stable adjacent-width comparisons;
- stable versus collapsed comparisons;
- stable versus reshuffled comparisons.

The five preregistered primary development directions remain those in the Gate-3 protocol:

1. H6 stable W64 > W1;
2. H8 stable W256 > W1;
3. H8 stable W256 > W64;
4. H8 stable W256 > collapsed W256;
5. H8 stable W256 > reshuffled W256.

## Development-only boundary

The first admitted run is training seed `0` only.

Its result must be labeled:

`DEVELOPMENT_ONLY_NOT_ASSIGNED`

No Gate-3 confirmation world may be inspected from this runner.

No development outcome, including a clean five-direction result, is sufficient to assign a positive Gate-3 verdict.

Any robustness-seed rule must be frozen **after** seeing seed 0 but **before** running additional development training seeds, exactly as a separate precommitted robustness step.

## Execution-variable boundary

The admitted capability-development baseline is eager PyTorch.

Compiler/runtime optimization remains independent and may only be tested after the capability protocol has produced evidence under this baseline. Compiler gains cannot be used to redefine or rescue a Gate-3 capability result.
