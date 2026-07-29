# Gate-3 numerical beam-order rule v0

## Status

**FROZEN BEFORE FIRST ADMITTED DEVELOPMENT RESULT**

This rule was added during pre-result qualification after the reference single-world and batched eager-GRU schedules produced a different beam path from numerically tiny FP32 score differences on one random-weight test world. No Gate-3 training or development result had been admitted or inspected.

## Problem

Gate-3 beam pruning is a discontinuous decision. Two mathematically matched eager PyTorch executions can accumulate slightly different FP32 values when the same recurrent computation is organized with different batch shapes. If two neural candidate scores are nearly equal, that irrelevant numerical drift can change which candidate survives even though the learned function and scientific work/information budget are unchanged.

The scientific experiment should measure hypothesis-population effects, not accidental BLAS/cuDNN reduction order.

## Frozen decision rule

Raw neural scores remain FP32 and are preserved descriptively.

For **beam ordering only**, every scalar neural score is mapped to the integer rank:

```text
q(score) = round(score / 0.001)
```

Candidates are ordered by:

1. descending `q(score)`;
2. the already-frozen deterministic SHA-based tie-break derived only from world seed, phase index and candidate-path identity.

Thus scores that differ by less than the frozen milliscale numerical resolution can become deterministic ties rather than causing schedule-specific branch divergence.

## Scientific scale

The quantization step is `1e-3` score units.

For context, the frozen supervised target separates one noisy-hint match from one mismatch by approximately:

```text
(log(0.70) - log(0.30)) / D
```

which at the hardest `D=8` tier is about `0.106` score units. A delayed-reveal mismatch contributes `16 / 8 = 2.0` target-score units at H8.

Therefore the numerical decision quantum is roughly two orders of magnitude below the smallest intended one-bit evidence separation in the frozen target and three orders below the H8 reveal penalty.

These scale comparisons explain the pre-result choice; they are not post-hoc tuning to an observed capability result.

## Invariants

- The rule is identical for every width and control mode.
- It changes no learned parameter.
- It changes no world observation.
- It changes no recurrent-update count.
- It changes no state-bank size.
- It provides no answer information.
- It is applied in both the single-world reference runtime and batched evaluator.
- Raw scores are not rounded inside the neural recurrence or training loss.
- Changing `0.001` after the first admitted Gate-3 development result creates a new protocol version.

## Qualification requirement

Before Gate-3 development seed 0 is admitted, qualification must show that the batched evaluator and single-world reference runtime agree on candidate decisions and outputs on the frozen regression corpus under this rule, while preserving exact work/information accounting.
