# Gate-1 v0 CUDA equivalence result

## Status

**Gate-1 v0 is closed as a correctness-gate failure. It produced no admissible performance result.**

The first real target-device run reached the frozen pre-timing correctness check on the local NVIDIA GeForce RTX 4060 Ti under PyTorch 2.9.1+cu130 / CUDA 13.0 and correctly refused to time the matrix because the three mathematically equivalent execution schedules did not satisfy the preregistered FP32 tensor-allclose rule in every frozen condition.

This does **not** negate Gate v0 capability evidence and does **not** establish that parallel population execution is slower or faster. No Gate-1 timing result was admitted.

## Frozen v0 rule that failed

Gate-1 v0 required, for every one of the 30 frozen `(difficulty, workers, batch)` cells:

- FP32 final shared state close at `rtol=2e-5`, `atol=2e-5`;
- FP32 final logits close at the same tolerance;
- exact decoded prediction equality;
- matched recurrent-work accounting;
- matched parallel/cached-serial static projection accounting.

The runner aborted before resource timing when one condition violated the FP32 tensor rule.

## Untimed CUDA equivalence diagnostic

A full untimed diagnostic over the same frozen 30-cell matrix found:

- conditions: `30`;
- FP32 allclose failures: `4`;
- decoded failures: `0`;
- all four allclose failures occurred at `relay-8`, `batch=64`, workers `4 / 16 / 64 / 256`;
- every `batch=1` condition passed, including `relay-8`, workers `256`;
- worst FP32 schedule-pair logits drift: `5.199909210205078e-4`;
- worst FP32 schedule-pair shared-state drift: `9.149312973022461e-5`.

The diagnostic was repeated and reproduced the same values on the same target environment.

## Untimed precision triangulation

The same frozen matrix was then evaluated in FP32 and FP64, still without measuring latency, throughput, or memory.

Global decoded agreement:

- FP32 schedule-pair decoded equality: `true`;
- FP64 schedule-pair decoded equality: `true`;
- FP32↔FP64 per-schedule decoded equality: `true`.

Worst schedule-pair drift:

| Precision | Logits | Shared state |
| --- | ---: | ---: |
| FP32 | `5.199909210205078e-4` | `9.149312973022461e-5` |
| FP64 | `6.203926261605375e-13` | `1.2045919817182948e-13` |

Worst FP32↔FP64 same-schedule drift:

- logits: `1.3133920937238308e-3`;
- shared state: `2.5444045154221495e-4`.

All decoded outputs remained identical despite those FP32 numerical differences.

## Interpretation

The precision result is strong evidence that the three schedules implement the same mathematical relay computation and that the v0 failure is caused by finite-precision execution-order / kernel-shape effects on CUDA rather than a schedule-semantic disagreement:

- different vectorization shapes legitimately change FP32 GEMM/reduction accumulation order;
- recurrent relay updates can amplify those small numerical differences across eight hops;
- when the same schedules are evaluated in FP64, pairwise drift collapses by roughly nine orders of magnitude to near machine precision;
- no decoded decision changes in FP32, FP64, or across precision.

The v0 `2e-5` rule is therefore retained as a historical failed preregistration. It is not enlarged post hoc.

## Consequence

Gate-1 v1 is a separately versioned protocol frozen **after** these untimed correctness diagnostics and **before** any admitted target-device performance measurement.

V1 uses complete-matrix FP64 shadow corroboration plus exact decoded agreement as its correctness gate, while continuing to record FP32 tensor drift descriptively. The FP64 reference model is removed from CUDA before timing so it cannot contaminate memory measurements.
