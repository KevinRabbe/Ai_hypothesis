# Gate-1 v1 first target-GPU result

Status: **AUDITED POSITIVE EXECUTION/RESOURCE RESULT; post-audit packaging repair pending**

This document records the first admitted Gate-1 v1 timing result. It is intentionally separate from the earlier Gate-1 v0 CUDA correctness failure.

## Frozen identity

The measured run used:

- measurement Git head: `18b201e22ca0a33feb4644c8ed8a09375e8e23ea`;
- GPU: NVIDIA GeForce RTX 4060 Ti 16 GB;
- PyTorch: `2.9.1+cu130`;
- CUDA runtime reported by PyTorch: `13.0`;
- training seed: `1`;
- learned parameters: `26,669`;
- parameter fingerprint: `c227ade9006e47bec17a2a3d5aedf6ac95a6a94607b96b9f52ab759905536c12`;
- checkpoint SHA-256: `0b7c1f2a14fe9d2987819ed53fc0b55c04f3bb00bce356c1023778830a08ad26`;
- execution mode: eager;
- matrix: relay-2 / relay-4 / relay-8 × populations `1 / 4 / 16 / 64 / 256` × batches `1 / 64`;
- 20 warmups + 100 measured iterations per schedule/cell;
- world seed `0`.

The complete FP32+FP64 Gate-1 v1 correctness preflight passed before timing, and the FP64 reference model was offloaded before resource measurement.

## Independent audit

The independent v1 audit completed before the later packaging failure and reported:

```text
protocol_valid = true
reasons = []
expected_condition_count = 30
observed_condition_count = 30
observed_measurement_rotations = [0, 1, 2]
```

Precision diagnostics inside the audited result remained consistent with the pre-timing v1 rule:

- worst FP32 schedule-pair logits drift: `5.199909210205078e-4`;
- worst FP32 schedule-pair shared drift: `9.149312973022461e-5`;
- worst FP32↔FP64 logits drift: `1.3133920937238308e-3`;
- worst FP32↔FP64 shared drift: `2.5444045154221495e-4`;
- worst FP64 schedule-pair logits drift: `6.203926261605375e-13`;
- worst FP64 schedule-pair shared drift: `1.2045919817182948e-13`.

All required decoded-agreement and FP64 corroboration checks passed.

## Complete-matrix result

Speedup is `serial median CUDA-event latency / parallel median CUDA-event latency`; values above 1 mean parallel execution was faster.

### Batch 1

Across all 15 batch-1 cells:

- parallel faster than compute-matched cached serial: **15 / 15**;
- parallel faster than low-memory serial: **15 / 15**;
- cached-serial / parallel geometric-mean speedup: **16.6186×**;
- cached-serial / parallel range: **1.6684× – 228.2805×**;
- low-memory-serial / parallel geometric-mean speedup: **20.8278×**;
- low-memory-serial / parallel range: **1.7320× – 296.2945×**.

### Batch 64

Across all 15 batch-64 cells:

- parallel faster than compute-matched cached serial: **15 / 15**;
- parallel faster than low-memory serial: **15 / 15**;
- cached-serial / parallel geometric-mean speedup: **15.4181×**;
- cached-serial / parallel range: **1.9436× – 147.1069×**;
- low-memory-serial / parallel geometric-mean speedup: **19.0010×**;
- low-memory-serial / parallel range: **2.0290× – 193.7522×**.

Across the full 30-cell matrix, the descriptive geometric mean is approximately **16.01× vs cached serial** and **19.89× vs low-memory serial**.

No pass/fail speed threshold was introduced after seeing these numbers. The result is interpreted from the complete frozen matrix.

## Width scaling

The strongest systems observation is not only pairwise speedup: parallel latency remained nearly flat while the serialized controls scaled strongly with population width.

For batch 1, width `1 → 256`:

| Difficulty | Parallel median | Parallel growth | Cached serial median | Cached growth |
| --- | ---: | ---: | ---: | ---: |
| relay-2 | `0.6196 → 0.6466 ms` | `1.04×` | `1.0337 → 89.3174 ms` | `86.4×` |
| relay-4 | `0.9246 → 0.9241 ms` | `1.00×` | `2.0884 → 207.0344 ms` | `99.1×` |
| relay-8 | `1.3440 → 1.8985 ms` | `1.41×` | `3.6070 → 433.3896 ms` | `120.2×` |

For batch 64, width `1 → 256`:

| Difficulty | Parallel median | Parallel growth | Cached serial median | Cached growth |
| --- | ---: | ---: | ---: | ---: |
| relay-2 | `0.6001 → 1.0424 ms` | `1.74×` | `1.1663 → 120.3537 ms` | `103.2×` |
| relay-4 | `0.9467 → 1.8432 ms` | `1.95×` | `2.1345 → 238.7912 ms` | `111.9×` |
| relay-8 | `1.6297 → 3.3556 ms` | `2.06×` | `4.4083 → 493.6391 ms` | `112.0×` |

This is the clearest evidence in Gate 1 that the target GPU can absorb a large increase in simultaneous runtime population at strongly sublinear latency cost for this relay computation.

## High-width examples

At relay-8, population 256, batch 1:

- parallel: `1.8985 ms` median;
- cached serial: `433.3896 ms` median;
- cached/parallel: **228.2805×**;
- low-memory serial: `559.4839 ms` median;
- low-memory/parallel: **294.6985×**.

At relay-8, population 256, batch 64:

- parallel: `3.3556 ms` median;
- cached serial: `493.6391 ms` median;
- cached/parallel: **147.1069×**;
- low-memory serial: `650.1643 ms` median;
- low-memory/parallel: **193.7522×**;
- parallel throughput: approximately `18,688 samples/s`;
- cached-serial throughput: approximately `128 samples/s`;
- low-memory-serial throughput: approximately `97 samples/s`.

## Memory trade-off

The speed advantage is not free.

At population 256:

- batch 1 peak allocated delta:
  - parallel: approximately `0.95 MiB`;
  - cached serial: `0.125 MiB`;
  - low-memory serial: approximately `0.009 MiB`;
- batch 64 peak allocated delta:
  - parallel: approximately `65.0 MiB`;
  - cached serial: `8.0 MiB`;
  - low-memory serial: approximately `0.281 MiB`.

For this 26.7K-parameter relay on a 16 GB GPU, that absolute cost is small. The relative O(N) residency difference is real and must be re-evaluated at larger state/model sizes.

## Interpretation

Gate-1 v1 is **positive for the practical eager-CUDA execution question**.

The result supports:

> On the RTX 4060 Ti target, simultaneous/batched execution of the frozen relay population gives a robust practical latency and throughput advantage over both mathematically equivalent serial schedules, including the compute-matched cached-serial control, across the entire preregistered 30-cell matrix.

The evidence is especially strong because:

1. parallel wins every frozen cell, not only selected large widths;
2. the weakest cached-serial comparison is still `1.6684×`;
3. the advantage grows strongly with width;
4. the parallel schedule keeps width-scaling latency close to flat over much of this matrix;
5. both batch-1 latency-oriented and batch-64 throughput-oriented regimes show the same direction;
6. the independent auditor accepted all correctness, work-accounting, CUDA-timing, memory-accounting and schedule-order invariants.

## What this does not establish

This result does **not** establish:

- general intelligence scaling;
- a real-workload advantage;
- a per-FLOP superiority claim;
- that every population architecture should execute simultaneously;
- that compiler/graph optimization cannot narrow the serial gap;
- a distributed/multi-machine result;
- population scaling beyond 256;
- that the observed eager-PyTorch speedup is entirely hardware compute rather than also launch/orchestration amortization.

The serial schedules expose much larger sequential execution span and many more small eager operations. Compiler/graph execution therefore remains a high-value **independent next systems variable**, exactly as originally planned.

## Packaging-only failure after the valid audit

After the benchmark and independent audit had completed successfully, the Windows PowerShell runner failed while constructing `result-manifest.sha256` because a clean `$gitStatus` is an empty array. Piping that empty array to `Set-Content` did not invoke `Set-Content`, so `git-status.txt` was absent when the manifest tried to hash it.

This defect occurred **after** measurement and audit. It did not affect timings, correctness, work accounting, memory accounting, or the independent audit result.

The runner has since been repaired to explicitly create a zero-byte `git-status.txt` for a clean tree, with CI regression coverage. The first result must be finalized without rerunning timing; a dedicated packaging-only finalizer verifies the stored measurement head and audit, rejects scientific-source drift, reconstructs only the missing empty status snapshot, records the repair, and builds the manifest.

The first measured result remains the canonical Gate-1 v1 timing evidence. A later execution would be a replication, not a replacement for this first admitted result.
