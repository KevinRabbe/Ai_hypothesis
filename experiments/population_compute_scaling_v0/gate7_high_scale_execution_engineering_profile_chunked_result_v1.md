# Gate-7 chunked high-scale execution engineering profile v1 — result

## Status

**VALID ENGINEERING EVIDENCE — `COMPLETE` THROUGH N131072 WITH PHYSICAL/LOGICAL WORLD BATCH 64 PRESERVED.**

Measured engineering head:

`a43d0f1caf2bd9c40f0179461083eb433cf1e6e6`

This was an engineering-only CUDA profile using:

- one deterministic randomly initialized 19,649-parameter scale-neutral scorer;
- synthetic public-only hints and public seeds;
- no trained checkpoint loading;
- no hidden path or scientific world;
- physical/logical world batch 64;
- fixed maximum 1,048,576 recurrent rows per complete-frontier chunk;
- 128 terminal Stage-B activations;
- compiler, CUDA graphs and mixed precision disabled.

The profile completed the full N1024 → N131072 ladder and every prepared selector/executor condition. This is **not Gate-7 scientific evidence** and does not estimate `K_required(N)`.

## Completed population tiers

| N | Frontier seconds | Frontier peak GiB | Global score s | Global hash s | Score K16 s | Hash K16 s | Score K512 s | Hash K512 s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1,024 | 0.277 | 0.23 | 0.153 | 0.094 | 0.173 | 0.125 | 0.143 | 0.126 |
| 2,048 | 0.359 | 0.44 | 0.106 | 0.090 | 0.146 | 0.126 | 0.141 | 0.125 |
| 4,096 | 0.701 | 0.87 | 0.108 | 0.091 | 0.146 | 0.127 | 0.142 | 0.127 |
| 8,192 | 1.374 | 1.73 | 0.106 | 0.091 | 0.146 | 0.122 | 0.143 | 0.124 |
| 16,384 | 2.735 | 3.44 | 0.105 | 0.089 | 0.144 | 0.126 | 0.140 | 0.122 |
| 32,768 | 5.469 | 6.86 | 0.200 | 0.116 | 0.146 | 0.128 | 0.143 | 0.124 |
| 65,536 | 10.721 | 7.62 | 0.534 | 0.344 | 0.146 | 0.124 | 0.143 | 0.124 |
| 131,072 | 21.217 | 9.15 | 1.103 | 0.672 | 0.143 | 0.128 | 0.143 | 0.126 |

All six prepared selector/executor conditions completed at every tier:

- `global_score`;
- `global_hash`;
- `bounded_score_k16`;
- `bounded_hash_k16`;
- `bounded_score_k512`;
- `bounded_hash_k512`.

## v0 → v1 engineering comparison

At N65536, fixed recurrent-row chunking changed immutable-frontier construction from:

```text
v0: 20.572 seconds, 13.70 GiB peak
v1: 10.721 seconds,  7.62 GiB peak
```

That is approximately:

- 47.9% lower frontier wall time;
- 44.4% lower peak allocated memory.

The previously unreachable N131072 tier completed in 21.217 seconds at 9.15 GiB peak. Its terminal conditions peaked between 2.12 GiB and 2.46 GiB, confirming that complete-frontier construction—not selector/terminal execution—was the original resource bottleneck.

## Engineering interpretation

The qualified B64-preserving row partition removed the measured RTX 4060 Ti resource frontier without changing:

- world count or world identity;
- complete frontier geometry or lexicographic order;
- FP32 model arithmetic;
- eight recurrent updates per child;
- learned parameters;
- N/K ladders;
- Stage-B activation count;
- condition semantics;
- scientific classification rules.

The fixed execution contract is:

```text
world batch preserved:                64
maximum recurrent rows per chunk:     1,048,576
N65536 final chunks per action:       2
N131072 final chunks per action:      4
```

The full prepared high-scale execution range is therefore feasible on the measured RTX 4060 Ti under the frozen engineering substrate.

## Provenance

- profile summary SHA-256: `e40823e3e2787151f2a63607aa3d396f18e03428b715b8864af4f549631e2953`
- recursive manifest SHA-256: `8393f9b4f11aa90aa333c3443669306675d1e9cc746e1f1dc3aa5acd1523afe4`
- local output root: `F:\gate7_high_scale_execution_engineering_profile_chunked_v1`

## Scientific boundary

This result establishes only that the frozen B64 high-scale execution substrate is computationally feasible through N131072 on the measured RTX 4060 Ti.

It does not establish:

- any Gate-7 coverage outcome;
- `K_required(N)`;
- learned routing superiority;
- a scientific routing frontier;
- an asymptotic scaling law;
- a maximum useful population.

High-scale Gate-7 science remains closed until a separately qualified admitted runner binds the exact transition checkpoints, fresh high-scale world namespace, frozen sequential K exposure, paired controls, bootstrap intervals and independent audit.