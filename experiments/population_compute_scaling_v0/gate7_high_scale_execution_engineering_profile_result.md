# Gate-7 high-scale execution engineering profile v0 — result

## Status

**VALID ENGINEERING EVIDENCE — `RESOURCE_FRONTIER_REACHED` AT N131072 DURING IMMUTABLE FRONTIER BUILD.**

Measured engineering head:

`534a58d26d19f82bd108563f5328212759b59d9a`

This was an engineering-only CUDA profile using:

- one deterministic randomly initialized 19,649-parameter scale-neutral scorer;
- synthetic public-only hints and public seeds;
- no trained checkpoint loading;
- no hidden path or scientific world;
- physical/logical world batch 64;
- 128 terminal Stage-B activations;
- compiler, CUDA graphs and mixed precision disabled.

The profile completed every condition through N65536 and reached the RTX 4060 Ti resource frontier only while constructing the N131072 immutable Stage-A frontier.

This is **not Gate-7 scientific evidence** and does not estimate `K_required(N)`.

## Completed population tiers

| N | Frontier seconds | Frontier peak GiB | Global score s | Global hash s | Score K16 s | Hash K16 s | Score K512 s | Hash K512 s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1,024 | 0.305 | 0.23 | 0.218 | 0.094 | 0.203 | 0.128 | 0.146 | 0.126 |
| 2,048 | 0.357 | 0.44 | 0.105 | 0.092 | 0.143 | 0.125 | 0.142 | 0.121 |
| 4,096 | 0.704 | 0.87 | 0.107 | 0.089 | 0.141 | 0.127 | 0.141 | 0.121 |
| 8,192 | 1.383 | 1.73 | 0.107 | 0.093 | 0.145 | 0.124 | 0.141 | 0.121 |
| 16,384 | 2.848 | 3.44 | 0.106 | 0.092 | 0.146 | 0.122 | 0.140 | 0.125 |
| 32,768 | 5.693 | 6.86 | 0.203 | 0.118 | 0.144 | 0.124 | 0.142 | 0.125 |
| 65,536 | 20.572 | 13.70 | 0.732 | 0.342 | 0.264 | 0.140 | 0.151 | 0.124 |
| 131,072 | resource stop | frontier build OOM | — | — | — | — | — | — |

All six prepared selector/executor conditions completed at every successful tier:

- `global_score`;
- `global_hash`;
- `bounded_score_k16`;
- `bounded_hash_k16`;
- `bounded_score_k512`;
- `bounded_hash_k512`.

## Engineering interpretation

The selector and terminal Stage-B path are not the resource bottleneck. At N65536:

- immutable-frontier construction peaked at 13.70 GiB;
- all condition-local executions peaked at no more than 1.24 GiB;
- bounded K16/K512 condition times remained approximately flat relative to N;
- global modes increased with N as expected because they inspect the live population.

The N131072 failure therefore localizes the engineering frontier to transient complete-frontier construction rather than condition-local index-bank storage or terminal routing.

The persistent N131072 B64 frontier geometry is approximately:

```text
states: 64 * 131072 * 64 * 4 = 2,147,483,648 bytes
scores: 64 * 131072 * 4      =    33,554,432 bytes
total:                         2,181,038,080 bytes (~2.03 GiB)
```

The condition-local B64 index bank remains only 67,109,376 bytes (~64 MiB).

The resource stop arises from transient last-layer recurrent execution over 4,194,304 parent-action rows, not from the persistent frontier itself. The next engineering slice may therefore chunk recurrent rows inside each frontier layer while preserving:

- all 64 worlds;
- exact output frontier order and FP32 states/scores;
- exact eight recurrent updates per child;
- exact logical work accounting;
- the frozen scientific protocol.

It may not reduce the scientific world count, change the N/K ladders, enable mixed precision, load checkpoints, or open a high-scale scientific namespace.

## Provenance

- profile summary SHA-256: `bd9dbe78af2ad165fb6bd53823c5d8d956d201df04f1b26b5f965b56b84f7098`
- recursive manifest SHA-256: `a9900daec78495e9635d3d506132f4f183f801e18c7aca02f4eb5e86660632dc`
- local output root: `F:\gate7_high_scale_execution_engineering_profile_v0`

## Scientific boundary

This result establishes only that the unchunked B64 execution substrate is feasible through N65536 on the measured RTX 4060 Ti and reaches a memory resource frontier during the N131072 frontier build.

It does not establish:

- any Gate-7 coverage outcome;
- `K_required(N)`;
- learned routing superiority;
- a scientific routing frontier;
- an asymptotic scaling law;
- a maximum useful population.

High-scale Gate-7 science remains closed.
