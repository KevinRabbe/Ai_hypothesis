# Gate-7 native-bank scaling profile — engineering result

Status: **ENGINEERING ONLY — NOT SCIENTIFIC EVIDENCE**

This record preserves the local RTX 4060 Ti execution-mechanics profile run from exact repository head:

`e93949e3c4d494bad9d2d1b81020f89dae06d197`

No checkpoint, hidden answer/path, training, Gate-7 scientific world namespace, capability outcome, compiler, CUDA graph, or mixed precision was used.

## Environment

- GPU: NVIDIA GeForce RTX 4060 Ti
- PyTorch: 2.9.1+cu130
- CUDA runtime: 13.0
- fixed physical batch: 32
- routing slots: 128
- repeats: 3
- population ladder: 512 -> 1,024 -> 2,048 -> 4,096 -> 8,192 -> 16,384 -> 32,768 -> 65,536 -> 131,072
- bounded K profile: 16 / 64 / 256 / 512 where K < N

Full local JSON was written to:

`F:\gate7_native_bank_scaling_profile_v0_retry2\summary.json`

The repository does not claim values from that local JSON which were not present in the captured terminal transcript.

## Measured mean routing latency

Values are microseconds per world routing decision. Bounded entries are `learned-score / matched-hash`.

| N | global | K16 | K64 | K256 | K512 |
|---:|---:|---:|---:|---:|---:|
| 512 | 36.348 | 60.560 / 41.658 | 46.780 / 43.453 | 47.695 / 41.789 | — |
| 1,024 | 33.292 | 47.792 / 41.948 | 45.798 / 40.487 | 44.455 / 40.100 | 45.332 / 40.498 |
| 2,048 | 32.263 | 45.556 / 41.246 | 45.040 / 42.153 | 47.414 / 42.769 | 46.728 / 41.529 |
| 4,096 | 32.282 | 45.688 / 42.203 | 55.080 / 42.440 | 47.319 / 41.726 | 48.077 / 41.621 |
| 8,192 | 32.251 | 46.725 / 42.134 | 45.851 / 43.292 | 47.018 / 42.184 | 44.335 / 41.083 |
| 16,384 | 32.155 | 44.866 / 40.538 | 44.937 / 41.649 | 45.782 / 41.593 | 45.460 / 41.605 |
| 32,768 | 32.505 | 45.614 / 40.286 | 44.740 / 39.790 | 47.098 / 39.902 | 44.785 / 39.614 |
| 65,536 | 48.153 | 47.945 / 52.042 | 58.886 / 50.797 | 60.806 / 41.023 | 48.603 / 47.088 |
| 131,072 | 105.397 | 119.332 / 56.872 | 79.094 / 52.231 | 67.471 / 51.057 | 93.941 / 115.889 |

## Largest-N / smallest-N latency ratios

- global: 512 -> 131,072 = **2.900x**
- K16 learned-score: 512 -> 131,072 = **1.970x**
- K64 learned-score: 512 -> 131,072 = **1.691x**
- K256 learned-score: 512 -> 131,072 = **1.415x**
- K512 learned-score: 1,024 -> 131,072 = **2.072x**

## Engineering interpretation

The native bounded-routing bank does not exhibit catastrophic N-dependent slowdown through N=131,072. Its fixed-K latency growth is materially smaller than the full-population global reference for K16/K64/K256.

However, the profile also exposes a second execution bottleneck: small bounded K does not automatically produce lower wall time on the GPU. At N=131,072, learned K256 (`67.471 us`) is materially faster than learned K16 (`119.332 us`) even though K256 examines sixteen times as many neural scores. Global routing is also faster than bounded learned routing over much of the ladder despite inspecting more scores.

That inversion is consistent with launch/dispatch overhead, low physical occupancy, and/or poor amortization of the multi-kernel bounded-selection sequence dominating arithmetic work. It is **not** evidence that K256 is scientifically better than K16; this profile contains no capability task.

The next engineering boundary is therefore physical execution density: sweep physical batch at fixed N=131,072 while keeping logical K unchanged. Compiler/fusion/CUDA-graph variants remain separate variables and are not admitted into the baseline until the eager physical-batch frontier is measured.
