# Gate-7 native physical-batch scaling profile — engineering result

Status: **ENGINEERING ONLY — NOT SCIENTIFIC EVIDENCE**

This record preserves the local RTX 4060 Ti execution-mechanics profile run from exact repository head:

`2e1880c403de7462d868a4f38275f3545cd24228`

No checkpoint, hidden answer/path, training, Gate-7 scientific world namespace, capability outcome, compiler, CUDA graph, or mixed precision was used.

## Environment

- GPU: NVIDIA GeForce RTX 4060 Ti
- PyTorch: 2.9.1+cu130
- CUDA runtime: 13.0
- population: 131,072 fixed
- routing slots: 128
- repeats: 5
- physical batch ladder: 8 / 16 / 32 / 64 / 128 / 256
- logical learned routing: K16 / K64 / K256
- matched control: K16 hash
- full-population reference: global score
- compiler: OFF
- CUDA graphs: OFF
- mixed precision: OFF

Full local JSON was written to:

`F:\gate7_native_batch_scaling_profile_v0\summary.json`

The repository does not claim values from that local JSON which were not present in the captured terminal transcript.

## Measured routing latency and throughput

Latency is mean microseconds per world routing decision. Throughput is world routing decisions/second.

| Physical batch | K16 score | K16 hash | K64 score | K256 score | Global |
|---:|---:|---:|---:|---:|---:|
| 8 | 194.559 us / 5,143 s^-1 | 177.059 us / 5,663 s^-1 | 191.084 us / 5,243 s^-1 | 190.215 us / 5,260 s^-1 | 138.249 us / 7,239 s^-1 |
| 16 | 96.813 us / 10,342 s^-1 | 86.516 us / 11,590 s^-1 | 93.672 us / 10,680 s^-1 | 95.518 us / 10,491 s^-1 | 97.707 us / 10,237 s^-1 |
| 32 | 48.425 us / 20,669 s^-1 | 43.080 us / 23,221 s^-1 | 45.918 us / 21,808 s^-1 | 46.251 us / 21,641 s^-1 | 101.791 us / 9,826 s^-1 |
| 64 | 23.578 us / 42,437 s^-1 | 21.474 us / 46,638 s^-1 | 22.821 us / 43,833 s^-1 | 23.046 us / 43,397 s^-1 | 92.282 us / 10,846 s^-1 |
| 128 | 12.447 us / 81,163 s^-1 | 10.546 us / 94,860 s^-1 | 11.267 us / 88,793 s^-1 | 11.408 us / 87,770 s^-1 | 85.565 us / 11,698 s^-1 |
| 256 | 8.136 us / 123,279 s^-1 | 10.037 us / 105,598 s^-1 | 7.200 us / 143,703 s^-1 | 9.433 us / 117,159 s^-1 | 83.781 us / 11,976 s^-1 |

## Batch-8 -> batch-256 scaling

- K16 learned score: throughput **23.971x**, latency **0.042x**, peak CUDA allocation **8.38 GiB**
- K16 matched hash: throughput **18.648x**, latency **0.057x**, peak CUDA allocation **8.38 GiB**
- K64 learned score: throughput **27.410x**, latency **0.038x**, peak CUDA allocation **8.38 GiB**
- K256 learned score: throughput **22.275x**, latency **0.050x**, peak CUDA allocation **8.38 GiB**
- global score: throughput **1.654x**, latency **0.606x**, peak CUDA allocation **9.16 GiB**

## Engineering interpretation

This profile strongly supports the execution hypothesis that the bounded-routing path was limited by insufficient physical work/launch amortization rather than by its logical sparse-compute semantics.

At fixed N=131,072 and unchanged logical K, pooling more independent worlds improves bounded-routing throughput by roughly 19x–27x from physical batch 8 to 256. The full-population global reference improves only 1.65x over the same physical-batch range.

At batch 256 the learned bounded paths require:

- K16: 8.136 us/world decision
- K64: 7.200 us/world decision
- K256: 9.433 us/world decision
- global: 83.781 us/world decision

Thus, once enough independent ready work is available, bounded routing is roughly an order of magnitude faster per world decision than the global reference at N=131,072 while retaining lower measured peak CUDA allocation in this mechanics-only workload.

This is **not** capability evidence and does not establish which K is scientifically sufficient. It establishes only that sparse logical routing can be executed densely enough to avoid the small-batch GPU-starvation bottleneck on this hardware.

## Consequence for Gate-7 preparation

The eager physical-batching baseline is now strong enough that compiler, CUDA-graph, fusion, and mixed-precision variants can remain separate optional engineering variables rather than prerequisites for the first high-scale capability experiment.

The next scientific-preparation boundary is the scale-neutral recurrent scorer transition. The original frozen scorer has a depth-10 one-hot representation and cannot honestly represent the N>512 frontier. A replacement scorer must therefore be frozen and qualified separately before any Gate-7 high-scale capability world is opened.