# Gate-7 execution-engine engineering profile

## Status

**ENGINEERING EVIDENCE ONLY — NOT GATE-7 SCIENTIFIC EVIDENCE.**

Measured Git head:

`57d0d5a9fcc78462e8a72964edd1aeb5a5577fc8`

Local hardware/runtime:

- GPU: NVIDIA GeForce RTX 4060 Ti
- PyTorch: 2.9.1+cu130
- CUDA runtime: 13.0
- compiler: OFF
- CUDA graphs: OFF
- mixed precision: OFF
- 64 deterministic synthetic/public engineering worlds
- frontier depth: 8
- repeats: 3
- generated children/run: 32,640
- frozen checkpoint C0 reused; no training

No Gate-7 scientific world namespace or result was used.

## Measured execution result

### Historical eager object path

```text
mean wall time:      6117.6433 ms
min wall time:       5546.3651 ms
max wall time:       6425.7195 ms
children/second:     5,335.3879
peak CUDA allocated: 688,479,232 bytes
```

### Tensorized eager path

```text
mean wall time:      126.8137 ms
min wall time:       96.7575 ms
max wall time:       147.6557 ms
children/second:     257,385.5079
peak CUDA allocated: 670,555,648 bytes
```

### Direct measured ratio

```text
wall-speedup tensor/object = 48.24119876837442x
throughput ratio           = ~48.24x
peak allocated reduction   = 17,923,584 bytes (~2.60%)
```

## Interpretation boundary

This result demonstrates a very large execution-efficiency gain from the first semantic-preserving tensorization slice on the measured local RTX 4060 Ti setup.

It is consistent with the previous eager path being dominated by execution overhead such as Python object handling, fragmented small GPU submissions and synchronization rather than raw recurrent-state memory capacity.

It does **not** establish:

- a Gate-7 capability result;
- a scientific population-scaling result;
- a compiler/CUDA-graph speedup;
- per-FLOP or per-joule superiority;
- that every remaining Stage-B bottleneck is removed.

The next engineering target is full dynamic Stage-B tensor-bank equivalence, including parent removal, bounded K selection across multiple K values, answer-blind capacity pruning, child insertion and terminal transcript preservation.

## Source

Local engineering output root:

`F:\gate7_execution_engine_profile_v0`

The user-provided terminal output is the measurement source for the values above.
