# Gate-7 preparation — execution-scaling bottleneck audit

**STATUS: ENGINEERING PREPARATION ONLY. NO GATE-7 SCIENTIFIC DATA.**

This document separates implementation/resource scaling from the scientific routing-bandwidth hypothesis. The purpose is to remove accidental runtime/memory bottlenecks before a high-scale campaign so a resource stop is not mistaken for a scientific routing limit.

Compiler/`torch.compile`/CUDA-graph choices remain a separate experimental variable. The first engineering pass preserves eager FP32 model semantics and routing decisions.

## Measured local profile — first tensorization slice

The first qualified tensorization slice was measured on the user's actual local machine:

- NVIDIA GeForce RTX 4060 Ti;
- PyTorch 2.9.1+cu130;
- CUDA runtime 13.0;
- compiler OFF;
- CUDA graphs OFF;
- mixed precision OFF;
- 64 deterministic synthetic/public engineering worlds;
- frontier depth 8;
- checkpoint C0;
- 3 synchronized repeats;
- 32,640 generated children/run.

Measured result:

```text
historical eager object path
mean wall time:      6117.6433 ms
throughput:          5,335.39 children/s
peak CUDA allocated: 688,479,232 bytes

tensorized eager Stage-A
mean wall time:      126.8137 ms
throughput:          257,385.51 children/s
peak CUDA allocated: 670,555,648 bytes

wall speedup:        48.2412x
peak allocation drop: 17,923,584 bytes (~2.60%)
```

This is engineering evidence only. It strongly supports that the historical Stage-A execution path was dominated by execution organization/host overhead rather than raw recurrent-state memory capacity. It does not yet measure the fully dynamic Stage-B scheduler.

Permanent record:

`experiments/population_compute_scaling_v0/gate7_execution_engine_profile_result.md`

## Historical bottlenecks and current state

### P0 — CUDA scalar extraction in the historical hot path

Gate-6 child construction converts every CUDA score to a Python float using `scores[row_index].item()`. The qualified tensorized Stage-A removes per-child CUDA-to-Python scalar extraction.

Remaining requirement: dynamic Stage-B must preserve scores/selected indices on device and allow only compact aggregate telemetry to cross to CPU after decisions.

### P0 — object-per-candidate representation

Gate-6 candidates carry Python tuple paths, one tensor object/state and a Python float score. The Stage-A tensor slice replaces this with dense banks.

Dynamic Stage-B target:

```text
states       [B, capacity, 64] float32
scores       [B, capacity]     float32
path_bits    [B, capacity]     int64
path_depth   [B, capacity]     compact integer
live_mask    [B, capacity]     bool
```

Use gather/index-select/scatter instead of candidate-object construction and O(N) tuple rebuilding.

### P0 — bounded K selection still sorts the full population in Gate-6

Gate-6 path-orders the entire reserve before sampling K, making bounded routing computationally O(N log N) per Stage-B slot even though causal score visibility is O(K).

Target:

- persistent canonical compact path/slot order;
- deterministic bounded sampler emits K live candidate indices directly;
- gather exactly those K scores;
- score/hash paired controls share sampled candidate identities when incoming banks are identical;
- no full-reserve ordering before causal bounded selection.

### P0 — post-decision global-rank telemetry

Gate-6 performs a second full population score sort after bounded selection only to calculate global-rank telemetry.

For high-scale Gate-7, either compute rank with vectorized comparisons/reduction or move complete rank diagnostics outside the hot path. Evaluation telemetry must never force O(N log N) work per activation.

### P0 — answer-blind capacity pruning

Gate-6 hashes every path with SHA-256 and fully sorts the live reserve every slot for answer-blind retention.

A future replacement must be frozen before scientific exposure and remain deterministic, score blind and answer blind. It may use vectorized/precomputed integer priority and indexed retention rather than Python cryptographic sorting.

### P1 — global reference selection

Global mode needs only the best parent. Replace complete score sorting with a tensor reduction implementing the frozen quantized-score + deterministic tie ordering.

### P1 — sparse logical activation should use dense physical execution

Sparse activation is a per-world scientific property; it does not require tiny physical GPU submissions. The runtime should gather selected work from many independent worlds into dense CUDA batches while preserving each world's exact logical transcript.

Target principle:

```text
sparse logical activation + dense physical GPU batching
```

### P1 — Stage-A reuse

For capability-only scheduler comparisons, the common Stage-A frontier may be materialized once per checkpoint/world batch and copied into treatments after regressions prove state identity, no storage aliasing and unchanged scientific results. Logical learned-work accounting remains attached to every treatment.

## Memory note

Raw FP32 recurrent-state storage at batch 64 / width 64 is approximately:

```text
N512       8 MiB
N1K       16 MiB
N2K       32 MiB
N4K       64 MiB
N8K      128 MiB
N16K     256 MiB
N32K     512 MiB
N64K       1 GiB
N128K      2 GiB
```

The first local profile changed peak CUDA allocation only from ~688 MB to ~671 MB while improving wall time by 48.24x. That strongly indicates execution overhead was the first major practical bottleneck, not state-bank storage.

## Qualified first tensor-engine slice

`gate7_tensor_engine_prep.py` currently provides:

- generation-synchronous dense recurrent state bank `[B,N,64]`;
- score bank `[B,N]`;
- compact integer path identity;
- vectorized productive child-input construction;
- dense Stage-A recurrent execution;
- bounded score selection that gathers only K score values once metadata is canonically ordered;
- selected indices kept device-side.

Data-blind CI proves synthetic Stage-A path identities, recurrent states and scores match the eager reference exactly. It checks K16 selection including quantized-score ties and rejects `.item()`, `.cpu()`, `.tolist()`, Python float extraction and `sorted()` inside the qualified hot tensor selection path.

This does not yet constitute full dynamic Stage-B equivalence.

## Engineering sequence

1. tensorized Stage-A + no per-child CUDA/Python sync — **DONE / QUALIFIED**;
2. local eager-object vs eager-tensorized profile — **DONE: 48.24x measured Stage-A wall speedup**;
3. implement full dynamic Stage-B tensor bank;
4. prove exact small-scale transcript equivalence for global, score-K and hash-K across K16/K32/K64;
5. remove full-reserve sort from bounded causal selection;
6. vectorize/decouple answer-blind capacity handling and global-rank telemetry;
7. enable immutable Stage-A reuse for capability-only sweeps after equivalence;
8. profile the full scheduler again;
9. only then test compiler/CUDA-graph modes as separate execution variables.

## Resource-frontier interpretation

A future high-scale stop must be classified by cause:

- routing statistics fail while execution remains healthy -> scientific routing-bandwidth frontier candidate;
- global reference/task collapses too -> task/reference frontier;
- OOM/host bottleneck/kernel-launch starvation -> engineering/resource frontier;
- numerical divergence -> numerical/runtime frontier.

Only the first category is evidence about the routing-bandwidth hypothesis itself.
