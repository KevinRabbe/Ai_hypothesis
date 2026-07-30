# Gate-7 preparation — execution-scaling bottleneck audit

**STATUS: ENGINEERING PREPARATION ONLY. NO GATE-7 SCIENTIFIC DATA.**

This document separates implementation/resource scaling from the scientific fixed-K population hypothesis. The purpose is to remove accidental runtime/memory bottlenecks before a high-scale frontier campaign so a resource stop is not mistaken for a scientific routing limit.

Compiler/`torch.compile`/CUDA-graph choices remain a separate experimental variable. The first engineering pass must preserve eager FP32 model semantics and routing decisions.

## Current code audit: highest-priority structural bottlenecks

### P0 — CUDA scalar extraction in the hot path

Current child construction converts every CUDA score to a Python float using `scores[row_index].item()`.

That introduces a CPU/GPU synchronization boundary in the child-generation hot path. At large frontiers, this happens once per generated child and can prevent the CPU from running ahead of the GPU.

Target architecture:

- scores remain CUDA tensors in the candidate bank;
- parent selection uses tensor indices;
- no per-candidate `.item()`, `.cpu()`, Python float conversion, or CUDA-dependent Python control flow in the hot path;
- only compact aggregate telemetry crosses to CPU at explicit synchronization points.

### P0 — object-per-candidate representation

Current candidates carry Python tuple paths, one tensor object per state, and a Python float score. Stage-A and Stage-B repeatedly create tuples/lists and clone tensors.

This cannot be the primary representation at 1K–128K population.

Prepared replacement:

```text
states       [B, N, 64] float32
scores       [B, N]     float32
path_ids     [B, N]     integer
path_depth   [B, N]     compact integer or common scalar where possible
live_mask    [B, N]     bool
```

Use gather/index-select/scatter instead of Python candidate-object construction.

### P0 — bounded K selection still sorts the full population

Current `_bounded_visible_candidates` first sorts the entire reserve by candidate path, then samples K indices.

This makes bounded K routing computationally O(N log N) per Stage-B slot even though its score-information visibility is O(K).

Target:

- keep a stable tensor slot identity;
- deterministic bounded sampler directly emits K integer slot indices;
- gather exactly those K scores;
- no full-reserve ordering before bounded selection.

The K16 score/hash pair must continue to use the same deterministic sampler whenever their incoming live banks are identical.

### P0 — post-decision global-rank telemetry performs another full sort

For every bounded Stage-B decision, the current runtime sorts the entire population after selection to compute `selected_global_score_rank` telemetry.

This is evaluation-only and cannot affect the selected parent, but it destroys bounded computational scaling.

Scale-campaign replacement options, frozen before exposure:

1. remove full-rank telemetry from every slot and retain only preregistered sparse diagnostic slots/worlds; or
2. compute selected rank using tensor comparisons against the selected quantized score plus deterministic tie key, which is O(N) rather than O(N log N); or
3. move full-rank diagnostics to a separate profiling/validation lane not used by the high-scale scientific runner.

The scientific audit must retain enough information to prove the bounded score-visibility channel without requiring a full sort every activation.

### P0 — answer-blind capacity pruning sorts + SHA-256 hashes every candidate every slot

Current overflow pruning deterministically sorts the complete live population using a SHA-256 key derived from `(world_seed, slot, path)`.

At high N this is an avoidable O(N log N) Python/cryptographic cost.

Target:

- freeze an answer-blind integer priority/permutation primitive;
- compute it vectorially or precompute priorities/retention order;
- select the retained N using indices/partition, not Python object sorting;
- prove score blindness by API and regression tests.

Cryptographic strength is not scientifically required; determinism and answer/score blindness are.

### P0 — selected-parent removal rebuilds the whole Python tuple

Current Stage-B removal filters every candidate path into a new tuple each slot.

Target:

- live-mask update or swap-delete in a tensor bank;
- O(1) logical removal after parent index selection;
- periodic compaction only when mechanically useful.

### P1 — global reference uses full sorting when only the best parent is required

Current global scheduler sorts all candidates by quantized score + deterministic tie-break and takes element zero.

Target:

- encode the frozen score/tie ordering as a tensor key;
- use argmax/reduction rather than full sort;
- compute additional rank telemetry separately if needed.

This preserves global score visibility while reducing scheduler selection from sorting to reduction.

### P1 — repeated tensor cloning/stacking and per-child input construction

Current `_advance_parent_batch`:

- clones each selected parent state;
- builds input vectors one child at a time in Python;
- stacks Python lists into tensors;
- clones every output state back into an individual candidate object.

Target:

- gather selected parent states directly from `[B,N,64]`;
- construct both branch inputs vectorially;
- run one batched recurrent update;
- write children into preallocated/double-buffer tensor storage.

### P1 — Stage A is recomputed independently for scheduler conditions

For capability experiments, every scheduler at a fixed `(checkpoint, world, N)` starts from the exact same Stage-A frontier. Recomputing it separately is scientific-work identity but unnecessary evaluator work when no wall-clock claim is being made.

High-scale campaign preparation should separate:

- **logical learned-work accounting** — still records the frozen Stage-A work each condition conceptually receives;
- **evaluator execution** — may construct an immutable frontier once and reuse/copy its state bank across scheduler conditions.

Any throughput benchmark must disclose whether Stage-A reuse is enabled and cannot compare cached execution against historical eager timings as if they were identical schedules.

Across the geometric N sweep, a scale-neutral scorer also allows Stage-A frontier construction to be extended one generation at a time rather than rebuilt from root for every larger N.

### P1 — fixed evaluation batch size will eventually become a resource artifact

At batch 64 and FP32 state width 64, the raw recurrent-state bank alone is approximately:

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

This excludes children, GRU temporaries/workspaces, scores, masks and telemetry.

The high-scale scientific campaign should therefore permit a frozen resource-safe execution batch policy that may decrease batch size with N while keeping the same worlds, checkpoints and scheduler semantics. No cross-N wall-clock claim may be made from those varying execution batches.

## Measurement before optimization

Before replacing the runtime, collect an engineering-only profile on synthetic worlds using:

- `torch.profiler` for CPU/CUDA operator time and memory;
- CUDA synchronization count / host gaps;
- GPU active-vs-idle timeline;
- Python time in sorting, hashing, tuple/list construction and telemetry;
- peak allocated/reserved CUDA memory;
- candidate expansions/sec and Stage-B decisions/sec.

Use NVTX ranges for:

```text
stage_a_expand
initial_thinning
stage_b_sample
stage_b_select
stage_b_advance
stage_b_prune
telemetry
```

The profile is not scientific evidence and must not use a Gate-7 hidden scientific namespace.

## Engineering sequence

Do not start with compiler tricks. Remove asymptotically unnecessary work first:

1. tensorized candidate bank + no `.item()` hot-path sync;
2. O(K) bounded sampler without full-reserve sort;
3. answer-blind tensorized capacity handling;
4. reduction-based global selection;
5. vectorized child encoding/update and preallocated buffers;
6. immutable Stage-A reuse for capability-only sweeps;
7. profile again and identify the new bottleneck;
8. only then test compiler/CUDA-graph execution as a **separate variable**.

## Compiler/runtime-graph track remains independent

After the eager tensorized baseline is correct and qualified, separately compare:

```text
eager tensorized
vs torch.compile(default)
vs torch.compile(reduce-overhead)
vs explicit CUDA-graph-compatible path (where shapes permit)
```

Do not change scientific conclusions based on compiled timing alone. Compiler mode is an execution variable, exactly as frozen in the broader project methodology.

## Resource-frontier interpretation

A future high-scale stop must be classified by cause:

- routing statistics fail while execution remains healthy -> scientific routing frontier candidate;
- global reference/task collapses too -> task/search-budget frontier;
- OOM/host bottleneck/kernel-launch starvation -> engineering/resource frontier;
- numerical divergence -> numerical/runtime frontier.

Only the first category is evidence about the fixed-K routing hypothesis itself.
