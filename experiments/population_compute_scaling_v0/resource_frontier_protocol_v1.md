# Relay work/span resource frontier protocol v1

## Status

**Gate-1 v1 measurement protocol. No target-hardware performance result is claimed by this document.**

V1 is a revision of Gate-1 v0 made after the first real target-CUDA run failed the frozen v0 FP32 tensor-allclose correctness rule and after two untimed correctness diagnostics characterized that failure. No latency, throughput, speedup, or memory result was admitted before this v1 protocol was frozen.

The v0 failure remains part of the scientific record in `gate1_v0_cuda_equivalence_result.md`; v1 does not rewrite or reinterpret v0 as a pass.

## Scientific question

> **Does simultaneous parallel population execution provide a useful latency/throughput frontier over strong equivalent serial schedules on the real target GPU once recurrent work, static learned projection work, memory residency and communication are explicit?**

Gate v0 already established reproducible fixed-parameter runtime-compute/source-scope scaling on the repaired relay function. Gate-1 remains a systems question, not another capability benchmark.

## Why v1 exists

The three Gate-1 schedules are mathematically equivalent but execute different tensor shapes:

- parallel performs batched recurrent/GEMM/reduction operations;
- serial-low-memory performs one record at a time and recomputes immutable projections;
- serial-cached performs one recurrent record at a time while retaining compute-matched immutable projection caches.

Gate-1 v0 required FP32 tensor allclose at `rtol=2e-5`, `atol=2e-5` before timing. On the target RTX 4060 Ti, 4/30 deep batched cells exceeded that rule while all decoded outputs remained identical.

An untimed precision diagnostic then found:

- exact decoded equality across all FP32 schedules in all 30 cells;
- exact decoded equality across all FP64 schedules in all 30 cells;
- exact per-schedule decoded equality between FP32 and FP64 in all 30 cells;
- worst FP32 pairwise logits drift `5.199909210205078e-4`;
- worst FP64 pairwise logits drift `6.203926261605375e-13`;
- worst FP64 pairwise shared-state drift `1.2045919817182948e-13`.

That result supports finite-precision execution-order / kernel-shape effects rather than semantic schedule disagreement.

V1 therefore does **not** enlarge the FP32 tolerance after observing the failure. It replaces the v0 admission rule with a separately versioned precision-aware rule.

## Frozen neural function and checkpoint

No training or learned parameter change occurs.

Canonical decisive checkpoint:

```text
training_seed = 1
parameter_fingerprint = c227ade9006e47bec17a2a3d5aedf6ac95a6a94607b96b9f52ab759905536c12
checkpoint_file_sha256 = 0b7c1f2a14fe9d2987819ed53fc0b55c04f3bb00bce356c1023778830a08ad26
source run = 30239005530
path = run/seed_1/model-v1.pt
learned parameters = 26669
```

The same checkpoint is loaded twice for the untimed correctness preflight:

- one FP32 execution model;
- one FP64 shadow reference.

Only the FP32 model is timed.

## Compared schedules

The schedules and work accounting are unchanged from v0.

### Parallel normalized

`normalized_parallel_forward(...)`

For `N` active records and `H` relay hops:

- recurrent learned worker updates: `N × H`;
- candidate evaluations: `N × H`;
- input projections: `N`;
- value projections: `N`;
- peak simultaneously updated neural states: `N`;
- cached initial-state vectors: `N`;
- cached candidate-message vectors: `N`.

### Serial low-memory normalized

`normalized_serial_forward(...)`

- recurrent learned worker updates: `N × H`;
- candidate evaluations: `N × H`;
- input projections: `N × H`;
- value projections: `N × H`;
- peak simultaneously updated neural states: `1`;
- no O(N) projection cache;
- zero simultaneous inter-state transfer accounting.

### Serial cached normalized

`normalized_serial_cached_forward(...)`

- recurrent learned worker updates: `N × H`;
- candidate evaluations: `N × H`;
- input projections: `N`;
- value projections: `N`;
- peak simultaneously updated neural states: `1`;
- cached initial-state vectors: `N`;
- cached candidate-message vectors: `N`;
- zero simultaneous inter-state transfer accounting.

This remains the primary compute-matched serial control.

## Frozen v1 correctness gate

The **complete 30-cell matrix** must pass this gate before any schedule is timed.

For every `(difficulty, active_workers, batch_size)` condition:

1. execute all three schedules in FP32;
2. execute all three schedules in FP64 using the same checkpoint values cast to FP64;
3. require exact decoded prediction equality between FP32 parallel and both FP32 serial schedules;
4. require exact decoded prediction equality between FP64 parallel and both FP64 serial schedules;
5. require exact decoded prediction equality between each FP32 schedule and its FP64 counterpart;
6. require FP64 parallel vs each FP64 serial schedule to satisfy:

```text
rtol = 1e-10
atol = 1e-10
```

for both final logits and final shared state;
7. require the original recurrent/static-work accounting invariants.

The `1e-10 / 1e-10` FP64 corroboration rule is frozen in v1 after the untimed precision diagnostic and before any admitted performance run. V1 is therefore a diagnostic-informed revised protocol, not a replication of the original v0 preregistration.

A failure anywhere in the complete correctness matrix aborts the run before timing.

## FP32 drift policy

V1 records but does **not** gate on a newly enlarged FP32 tensor tolerance.

Per cell it records:

- maximum FP32 schedule-pair logits drift;
- maximum FP32 schedule-pair shared-state drift;
- maximum FP64 schedule-pair logits drift;
- maximum FP64 schedule-pair shared-state drift;
- maximum same-schedule FP32↔FP64 logits drift;
- maximum same-schedule FP32↔FP64 shared-state drift.

This prevents the v0 target result from being rescued by selecting a convenient new FP32 threshold after observation.

## FP64 memory isolation before timing

The FP64 model is correctness-only infrastructure.

After the complete correctness matrix passes and before any resource measurement:

1. move the FP64 reference model off the target GPU;
2. synchronize CUDA;
3. release unused cached CUDA blocks;
4. retain only the FP32 model for timed execution.

The result provenance must record that this occurred.

This is required because leaving an FP64 model resident would contaminate allocator baselines, peak memory, and possibly execution behavior.

## Frozen resource matrix

Population sizes:

```text
1 / 4 / 16 / 64 / 256
```

Relay difficulties:

```text
relay-2 / relay-4 / relay-8
```

Batch sizes:

```text
1  — latency-oriented
64 — throughput-oriented
```

Execution settings:

```text
warm-up iterations   = 20
measured iterations  = 100
world seed            = 0
execution mode        = eager
```

The matrix is unchanged from v0 so the corrected admission protocol does not alter the systems question.

## Timed-region boundary

Excluded from timing:

- world generation;
- integer/bit encoding;
- tensor-batch construction;
- checkpoint loading;
- all FP32/FP64 correctness-preflight executions;
- FP64 offload and CUDA-cache cleanup;
- benchmark input device transfer;
- JSON serialization.

Included:

- complete selected FP32 relay schedule call;
- eager PyTorch orchestration required by that schedule;
- reducer work;
- completion of all scheduled device work for the measured block.

## CUDA timing rule

Unchanged from v0:

1. warm each FP32 schedule independently;
2. synchronize once before its measured block;
3. enqueue a CUDA event pair around each invocation;
4. do not synchronize between measured invocations;
5. record host enqueue completion;
6. synchronize once at the end of the measured block;
7. derive per-call device latency from event pairs;
8. derive throughput from the full synchronized block.

Schedule measurement order rotates deterministically by the full condition and the exact order is stored in each result row.

## Resource measurements

Unchanged from v0:

- median / p95 / minimum batch latency;
- CUDA-event device latency;
- host enqueue/orchestration time;
- synchronized measured wall time;
- samples/second;
- CUDA allocator baseline/peak/delta;
- recurrent worker updates;
- candidate evaluations;
- input/value/static projection counts;
- communicated scalars;
- peak simultaneously updated neural states;
- cached state/message vector counts;
- low-memory serial / parallel latency ratio;
- cached serial / parallel latency ratio.

No speedup threshold is preregistered. A negative result remains valid.

## Interpretation rules

The primary systems comparison remains `serial_cached_normalized` vs `parallel_normalized` because both perform `N` input projections, `N` value projections and the same `N × H` recurrent worker updates.

Interpret batch sizes separately:

- batch `1`: latency-oriented;
- batch `64`: throughput-oriented.

Possible valid outcomes include:

- cached serial matches or beats parallel across the frontier → simultaneous population execution has no practical advantage for this eager relay implementation;
- parallel wins only at large `N` or batch `64` → throughput-specific benefit rather than universal latency benefit;
- device-time benefit is erased by host enqueue/orchestration → execution orchestration is the next systems bottleneck;
- speed gains are outweighed by memory residency → serial or hybrid scheduling is preferable.

Compiler/graph optimization remains a separate later experimental variable.

## Scientific boundary

Gate-1 v1 can answer only the resource-organization question for this frozen relay function, checkpoint, eager PyTorch implementation and target GPU environment.

It cannot establish:

- a new capability gain;
- per-FLOP superiority;
- general superiority of population computation;
- 1K+/100K population scaling;
- compiler advantage;
- real-workload advantage.

No performance result exists until the complete v1 target-CUDA run passes the independent v1 auditor.
