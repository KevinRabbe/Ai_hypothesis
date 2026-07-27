# Relay work/span resource frontier protocol v0

## Status

**Gate 1 measurement protocol. No target-hardware performance result is claimed by this document.**

Gate v0 established reproducible fixed-parameter runtime-compute/source-scope scaling on canonical relay-v1. #77 then established that the repaired normalized relay function is mathematically serializable at the same `N × H` recurrent worker-update count.

Gate 1 asks the remaining systems question:

> **Does simultaneous parallel population execution provide a useful latency/throughput frontier over strong equivalent serial schedules on real target hardware?**

## Frozen neural function

The benchmark does not train or modify a model.

It accepts only a canonical relay-v1 checkpoint loadable through `load_relay_checkpoint_v1(...)` and records:

- experiment version;
- protocol version;
- benchmark version;
- training seed;
- learned parameter count;
- parameter fingerprint.

Every schedule uses that exact loaded model, source scope, input batch and recurrent hop count.

## Why two serial controls are required

The original serial-control proof matched recurrent worker updates but intentionally minimized live state. That implementation recomputes immutable input/value projections every relay hop.

Parallel execution computes those immutable learned projections once and reuses them.

Therefore `N × H` recurrent worker-update equality is **not by itself total learned-work equality**.

Gate 1 must expose this rather than silently biasing the timing comparison. It measures two serial controls:

1. a low-memory control that recomputes immutable projections;
2. a compute-matched cached control that computes them once like parallel, then serializes recurrent updates.

This creates an explicit time–memory trade-off instead of hiding it inside one baseline.

## Compared schedules

### Parallel normalized

`normalized_parallel_forward(...)`

For `N` active records and `H` relay hops:

- recurrent learned worker updates per sample: `N × H`;
- candidate evaluations: `N × H`;
- input projections: `N`;
- value projections: `N`;
- peak simultaneously updated neural states: `N`;
- cached initial-state vectors: `N`;
- cached candidate-message vectors: `N`;
- normalized active-record reduction once per hop.

### Serial low-memory normalized

`normalized_serial_forward(...)`

- recurrent learned worker updates: the same `N × H`;
- candidate evaluations: the same `N × H`;
- input projections: `N × H`;
- value projections: `N × H`;
- peak simultaneously updated neural states: `1`;
- no O(N) cached initial/message projection vectors;
- online numerically stable normalized reducer;
- zero simultaneous inter-state transfer accounting.

This is the existing minimum-live-state schedule from #77.

### Serial cached normalized

`normalized_serial_cached_forward(...)`

- recurrent learned worker updates: the same `N × H`;
- candidate evaluations: the same `N × H`;
- input projections: `N`, matching parallel;
- value projections: `N`, matching parallel;
- recurrent record updates remain serial;
- peak simultaneously updated neural states: `1`;
- cached initial-state vectors: `N`;
- cached candidate-message vectors: `N`;
- online numerically stable normalized reducer.

This is the stronger compute-matched serial baseline. It deliberately gives up part of the low-memory advantage to remove repeated static learned projection work.

## Correctness gate before timing

For every `(difficulty, active_workers, batch_size)` condition the harness first executes all three schedules and requires:

- final shared states close under `rtol=2e-5`, `atol=2e-5` by default;
- final logits close under the same tolerance;
- decoded node predictions exactly equal;
- recurrent worker-update counts equal across all schedules;
- static input/value projection counts equal between parallel and serial-cached.

A failed equivalence/accounting condition is invalid and must not produce a performance comparison.

## Timed-region boundary

Excluded from timing:

- world generation;
- integer/bit encoding;
- relay tensor-batch construction;
- checkpoint loading;
- benchmark input device transfer;
- JSON serialization.

Included:

- the complete selected relay schedule call;
- eager PyTorch orchestration required by that schedule;
- reducer work;
- completion of all scheduled device work for the measured block.

The first protocol is intentionally **eager**. Compiler/graph optimization remains a separate later systems variable.

## CUDA timing rule

CUDA profiling must not force a synchronization barrier after every measured batch.

For CUDA:

1. warm the schedule;
2. synchronize once before the measured block;
3. enqueue one CUDA event pair around each schedule invocation;
4. do **not** synchronize between measured invocations;
5. record host enqueue time after all invocations are submitted;
6. synchronize once at the end;
7. derive per-call device latency from stream-event pairs;
8. derive throughput from the full synchronized measured block.

This preserves normal stream/batching behavior instead of turning profiling itself into a serial barrier.

For CPU, per-call latency uses `perf_counter` directly.

The result therefore records both:

- latency source (`cuda_event` or `host_perf_counter`);
- host enqueue/orchestration time;
- total synchronized measured time.

## Schedule-order rule

Each schedule is independently warmed. The order of the three measured schedules rotates deterministically by `(active_workers, batch_size, hop_count)` so one schedule is not always measured first or last across the matrix.

The exact per-condition measurement order is stored in the result artifact.

If target-hardware measurements show order sensitivity large enough to affect conclusions, repeat the full run in a fresh process and report both rather than choosing the favorable order.

## Primary matrix

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

Primary local run defaults:

```text
warm-up iterations   = 20
measured iterations  = 100
world seed            = 0
execution mode        = eager
```

Do not compare unlabelled runs with different settings.

## Measurements

Per schedule and condition:

- median batch latency;
- p95 batch latency;
- minimum observed batch latency;
- latency source;
- host enqueue/orchestration time;
- total synchronized measured wall time;
- achieved samples/second;
- CUDA-event median device latency when CUDA is used;
- CUDA baseline allocated/reserved memory;
- CUDA peak allocated/reserved memory;
- peak allocated-memory delta above post-warm-up baseline;
- recurrent worker updates per sample;
- candidate evaluations per sample;
- input projection evaluations per sample;
- value projection evaluations per sample;
- total static learned projection evaluations per sample;
- communicated scalars per sample;
- peak simultaneously updated neural states;
- cached state/message vector counts.

Per condition:

- maximum absolute logits difference across serial controls versus parallel;
- maximum absolute final-shared difference;
- decoded-prediction equality;
- recurrent worker-update equality;
- parallel/serial-cached static-projection equality;
- learned recurrent-update depth proxy:
  - parallel: `H`;
  - both serial controls: `N × H`;
- low-memory serial speedup ratio:

```text
serial_low_memory median latency / parallel median latency
```

- compute-matched cached serial speedup ratio:

```text
serial_cached median latency / parallel median latency
```

The span value is explicitly a **recurrent learned-update proxy**, not a claim about exact kernel-level critical path. Device/wall latency remains the systems outcome.

## Memory interpretation

Do not reduce memory to “parallel N states, serial one state.”

The controls intentionally expose different residency choices:

- parallel keeps N recurrent states plus N immutable projection caches;
- serial-low-memory keeps one recurrent state and recomputes static projections;
- serial-cached keeps one simultaneously updated recurrent state but retains N cached initial/message projections.

CUDA allocator metrics include real eager implementation/caching behavior. The harness records baseline and peak values instead of pretending one allocator number is pure neural-state memory.

## Primary hardware

The decisive Gate-1 result must run on the actual local consumer GPU target, not only on GitHub CPU runners.

The result artifact records:

- Python version;
- OS/platform;
- machine/processor;
- PyTorch version;
- PyTorch thread count;
- execution mode;
- device type;
- CUDA runtime;
- CUDA device name/index;
- compute capability when CUDA is used;
- schedule timing policy.

GitHub CPU qualification/preflight proves mechanics only. It cannot pass or fail Gate 1.

## Canonical local command

```powershell
python -m ai_hypothesis.population_compute.run_relay_resource_frontier `
    --checkpoint <relay-v1-model.pt> `
    --device cuda `
    --population-sizes 1 4 16 64 256 `
    --batch-sizes 1 64 `
    --difficulties relay-2 relay-4 relay-8 `
    --warmup-iterations 20 `
    --measured-iterations 100 `
    --world-seed 0 `
    --output results\population_compute_scaling_v0\relay_resource_frontier_v0.json
```

A preserved frozen-confirmation checkpoint is appropriate because Gate 1 measures execution organization, not retraining quality.

## Interpretation

### Useful parallel frontier

Parallel population execution is practically useful for this relay function if it produces meaningful lower latency and/or higher throughput on target hardware relative to the **stronger applicable serial control**, while its extra residency/communication costs remain acceptable.

The low-memory serial control answers a memory-minimizing trade-off. The cached serial control answers a closer learned-work-matched trade-off. Neither should be hidden when interpreting the frontier.

Perfect `N×` speedup is not required.

### Negative systems result

If a strong serial control matches or beats parallel execution after fair runtime treatment, or if memory/synchronization costs dominate the speedup, then this relay function is better implemented serially despite Gate v0's positive capability scaling.

That is a valid negative Gate-1 result and does not invalidate Gate v0.

## Compiler follow-up

Do not silently compile only one schedule.

After the eager frontier is measured, profiling may justify a separate matched compiler ablation. `torch.compile`, CUDA Graphs, fusion or custom kernels must be reported as systems variables, not neural capability gains.

## Non-goals

This protocol does not test:

- new learned parameters;
- retraining;
- populations larger than 256;
- dense-model superiority;
- organization-specific function-level capability;
- dynamic activation;
- a real workload;
- a learned scheduler.

Those remain later gates.
