# Relay work/span resource frontier protocol v0

## Status

**Gate 1 measurement protocol. No performance result is claimed by this document.**

Gate v0 already established reproducible fixed-parameter runtime-compute/source-scope scaling on the canonical relay-v1 task. The equal-work serial-control result also established that the repaired relay function is mathematically serializable.

The remaining question for this relay function is therefore a systems/resource question:

> **Does simultaneous parallel population execution provide a useful latency/throughput frontier over the exactly equivalent serial schedule on real target hardware?**

## Frozen neural function

The benchmark does not train or modify a model.

It accepts only a canonical relay-v1 checkpoint loadable through `load_relay_checkpoint_v1(...)` and records its:

- experiment version;
- protocol version;
- benchmark version;
- training seed;
- learned parameter count;
- parameter fingerprint.

Every measured schedule uses that exact loaded model.

## Compared schedules

### Parallel normalized

`normalized_parallel_forward(...)`

For `N` active records and `H` relay hops:

- learned worker updates per sample: `N × H`;
- candidate evaluations per sample: `N × H`;
- peak live learned states per sample: `N`;
- one normalized active-record reduction per hop.

### Serial normalized

`normalized_serial_forward(...)`

- learned worker updates per sample: the same `N × H`;
- candidate evaluations per sample: the same `N × H`;
- peak live learned states per sample: `1`;
- active records are time-multiplexed through one learned state;
- an online numerically stable reducer reproduces the same normalized result.

The serial implementation is the existing qualified schedule, not a different model or reduced-scope approximation.

## Correctness gate before timing

For every `(difficulty, active_workers, batch_size)` condition the harness first executes both schedules and requires:

- final shared states close under the existing serial-control tolerance (`rtol=2e-5`, `atol=2e-5` by default);
- final logits close under the same tolerance;
- decoded node predictions exactly equal;
- learned worker-update counts equal.

A failed equivalence condition is invalid and must not produce a performance comparison.

## Timed-region boundary

Excluded from timing:

- world generation;
- integer/bit encoding;
- relay tensor-batch construction;
- checkpoint loading;
- device transfer of benchmark inputs;
- JSON serialization.

Included:

- the complete selected relay schedule call;
- eager PyTorch orchestration needed by that schedule;
- required device synchronization before latency completion is recorded.

The first protocol is intentionally **eager**. Compiler/graph optimization is a separate later systems variable.

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

These values may be changed only as an explicitly recorded measurement configuration; do not compare unlabelled runs with different settings.

## Measurements

Per schedule and condition:

- median batch latency;
- p95 batch latency;
- minimum observed batch latency;
- total measured wall time;
- achieved samples/second;
- CUDA-event median device latency when CUDA is used;
- CUDA baseline allocated/reserved memory;
- CUDA peak allocated/reserved memory;
- peak allocated-memory delta above the post-warm-up baseline;
- learned worker updates per sample;
- candidate evaluations per sample;
- communicated scalars per sample;
- peak live neural states per sample.

Per paired condition:

- maximum absolute logits difference;
- maximum absolute final-shared difference;
- decoded-prediction equality;
- matched-work status;
- learned sequential-update depth proxy:
  - parallel: `H`;
  - serial: `N × H`;
- latency speedup:

```text
serial median batch latency / parallel median batch latency
```

The span values are explicitly a **learned sequential-update proxy**, not a claim about exact kernel-level critical path. Wall/device latency remains the measured systems outcome.

## GPU memory interpretation

CUDA memory measurements include the real eager implementation and caching allocator behavior.

The harness records both baseline and peak values rather than pretending a single allocator number is pure neural-state memory. The logical live-state difference is already known structurally (`N` versus `1`) and should be interpreted alongside measured allocation peaks.

## Primary hardware

The decisive Gate-1 result should be run on the actual local consumer GPU target.

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
- compute capability when CUDA is used.

GitHub CPU qualification proves mechanics only. It is not the practical GPU frontier result.

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

A confirmation checkpoint from the preserved #80 artifact is suitable because Gate 1 measures schedule organization, not retraining quality.

## Interpretation

### Useful parallel frontier

Parallel width is practically useful for this relay function if equal-work parallel execution produces meaningful lower latency and/or higher throughput on target hardware while its extra state/memory/communication cost remains acceptable.

Perfect `N×` speedup is not required.

### Negative systems result

If the serial schedule matches or beats parallel execution after fair implementation/runtime treatment, or if memory/synchronization costs dominate the speedup, then the relay function should be considered better implemented serially despite Gate v0's positive capability scaling.

That is a valid negative Gate-1 result and does not invalidate Gate v0.

## Compiler follow-up

Do not silently compile only one schedule.

After the eager frontier is measured, profiling may justify a separate matched compiler ablation. Any `torch.compile`, CUDA Graph, fusion or custom-kernel result must be reported as a systems optimization variable, not as a neural capability gain.

## Non-goals

This protocol does not test:

- new learned parameters;
- retraining;
- larger than 256 populations;
- dense-model superiority;
- organization-specific function-level capability;
- dynamic activation;
- a real workload;
- a learned scheduler.

Those remain later gates.
