# Persistent Neural Profile v0

## Purpose

This diagnostic answers the first performance question raised by the normalized direct-versus-persistent comparison:

> Is persistent overhead primarily coming from the selected-worker learned execution path, or from everything around it?

It deliberately stops at that coarse decomposition. Deeper scheduler/ledger/projection phase instrumentation is deferred until real measurements show that the non-neural portion is large enough to justify splitting further.

## Why profile the selected-worker boundary

`HomogeneousWorkerBank.forward_selected(...)` is the stable learned-execution boundary used by both:

- the direct large-scope benchmark;
- the persistent Worker Runtime adapter.

Timing that exact boundary gives one common measure of neural/device work without adding clocks to scheduler or ledger hot paths.

The residual is:

```text
condition wall time - selected-worker learned time
```

For the persistent run this residual includes, among other things:

- Work Thread projection;
- scheduler/control logic;
- Work Item construction;
- Research Ledger attempt/evidence writes;
- scope coverage reconstruction;
- CPU tensor/evidence handling outside the selected-worker call;
- final persistent evaluation projection.

It is intentionally called **non-selected-worker time**, not "scheduler time" or "ledger time", because v0 has not measured those subcomponents separately.

## Opt-in timing wrapper

`TimedSelectedWorkerBank` wraps any selected-worker bank while delegating its normal API and metadata.

It does not change worker selection, model inputs, evidence conversion, or persistence.

### CPU

CPU selected-worker calls are synchronous, so the wrapper uses `perf_counter()` around `forward_selected(...)`.

### CUDA

CUDA calls record start/end `torch.cuda.Event` objects on the active device stream.

The wrapper does **not** synchronize after every call.

The caller synchronizes once at the existing condition boundary, then reads the accumulated event durations through `snapshot_after_synchronize()`.

This preserves batching/launch behavior while measuring queued device work, including stream-ordered copies/operations performed inside the selected-worker boundary.

## Measurement counters

Each timing snapshot records:

- selected-worker call count;
- selected-worker sample count;
- elapsed selected-worker seconds;
- selected-worker samples/second.

`reset_timing()` starts a fresh measurement window after prior CUDA work has been synchronized.

## Diagnostic runner

Run:

```bash
python -m ai_hypothesis.large_scope.run_persistent_neural_profile \
  --checkpoints checkpoint_1.pt checkpoint_2.pt checkpoint_3.pt checkpoint_4.pt \
  --device cuda \
  --backend vmap \
  --split development \
  --world-count 64 \
  --window-count 16 \
  --step-width 2 \
  --rounds 4 \
  --mode diverse_workers
```

This compares:

```text
direct width 8
```

against:

```text
persistent width 2 × 4 rounds
```

on the exact same generated worlds and worker sequence.

The profiler handles one bounded world chunk intentionally. Large-corpus statistical comparison remains the job of `persistent_scope_comparison_v0`; this tool is for diagnosing where one representative persistent chunk spends time.

## Timing boundaries

An unmeasured direct selected-worker pass warms the exact path first.

Then:

### Direct

The runner measures:

- total direct wall time;
- selected-worker learned time;
- residual non-selected-worker time.

### Persistent

The runner measures separately:

- persistent setup time;
- persistent run wall time;
- selected-worker learned time inside the run;
- residual non-selected-worker run time;
- persistent setup + run end-to-end time.

The persistent condition also records:

- Research Ledger event count;
- SQLite main/WAL/SHM bytes;
- stable worker-bank identity.

## Derived ratios

The JSON includes:

- persistent/direct total-time ratio;
- persistent-run/direct total-time ratio;
- persistent/direct selected-worker-time ratio;
- non-selected-worker seconds per local evaluation for both conditions.

Interpretation examples:

### Persistent total high, selected-worker ratio near 1

The learned compute itself is comparable; persistent organization is the likely bottleneck.

Only then should deeper CPU-side phase profiling be added.

### Selected-worker ratio also high

Persistent execution is changing batch shape/device efficiency enough to make the learned path slower.

Likely next checks are batch size, tensor preparation, worker-index selection, kernel launch behavior, compiler mode, or CUDA graphs—not ledger optimization.

### Direct and persistent are not equivalent

Do not interpret the timing decomposition.

The runner reuses the same structural/output equivalence gate as the normalized comparison and exits nonzero on equivalence failure.

## Important limitation

For CUDA, selected-worker event time and CPU wall time are different clocks. The residual is therefore a diagnostic approximation of host/organization overhead, not a mathematically exact exclusive CPU profile.

If the first real result makes that distinction decision-relevant, deeper profiling can use explicit CPU phase spans and/or a system profiler.

Do not build that instrumentation before the coarse result requires it.

## Non-goals

This diagnostic does not establish:

- an adaptive routing advantage;
- the best persistent step width;
- the best world batch size;
- detailed scheduler versus ledger versus projector percentages;
- compiler benefit;
- a Gate 5 result.

It exists to decide what should be profiled next, if anything.
