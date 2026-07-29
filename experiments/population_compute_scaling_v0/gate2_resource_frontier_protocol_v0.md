# Gate 2 persistent-state target-GPU resource frontier protocol v0

Status: **FROZEN BEFORE GATE-2 CONFIRMATION EXPOSURE OR RESOURCE TIMING**

Purpose:

> Measure whether the confirmed stable-persistent Gate-2 computation has a useful practical execution frontier on the local RTX 4060 Ti when simultaneous state-slot updates are executed in parallel rather than time-multiplexed serially.

This is a systems/resource experiment. It must not alter the capability protocol or be interpreted as additional neural capability.

## 1. Checkpoint selection frozen in advance

If and only if the Gate-2 capability confirmation protocol passes, use the **confirmation training seed 3 checkpoint** for the primary resource measurement.

Seed 3 is selected now, before confirmation results, so no fastest/best checkpoint may be chosen post hoc.

Seeds 4 and 5 remain available for descriptive replication if needed, but the primary resource pass/fail rule is bound to seed 3.

## 2. Target hardware/environment

Primary target:

- local NVIDIA GeForce RTX 4060 Ti 16 GB;
- same local machine used for Gate-1/Gate-2 development;
- eager PyTorch CUDA execution;
- machine intentionally idle except normal OS/background services;
- no Factorio, games, browsers with heavy GPU workloads, local model inference or unrelated CUDA processes during admitted timing.

Capture:

- Git HEAD / clean-tree state;
- PyTorch version;
- CUDA runtime;
- GPU name;
- `nvidia-smi` before and after timing;
- execution-mode metadata.

Compiler/graph execution is a separate future systems variable and is excluded from this baseline protocol.

## 3. Computation under comparison

Use stable-persistent Gate-2 worlds only.

Compare the existing schedules:

1. `parallel_persistent_forward` — updates all independent runtime state slots simultaneously within each collision lane;
2. `serial_persistent_forward` — time-multiplexes the exact same persistent state bank slot-by-slot.

Both schedules must retain:

- identical checkpoint;
- identical observations/worlds;
- identical routing;
- identical persistent state-bank width;
- identical `8 × C` learned recurrent updates per sample;
- identical final query/readout semantics.

The schedules differ only in execution span / simultaneous update width.

## 4. Frozen resource matrix

Entity tiers:

- `C=64` with `W=1 / 4 / 16 / 64`;
- `C=256` with `W=1 / 4 / 16 / 64 / 256`.

Batch sizes:

- `B=1`;
- `B=64`.

This produces 18 width × batch cells, each measured under both schedules.

C16 is omitted from the primary resource matrix because Gate 2's decision-relevant organization effect is defined at C64/C256 and the target-hardware question is the practical frontier at the larger workloads.

## 5. Frozen timing corpus

Generate a separate deterministic **resource-only** world corpus that is not used for capability acceptance.

Resource timing worlds are allowed to reuse the frozen world generator but must not alter/train/evaluate the capability checkpoint.

Use a fixed public timing seed domain documented by the runner and preserve exact world seeds in the result artifact.

Timing data cannot become training/confirmation feedback.

## 6. Correctness preflight before admitted timing

For every matrix cell before timing:

- run both schedules on the same preflight worlds;
- require exact decoded 4-bit payload identity for every sample;
- require equal learned-update telemetry;
- require equal persistent-state-bank size;
- record FP32 maximum absolute logits/final-state drift descriptively.

Existing structural regression already requires CPU tensor agreement at `rtol=2e-5, atol=2e-5` plus exact decoded identity. On target CUDA, the admitted resource criterion is **exact decoded identity**, not a post hoc floating-point tolerance chosen after observation.

If any decoded output differs, that cell is inadmissible for timing and the resource protocol fails mechanically until a new version is justified. Do not loosen correctness after seeing timing.

## 7. Warmup and trials

For each schedule/cell:

- warmup iterations: `10`;
- admitted timed iterations: `50`;
- use CUDA events for device elapsed time;
- synchronize before/after each admitted measurement boundary;
- do not include world generation, checkpoint loading or tensor construction in neural forward timing;
- tensor construction may be measured separately as descriptive orchestration cost.

Alternate or deterministically interleave schedule order to avoid measuring all parallel cells at one thermal state and all serial cells at another.

## 8. Reported metrics

Per schedule/cell report:

- median CUDA-event forward latency;
- p25/p75 latency;
- descriptive min/max;
- samples/second derived from median latency;
- peak CUDA allocated memory;
- peak CUDA reserved memory;
- learned recurrent updates/sample;
- peak simultaneous updates/sample;
- persistent state vectors/sample;
- collision load;
- decoded correctness identity;
- FP32 numerical drift summaries.

Derived comparison:

`serial median latency / parallel median latency`

for every cell.

## 9. Frozen practical resource pass rule

The Gate-2 resource half passes only if all of the following hold on the preregistered seed-3 checkpoint:

1. every primary matrix cell passes decoded schedule-identity preflight;
2. neither schedule has an OOM/unrecoverable memory failure in the primary matrix;
3. at C64/W64, parallel median latency is lower than serial at both B1 and B64;
4. at C256/W256, parallel median latency is lower than serial at both B1 and B64.

Thus all **4 / 4 decision-relevant largest-width endpoint cells** must favor parallel execution.

No minimum numerical speedup factor is added after timing. The measured magnitude is reported descriptively.

Width-1 is expected to provide little/no population parallelism and is not itself a pass/fail endpoint.

## 10. Final Gate-2 positive rule

Gate 2 becomes positive only if both independently frozen halves pass:

### Capability half

`gate2_confirmation_protocol_v0.md`

requires all 12 primary seed × comparison confirmation CIs to have `CI_low > 0` with all mechanics/information/work invariants intact.

### Resource half

This protocol requires exact decoded parallel/serial correctness and parallel latency advantage in all four decision-relevant largest-width endpoint cells without unacceptable memory failure.

Failure of either half means Gate 2 is not positive under v0.

## 11. Environmental contamination rule

Any admitted timing run with known concurrent game/GPU-heavy workload is invalid for resource claims and must be preserved as contaminated/non-admitted evidence rather than silently overwritten.

Capability prediction artifacts generated under such a condition may remain scientifically useful if their semantics/provenance are intact, but their timing/power/utilization is not admitted.

## 12. No optimization before baseline

Do not add before the v0 measurement:

- `torch.compile`;
- CUDA graphs;
- custom kernels;
- manual GRU fusion;
- altered tensor layouts solely for speed;
- reduced numerical precision;
- a different model architecture.

After the eager baseline is recorded, compiler/execution optimization may become a separate experiment using the exact same confirmed function.
