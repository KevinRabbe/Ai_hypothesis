# Gate 2 development seed-robustness plan v0

Status: **FROZEN AFTER SEED-0 OUTCOME-D DEVELOPMENT RESULT; CONFIRMATION REMAINS CLOSED**

Purpose: determine whether the first clean directional Gate-2 development pattern is robust to independent training initialization **without changing the recipe after seeing seed 0**.

## Fixed development recipe

Repeat exactly the seed-0 development runner configuration:

- measured code head: `06f359b2bc26bf3130552c0272d89f493abce636`;
- training seeds: `1` and `2`;
- steps: `1,000`;
- training batch size: `32`;
- evaluation world count: `256` per entity tier;
- evaluation batch size: `64`;
- paired bootstrap samples: `2,000`;
- device: CUDA;
- optimizer/model/training-condition cycle unchanged from the measured seed-0 head;
- evaluation split: development only;
- confirmation remains locked.

## No-tuning rule

Before seed 1 and seed 2 are captured, do not change:

- model architecture;
- state width;
- query width;
- optimizer;
- learning rate;
- weight decay;
- gradient clipping;
- training condition order;
- number of steps;
- training batch size;
- development world generator/split;
- evaluation matrix;
- evaluation world count;
- evaluation batch size;
- bootstrap method/sample count;
- stable/reshuffled/reset semantics.

The purpose is robustness measurement, not optimization.

## Concurrent-load handling

A user-reported Factorio session may have overlapped one or more local development runs.

This does **not** alter the benchmark worlds, learned-update count, inspected information, checkpoint parameter count, or evaluation matrix. The Gate-2 development result is a capability result, not a timing/resource result.

However, a concurrent foreground workload can change:

- GPU/CPU scheduling;
- clocks/thermal state;
- available VRAM/system memory;
- wall-clock duration;
- the exact floating-point path when underlying CUDA operations are not guaranteed bitwise deterministic.

Therefore:

1. do not interrupt a development run solely because Factorio is active;
2. record any run with possible overlap as `CONCURRENT_LOAD_POSSIBLE`;
3. do not use such a run for target-GPU latency/throughput/resource claims;
4. retain it as development capability evidence if all mechanics/provenance checks pass and the run completes without CUDA/OOM errors;
5. before final recipe freeze, repeat any robustness seed whose clean-room status is uncertain under an otherwise idle machine if that seed is decision-relevant;
6. confirmation/resource-frontier runs must use an explicitly controlled idle-target environment and record that boundary before execution.

A clean rerun must be treated as a separate artifact, not as a silent replacement. Preserve both outcomes if they differ.

## Artifact isolation

Use distinct output roots:

- `results/population_compute_scaling_v0/gate2_persistent_state_capacity_v0/development_seed_1`
- `results/population_compute_scaling_v0/gate2_persistent_state_capacity_v0/development_seed_2`

Each run must preserve its own:

- checkpoint;
- checkpoint SHA-256;
- result JSON;
- result SHA-256;
- run config;
- Git head/status;
- runtime/GPU provenance;
- result manifest;
- external ZIP + ZIP SHA-256 when archived.

Do not overwrite seed 0.

## Analysis

Run the same read-only analyzer from `agent/gate2-result-analysis-v0` against each result.

For every seed record:

- stable width curves;
- largest-width control performance;
- primary paired comparisons;
- width-1 stable/reshuffled identity;
- parameter count/fingerprint consistency;
- equal information/work identities;
- concurrent-load status if known.

## Robustness outcome

The seed-robustness check is considered **directionally replicated** only if seeds 0, 1 and 2 each show all four primary comparisons in the same positive direction:

1. `C64/W64 stable > C64/W1`;
2. `C256/W256 stable > C256/W1`;
3. `C256/W256 stable > reshuffled locality`;
4. `C256/W256 stable > reset state`.

This development robustness check does **not** itself require every seed-level bootstrap interval to exclude zero; those intervals are descriptive development diagnostics and the sample is only 256 worlds/tier. However, a sign reversal in any primary comparison means the seed-0 pattern is not robust enough to freeze confirmation without further development diagnosis.

After all three seeds exist, report the per-seed effects and a simple across-seed summary. Do not pool the three checkpoints as though they were one larger confirmation sample.

If a decision-relevant seed carries `CONCURRENT_LOAD_POSSIBLE`, run one explicitly idle-machine clean repeat before final recipe freeze. Do not overwrite or discard the original run.

## Next branch if replicated

If all three development seeds preserve the same causal direction:

1. resolve any decision-relevant concurrent-load uncertainty with preserved clean reruns;
2. freeze the final Gate-2 architecture/training/evaluation recipe;
3. freeze untouched confirmation training seeds that are **not 0, 1 or 2**;
4. freeze confirmation world counts and acceptance rule;
5. freeze the target-GPU parallel-vs-serial persistent resource protocol;
6. only then open confirmation.

## Next branch if not replicated

Remain in development.

Diagnose seed sensitivity before tuning, and record any recipe change as a new development version. Do not silently modify the seed-0 protocol and call it the same experiment.

## Scientific boundary

This plan exists to prevent favorable seed-0 evidence from immediately becoming a tuned confirmation protocol.

No Gate-2 verdict is assigned here.
