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
- equal information/work identities.

## Robustness outcome

The seed-robustness check is considered **directionally replicated** only if seeds 0, 1 and 2 each show all four primary comparisons in the same positive direction:

1. `C64/W64 stable > C64/W1`;
2. `C256/W256 stable > C256/W1`;
3. `C256/W256 stable > reshuffled locality`;
4. `C256/W256 stable > reset state`.

This development robustness check does **not** itself require every seed-level bootstrap interval to exclude zero; those intervals are descriptive development diagnostics and the sample is only 256 worlds/tier. However, a sign reversal in any primary comparison means the seed-0 pattern is not robust enough to freeze confirmation without further development diagnosis.

After all three seeds exist, report the per-seed effects and a simple across-seed summary. Do not pool the three checkpoints as though they were one larger confirmation sample.

## Next branch if replicated

If all three development seeds preserve the same causal direction:

1. freeze the final Gate-2 architecture/training/evaluation recipe;
2. freeze untouched confirmation training seeds that are **not 0, 1 or 2**;
3. freeze confirmation world counts and acceptance rule;
4. freeze the target-GPU parallel-vs-serial persistent resource protocol;
5. only then open confirmation.

## Next branch if not replicated

Remain in development.

Diagnose seed sensitivity before tuning, and record any recipe change as a new development version. Do not silently modify the seed-0 protocol and call it the same experiment.

## Scientific boundary

This plan exists to prevent favorable seed-0 evidence from immediately becoming a tuned confirmation protocol.

No Gate-2 verdict is assigned here.
