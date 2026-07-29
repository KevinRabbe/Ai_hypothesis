# Gate 2 persistent-state capacity confirmation protocol v0

Status: **FROZEN BEFORE ANY GATE-2 CONFIRMATION TRAINING OR WORLD EXPOSURE**

Protocol version:

`gate2-persistent-state-confirmation-v0`

Scientific question:

> With learned parameters, inspected information and total learned recurrent-update count held fixed, does increasing the population of persistent runtime neural states reproducibly improve delayed keyed-trace capability by reducing state interference, with stable locality and persistence causally required?

This protocol is frozen after three development training seeds (`0 / 1 / 2`) independently reproduced the same directional pattern. Those development seeds and development worlds are permanently excluded from confirmation.

## 1. Confirmation training seeds

Use exactly:

`3 / 4 / 5`

Each seed trains one independent checkpoint from initialization.

No checkpoint selection, restart selection, best-of-N selection or seed replacement is allowed.

If a run fails for an environmental/mechanical reason, preserve the failed attempt and rerun the **same seed** only after the reason is documented. A capability-poor completed seed may not be replaced.

## 2. Frozen model and training recipe

The confirmation recipe is exactly the seed-robustness development recipe:

- shared persistent-state model;
- state width: `64`;
- query width: `24`;
- learned parameter count expected from this architecture: `21,580`;
- optimizer: AdamW;
- learning rate: `3e-4`;
- weight decay: `1e-4`;
- gradient clipping norm: `1.0`;
- optimizer steps: `1,000`;
- training batch size: `32`;
- training conditions: the same deterministic 12 stable-persistent `entity_count × width` cycle;
- reshuffled/reset controls never enter training;
- training worlds remain generated exclusively from the reserved training seed domain.

No curriculum, architecture, optimizer or hyperparameter tuning is allowed after confirmation begins.

## 3. Frozen confirmation world split

Use the existing reserved confirmation world domain beginning at:

`2^31`

For every entity tier, evaluate the first exactly:

`512`

confirmation worlds from that reserved domain.

The same ordered 512 worlds for a given entity count must be reused across every width and every control for every confirmation checkpoint.

Development worlds beginning at `2^30` are not confirmation evidence.

## 4. Frozen evaluation matrix

Evaluate the complete existing 36-cell matrix.

Entity counts:

- `16`;
- `64`;
- `256`.

Widths:

- C16: `1 / 4 / 16`;
- C64: `1 / 4 / 16 / 64`;
- C256: `1 / 4 / 16 / 64 / 256`.

Controls at every width:

- stable persistent;
- reshuffled locality;
- reset state.

Evaluation batch size:

`64`

Paired bootstrap samples:

`2,000`

The existing deterministic bootstrap seed derivation and percentile procedure remain unchanged.

## 5. Frozen information/work invariants

Every confirmation cell must prove:

- exactly one unchanged checkpoint per training seed across all widths/controls;
- learned parameter count/fingerprint unchanged throughout evaluation;
- identical ordered confirmation worlds within one entity tier;
- every entity inspected at every width;
- exactly `8 × C` inspected observations per world;
- exactly `8 × C` learned recurrent updates per world;
- width only changes runtime-state collision load / simultaneous organization, not learned parameter count or information exposure.

Any invariant failure makes that run mechanically inadmissible rather than scientifically negative.

## 6. Exact identity control

For every confirmation training seed and every entity count:

`width 1 stable persistent == width 1 reshuffled locality`

must remain exact at the decoded world-outcome level.

The corresponding paired summary must have:

- exact-solve delta `0`;
- treatment-only `0`;
- reference-only `0`;
- bootstrap CI `[0,0]`.

Failure is a mechanics failure.

## 7. Four primary confirmation comparisons

For **each** confirmation training seed independently, require all four:

1. C64/W64 stable persistent > C64/W1 stable persistent;
2. C256/W256 stable persistent > C256/W1 stable persistent;
3. C256/W256 stable persistent > C256/W256 reshuffled locality;
4. C256/W256 stable persistent > C256/W256 reset state.

The acceptance statistic is paired exact-solve delta on the identical ordered confirmation worlds.

For each comparison, the deterministic paired 95% bootstrap confidence interval must satisfy:

`bootstrap_ci_low > 0`

No minimum absolute effect-size threshold is added after development.

## 8. Gate-2 capability confirmation rule

Capability confirmation passes only if:

- all mechanical/information/work invariants pass;
- all width-1 identity controls pass;
- every one of the four primary comparisons has `CI_low > 0` on seed 3;
- every one of the four primary comparisons has `CI_low > 0` on seed 4;
- every one of the four primary comparisons has `CI_low > 0` on seed 5.

This is an intersection rule: **12 / 12 primary seed × comparison tests must pass.**

There is no pooling across checkpoints and no majority-vote rule.

A single completed confirmation seed with a primary CI touching/crossing zero means the frozen capability-confirmation rule fails.

The result may still be scientifically informative, but the protocol may not be relaxed post hoc.

## 9. Secondary descriptive outputs

Report, but do not use to alter the frozen pass rule:

- full stable width curves;
- bit accuracy;
- largest-width stable/reshuffled/reset exact solve;
- collision load;
- treatment-only/reference-only discordances;
- descriptive across-seed means/ranges;
- training batch-loss summaries.

Random exact solve for a 4-bit payload remains `1/16 = 0.0625`; random bit accuracy remains `0.5`.

These chance references are descriptive rather than separate acceptance thresholds.

## 10. Machine/environment rule

Capability confirmation runs must be performed on the local RTX 4060 Ti machine with the machine intentionally idle except for normal OS/background services required for execution.

Before each run:

- close Factorio and other games;
- close GPU-heavy applications;
- capture `nvidia-smi` provenance;
- record Git HEAD and clean-tree state;
- preserve runtime/PyTorch/CUDA/device provenance.

Capability results are not timing results, but this rule removes avoidable environmental ambiguity before the same checkpoints are used for the separately frozen resource measurement.

## 11. Progress reporting

Human-visible progress reporting may be added as an **observational-only** execution feature before confirmation runs.

It may report:

- training step / 1,000;
- current training C/W condition;
- latest already-computed batch loss;
- evaluation cell / 36;
- current C/W/control;
- wall-clock elapsed time.

Progress instrumentation must:

- not consume random numbers used by training/evaluation/bootstrap;
- not alter tensor values, update ordering, batching or optimizer state;
- not be used for timing/resource claims;
- be regression-tested for checkpoint/output identity with instrumentation disabled/enabled on a small deterministic fixture.

## 12. Confirmation execution boundary

No confirmation run may start until:

1. this protocol exists on a committed branch;
2. the confirmation runner/result contract is implemented;
3. focused tests/CI qualify the exact protocol head;
4. the target-GPU resource protocol is separately frozen;
5. no confirmation world has been inspected manually or through ad-hoc scripts.

## 13. Capability is necessary but not sufficient for final Gate 2

Passing this confirmation protocol establishes the **capability/causal** half of Gate 2 only.

Final Gate-2 positive status additionally requires the separately frozen target-GPU resource test to show that parallel persistent execution has a useful practical frontier versus mathematically equivalent serial persistent execution without unacceptable memory failure.

No minimum resource speedup is invented after measurement.

## 14. Failure interpretation

A frozen confirmation failure does not invalidate Gate 0 or Gate 1.

It means the Gate-2 development pattern failed the precommitted untouched replication rule.

Do not:

- replace failed seeds;
- remove difficult confirmation worlds;
- reduce the number of required comparisons;
- change CI procedure;
- tune architecture/training on confirmation data;
- move confirmation worlds into development.

Any new recipe after a failed confirmation must become a new version with a new untouched confirmation boundary if scientifically justified.
