# Gate-3 v0 — Hypothesis-population breadth under fixed learned work

## Status

**FROZEN DEVELOPMENT PROTOCOL — NO RESULT INSPECTED**

This document freezes the first Gate-3 experiment before any Gate-3 development result is inspected.

Gate-3 is unlocked only because Gate-2 v0 completed positive under its frozen capability and eager-CUDA resource protocols.

## Research question

> With learned parameters, observed world information, and total learned recurrent-update count fixed within each difficulty tier, can allocating the same learned work across a larger population of persistent hypothesis states improve one-pass problem solving by preserving more live possibilities until delayed disambiguating evidence arrives?

This is deliberately different from Gate 2.

Gate 2 tested whether more persistent states reduce interference while many independent facts must be retained.

Gate 3 tests whether more persistent states improve **hypothesis breadth**: the system must make provisional branch choices before it receives the information that determines which branch was correct.

## Core causal variable

The experimental variable is the maximum number of simultaneously retained neural hypothesis states `W`.

Within each difficulty tier, all widths receive:

- the same learned parameter set;
- the same world and observation sequence;
- the same unique world information;
- the same total number of learned recurrent updates;
- the same branching factor;
- the same deterministic beam-management algorithm;
- the same scoring/readout function;
- the same final answer rule.

What changes is how the fixed learned-update budget is distributed:

- small `W` = few hypotheses, many recurrent refinement updates per retained branch;
- large `W` = many hypotheses, fewer recurrent refinement updates per branch.

This is a breadth-versus-depth allocation test at fixed learned work.

## Workload: delayed binary hypothesis tree

Each world contains a uniformly sampled hidden binary path of depth `D`.

Difficulty tiers:

| Tier | Hidden depth `D` | Hypothesis space | Widths |
|---|---:|---:|---|
| H4 | 4 | 16 | 1, 4, 16 |
| H6 | 6 | 64 | 1, 4, 16, 64 |
| H8 | 8 | 256 | 1, 4, 16, 64, 256 |

The answer is the complete hidden path.

### Early evidence

During the first `D` phases, one noisy hint is emitted for the corresponding hidden bit.

Frozen hint reliability:

`P(hint == hidden_bit) = 0.70`

Otherwise the hint is flipped.

The hint noise is sampled independently by phase from the world seed.

The hidden bit is **not** revealed during the branching phase.

### Delayed disambiguation

After all `D` branch decisions are complete, the world emits `D` exact reveal observations, one per hidden bit in original path order.

Thus the full evidence is sufficient to identify the hidden path exactly, but the exact evidence arrives only after the runtime has already decided which hypotheses to retain.

A discarded hypothesis cannot be reconstructed or replayed later.

No observation replay is allowed.

## Runtime hypothesis representation

All hypothesis states use the same learned neural function and shared learned parameters.

A hypothesis consists of:

- one persistent neural state vector;
- parameter-free candidate-path metadata used only by the deterministic beam manager and final answer extraction;
- one scalar neural score produced by the shared scorer.

The deterministic runtime may:

- clone a parent state when branching;
- append a branch bit to candidate metadata;
- sort candidates by neural score;
- apply a deterministic answer-independent tie break;
- retain the top `W` candidates;
- return the candidate metadata associated with the final highest-scoring neural state.

The deterministic runtime may **not**:

- compare candidate metadata directly with hidden bits or reveal values;
- compute an oracle likelihood;
- restore a pruned hypothesis;
- store a second semantic copy of a discarded neural state;
- replay prior world observations;
- inspect any information unavailable to the neural scorer.

Candidate metadata is routing identity, not a second inference engine.

## One-pass phase schedule

For depth `D`, the run contains exactly `2D` phases:

1. `D` branching phases with noisy hints;
2. `D` reveal-only phases with exact delayed evidence.

During branching phase `t`:

1. each retained hypothesis is cloned into branch-0 and branch-1 children;
2. each child receives the current noisy hint plus its proposed branch-action token;
3. the shared recurrent neural function updates/scorers each child;
4. the deterministic beam manager keeps the highest-scoring `W` children.

During reveal phase `t`:

1. no new branch is created;
2. every retained hypothesis receives the exact reveal observation;
3. the shared recurrent neural function updates/scorers the states;
4. all retained hypotheses remain available unless the control mode specifies otherwise.

After the final reveal phase, the highest-scoring retained hypothesis is returned as the answer.

## Fixed learned-update budget

For difficulty depth `D`, define the frozen per-phase learned-update budget:

`B(D) = 2^D`

Therefore:

- H4: `B = 16` learned recurrent updates per phase;
- H6: `B = 64`;
- H8: `B = 256`.

Each world has exactly `2D` phases, so the frozen learned-update totals are:

| Tier | Updates / phase | Phases | Total learned updates / world |
|---|---:|---:|---:|
| H4 | 16 | 8 | 128 |
| H6 | 64 | 12 | 768 |
| H8 | 256 | 16 | 4,096 |

These totals are identical across all widths and control modes within a tier.

### Budget allocation during branching

Let `E` be the number of child states produced before pruning in the current branch phase.

Because all frozen widths and hypothesis-space sizes are powers of two, `B(D) / E` is an integer.

Each child receives exactly:

`R = B(D) / E`

recurrent updates using the current branch-phase input.

Thus narrow populations receive more recurrent refinement per candidate; wide populations evaluate more distinct live candidates.

The sum of learned recurrent updates in the phase remains exactly `B(D)`.

### Budget allocation during reveal-only phases

Let `A` be the number of retained hypothesis states.

Each retained state receives exactly:

`R = B(D) / A`

recurrent updates on the current reveal observation.

Again the phase total is exactly `B(D)`.

No dummy learned updates are permitted.

## Information accounting

Unique world information is fixed within each tier.

For depth `D`, every width receives exactly:

- `D` noisy branch hints;
- `D` exact delayed reveals;
- the same phase ordering;
- the same hidden-path distribution;
- the same hint reliability.

Replicating an observation across multiple active hypotheses does not create new world information; it consumes the fixed learned-update budget.

The result artifacts must report both:

- unique world observations inspected;
- learned recurrent updates executed.

## Model boundary

Gate-3 v0 uses one shared recurrent hypothesis scorer.

Frozen architectural target:

- one shared GRU-style recurrent state update;
- persistent state width: 64;
- one scalar neural hypothesis score;
- branch/reveal/noisy-hint/phase encodings only;
- no width embedding;
- no learned per-slot identity embedding;
- no learned parameters that scale with `W`;
- no attention across the hypothesis population in v0;
- no compiler/CUDA-graph/fusion dependency in the capability protocol.

Exact learned parameter count is recorded from the implementation and must remain identical across all widths and controls for a checkpoint.

## Training semantics

Training is scorer training, not width-specific policy tuning.

The shared neural scorer is trained on stable, correctly associated hypothesis trajectories only.

Training examples contain:

- hidden paths from the training world domain;
- noisy hints generated by the frozen world process;
- candidate branch trajectories;
- delayed exact reveals;
- supervised candidate-consistency/ranking targets.

Training must not use the confirmation world domain.

Control modes are evaluation-only.

The final implementation must freeze the exact optimizer, loss, training-example sampler, training-step count and model parameterization before the first admitted Gate-3 development result.

## Runtime modes

### 1. Stable diverse beam — primary treatment

Each retained candidate keeps its own persistent neural state and candidate-path identity across phases.

This is the intended hypothesis-population runtime.

### 2. Collapsed diversity control

Same maximum width, state-bank allocation, world observations and learned-update budget.

After each branch-phase prune, all retained slots are replaced by copies of the current top-scoring hypothesis state and the same candidate identity.

Subsequent branching therefore spends the same learned work without preserving a diverse set of hypotheses.

This tests whether any width benefit depends on maintaining distinct live possibilities rather than merely having a larger state bank or more parallel tensor lanes.

At `W=1`, stable and collapsed are structurally identical.

### 3. Reshuffled hypothesis-continuity control

Same maximum width, candidate count, world observations and learned-update budget.

After each phase, persistent neural states are deterministically permuted among retained candidate identities using an answer-independent permutation derived from the world seed and phase index.

Candidate identities remain, but their neural histories no longer consistently correspond to the same hypothesis.

This tests whether persistent neural continuity of each live hypothesis contributes to the population effect.

At `W=1`, stable and reshuffled are structurally identical.

## Deterministic tie breaking

Beam-score ties use a stable deterministic ordering derived only from:

- world seed;
- phase index;
- candidate-path metadata.

The tie break must be independent of the hidden correct answer except through candidate identity itself and must never inspect reveal values outside the neural scorer.

The same rule is used for every width and mode.

## Evaluation domains

World domains must be non-overlapping and deterministic.

Frozen domain allocation:

- training worlds: seeds starting at `0`;
- development worlds: seeds starting at `2^30`;
- confirmation worlds: seeds starting at `2^31`.

The confirmation domain remains closed during Gate-3 development.

The development runner must not expose an option that opens confirmation.

## Development matrix

Evaluate all valid `(D, W)` cells for:

- stable diverse beam;
- collapsed diversity;
- reshuffled continuity.

This produces:

- H4: 3 widths × 3 modes = 9 cells;
- H6: 4 widths × 3 modes = 12 cells;
- H8: 5 widths × 3 modes = 15 cells;

Total: **36 evaluation cells per checkpoint**.

For every cell record at minimum:

- exact hidden-path solve rate;
- per-bit accuracy;
- world seeds;
- solved-by-world vector;
- hidden depth;
- maximum width;
- actual retained population by phase;
- unique candidate count by phase;
- learned updates by phase and total;
- unique world observations inspected;
- learned parameter count;
- checkpoint fingerprint;
- mode;
- hint reliability;
- candidate-survival diagnostics for the correct path.

## Paired statistics

All primary comparisons are paired on identical worlds.

Use paired bootstrap intervals over exact solve outcome differences.

The development bootstrap count is frozen at 2,000 unless implementation qualification discovers a purely mechanical defect before any development result is admitted.

## Primary Gate-3 capability comparisons

The following comparisons define the intended breadth mechanism:

1. H6 stable `W64 > W1`;
2. H8 stable `W256 > W1`;
3. H8 stable `W256 > W64`;
4. H8 `stable W256 > collapsed W256`;
5. H8 `stable W256 > reshuffled W256`.

The third comparison is important: Gate 3 is not satisfied merely by discovering that a tiny amount of beam width helps. The largest frozen population must provide additional held-out capability beyond the intermediate `W64` population on H8.

## Structural identities

The implementation must prove on paired worlds:

- `W1 stable == W1 collapsed` exactly;
- `W1 stable == W1 reshuffled` exactly;
- learned parameter count identical across all widths/modes for a checkpoint;
- unique world observations identical across widths/modes within a tier;
- total learned recurrent updates identical across widths/modes within a tier;
- no candidate can be restored after pruning;
- no control receives extra observations or learned work.

Any failure of these identities invalidates the corresponding result artifact.

## Development interpretation map

Gate-3 development uses one checkpoint/seed at a time and cannot assign a gate verdict.

### Outcome A — no breadth effect

Largest-width stable performance does not exceed W1 or W64 on H8.

Interpretation: under this workload and learned scorer, distributing fixed learned work across more simultaneous hypotheses does not improve capability.

### Outcome B — width helps, diversity controls do not separate

Stable width increases capability, but collapsed or reshuffled controls match the stable population.

Interpretation: the observed effect is not specifically attributable to persistent diverse hypothesis population. Diagnose execution/ranking/confounds before confirmation.

### Outcome C — diversity/continuity matter, but width saturates early

Stable population beats collapsed/reshuffled and beats W1, but H8 W256 does not improve over W64.

Interpretation: population breadth matters, but the measured useful population frontier saturates by W64. This is informative but not the intended Gate-3 positive pattern.

### Outcome D — clean Gate-3 directional pattern

All five primary directions are positive:

- H6 W64 > W1;
- H8 W256 > W1;
- H8 W256 > W64;
- H8 W256 stable > collapsed;
- H8 W256 stable > reshuffled.

This is development evidence only. It does not open confirmation automatically.

### Outcome E — structural/mechanical failure

Any fixed-information, fixed-work, width-1 identity, provenance or evaluation-domain invariant fails.

Interpretation: no scientific capability result is admitted until the mechanics are repaired and the protocol is re-frozen before new data exposure.

## Confirmation boundary

No Gate-3 confirmation world may be inspected until:

1. the implementation and development recipe are frozen;
2. at least one development checkpoint has been analyzed;
3. any robustness-seed policy is frozen before those additional seeds are run;
4. the final confirmation training seeds, world count and exact positive rule are frozen before confirmation exposure.

Development evidence alone cannot produce a positive Gate-3 verdict.

## Resource boundary

Gate-3 v0 capability development is logically separate from compiler/runtime optimization.

If capability evidence justifies confirmation, a Gate-3 target-GPU resource protocol may then be frozen separately.

Compiler, CUDA graphs, fusion and other execution optimizations remain independent experimental variables and may not be introduced retroactively to rescue a capability result.

## What a future positive Gate-3 result would support

A correctly confirmed positive result would support a narrow claim:

> On the frozen delayed-hypothesis workload, a larger persistent neural hypothesis population can use the same learned parameters, the same world information and the same total learned recurrent-update budget more effectively by preserving more live possibilities until delayed evidence resolves them.

It would not by itself establish general intelligence, arbitrary search superiority, asymptotic scaling, or usefulness on natural-language/code tasks.

## Freeze rule

Any change to the workload semantics, hint reliability, difficulty tiers, width ladder, work-budget formula, control definitions, evaluation domains or primary comparisons after the first Gate-3 development result is inspected creates a new protocol version.
