# Gate-3 v2 — ambiguity-driven latent-frontier scaling protocol

## Status

**FROZEN PRE-RESULT PROTOCOL — NO GATE-3 v2 FRONTIER RESULT INSPECTED**

Gate-3 v2 starts from the fully qualified Gate-3 v1 three-seed robustness result:

`R1_ROBUST_LATENT_POPULATION_MECHANISM`

The v1 result is preserved unchanged. v2 is a separately versioned experiment and cannot retroactively alter v1.

## Motivation

Gate-3 v1 established across three independently trained checkpoints that larger dormant persistent-hypothesis populations strongly improve no-replay exact search coverage when:

- learned parameters are fixed;
- active neural width is fixed at two child lanes;
- every evaluated child receives eight recurrent updates;
- total learned work is fixed within a condition;
- runtime cannot inspect the hidden answer;
- distinct hypothesis diversity and candidate-specific neural continuity are preserved.

The v1 useful frontier saturated by `L64`: `L256 - L64 = 0` on all 256 paired worlds for all three checkpoints.

The next question is therefore narrower:

> If the search evidence is made more ambiguous while the scorer, depth, active neural width, per-child refinement, search budget and learned parameters remain frozen, does useful dormant population extend beyond L64?

## Why ambiguity, not greater depth

The frozen v1 scorer encodes depths `6/8/10` using a fixed 19-feature input representation. Increasing depth would require changing the learned input representation or introducing an adapter, which would confound a pure population-frontier test.

v2 therefore keeps depth `10` and changes only the statistical ambiguity of the public noisy hints.

For any reliability above `0.5`, the ideal ordering of paths by number of hint agreements is unchanged; lowering reliability makes the hidden path less likely to remain near the top of that ordering and therefore increases the value of preserving alternative hypotheses without requiring a new scorer architecture.

## Frozen checkpoints

No training occurs in Gate-3 v2.

Exactly the three already-measured Gate-3 v1 checkpoints are reused and must be SHA-256 verified before execution:

### checkpoint 0

- source: Gate-3 v1 development seed 0
- learned parameters: `19,649`
- checkpoint SHA-256: `e63748a1182472d03c540f9123c3442ce44e130519e2176681034568826db590`
- parameter fingerprint: `e4f9990f08e85393a155637cfa50b5831d779770447ec3716fc9e67946992afc`

### checkpoint 1

- source: Gate-3 v1 robustness seed 1
- learned parameters: `19,649`
- checkpoint SHA-256: `8ce384627f5543fa4fe78498f9990f8214cf2a4afed0c9b734e86761ad13c989`
- parameter fingerprint: `2a57984f755ee1404fd828c2df36a5efff8f13a541b7d7c93a891014fbf4897c`

### checkpoint 2

- source: Gate-3 v1 robustness seed 2
- learned parameters: `19,649`
- checkpoint SHA-256: `103826bbd9451b965eced5134d1674cb8e893f5d3b378cb828312ffdb6fc9a37`
- parameter fingerprint: `8afaf956f200f41ea914eafdd1b5f151dd303cae1552165c74e12bb8c945af02`

No checkpoint may be retrained, fine-tuned, recalibrated or replaced under this protocol.

## Frozen workload

Depth remains exactly `10`.

Hidden solution space:

`2^10 = 1,024` terminal paths.

Search budget remains exactly:

`256` search rounds/world.

Each round still uses:

```text
2 active child lanes
x 8 recurrent updates / child
= 16 learned recurrent updates / round
```

Therefore every condition uses exactly:

`256 x 16 = 4,096 learned recurrent updates/world`.

This total is identical across capacities, controls, ambiguity tiers and checkpoints.

## Frozen ambiguity tiers

Two development-only evidence tiers are preregistered:

- `A60`: hint reliability `0.60`
- `A55`: hint reliability `0.55`

The v1 `0.70` distribution is not rerun as a v2 result; it already exists as the preserved v1 baseline.

The two new tiers are fixed now to avoid selecting a noise level after observing frontier results.

## New world namespace

Gate-3 v2 must not consume the untouched Gate-3 v1 confirmation domain.

v2 therefore uses a new deterministic namespace rather than the v1 numeric world-seed split.

Development worlds are generated from labels beginning with:

```text
gate3-v2-frontier-development-hidden
gate3-v2-frontier-development-hints
gate3-v2-frontier-development-runtime
```

Confirmation, if a later protocol ever opens it, must use a distinct namespace beginning with:

```text
gate3-v2-frontier-confirmation-...
```

The confirmation namespace remains mechanically unreachable during development.

Exactly `256` development worlds are used per ambiguity tier.

Within each tier, the exact same 256 worlds are reused across all capacities, controls and checkpoints for paired comparisons.

Across `A60` and `A55`, the hidden path for a given world index is held fixed while hint corruption is generated deterministically for that tier. This makes difficulty changes interpretable without exposing hidden answers to runtime.

## Runtime-facing information

Runtime receives only:

- depth `10`;
- the ten public noisy hints;
- an answer-independent deterministic runtime/tie-break seed.

Runtime never receives:

- the hidden solution;
- whether any generated terminal is correct;
- the ambiguity-tier label as a learned input;
- checkpoint identity as a learned input;
- reserve capacity as a learned input.

The hidden path exists only in post-execution evaluation instrumentation.

## Frozen population ladder

Stable-reserve conditions:

```text
L1
L16
L64
L256
```

`L1` preserves the narrow baseline.

`L16` shows the lower part of the useful frontier.

`L64` is the empirically saturated v1 frontier.

`L256` is the preregistered beyond-v1 frontier test.

No capacity between or above these values may be added after development results are inspected and still be called Gate-3 v2.

## Controls

Because the diversity/continuity mechanism already replicated across three v1 checkpoints, v2 keeps the expensive controls only at the frontier capacity where the new claim would be made:

- `collapsed_diversity`, `L256`;
- `reshuffled_continuity`, `L256`.

Thus each ambiguity tier contains six conditions per checkpoint:

```text
stable L1
stable L16
stable L64
stable L256
collapsed L256
reshuffled L256
```

Across two ambiguity tiers and three frozen checkpoints, the complete development matrix contains:

`2 x 3 x 6 = 36 cells`.

## Primary outcome

Primary capability outcome remains exact no-replay search coverage:

> Did the fixed 256-round search transcript generate the exact hidden terminal path at least once?

The runtime cannot stop early on success because it cannot inspect success.

## Capacity-pressure telemetry

To distinguish `L64 is sufficient` from `L64 is actively binding`, v2 additionally records answer-blind reserve-pressure telemetry for every stable condition:

- maximum reserve population reached;
- mean reserve population across rounds;
- fraction of rounds at nominal capacity;
- whether nominal capacity was reached at least once;
- productive-vs-sink round counts;
- generated terminal count;
- unique generated terminal count.

These are explanatory secondary metrics only. They cannot override the frozen primary coverage result.

## Frozen paired comparisons

For each checkpoint and ambiguity tier, reconstruct at minimum:

1. stable `L256 - L64` exact-coverage delta;
2. stable `L64 - L16` exact-coverage delta;
3. stable `L256 - L1` exact-coverage delta;
4. stable `L256 - collapsed L256` exact-coverage delta;
5. stable `L256 - reshuffled L256` exact-coverage delta.

Use deterministic paired bootstrap intervals over the raw 256-world coverage vectors.

Bootstrap samples: `2,000`.

## Frozen development interpretation

The central frontier comparison is `L256 - L64`.

### V2-F0 — no beyond-L64 extension

Assign `V2_F0_NO_BEYOND_L64_EXTENSION` if `L256 - L64` has CI low `<= 0` in all three checkpoints at both ambiguity tiers.

This means the v1 L64 frontier remains sufficient even under the preregistered ambiguity increase.

### V2-F1 — extension only at highest ambiguity

Assign `V2_F1_EXTENSION_AT_A55_ONLY` if all three checkpoints have:

- A55 `L256 - L64` CI low `> 0`;
- A55 stable-vs-collapsed CI low `> 0`;
- A55 stable-vs-reshuffled CI low `> 0`;

while A60 does not satisfy the all-three-checkpoint `L256 - L64` criterion.

### V2-F2 — robust beyond-L64 extension

Assign `V2_F2_ROBUST_BEYOND_L64_EXTENSION` if all three checkpoints have:

- A60 `L256 - L64` CI low `> 0`;
- A55 `L256 - L64` CI low `> 0`;
- stable-vs-collapsed CI low `> 0` at both tiers;
- stable-vs-reshuffled CI low `> 0` at both tiers.

### V2-F3 — checkpoint-sensitive frontier

Assign `V2_F3_CHECKPOINT_SENSITIVE_FRONTIER` if the mechanism controls remain positively separated but the sign/significance of `L256 - L64` differs among checkpoints within a tier.

### V2-F4 — mechanism degradation under ambiguity

Assign `V2_F4_MECHANISM_DEGRADES_UNDER_AMBIGUITY` if one or more checkpoints lose positive control separation at the frontier capacity.

No development classification is itself a positive confirmation verdict.

## Confirmation boundary

Gate-3 v2 confirmation is CLOSED.

If and only if development returns `V2_F1` or `V2_F2`, a separate confirmation protocol may be frozen later using:

- the exact same three checkpoints;
- unchanged ambiguity tier(s) selected mechanically by the development class;
- unchanged population ladder and work budget;
- a new untouched confirmation namespace;
- a preregistered confirmation acceptance rule.

No confirmation world may be generated or inspected before that later freeze.

## Compiler / execution boundary

Compiler optimization remains an independent experimental variable.

Gate-3 v2 capability development uses the same eager FP32 execution semantics as v1. No `torch.compile`, CUDA graphs, mixed precision, custom fusion or compiler-specific optimization may be enabled in the scientific capability run.

Runtime batching across independent worlds is allowed because it preserves the same per-world search semantics and is part of the already-qualified GPU-native execution path.

## Claims allowed from a positive development result

A positive frontier-development result may support only the narrow statement that, under the preregistered ambiguity regime and fixed learned-work budget, useful dormant hypothesis capacity extends beyond the previously observed L64 frontier across the frozen learned checkpoints.

It does not establish:

- AGI or general intelligence;
- arbitrary-task scaling;
- unlimited population scaling;
- superiority to every serial/replay algorithm;
- per-FLOP or per-joule superiority;
- confirmation;
- benefit from geographically distributed hardware.

## Stop rule

After the complete 36-cell development matrix is inspected:

- do not add ambiguity tiers;
- do not add capacities;
- do not retrain checkpoints;
- do not change search rounds;
- do not change score quantization;
- do not change the controls;
- do not open confirmation unless the frozen interpretation map permits a later separately frozen confirmation protocol.

Negative or plateau results are final evidence for Gate-3 v2 and must be preserved.
