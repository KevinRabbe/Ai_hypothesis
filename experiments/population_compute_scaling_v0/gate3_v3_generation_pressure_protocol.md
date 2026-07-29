# Gate-3 v3 — generation-synchronous frontier-pressure protocol

## Status

**FROZEN PRE-RESULT PROTOCOL — NO GATE-3 v3 RESULT INSPECTED**

Gate-3 v3 starts from the fully qualified Gate-3 v2 result:

`V2_F0_NO_BEYOND_L64_EXTENSION`

Gate-3 v2 showed that lowering public-hint reliability to 0.60/0.55 did not make `L64` capacity-binding under the depth-10 best-first topology: `L64` reached nominal capacity in 0% of worlds, and `L256-L64` remained exactly zero across all three frozen checkpoints.

v3 changes the **scheduler/topology that creates simultaneous live hypotheses**, not the learned scorer, checkpoint, model size or per-expansion neural computation.

## Scientific question

> When the search topology is constructed so that more than 64 candidate hypotheses must coexist before the final search phase, does a larger persistent population (`L256`) improve exact search coverage over `L64` at fixed learned parameters, fixed active neural width and fixed total learned recurrent work?

This is a frontier-pressure mechanism experiment. It is not a continuation/tuning of v2.

## Frozen checkpoints

No training occurs in Gate-3 v3.

Exactly the same three frozen Gate-3 v1 checkpoints are reused and SHA-256 verified:

### checkpoint 0

- SHA-256: `e63748a1182472d03c540f9123c3442ce44e130519e2176681034568826db590`
- fingerprint: `e4f9990f08e85393a155637cfa50b5831d779770447ec3716fc9e67946992afc`
- parameters: `19,649`

### checkpoint 1

- SHA-256: `8ce384627f5543fa4fe78498f9990f8214cf2a4afed0c9b734e86761ad13c989`
- fingerprint: `2a57984f755ee1404fd828c2df36a5efff8f13a541b7d7c93a891014fbf4897c`
- parameters: `19,649`

### checkpoint 2

- SHA-256: `103826bbd9451b965eced5134d1674cb8e893f5d3b378cb828312ffdb6fc9a37`
- fingerprint: `8afaf956f200f41ea914eafdd1b5f151dd303cae1552165c74e12bb8c945af02`
- parameters: `19,649`

No checkpoint may be retrained, fine-tuned, recalibrated or replaced.

## Why depth 8

The frozen scorer natively supports depths `6 / 8 / 10`. v3 uses depth `8` so the learned representation remains exactly in-distribution while allowing a layer-synchronous binary search to create a deterministic 128-hypothesis depth-7 frontier within a practical fixed work budget.

No model interface changes are permitted.

## Public evidence

Hint reliability is fixed at `0.70`, matching the original v1 training distribution.

v3 therefore does not combine frontier pressure with the v2 ambiguity shift.

The runtime receives only:

- depth `8`;
- eight public noisy hints;
- answer-independent deterministic tie-break seeds.

The runtime never receives the hidden terminal path and cannot stop on success.

## Generation-synchronous scheduler

Unlike v1/v2 best-first search, v3 processes hypotheses in **generations**.

For each depth layer:

1. start with the retained candidate population for that depth;
2. rank parents by the same frozen quantized neural score + deterministic tie break;
3. expand parents in ranked order;
4. every productive parent expansion creates exactly binary child `0` and child `1`;
5. each child receives exactly eight eager-FP32 GRU updates before scoring;
6. after a complete nonterminal generation, rank all generated child hypotheses and retain at most nominal capacity `L`;
7. only then advance to the next depth generation.

At the final parent layer (depth 7), the global expansion budget may stop part-way through the generation. Generated depth-8 children are terminal outputs and are not reinserted into the population.

This scheduling rule is answer-blind.

## Frozen population ladder

Stable population capacities:

```text
L16
L64
L256
```

Frontier controls at nominal `L256`:

```text
collapsed_diversity
reshuffled_continuity
```

No other capacity is part of v3.

## Structural pressure invariant

For stable `L64` and `L256`, the unpruned binary generations through depth 7 have widths:

```text
1, 2, 4, 8, 16, 32, 64, 128
```

The first seven parent generations require:

```text
1 + 2 + 4 + 8 + 16 + 32 + 64 = 127 parent expansions
```

After those 127 productive expansions, the depth-7 candidate generation contains exactly 128 unique hypotheses before capacity pruning.

Therefore:

- `L64` is mechanically binding at the depth-7 generation boundary and must retain only 64 of 128 hypotheses;
- `L256` is non-binding there and can retain all 128 hypotheses.

This is a preregistered structural property, not an empirical result.

## Frozen global learned-work budget

Total scheduled parent-expansion slots per world:

`223`

Each slot executes exactly:

```text
2 active child lanes
x 8 recurrent updates / child
= 16 learned recurrent updates
```

Total learned recurrent work per condition:

`223 x 16 = 3,568 learned recurrent updates/world`

This is identical across capacities, controls and checkpoints.

For stable L64/L256:

- the first 127 slots build the complete depth-7 frontier before capacity pruning;
- 96 scheduled slots remain for the final depth-7 parent generation.

Stable `L256` has 128 retained depth-7 parents and therefore can use all 96 final slots productively.

Stable `L64` has only 64 retained depth-7 parents. After those 64 are expanded, the remaining 32 scheduled slots execute answer-independent matched sink work.

The learned-update total remains exactly 3,568 in both conditions.

This difference in productive-vs-sink work is part of the causal mechanism being tested: can larger persistent population convert an identical neural-work budget into exploration of more distinct hypotheses instead of losing alternatives and later spending matched work without useful candidate state?

## Frozen controls

### stable_reserve

Distinct candidate identities retain their own persistent GRU state and score across generation boundaries.

### collapsed_diversity

Nominal `L256` backing capacity exists, but only one logical schedulable hypothesis survives each generation boundary.

### reshuffled_continuity

The same candidate identities and nominal capacity remain, but persistent neural histories/scores are deterministically permuted among candidate paths at each generation boundary before the next layer.

The reshuffle is answer-independent.

## Development worlds

A new deterministic namespace is used:

```text
gate3-v3-generation-pressure-development-hidden
gate3-v3-generation-pressure-development-hints
gate3-v3-generation-pressure-development-runtime
```

Exactly `256` development worlds are used.

The same 256 hidden paths and public worlds are reused across all capacities, controls and checkpoints for paired comparisons.

No v1/v2 confirmation namespace is touched.

## Development matrix

Per checkpoint:

```text
stable L16
stable L64
stable L256
collapsed L256
reshuffled L256
```

Across three frozen checkpoints:

`3 x 5 = 15 development cells`

## Primary capability outcome

Exact no-replay terminal coverage:

> Did the fixed 223-slot search transcript generate the exact hidden depth-8 terminal path at least once?

The runtime cannot inspect this outcome during execution.

## Required telemetry

For every condition record:

- productive parent-expansion slots;
- matched sink slots;
- total learned recurrent updates;
- population width at every generation boundary;
- unique population width at every generation boundary;
- whether nominal capacity was binding at each boundary;
- number of depth-7 parents expanded;
- generated terminal count;
- unique generated terminal count.

For stable L64/L256 specifically verify mechanically:

```text
pre-prune depth-7 candidates = 128
L64 retained depth-7 = 64
L256 retained depth-7 = 128
L64 final productive expansions = 64
L64 final sink slots = 32
L256 final productive expansions = 96
L256 final sink slots = 0
```

Any violation invalidates the artifact.

## Frozen paired comparisons

For each checkpoint reconstruct from raw 256-world coverage vectors:

1. stable `L256 - L64` — primary frontier-pressure effect;
2. stable `L64 - L16` — lower population effect;
3. stable `L256 - collapsed L256`;
4. stable `L256 - reshuffled L256`.

Use deterministic paired bootstrap intervals with `2,000` samples.

## Frozen interpretation map

### V3-G0 — no pressure benefit

Assign `V3_G0_NO_L256_PRESSURE_BENEFIT` if `L256-L64` has CI low `<= 0` in all three checkpoints.

### V3-G1 — robust generation-pressure benefit

Assign `V3_G1_ROBUST_GENERATION_PRESSURE_BENEFIT` only if all three checkpoints satisfy:

- `L256-L64` paired-bootstrap CI low `> 0`;
- stable-vs-collapsed CI low `> 0`;
- stable-vs-reshuffled CI low `> 0`;
- all structural pressure/work invariants pass.

This supports the narrow mechanism that larger persistent population can improve capability when the task/scheduler genuinely requires retaining more than 64 simultaneous alternatives under fixed learned work.

### V3-G2 — checkpoint-sensitive pressure benefit

Assign `V3_G2_CHECKPOINT_SENSITIVE_PRESSURE_BENEFIT` if structural invariants and controls remain valid but `L256-L64` significance differs across checkpoints.

### V3-G3 — mechanism/control degradation

Assign `V3_G3_CONTROL_OR_MECHANISM_DEGRADATION` if one or more checkpoints lose positive stable-vs-collapsed or stable-vs-reshuffled separation.

No development classification is a confirmation verdict.

## Confirmation boundary

Gate-3 v3 confirmation is CLOSED.

Only `V3_G1` may permit a later separately frozen confirmation protocol using an untouched namespace and unchanged checkpoints/scheduler/work budget.

No confirmation world may be generated or inspected during development.

## Claims boundary

A positive v3 development result may support only the narrow statement that under this generation-synchronous binary search regime, where a 128-hypothesis frontier is mechanically created, persistent capacity above 64 can improve exact search coverage while learned parameters, active neural width and total learned recurrent work remain fixed.

It does not establish:

- AGI or general intelligence;
- arbitrary-task population scaling;
- unlimited useful population scaling;
- superiority to every serial/replay algorithm;
- per-FLOP/per-joule superiority;
- that matched sink work is equivalent to productive candidate work;
- confirmation.

## Stop rule

After the 15-cell development matrix is inspected:

- do not change the 223-slot budget;
- do not change depth or reliability;
- do not add capacities;
- do not change generation pruning or ranking;
- do not retrain checkpoints;
- do not alter controls;
- do not open confirmation unless the frozen map permits a separately versioned confirmation protocol.

Negative, mixed or positive results are all permanent evidence.
