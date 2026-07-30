# Gate-7 candidate — high-scale bounded-routing frontier campaign

**STATUS: PREPARATION ONLY — NOT ADMITTED.**

This candidate may be prepared while Gate-6 remains unresolved. It must not generate Gate-7 scientific worlds, train a replacement scorer, execute an admitted scale sweep, or assign a Gate-7 outcome until a later admission commit explicitly freezes the scientific protocol.

Base preparation state: Gate-6 pre-result qualified head `02671dcd070201690aa71f7326b1f0779bc660c4`.

## Purpose

Locate the population scale at which fixed bounded learned routing (primary K16) stops retaining near-global routing quality, without spending a full confirmation experiment at every power-of-two population size.

This is a **frontier-localization campaign**, not a final confirmatory result. Its job is to rapidly bracket the first scaling failure. A later separately frozen confirmation experiment must test the last-pass / first-fail neighborhood on an untouched namespace.

## Why not stop at 1K

A single N1024 experiment is inefficient. If K16 still works there, the next scientific question is immediately N2048. The prepared ladder therefore doubles population geometrically until the first clear failure or a preregistered campaign ceiling.

Prepared high-scale ladder:

`512 -> 1024 -> 2048 -> 4096 -> 8192 -> 16384 -> 32768`

The campaign must report the largest tested N at which K16 remains established and near-global, plus the first tested N at which either criterion fails.

## Phase A — exact existing frozen checkpoints

The existing Gate-3 v1 scorer/checkpoints have a hard representation ceiling:

- maximum supported world depth: 10;
- child depth is encoded into a 10-position one-hot;
- world depth is indexed from the frozen `(6, 8, 10)` set.

Therefore the largest clean simultaneously live **nonterminal** frontier available without changing the scorer representation is depth 9 = **512 candidates**.

Phase A may therefore test N512 using the exact existing three checkpoints, but must not pretend they support N1024+ nonterminal search.

## Phase B — one architecture transition, then push hard

To avoid one architecture change per scale, prepare one **scale-neutral scorer** and freeze it before high-scale scientific exposure.

Candidate representation constraint:

- preserve total scorer input width at 19 so the neural architecture remains `Linear(19->32) + GRU(32->64) + LayerNorm + Linear(64->1)` and therefore retains the same 19,649 learned-parameter count;
- replace the hard-coded depth one-hots with fixed, non-learned depth features that support arbitrary positive depth;
- keep noisy-hint/action/sink semantics unchanged;
- no population-size input, slot ID, N embedding, routing-condition input, or hidden-answer channel;
- train a new set of three checkpoints once under a depth-diverse frozen recipe, then reuse those exact checkpoints for the entire 1K->32K sweep.

The exact fixed depth-feature transform and training recipe must be frozen in a later admission commit before any replacement-checkpoint training occurs.

## High-scale runtime preparation

The admitted high-scale runner should not use one Python object per candidate as its primary representation. Prepare a tensorized candidate bank:

- recurrent states: dense tensor `[batch, population, 64]`;
- scores: dense tensor `[batch, population]`;
- path/prefix identity: compact integer representation;
- live/terminal masks: boolean tensors;
- bounded K sampling: deterministic answer-blind integer indices;
- K16 parent selection may read only sampled candidate scores before selection;
- global-score mode may read all live scores;
- hash K16 control reads zero neural scores before selection.

Stage-A frontier construction should use generation-synchronous batching. This changes execution efficiency only; every generated child must still receive the frozen number of recurrent updates. No wall-clock/per-FLOP claim may be made against earlier eager object-based gates from this campaign alone.

## Frontier-localization ladder

After the scale-neutral three-checkpoint set is frozen, evaluate N geometrically:

- N1024
- N2048
- N4096
- N8192
- N16384
- N32768

At every N use the same primary scheduler trio:

1. `global_score`
2. `bounded_score_k16`
3. `bounded_hash_k16`

K8 may remain descriptive if frozen before exposure.

The primary K remains K16. The campaign may not increase K merely to keep a positive result while locating the K16 scaling frontier.

## Within-N causal comparisons

The high-scale frontier campaign does **not** require equal Stage-A construction work between different N values. Larger N necessarily requires a larger frontier. It therefore makes no capability-per-compute claim across N.

Instead, at each N independently compare schedulers on identical worlds/candidates/work:

- learned-routing effect: `K16 - hashK16`;
- near-global gap: `K16 - global`.

For a tested N to count as a **pass candidate**:

- `CI_low(K16 - hashK16) > 0`;
- `CI_low(K16 - global) > -0.05`;
- both conditions must hold for all three frozen high-scale checkpoints.

The five-percentage-point margin is inherited from Gates 5–6 rather than selected after seeing high-scale data.

## Sequential frontier-search rule

This is explicitly exploratory frontier localization.

Starting at N1024 after the scale-neutral scorer is frozen:

1. evaluate N;
2. if all three checkpoints satisfy both K16 criteria, proceed to the next doubled N;
3. at the first N where either criterion fails on any checkpoint, stop the primary ladder;
4. preserve all results including failures;
5. do not retrain, alter K16, change the margin, or rerun alternate namespaces to extend the frontier;
6. separately freeze a confirmation experiment around the **largest passing N and first failing N**.

If every tier through N32768 passes, report only a lower bound: `K16 frontier > 32768 within the tested regime`. Do not claim the scaling limit was found.

Because the ladder is sequential/exploratory, its repeated confidence intervals are not a final familywise-controlled confirmation claim. The later bracket confirmation supplies the inferential test.

## Resource stop boundary

A resource failure is not a scientific K16 failure.

Record separately if a tier cannot be completed because of VRAM, host RAM, numerical/runtime failure, or a preregistered wall-time ceiling. The campaign must distinguish:

- `ROUTING_FRONTIER_FAILURE`
- `RESOURCE_FRONTIER_REACHED`
- `CAMPAIGN_CEILING_REACHED`

A resource stop may motivate an execution/memory-engineering experiment but cannot be reclassified as evidence that bounded routing failed.

## What this campaign can answer

It can rapidly estimate how far a fixed K16 learned routing channel continues to work as the available population grows by powers of two.

It cannot by itself establish:

- physical decentralization;
- multi-machine scaling;
- optimal K asymptotics;
- general intelligence;
- capability-per-FLOP superiority;
- the true asymptotic scaling limit if the campaign ceiling is reached without failure.

## Preparation boundary

While this file remains preparation-only:

- no Gate-7 world namespace;
- no high-scale scientific data;
- no replacement scorer training;
- no result classifier execution;
- no confirmation namespace.

Only data-blind mechanics, tensor-bank infrastructure, work/resource accounting, and protocol checks may be prepared.