# Gate-7 candidate — high-scale bounded-routing frontier campaign

**STATUS: PREPARATION ONLY — NOT ADMITTED.**

This candidate may be prepared while Gate-6 remains unresolved. It must not generate Gate-7 scientific worlds, train a replacement scorer, execute an admitted scale sweep, or assign a Gate-7 outcome until a later admission commit explicitly freezes the scientific protocol.

Base preparation state: Gate-6 pre-result qualified head `02671dcd070201690aa71f7326b1f0779bc660c4`.

## Purpose

Locate the population scale at which fixed bounded learned routing (primary K16) stops retaining near-global routing quality, without spending a full confirmation experiment at every power-of-two population size.

This is a **frontier-localization campaign**, not a final confirmatory result. Its job is to rapidly bracket the first scaling failure. A later separately frozen confirmation experiment must test the last-pass / first-fail neighborhood on an untouched namespace.

## Why not stop at 1K

A single N1024 experiment is inefficient. If K16 still works there, the next scientific question is immediately N2048. The prepared ladder therefore doubles population geometrically until the first clear routing failure, a reference-search failure, a resource stop, or the campaign ceiling.

Prepared ladder:

`512 -> 1024 -> 2048 -> 4096 -> 8192 -> 16384 -> 32768 -> 65536 -> 131072`

The campaign must report the largest tested N at which K16 remains established and near-global, plus the first tested N at which either routing criterion fails. If no routing failure occurs, report only the tested lower bound.

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
- replace the hard-coded depth one-hots with fixed, non-learned depth features that support the whole prepared depth range in one representation;
- keep noisy-hint/action/sink semantics unchanged;
- no population-size input, slot ID, N embedding, routing-condition input, or hidden-answer channel;
- train a new set of three checkpoints once under one depth-diverse frozen recipe, then reuse those exact checkpoints for the entire N1024->N131072 sweep.

The exact fixed depth-feature transform, depth-training mixture and optimization recipe must be frozen in a later admission commit before any replacement-checkpoint training occurs.

For the prepared ceiling N131072, the scale-neutral representation must support at least a depth-17 nonterminal frontier and depth-18 terminal task without changing learned dimensions or parameter count.

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

Stage-A frontier construction should use generation-synchronous batching. This changes execution efficiency only; every generated child must still receive the frozen number of recurrent updates.

For evaluation speed, the common Stage-A frontier may be materialized **once per checkpoint/world batch** and copied into the scheduler conditions, provided CI proves the copies are state-identical before treatment and no condition can mutate another condition's bank. Logical learned-work accounting remains attached to every condition; physical evaluation reuse is not a capability-per-compute claim.

No wall-clock/per-FLOP claim may be made against earlier eager object-based gates from this campaign alone.

## Frontier-localization ladder

After the scale-neutral three-checkpoint set is frozen, evaluate N geometrically:

- N1024
- N2048
- N4096
- N8192
- N16384
- N32768
- N65536
- N131072

At every N use the same primary scheduler trio:

1. `global_score`
2. `bounded_score_k16`
3. `bounded_hash_k16`

K8 may remain descriptive if frozen before exposure.

The primary K remains K16. The campaign may not increase K merely to keep a positive result while locating the K16 scaling frontier.

## Fast screening before expensive confirmation

The intended campaign is deliberately cheaper than a confirmation matrix at every tier.

Prepared screening target: **64 fresh worlds/checkpoint/tier**, paired across schedulers, with a deterministic paired-bootstrap analysis. This count is exploratory and must be frozen before the first high-scale world is generated.

The first clear boundary then receives a separately frozen high-power confirmation on untouched worlds. The screening intervals are therefore frontier-localization evidence, not the final inferential claim.

If 64 worlds prove mechanically insufficient before exposure for a preregistered numerical reason, that count may be changed only in the later admission commit and must then stay fixed for the entire high-scale ladder.

## Within-N causal comparisons

The high-scale frontier campaign does **not** require equal Stage-A construction work between different N values. Larger N necessarily requires a larger frontier. It therefore makes no capability-per-compute claim across N.

Instead, at each N independently compare schedulers on identical worlds/candidates/logical work:

- learned-routing effect: `K16 - hashK16`;
- near-global gap: `K16 - global`.

For a tested N to count as a **routing pass candidate**:

- `CI_low(K16 - hashK16) > 0`;
- `CI_low(K16 - global) > -0.05`;
- both conditions must hold for all three frozen high-scale checkpoints.

The five-percentage-point margin is inherited from Gates 5–6 rather than selected after seeing high-scale data.

## Reference viability

A bounded router being near a useless global router is not evidence of useful scaling.

Therefore the campaign must separately classify whether the global learned reference itself remains useful at a tier. The exact preregistered reference-viability rule must be frozen at admission before high-scale data. At minimum it must use a paired comparison against an answer-blind reference and may also freeze an absolute coverage floor.

If the global reference fails that preregistered viability rule, stop the routing-frontier interpretation at that N and classify the boundary as `REFERENCE_FRONTIER_REACHED`, not as K16 success or failure.

## Sequential frontier-search rule

This is explicitly exploratory frontier localization.

Starting at N1024 after the scale-neutral scorer is frozen:

1. evaluate N;
2. verify the global reference remains viable;
3. if the reference is viable and all three checkpoints satisfy both K16 criteria, proceed to the next doubled N;
4. at the first N where K16 fails while the global reference remains viable, stop with `ROUTING_FRONTIER_FAILURE`;
5. at the first N where the global reference itself fails its frozen viability criterion, stop with `REFERENCE_FRONTIER_REACHED`;
6. preserve all results including failures;
7. do not retrain, alter K16, change the margin, or rerun alternate namespaces to extend the frontier;
8. separately freeze a confirmation experiment around the **largest passing N and first failing N** whenever a routing boundary is found.

If every tier through N131072 passes, report only a lower bound: `K16 frontier > 131072 within the tested regime`. Do not claim the scaling limit was found.

Because the ladder is sequential/exploratory, its repeated confidence intervals are not a final familywise-controlled confirmation claim. The later bracket confirmation supplies the inferential test.

## Resource stop boundary

A resource failure is not a scientific K16 failure.

Record separately if a tier cannot be completed because of VRAM, host RAM, numerical/runtime failure, or a preregistered wall-time ceiling. The campaign must distinguish:

- `ROUTING_FRONTIER_FAILURE`
- `REFERENCE_FRONTIER_REACHED`
- `RESOURCE_FRONTIER_REACHED`
- `CAMPAIGN_CEILING_REACHED`

A resource stop may motivate an execution/memory-engineering experiment but cannot be reclassified as evidence that bounded routing failed.

## What this campaign can answer

It can rapidly estimate how far a fixed K16 learned routing channel continues to work as the available population grows by powers of two, potentially into the 10K–100K regime on one local machine.

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