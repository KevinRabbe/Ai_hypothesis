# Gate-7 v0 — high-scale routing-bandwidth screening protocol

## Status

**PREREGISTERED / DATA-BLIND — HIGH-SCALE EXECUTION NOT YET OPEN.**

This protocol is stacked on the fully qualified scale-neutral transition evidence head:

`0d1bd683bae322a11a76b4d885f2efeb3c4a5099`

Required bridge outcome:

`GATE7_SCALE_NEUTRAL_TRANSITION_QUALIFIED`

No N>=1024 Gate-7 hidden path, hint stream, runtime seed, checkpoint execution, screening result, or confirmation namespace has been generated while this protocol is being frozen.

## Scientific question

> As persistent population N increases, what is the smallest preregistered score-visibility budget K that preserves learned-routing value relative to a matched answer-blind K control and remains within five percentage points of global learned routing?

The finite-range object is:

`K_required(N)`

This experiment does not preregister an asymptotic fit. It localizes a routing-bandwidth frontier over a fixed geometric ladder.

## Exact checkpoint family

The entire campaign must reuse all three qualified transition checkpoints without selection, retraining or fine-tuning:

```text
T0 be1b18e22a1f713b48de8934aca1c0302f1255342684158c6ae31a7c3618a719
T1 a9b712736f440168ccd86ee070b8d301be0eb834b9d17418d40c0175207cadfb
T2 cd7869511d07679cb1ad430743cad031299a575413e8a853ef9116d545c7480a
```

Every neural condition uses exactly 19,649 learned parameters.

## Frozen population and K ladders

Population ladder:

```text
N1024
N2048
N4096
N8192
N16384
N32768
N65536
N131072
```

Bounded score-visibility ladder:

```text
K16
K32
K64
K128
K256
K512
```

Only K<N is valid. No intermediate or larger K may be inserted after exposure.

## Fresh high-scale worlds

For population `N=2^d`:

- complete Stage-A frontier depth: `d`;
- public task depth: `d+1`;
- fresh hidden, hint and runtime namespaces include the population, world index and task depth;
- 64 worlds/checkpoint/tier;
- the same 64 worlds are paired across global, bounded-score and answer-blind conditions;
- hint reliability remains 0.70.

The one-step terminal horizon is deliberate. It isolates score-based selection from a complete N-candidate hypothesis frontier: each Stage-B activation removes one frontier candidate and evaluates both terminal actions. The experiment therefore measures how much score visibility is needed to route a fixed 128-parent activation budget across N persistent alternatives.

## Frozen learned-work identity

Per world and condition:

```text
Stage A parent slots = N - 1
Stage B parent slots = 128
active child lanes   = 2
updates / child      = 8
```

Therefore:

```text
Stage A learned updates = (N - 1) * 16
Stage B learned updates = 128 * 16 = 2,048
total logical updates   = (N - 1 + 128) * 16
```

The complete Stage-A frontier is identical across routing conditions for the same checkpoint/N/world batch. An implementation may physically compute and cache that frontier once, then clone/reload it for each Stage-B condition. That execution reuse does not change the logical work identity of any condition.

Physical batching is fixed at 64 worlds. Compiler, CUDA graphs and mixed precision remain off for the scientific baseline.

## Frozen condition families

Every completed population tier begins with:

1. `global_score` — reads all live neural scores;
2. `global_hash` — full-population answer-blind selection using the same live bank but zero neural-score observations.

Then K is exposed in ascending order. Every K has both:

- `bounded_score_kK`;
- `bounded_hash_kK`.

The score/hash pair must use the exact same deterministic sampled positions at every world and Stage-B slot. The hash condition reads candidate metadata only and zero neural scores before selection.

Thus there is no unmatched K and no possibility of declaring K_required from a score treatment lacking its own answer-blind control.

## Reference viability

Before any K is interpreted at population N, the global learned reference must remain viable.

Define paired per-world difference:

`global_score - global_hash`

Reference viability requires both:

1. the point delta is strictly positive on each of T0, T1 and T2;
2. a deterministic stratified paired bootstrap pooling the three checkpoint strata has CI low > 0.

If this fails, the tier outcome is:

`G7_REFERENCE_FRONTIER_REACHED`

No K result at that N or any larger N is scientifically interpreted.

This reference rule avoids an arbitrary absolute coverage threshold while preventing a near-global criterion from becoming vacuous after the learned global scorer itself loses value.

## Per-K primary criteria

For checkpoint C, population N and K, compute paired bootstrap intervals on the exact same worlds.

K passes checkpoint C only if both hold:

1. `CI_low(score_K - hash_K) > 0`;
2. `CI_low(score_K - global_score) > -0.05`.

The five-point non-inferiority margin is inherited unchanged from Gates 5–6 and the transition bridge.

K is `K_required(N)` only if it passes both criteria on all three checkpoints.

## Sequential exposure and first-pass rule

For each population:

1. evaluate global score and global hash on all three checkpoints;
2. apply the frozen reference-viability rule;
3. if viable, evaluate K16 score/hash on all three checkpoints;
4. if K16 passes all six primary checkpoint criteria, record `K_required(N)=16` and do not expose K32+;
5. otherwise evaluate K32, then K64, K128, K256 and K512 in order;
6. stop at the first all-checkpoint passing K;
7. if all six K values fail, record `K_required(N)>512` and `G7_ROUTING_BANDWIDTH_FRONTIER_REACHED`;
8. continue to the next N only after a valid K_required result at the current tier.

The tested K set must always be one contiguous prefix of the frozen ladder. Skipping a failed K, testing a larger K first, or adding a rescue K after exposure is forbidden.

## Statistical summaries

Per checkpoint/pair:

- 64 paired worlds;
- 2,000 deterministic paired bootstrap samples;
- point delta;
- 95% bootstrap CI low/high.

Reference viability additionally uses a deterministic stratified paired bootstrap across the three checkpoint strata, resampling 64 pairs independently within each checkpoint and averaging the three stratum means.

This is sequential exploratory frontier localization. The repeated intervals are not presented as a familywise-controlled final confirmation.

## Required reporting per completed N

- global and global-hash coverage by checkpoint;
- global-reference point deltas and pooled CI;
- every exposed score-K and hash-K coverage rate;
- every exposed score-K/hash-K paired interval;
- every exposed score-K/global paired interval;
- `K_required(N)` or `>512`;
- `K_required(N)/N` where defined;
- score observations/world by condition;
- logical learned updates/world;
- wall time and peak VRAM as engineering telemetry only.

Unexposed larger K values are recorded explicitly as `NOT_RUN_BY_FIRST_PASS_RULE`, not omitted silently.

## Campaign continuation and stop classes

After a valid `K_required(N)`:

- continue to the next population in the fixed ladder;
- if N131072 completes, stop with `G7_CAMPAIGN_CEILING_REACHED`.

Other terminal classes:

- `G7_REFERENCE_FRONTIER_REACHED` — global learned routing no longer beats full-population answer-blind routing under the frozen viability rule;
- `G7_ROUTING_BANDWIDTH_FRONTIER_REACHED` — no K<=512 passes at the current N;
- `G7_RESOURCE_FRONTIER_REACHED` — the fixed runner cannot complete a tier due to VRAM/runtime/resource limits;
- `G7_CAMPAIGN_CEILING_REACHED` — the full prepared ladder completes.

A resource stop is not a scientific routing failure. An incomplete tier is not assigned K_required.

## Confirmation boundary

This screening campaign cannot open its own confirmation automatically.

After the complete screening result is recorded, a separately versioned confirmation protocol may select representative preregistered boundary points using the screening evidence. The confirmation must use a new untouched world namespace, higher world count, unchanged checkpoint family and unchanged condition definitions.

## Claims boundary

A successful campaign may support only a finite-range empirical relation between population N and required score visibility K under this controlled task/scheduler/work regime.

It does not establish:

- an asymptotic Big-O law;
- a universal maximum population;
- physical decentralization or multi-machine scalability;
- capability-per-FLOP or capability-per-joule superiority;
- AGI or general intelligence;
- superiority to arbitrary serial/replay algorithms.

## Pre-exposure lock

Before high-scale execution is admitted, CI must prove:

- the exact checkpoint identities and qualified bridge head remain bound;
- population and K ladders are immutable;
- every K has a matched hash control;
- K exposure can only be an ascending contiguous prefix;
- the first-pass rule suppresses all larger K;
- reference viability uses global score versus global hash;
- no scientific world generator or high-scale runner is reachable from the protocol-only branch.
