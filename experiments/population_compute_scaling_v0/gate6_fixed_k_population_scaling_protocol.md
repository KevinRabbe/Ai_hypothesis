# Gate-6 v0 — fixed-K routing under larger live populations

## Status

**FROZEN BEFORE ANY GATE-6 DEVELOPMENT WORLD IS GENERATED OR INSPECTED.**

Gate-6 is admitted only because Gate-5 independently confirmed:

`GATE5_CONFIRMED_BOUNDED_SCORE_ACTIVATION`

on fully qualified final Gate-5 evidence head:

`0705afc8f31d49a70158b4011631a31dc075c681`

The preparation-only Gate-6 candidate did not define or instantiate a scientific Gate-6 namespace.

## Scientific question

> With the learned scorer, learned parameters, candidate depth, public information, active neural width, per-candidate refinement, frontier-construction work and Stage-B work held fixed, does a constant bounded routing visibility K remain useful as the number of simultaneously live hypotheses increases?

The causal variable is **Stage-B live population capacity**.

## Frozen checkpoints/model

Reuse the exact three frozen Gate-3 v1 checkpoints used by Gates 3–5:

- C0 SHA-256 `e63748a1182472d03c540f9123c3442ce44e130519e2176681034568826db590`, fingerprint `e4f9990f08e85393a155637cfa50b5831d779770447ec3716fc9e67946992afc`;
- C1 SHA-256 `8ce384627f5543fa4fe78498f9990f8214cf2a4afed0c9b734e86761ad13c989`, fingerprint `2a57984f755ee1404fd828c2df36a5efff8f13a541b7d7c93a891014fbf4897c`;
- C2 SHA-256 `103826bbd9451b965eced5134d1674cb8e893f5d3b378cb828312ffdb6fc9a37`, fingerprint `8afaf956f200f41ea914eafdd1b5f151dd303cae1552165c74e12bb8c945af02`.

Learned parameter count remains exactly **19,649/checkpoint**.

No training or fine-tuning is permitted.

## Frozen task and namespace

- hidden binary path depth: `10`;
- noisy-hint reliability: `0.70`;
- public/hidden separation unchanged;
- hidden path is available only to the evaluator after search execution;
- runtime may inspect only public hints plus scheduler-allowed candidate information.

Use exactly **256 development worlds** from fresh deterministic Gate-6 namespaces:

- `gate6-fixed-k-population-scaling-development-hidden`;
- `gate6-fixed-k-population-scaling-development-hints`;
- `gate6-fixed-k-population-scaling-development-runtime`.

These namespaces are disjoint from Gates 0–5.

Evaluation batch size: **64 worlds**.

Use **2,000 deterministic paired bootstrap samples**.

Confirmation remains CLOSED.

## Common Stage A — identical across every condition

Every condition first builds the same complete depth-8 frontier generation-synchronously.

- frontier size: `256` persistent scored candidates;
- parent expansions: `1 + 2 + ... + 128 = 255`;
- two child lanes per parent expansion;
- eight recurrent updates per child;
- Stage-A learned recurrent updates/world: `255 × 2 × 8 = 4,080`.

No population-size treatment or scheduler-specific score visibility is applied during Stage A.

## Population treatment

Frozen population ladder:

- `N64`;
- `N128`;
- `N256`.

After the common 256-candidate frontier exists, derive the initial Stage-B population by one deterministic answer-blind permutation keyed only by public runtime seed + candidate path. Retain the first N candidates.

Therefore, for the same world:

`N64 ⊂ N128 ⊂ N256`.

The thinning rule must not inspect candidate score, hidden answer, label or scheduler mode.

### Hard Stage-B capacity

N is also the hard live-reserve capacity during Stage B.

After every productive expansion, if adding children would make the live reserve exceed N, prune back to N with a deterministic answer-blind retention rule keyed only by public runtime seed + Stage-B slot + candidate path.

This retention rule must not inspect candidate score, hidden answer or scheduler mode.

This prevents smaller-N conditions from silently growing into larger populations.

## Common Stage B work

Every condition receives exactly **128 scheduled parent activations** after thinning.

Each activation:

- selects one live nonterminal parent;
- expands two children;
- gives each child exactly eight recurrent updates;
- consumes exactly `16` learned recurrent updates.

Stage-B learned recurrent updates/world:

`128 × 16 = 2,048`.

Total learned recurrent work/world:

`4,080 + 2,048 = 6,128`.

This total is identical across every population tier and scheduler condition.

The frozen topology guarantees a non-empty productive reserve across all 128 Stage-B slots for N ≥ 64. Sink work is therefore not part of the admitted v0 execution path; any unexpected reserve exhaustion invalidates the run.

## Frozen scheduler matrix

For each N evaluate exactly four schedulers:

1. `global_score` — full live-reserve score visibility;
2. `bounded_score_k16` — primary fixed-K learned treatment;
3. `bounded_hash_k16` — matched K16 sampling with zero neural-score comparison before parent selection;
4. `bounded_score_k8` — descriptive frontier only.

`3 N tiers × 4 schedulers = 12 conditions/checkpoint`.

Across three checkpoints:

`36 development cells`.

K16 is primary because it was Gate-5's preregistered primary bounded treatment. K8 remains descriptive and cannot rescue a failed K16 result.

## Strict score-visibility boundary

For bounded conditions, only the sampled K candidate scores may be read before parent selection.

The parent must be irrevocably selected before any non-sampled candidate score is read.

A full live-reserve score ranking may be computed only after that decision for evaluation telemetry and must never feed back into search state, pruning, routing or future sampling.

`bounded_score_k16` and `bounded_hash_k16` use the same deterministic `k16` sampling-group rule whenever their incoming reserves are identical. Later divergence caused by different parent selections is part of the treatment effect.

## Frozen primary effects

For each checkpoint C0/C1/C2 and each N ∈ {64,128,256}, reconstruct from raw paired per-world exact-coverage vectors:

### Learned bounded-routing effect

`bounded_score_k16 - bounded_hash_k16`

K16 learned routing is established at a tier only when paired-bootstrap 95% CI low is strictly `> 0`.

### Non-inferiority to global learned routing

`bounded_score_k16 - global_score`

Retain the Gate-5 frozen non-inferiority margin:

`delta_NI = 0.05`.

K16 is non-inferior at a tier only when paired-bootstrap 95% CI low is strictly `> -0.05`.

### Descriptive K8 frontier

Also report:

`bounded_score_k8 - global_score`

with paired CI and score-observation cost. K8 is never part of the Gate-6 v0 acceptance rule.

## Frozen mutually exclusive development outcomes

For each checkpoint/tier define `PASS(C,N)` iff both K16 primary criteria pass at that checkpoint/tier.

Outcome precedence is frozen as follows:

1. `G6_S4_BOUNDED_ROUTING_HARMFUL_AT_SCALE` — if the 95% CI high of `K16 - hashK16` is strictly `< 0` at any checkpoint/tier.
2. `G6_S3_CHECKPOINT_SENSITIVE_SCALING` — if PASS status is mixed across checkpoints at N64, N128 or N256.
3. `G6_S0_FIXED_K_NOT_ESTABLISHED` — if PASS is false on all three checkpoints at N64.
4. `G6_S2_ROBUST_FIXED_K_POPULATION_SCALING` — if PASS is true on all three checkpoints at N64, N128 and N256.
5. `G6_S1_FIXED_K_DEGRADES_WITH_POPULATION` — otherwise; this requires N64 to pass on all checkpoints while at least one larger N tier uniformly fails the combined primary criteria.

These classes are mutually exclusive under the stated precedence.

Development alone never assigns a final confirmed Gate-6 verdict.

A separately versioned confirmation protocol may be opened only after `G6_S2_ROBUST_FIXED_K_POPULATION_SCALING`.

## Required telemetry

Per world/condition preserve at minimum:

- exact hidden-path generation coverage;
- common Stage-A frontier width;
- initial thinned Stage-B population;
- hard Stage-B population capacity;
- live population by Stage-B slot;
- activated parent depth by slot;
- visible candidate count by slot;
- score observations by slot;
- total/max score observations;
- selected visible score rank;
- selected global score rank computed only after bounded selection;
- selected parent path;
- deterministic overflow-prune count;
- generated terminal count and unique terminal count;
- productive Stage-A and Stage-B slot counts;
- total learned recurrent updates;
- runtime seed/world index;
- checkpoint parameter count/fingerprint.

## Claims boundary

Even `G6_S2_ROBUST_FIXED_K_POPULATION_SCALING` would support only the narrow statement that fixed K16 score visibility retained useful learned routing while a controlled live population increased from 64 to 256 under the frozen task and compute budget.

It would not establish:

- physical decentralization;
- arbitrary communication graphs;
- 1K/10K/100K-worker scaling;
- asynchronous distributed execution;
- per-FLOP/per-joule superiority;
- general intelligence;
- universal sufficiency of K8/K16.

Those require later gates.

## Stop rule

After the first admitted 256-world development matrix is inspected:

- do not change checkpoints;
- do not change N tiers;
- do not change K16 primary treatment or 0.05 margin;
- do not change 255/128 slot budgets;
- do not change 6,128 learned updates/world;
- do not change depth/reliability;
- do not change thinning, capacity-pruning or score-visibility semantics;
- do not add rescue conditions or alternate development namespaces.
