# Gate-7 candidate — routing-bandwidth scaling frontier campaign

**STATUS: PREPARATION ONLY — NOT ADMITTED.**

This candidate was originally prepared before Gate-6 resolved. Gate-6 has now returned valid development outcome `G6_S3_CHECKPOINT_SENSITIVE_SCALING`: fixed K16 remains strongly useful relative to its matched hash control at N64/N128/N256, but the frozen five-percentage-point near-global criterion is robust through N128 and becomes checkpoint-sensitive at N256.

Therefore Gate-7 must **not** continue as a blind fixed-K16 doubling campaign. The first fixed-K16 near-global boundary has already been localized. Gate-7 is repurposed, before any Gate-7 scientific exposure, to ask how the routing-visibility budget required for near-global learned routing scales with population size.

No Gate-7 scientific world, replacement-checkpoint training run, admitted sweep, or Gate-7 outcome exists while this file remains preparation-only.

## Gate-6 motivation

Gate-6 primary K16 status:

```text
N64   C0 PASS  C1 PASS  C2 PASS
N128  C0 PASS  C1 PASS  C2 PASS
N256  C0 PASS  C1 FAIL  C2 FAIL
```

The N256 failures are non-inferiority failures, **not** loss of learned routing signal. At N256, `K16 - hashK16` remains strongly positive on all three checkpoints, while `K16 - global` crosses the frozen -0.05 NI boundary on C1/C2.

This suggests a routing-bandwidth frontier: constant K16 is no longer robustly sufficient, but learned bounded routing itself remains highly informative.

## Scientific question

> As live population N grows, how slowly can score-visibility budget K grow while learned bounded routing remains useful relative to a matched answer-blind K control and near-global under the inherited five-percentage-point margin?

The object of study is now the **K_required(N)** frontier, not whether K16 works forever.

The desired scalable regime is sublinear routing visibility:

`K_required(N) << N`.

A future positive result may support only the tested finite-range relation. It must not be presented as a proof of asymptotic complexity.

## Why this is faster than one full gate per doubling

Prepare one geometric population ladder and one geometric K ladder, use a cheap exploratory screening namespace to locate the routing-bandwidth frontier, then spend high-power confirmation only on selected boundary points.

Prepared population ladder:

`512 -> 1024 -> 2048 -> 4096 -> 8192 -> 16384 -> 32768 -> 65536 -> 131072`

Prepared bounded-score ladder:

`K = 16 -> 32 -> 64 -> 128 -> 256 -> 512`

K values greater than or equal to N are not evaluated as bounded conditions; `global_score` remains the full-information reference.

The exact admitted ladder may be reduced for resource reasons **before exposure**, but no K may be inserted after results merely to rescue a tier.

## Phase A — bridge using exact existing frozen checkpoints

The current Gate-3 v1 scorer/checkpoints have a hard representation ceiling:

- maximum supported world depth: 10;
- child depth is encoded into a 10-position one-hot;
- world depth is indexed from the frozen `(6, 8, 10)` set.

Therefore the largest clean simultaneously live nonterminal frontier available without changing scorer representation is depth 9 = **512 candidates**.

Phase A should use the exact existing three 19,649-parameter checkpoints to test a **bandwidth bridge at N512**, with K16/K32/K64/K128 (subject to final pre-exposure freeze), plus global and matched hash controls.

N256 does not need to be rerun merely to rescue Gate-6. Gate-6 remains closed. Any N256 point included in Gate-7 must use a fresh Gate-7 namespace and be justified as a bridge/comparability condition, not as a Gate-6 reinterpretation.

## Phase B — one scale-neutral scorer transition

N1024+ nonterminal search requires a representation that is not hard-coded to depth <=10.

Prepare one scale-neutral scorer and freeze it before high-scale scientific exposure.

Candidate representation constraint:

- preserve scorer input width at 19 so the neural architecture remains `Linear(19->32) + GRU(32->64) + LayerNorm + Linear(64->1)` and therefore retains 19,649 learned parameters;
- replace hard-coded depth one-hots with fixed non-learned depth features that support the entire prepared depth range;
- keep noisy-hint/action/sink semantics unchanged;
- no population-size input, slot ID, N embedding, routing-condition input, or hidden-answer channel;
- train one new set of three checkpoints under one frozen depth-diverse recipe;
- reuse those exact checkpoints for the entire N1024->N131072 campaign.

For N131072, the representation must support at least a depth-17 nonterminal frontier and depth-18 terminal task without changing learned dimensions or parameter count.

### Mandatory low-scale bridge for the replacement scorer

Before its high-scale frontier can be interpreted, the new checkpoint set must demonstrate the already-established mechanism on fresh bridge worlds at smaller N. The exact bridge criteria must be frozen before replacement-checkpoint training or evaluation.

At minimum the bridge must establish:

- learned bounded-score routing > matched answer-blind bounded control;
- a sensible monotone or near-monotone K frontier at N64/N128/N256;
- global learned reference remains viable.

Failure of this bridge is `MODEL_TRANSITION_NOT_QUALIFIED`, not a population-scaling result.

## High-scale tensor runtime

The high-scale runner must not use one Python object per candidate as its primary representation.

Prepare a tensorized candidate bank:

- recurrent states `[batch, population, 64]`;
- scores `[batch, population]`;
- compact integer path/prefix identity;
- live/terminal masks;
- deterministic answer-blind integer sampling;
- bounded parent selection reads only its sampled K neural scores before selection;
- global mode may read all live scores;
- matched hash control reads zero neural scores before selection.

Logical sparse activation and physical GPU execution are separate:

- each world retains the frozen logical active-parent/child budget;
- selected neural updates from many independent worlds may be gathered into large GPU batches and scattered back;
- this is execution batching, not additional learned work.

The tensor engine must pass exact small-scale transcript equivalence against the qualified eager reference before high-scale admission.

## Remove accidental N-scaling overhead before frontier search

The high-scale baseline must remove known implementation costs that should not define the scientific frontier:

- no full-reserve path sort merely to sample K candidates;
- no full-reserve score sort after every bounded decision solely for telemetry in the hot path;
- no per-child CUDA `.item()` extraction;
- no one-Python-object-per-candidate hot representation;
- no O(N) tuple rebuild to delete one parent;
- no cryptographic SHA-256 + full sort for routine answer-blind capacity pruning;
- no repeated state clone/list/stack churn where indexed tensor gather/scatter suffices.

Diagnostic telemetry that requires O(N) work may be sampled or executed out-of-band if the frozen protocol permits, but it must never alter scientific decisions or runtime state.

Compiler/CUDA-graph modes remain separate execution variables. First establish a clean eager tensor baseline.

## Exploratory screening design

The high-scale campaign is frontier localization, not final inference.

Prepared screening target:

- 64 fresh worlds/checkpoint/tier;
- same worlds paired across K/global/control conditions;
- deterministic paired-bootstrap summaries;
- preserve every tested tier and failure.

At each N, evaluate:

1. `global_score`;
2. bounded-score K ladder values valid for that N;
3. matched answer-blind hash control(s) for the K values used in primary bandwidth decisions.

To keep the matrix tractable, admission may freeze one matched hash control for each candidate K or a preregistered subset sufficient to establish learned-score value. This choice must be fixed before any Gate-7 world is exposed.

## Per-K criteria

For checkpoint C and population N, a bounded score treatment K is a **routing pass candidate** only if both hold:

- `CI_low(score_K - hash_K) > 0`;
- `CI_low(score_K - global_score) > -0.05`.

The 0.05 NI margin is inherited from Gates 5–6.

For a population N, define exploratory `K_required(N)` as the **smallest preregistered K** whose two criteria pass on all three frozen checkpoints.

If no tested K passes, report `K_required(N) > K_max_tested`; do not insert a new K after exposure for rescue.

Because this is sequential exploratory localization, its repeated intervals are not a final familywise-controlled confirmation claim.

## Scaling summaries

For every completed N report:

- `K_required(N)` or lower bound;
- K/N fraction;
- score observations/world for each K;
- global score observations/world;
- learned-routing delta versus matched hash;
- near-global delta;
- coverage/reference viability;
- wall time and peak VRAM as engineering telemetry only.

Useful descriptive ratios include:

`K_required(N) / N`

and doubling ratios:

`K_required(2N) / K_required(N)`.

Do not fit or claim an asymptotic law unless enough independent N tiers support it and a later protocol preregisters that analysis.

## Sequential high-scale rule

After the scale-neutral scorer and tensor engine are separately qualified:

1. start at N1024;
2. verify global reference viability;
3. evaluate the frozen K ladder from smallest upward;
4. record the smallest passing K, or the tested lower bound if none passes;
5. continue doubling N while a bounded K materially smaller than N remains viable and resource limits allow;
6. stop scientific routing interpretation if the global reference fails;
7. stop engineering execution separately on a resource limit;
8. preserve all failures;
9. do not retrain checkpoints, alter the NI margin, or add rescue K values after exposure;
10. later freeze a high-power confirmation around representative population/K points that define the observed bandwidth frontier.

## Distinct stop classes

A stop must be classified as one of:

- `MODEL_TRANSITION_NOT_QUALIFIED`;
- `ROUTING_BANDWIDTH_FRONTIER_REACHED`;
- `REFERENCE_FRONTIER_REACHED`;
- `RESOURCE_FRONTIER_REACHED`;
- `CAMPAIGN_CEILING_REACHED`.

A VRAM/OOM/runtime stop is not a scientific routing failure.

## Claims boundary

This campaign can estimate over the tested finite range how much score visibility is needed as population grows.

It cannot by itself establish:

- an asymptotic Big-O law for K_required(N);
- physical decentralization;
- communication-network scalability;
- multi-machine scaling;
- capability-per-FLOP/per-joule superiority;
- AGI/general intelligence;
- a true universal maximum population.

## Preparation boundary

While this file remains preparation-only:

- no Gate-7 world namespace;
- no high-scale scientific data;
- no replacement scorer training;
- no result classifier execution;
- no confirmation namespace.

Only data-blind mechanics, tensor-bank infrastructure, work/resource accounting, profiler preparation, and protocol checks may be prepared.
