# Gate-5 v0 — bounded score-visibility adaptive activation

## Status

**FROZEN PRE-RESULT PROTOCOL — NO GATE-5 DEVELOPMENT WORLD GENERATED OR INSPECTED.**

Gate-5 starts from the fully qualified Gate-4 confirmation evidence head:

`b5407dc917c096af961a11e34ce379c83d637183`

Gate-4 established that a frozen learned score can route a fixed active neural-work budget more effectively than both a matched static schedule and an answer-blind dynamic-priority control when the scheduler may rank the complete live reserve.

Gate-5 does not retest that result. It asks whether useful learned routing survives when the scheduler is allowed to inspect only a **bounded subset of live candidate scores per activation**.

This is a controlled information-routing/scalability experiment. It is not yet a claim about physical multi-device decentralization or network communication.

## Primary question

With learned parameters, public information, latent capacity, active neural width, total learned recurrent work and search topology fixed, can a learned adaptive scheduler retain near-global capability while observing only a bounded number of candidate scores per activation?

## Frozen model/checkpoints

Reuse the exact three already-confirmed checkpoints:

- C0 SHA-256 `e63748a1182472d03c540f9123c3442ce44e130519e2176681034568826db590`, fingerprint `e4f9990f08e85393a155637cfa50b5831d779770447ec3716fc9e67946992afc`;
- C1 SHA-256 `8ce384627f5543fa4fe78498f9990f8214cf2a4afed0c9b734e86761ad13c989`, fingerprint `2a57984f755ee1404fd828c2df36a5efff8f13a541b7d7c93a891014fbf4897c`;
- C2 SHA-256 `103826bbd9451b965eced5134d1674cb8e893f5d3b378cb828312ffdb6fc9a37`, fingerprint `8afaf956f200f41ea914eafdd1b5f151dd303cae1552165c74e12bb8c945af02`.

Learned parameter count remains exactly **19,649/checkpoint**.

No training or fine-tuning is permitted.

## Frozen workload

Use a new deterministic Gate-5 development namespace disjoint from every Gate-3/Gate-4 development and confirmation namespace.

Per world:

- hidden binary path depth: `8`;
- noisy-hint reliability: `0.70`;
- public/hidden separation unchanged;
- runtime may inspect only public noisy hints, candidate paths, candidate neural states/scores allowed by the active scheduler condition, and deterministic scheduler metadata;
- hidden path is available only to the evaluator after search execution.

Development world count: **256**.

Evaluation batch size: **64 worlds**.

Use **2,000 deterministic paired bootstrap samples**.

## Fixed population and active neural work

Every condition uses:

- latent reserve capacity: **L256**;
- persistent candidate-specific recurrent state;
- two active child lanes per productive parent expansion;
- eight recurrent updates per generated child;
- exactly **159 scheduled parent-expansion slots/world**;
- exactly **2,544 learned recurrent updates/world**.

There is no compiler, CUDA graph, fusion or mixed-precision treatment in Gate-5.

If a scheduled slot cannot be used productively, matched sink work must preserve the exact recurrent-update total and must not enter search state.

## Fixed two-stage search topology

All conditions share the same first stage.

### Stage A — generation-synchronous warm-up

Use exactly **63 parent-expansion slots** to build the complete depth-6 frontier:

`1 + 2 + 4 + 8 + 16 + 32 = 63`

This produces exactly **64 distinct depth-6 live hypotheses** before bounded/global adaptive selection begins.

No treatment-specific score visibility is used during this breadth warm-up.

### Stage B — adaptive activation

Exactly **96 parent-expansion slots** remain:

`159 - 63 = 96`

These 96 slots use the scheduler condition to choose one live nonterminal parent at a time. Children are processed by the exact same frozen recurrent scorer and returned to the common L256 reserve unless terminal.

Because expanding a depth-6 parent replaces one live candidate with two depth-7 children, while expanding a depth-7 parent removes one live candidate and generates terminal paths, the adaptive phase retains a substantial live population and therefore creates a meaningful score-visibility bottleneck.

## Canonical live-reserve ordering

For bounded-visibility conditions only, maintain live hypotheses in deterministic lexicographic path order for scheduler indexing.

At each adaptive slot:

1. let `N` be the current live nonterminal count;
2. derive an answer-blind start index and stride from SHA-256 over experiment version, runtime seed, slot index and condition identifier;
3. choose `min(K,N)` distinct reserve indices by modular traversal using a stride coprime to `N`;
4. only those candidate scores are visible to the bounded-score selector for that slot.

The index-selection rule may inspect `N` and candidate path-order metadata but may not inspect non-sampled neural scores.

This protocol treats score visibility as the constrained information channel. It does **not** claim that lexicographic indexing itself is a physical distributed-network implementation.

## Frozen scheduler conditions

### 1. `global_score`

Gate-4-style full-visibility adaptive scheduler.

At each Stage-B slot, inspect every live candidate score and activate the globally highest candidate under the existing frozen score quantization and deterministic tie-break rule.

This is the capability reference, not the scalability target.

### 2. `bounded_score_k4`

At each Stage-B slot, expose exactly `min(4,N)` candidates through the frozen bounded sampler and activate the highest learned-score candidate among only those visible candidates.

### 3. `bounded_score_k8`

Same, with `K=8`.

### 4. `bounded_score_k16`

Same, with `K=16`.

This is the **primary bounded-visibility treatment**.

### 5. `bounded_score_k32`

Same, with `K=32`.

### 6. `bounded_hash_k16`

Use the **exact same K=16 visible candidate subset** as `bounded_score_k16`, but ignore neural scores when choosing the parent. Select the parent using a separately namespaced deterministic answer-blind SHA ordering over runtime seed, slot index and candidate path.

This isolates learned routing information from the mechanics of bounded candidate visibility.

## Fixed numerical semantics

Where learned scores are compared, reuse the already-qualified Gate-3/Gate-4 score quantization and deterministic answer-blind tie-break rules unchanged.

No Gate-5-specific score tolerance, ranking threshold or sampling heuristic may be tuned after results.

## Required invariants

Within checkpoint/world, every condition must have identical:

- hidden path;
- noisy hints;
- recurrent checkpoint;
- candidate encoder;
- initial state;
- L256 maximum reserve capacity;
- Stage-A 63-slot breadth warm-up;
- Stage-B 96-slot adaptive budget;
- two child lanes per productive slot;
- eight recurrent updates/child;
- total 159 scheduled slots;
- total 2,544 learned recurrent updates/world.

Only **Stage-B score visibility / parent-selection policy** differs.

## Required telemetry

Per world/condition record at minimum:

- exact-solution coverage;
- generated and unique terminal count;
- productive/sink slot counts;
- total learned recurrent updates;
- live nonterminal population by Stage-B slot;
- activated parent depth by Stage-B slot;
- visible candidate count by Stage-B slot;
- total candidate-score observations during Stage B;
- maximum candidate-score observations in one Stage-B slot;
- selected candidate rank within the visible subset where applicable;
- selected candidate global score rank as **evaluation-only telemetry** computed after the decision, never available to bounded conditions at runtime;
- number of Stage-B selections that match the global-score condition on the paired world;
- runtime seed/world index;
- parameter count/fingerprint.

The independent auditor must reconstruct the 159-slot / 2,544-update work identity and bounded score-visibility limits from raw telemetry.

## Development matrix

Evaluate all six conditions on the same 256 worlds for each frozen checkpoint:

`3 checkpoints × 6 conditions = 18 cells`.

## Frozen primary comparisons

For each checkpoint reconstruct paired exact-coverage effects from raw per-world vectors.

### Learned bounded-routing effect

`bounded_score_k16 - bounded_hash_k16`

This must be positive to show the learned score remains useful under the bounded information channel.

### Non-inferiority to global routing

`bounded_score_k16 - global_score`

Freeze non-inferiority margin:

`delta_NI = 0.05`

Gate-5 treats K16 as non-inferior only if the paired-bootstrap 95% CI low is strictly greater than `-0.05`.

The margin is frozen before Gate-5 data exposure. It means K16 may lose at most five absolute percentage points of exact coverage at the lower confidence bound while reducing per-activation score visibility from the full live reserve to at most 16 candidates.

## Secondary bounded-visibility frontier

K4, K8 and K32 are preregistered descriptive frontier points.

For each checkpoint report:

- coverage;
- paired delta vs `global_score`;
- paired delta vs `bounded_hash_k16` only where scientifically interpretable;
- total score observations;
- score-observation reduction vs global.

Also report the smallest `K ∈ {4,8,16,32}` whose `K - global_score` paired CI low exceeds `-0.05` on **all three checkpoints**. This is descriptive development evidence and may be `none`.

## Frozen development outcome precedence

Define for the primary K16 treatment:

- `L_i`: CI low of `bounded_score_k16 - bounded_hash_k16` is `> 0` on checkpoint `i`;
- `N_i`: CI low of `bounded_score_k16 - global_score` is `> -0.05` on checkpoint `i`.

Evaluate outcomes in this exact order:

### G5-B4 — bounded learned routing harmful

Assign `G5_B4_BOUNDED_LEARNED_ROUTING_HARMFUL` if `bounded_score_k16 - bounded_hash_k16` has CI high `< 0` on one or more checkpoints.

### G5-B2 — robust bounded-score activation

Assign `G5_B2_ROBUST_BOUNDED_SCORE_ACTIVATION` if all three checkpoints satisfy both `L_i` and `N_i`, with every work/information/visibility invariant valid.

### G5-B3 — checkpoint-sensitive bounded effect

Assign `G5_B3_CHECKPOINT_SENSITIVE_BOUNDED_EFFECT` if either the `L_i` truth values differ across checkpoints or the `N_i` truth values differ across checkpoints.

### G5-B1 — learned local signal but material global gap

Assign `G5_B1_LEARNED_SIGNAL_WITH_GLOBAL_GAP` if all three `L_i` are true but all three `N_i` are false.

### G5-B0 — bounded learned routing not established

Assign `G5_B0_BOUNDED_LEARNED_ROUTING_NOT_ESTABLISHED` if all three `L_i` are false.

Otherwise assign `G5_MIXED_BOUNDED_SCORE_PATTERN`.

The classifier is frozen before development data exposure.

## Confirmation boundary

Gate-5 confirmation is **CLOSED**.

Only `G5_B2_ROBUST_BOUNDED_SCORE_ACTIVATION` may permit a later separately frozen confirmation protocol on an untouched namespace with unchanged checkpoints, work, warm-up, visibility and scheduler semantics.

No Gate-5 confirmation world may be generated or inspected during development.

## Claims boundary

A positive Gate-5 result would support only the narrow statement that useful learned work allocation can survive a bounded candidate-score visibility channel under this controlled search regime.

It would not establish:

- physical decentralization;
- distributed-machine latency tolerance;
- arbitrary graph/locality robustness;
- AGI/general intelligence;
- optimal communication complexity;
- per-FLOP/per-joule superiority;
- scaling to 20K/100K workers;
- that K16 is universally sufficient.

Those require later experiments.

## Stop rule

After the first admitted 18-cell development matrix is inspected:

- do not change K values;
- do not change the 0.05 non-inferiority margin;
- do not change the 63/96 slot split;
- do not change total 159 slots / 2,544 learned updates;
- do not change L256;
- do not change depth/reliability;
- do not change bounded sampling semantics;
- do not add conditions;
- do not retrain checkpoints;
- do not open confirmation except through the frozen B2 path.

Negative, mixed and positive outcomes are all permanent evidence.
