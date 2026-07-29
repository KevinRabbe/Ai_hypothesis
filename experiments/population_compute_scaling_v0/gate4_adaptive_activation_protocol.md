# Gate-4 v0 — adaptive activation under fixed population and work

## Status

**FROZEN PRE-RESULT PROTOCOL — NO GATE-4 DEVELOPMENT RESULT INSPECTED.**

Gate-4 starts from the fully qualified Gate-3 v3 confirmation evidence head:

`b77748314789550b3ba141b5fbfeadcb75dc1e1b`

Gate-3 established that larger persistent hypothesis capacity can improve capability when the smaller population is genuinely capacity-binding. Gate-4 does not retest population capacity. Instead, it fixes latent capacity and asks whether the organism can allocate its limited active neural work more intelligently.

## Primary question

With learned parameters, public information, latent population capacity, active neural width and total learned recurrent work fixed, can a score-driven **adaptive activation scheduler** improve exact search coverage over a matched static generation-synchronous scheduler by choosing which live hypotheses receive the next expensive neural expansion?

This is a controlled test of **where active neural work is spent**, not how much work exists.

## Fixed model/checkpoints

Reuse the exact three Gate-3 v1 checkpoints already confirmed in Gate-3 v3:

- checkpoint 0 SHA-256: `e63748a1182472d03c540f9123c3442ce44e130519e2176681034568826db590`;
- checkpoint 1 SHA-256: `8ce384627f5543fa4fe78498f9990f8214cf2a4afed0c9b734e86761ad13c989`;
- checkpoint 2 SHA-256: `103826bbd9451b965eced5134d1674cb8e893f5d3b378cb828312ffdb6fc9a37`.

Learned parameter count remains exactly **19,649/checkpoint**.

No training or fine-tuning is permitted in Gate-4 v0.

## Workload

Use a new deterministic development namespace.

Per world:

- binary hidden path depth: `8`;
- noisy-hint reliability: `0.70`;
- same public/hidden separation as Gate-3;
- runtime may inspect only public noisy hints and its own neural states/scores;
- hidden path is available only to the evaluator after search execution.

Development world count: **256**.

The Gate-4 development namespace must be disjoint from all Gate-3 development and confirmation namespaces.

## Fixed latent population

All Gate-4 conditions use:

- latent reserve capacity `L = 256`;
- persistent candidate-specific recurrent states;
- two active child lanes per productive parent expansion;
- eight recurrent updates per generated child.

Population capacity is therefore **not** a treatment variable in Gate-4.

## Fixed active-work budget

Every condition receives exactly **159 scheduled parent-expansion slots/world**.

Each scheduled productive slot evaluates two children, each with eight recurrent updates:

`159 × 2 × 8 = 2,544 learned recurrent updates/world`.

If a scheduler cannot use a scheduled slot productively, it must execute matched sink work so the learned recurrent-update total remains exactly `2,544`.

No dummy work may enter the candidate population or affect search decisions.

### Why 159 slots

A full generation-synchronous build through the depth-7 frontier requires:

`1 + 2 + 4 + 8 + 16 + 32 + 64 = 127` parent expansions.

The frozen budget therefore leaves exactly `32` final parent expansions for the static breadth schedule, generating at most `64` terminal paths. This avoids the near-ceiling behavior of Gate-3 v3 while leaving the adaptive scheduler enough work to trade breadth against selective depth.

The value `159` is frozen before any Gate-4 capability result.

## Frozen scheduler conditions

All conditions operate on the same worlds, same checkpoint, same L256 reserve, same recurrent model and same 159-slot work budget.

### 1. `adaptive_score`

A global sparse-activation scheduler.

At each scheduled parent slot:

1. consider every currently live nonterminal hypothesis in the reserve;
2. order candidates by the existing frozen quantized neural score, then the existing deterministic answer-blind tie break;
3. activate exactly the highest-ranked parent;
4. generate its two children using the frozen recurrent scorer;
5. return nonterminal children to the persistent reserve;
6. record terminal children as generated solutions and remove them from the expandable reserve;
7. if insertion would exceed L256, retain the top L256 nonterminal candidates by the same frozen score/tie-break rule.

The scheduler sees no hidden-answer information.

### 2. `static_generation`

Matched static breadth baseline.

Build complete generations synchronously from depth 0 through depth 7 while work remains. With 159 slots:

- the first 127 slots construct the unpruned depth-7 frontier;
- the remaining 32 slots expand the top 32 depth-7 parents according to the same frozen neural score/tie-break rule;
- all unused scheduled slots, if any, become matched sink work.

This baseline uses the scorer for within-generation ranking but cannot move work between depths dynamically.

### 3. `adaptive_hash`

Answer-blind dynamic-scheduler control.

Use the same global one-parent-at-a-time adaptive mechanics as `adaptive_score`, but parent activation priority is a deterministic SHA-based ordering over:

- Gate-4 runtime seed;
- scheduled slot index;
- candidate path.

Neural states and scores are still computed exactly as in the other conditions, but the neural score is **not used to decide which parent becomes active next**.

This tests whether any benefit comes from learned score-guided work allocation rather than merely from asynchronous queue mechanics.

## Frozen numerical ordering

Where neural score ordering is used, retain the already-qualified Gate-3 score quantization and deterministic answer-blind tie-break semantics unchanged.

No Gate-4-specific score tolerance or ranking threshold may be tuned after results.

## Fixed information and causality

Across all three conditions, within checkpoint/world:

- same hidden path;
- same public noisy hints;
- same candidate encoder;
- same recurrent scorer;
- same initial state;
- same learned parameters;
- same L256 reserve capacity;
- same two active child lanes per productive slot;
- same eight recurrent updates per child;
- same 159 scheduled slots;
- same 2,544 learned recurrent updates/world.

Only the **activation schedule** differs.

## Required telemetry

Per world/condition record at minimum:

- exact-solution coverage;
- generated terminal count;
- unique generated terminal count;
- productive slot count;
- sink slot count;
- total learned recurrent updates;
- maximum live nonterminal population;
- mean live nonterminal population;
- number of distinct parent depths activated;
- productive activations by parent depth;
- terminal-generation slot index for every generated terminal;
- parameter count and fingerprint;
- runtime seed and world index.

The auditor must reconstruct work identity from raw telemetry.

## Development matrix

For each of the three frozen checkpoints evaluate all three scheduler conditions on the same 256 Gate-4 development worlds:

`3 checkpoints × 3 scheduler conditions = 9 cells`.

Evaluation batch size: **64 worlds**.

Use deterministic paired bootstrap intervals with **2,000** samples.

## Frozen paired comparisons

For each checkpoint reconstruct from raw per-world exact-coverage vectors:

1. `adaptive_score - static_generation` — primary dynamic-activation effect;
2. `adaptive_score - adaptive_hash` — learned-routing-signal control;
3. `static_generation - adaptive_hash` — descriptive scheduler baseline comparison.

## Frozen development interpretation

### G4-A0 — no adaptive allocation benefit

Assign `G4_A0_NO_ADAPTIVE_ALLOCATION_BENEFIT` if `adaptive_score - static_generation` has CI low `<= 0` in all three checkpoints.

### G4-A1 — learned routing signal without static-schedule advantage

Assign `G4_A1_ROUTING_SIGNAL_ONLY` if all three checkpoints have `adaptive_score - adaptive_hash` CI low `> 0`, but the primary `adaptive_score - static_generation` comparison does not have CI low `> 0` in all three checkpoints.

### G4-A2 — robust adaptive activation benefit

Assign `G4_A2_ROBUST_ADAPTIVE_ACTIVATION_BENEFIT` only if all three checkpoints satisfy:

- `adaptive_score - static_generation` paired-bootstrap CI low `> 0`;
- `adaptive_score - adaptive_hash` paired-bootstrap CI low `> 0`;
- all work/population/information invariants pass.

This would support the narrow mechanism that a persistent population can use learned hypothesis scores to decide **where** to spend a fixed active neural-work budget more effectively than a matched static breadth schedule.

### G4-A3 — checkpoint-sensitive adaptive effect

Assign `G4_A3_CHECKPOINT_SENSITIVE_ADAPTIVE_EFFECT` if the primary significance differs across checkpoints while artifact and work invariants remain valid.

### G4-A4 — learned routing is harmful

Assign `G4_A4_LEARNED_ROUTING_HARMFUL` if `adaptive_score - adaptive_hash` is negative with bootstrap CI high `< 0` in one or more checkpoints.

Otherwise assign `G4_MIXED_ADAPTIVE_PATTERN`.

## Confirmation boundary

Gate-4 confirmation is **CLOSED**.

Only `G4_A2_ROBUST_ADAPTIVE_ACTIVATION_BENEFIT` may permit a later separately frozen confirmation protocol using an untouched namespace and unchanged checkpoints/work/scheduler semantics.

No confirmation world may be generated or inspected during Gate-4 development.

## Claims boundary

A positive Gate-4 development result would not establish AGI, arbitrary-task generalization, optimal scheduling, per-FLOP superiority, or that dynamic activation always beats breadth-first search.

It would support only a narrow controlled statement about learned score-guided allocation of a fixed active neural-work budget within the frozen hypothesis-search regime.

## Stop rule

After the nine-cell development matrix is inspected:

- do not change the 159-slot budget;
- do not change depth/reliability;
- do not add scheduler conditions;
- do not change score quantization/tie breaks;
- do not retrain checkpoints;
- do not alter L256 capacity;
- do not open confirmation unless the frozen interpretation map permits a separately versioned confirmation protocol.

Negative, mixed and positive outcomes are all permanent evidence.
