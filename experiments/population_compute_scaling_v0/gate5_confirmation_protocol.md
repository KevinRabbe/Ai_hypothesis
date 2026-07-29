# Gate-5 v0 — bounded score-visibility confirmation protocol

## Status

**FROZEN BEFORE ANY GATE-5 CONFIRMATION WORLD IS GENERATED OR INSPECTED.**

This confirmation protocol starts from the fully qualified Gate-5 B2 development-evidence head:

`17ff9d0ace35a9b7b97f240154050251dca5d630`

The first admitted development run returned:

`G5_B2_ROBUST_BOUNDED_SCORE_ACTIVATION`

which is the only frozen Gate-5 development outcome allowed to open a separately versioned confirmation test.

## Confirmation question

On an untouched world namespace, does bounded learned score visibility again preserve near-global adaptive-routing capability while strongly outperforming a matched answer-blind bounded-routing control, with learned parameters, public information, L256 population, active neural width, search topology and total learned recurrent work fixed?

## Frozen checkpoints/model

Reuse the exact three already-confirmed checkpoints:

- C0 SHA-256 `e63748a1182472d03c540f9123c3442ce44e130519e2176681034568826db590`, fingerprint `e4f9990f08e85393a155637cfa50b5831d779770447ec3716fc9e67946992afc`;
- C1 SHA-256 `8ce384627f5543fa4fe78498f9990f8214cf2a4afed0c9b734e86761ad13c989`, fingerprint `2a57984f755ee1404fd828c2df36a5efff8f13a541b7d7c93a891014fbf4897c`;
- C2 SHA-256 `103826bbd9451b965eced5134d1674cb8e893f5d3b378cb828312ffdb6fc9a37`, fingerprint `8afaf956f200f41ea914eafdd1b5f151dd303cae1552165c74e12bb8c945af02`.

Learned parameter count remains exactly **19,649/checkpoint**.

No training or fine-tuning is permitted.

## Frozen task and namespace

Per confirmation world:

- hidden binary path depth: `8`;
- noisy-hint reliability: `0.70`;
- public/hidden separation unchanged;
- hidden path available only to the evaluator after search execution;
- runtime may inspect only the same public information and scheduler-allowed candidate information as development.

Use exactly **512 untouched confirmation worlds** from deterministic namespaces disjoint from Gate-5 development and every Gate-3/Gate-4 namespace.

No confirmation world may be generated, executed or inspected during protocol qualification.

## Fixed population, topology and learned work

Every condition uses:

- latent reserve capacity: **L256**;
- persistent candidate-specific recurrent state;
- two active child lanes/productive parent expansion;
- eight recurrent updates/generated child;
- exactly **159 scheduled parent-expansion slots/world**;
- exactly **2,544 learned recurrent updates/world**.

The search topology remains exactly:

### Stage A — common breadth warm-up

- exactly **63** generation-synchronous parent-expansion slots;
- builds the complete depth-6 frontier;
- produces exactly **64 distinct depth-6 live hypotheses**;
- no treatment-specific score visibility is used.

### Stage B — adaptive activation

- exactly **96** remaining parent-expansion slots;
- same scheduler semantics as development;
- matched sink work must preserve exact total learned work if a productive expansion is unavailable.

## Frozen six-condition matrix

Evaluate all six development conditions unchanged:

1. `global_score`
2. `bounded_score_k4`
3. `bounded_score_k8`
4. `bounded_score_k16` — **primary bounded treatment**
5. `bounded_score_k32`
6. `bounded_hash_k16` — matched learned-routing control

For bounded score conditions, only the sampled K candidate scores may be read before parent selection. Full-reserve score ranking is permitted only after the parent is irrevocably chosen and only as evaluation telemetry.

The K16 score/hash control pair uses the same deterministic `k16` sampling-group rule whenever their incoming reserves are identical. Later reserve divergence caused by different parent choices is part of the treatment effect and does not redefine the sampler.

## Confirmation matrix

For each of the three frozen checkpoints evaluate all six conditions on the same 512 confirmation worlds:

`3 checkpoints × 6 conditions = 18 cells`.

Evaluation batch size: **64 worlds**.

Use **4,000 deterministic paired bootstrap samples**.

## Frozen primary confirmation effects

For each checkpoint reconstruct from raw per-world exact-coverage vectors:

### Learned bounded-routing effect

`bounded_score_k16 - bounded_hash_k16`

Acceptance requires paired-bootstrap 95% CI low strictly `> 0` on C0, C1 and C2.

### Non-inferiority to global learned routing

`bounded_score_k16 - global_score`

The development-frozen non-inferiority margin remains unchanged:

`delta_NI = 0.05`

Acceptance requires paired-bootstrap 95% CI low strictly `> -0.05` on C0, C1 and C2.

No post-development change to the primary K or margin is permitted.

## Descriptive frontier

K4, K8 and K32 remain preregistered descriptive frontier points.

For each checkpoint report:

- coverage;
- paired delta vs `global_score`;
- paired-bootstrap CI;
- Stage-B score-observation count and reduction vs global.

Also report the smallest `K ∈ {4,8,16,32}` whose `K - global_score` paired CI low exceeds `-0.05` on **all three confirmation checkpoints**. This remains descriptive and is not an acceptance gate.

Development found `smallest_noninferior_k = 8`; confirmation may replicate, improve or fail that descriptive frontier without changing the primary K16 confirmation verdict.

## Confirmation acceptance

Assign final outcome:

`GATE5_CONFIRMED_BOUNDED_SCORE_ACTIVATION`

**only if all of the following hold:**

1. independent artifact audit accepts the artifact with no errors;
2. exact checkpoint identities/fingerprints/parameter counts match frozen values;
3. world namespace, public/hidden separation, L256 capacity, Stage-A/Stage-B topology and 159-slot / 2,544-update work invariants pass;
4. strict bounded-score visibility invariants pass for every bounded condition;
5. on every checkpoint C0/C1/C2, `bounded_score_k16 - bounded_hash_k16` paired-bootstrap CI low is strictly `> 0`;
6. on every checkpoint C0/C1/C2, `bounded_score_k16 - global_score` paired-bootstrap CI low is strictly `> -0.05`.

If artifact/mechanical/information invariants fail, classify the evidence as invalid rather than positive or negative.

If the artifact is valid but any preregistered benefit/non-inferiority condition fails, assign:

`GATE5_CONFIRMATION_NOT_ESTABLISHED`.

There is no post-result retuning, alternate-K rescue, subset selection, checkpoint rescue or alternate confirmation namespace in v0.

## Required telemetry

Per world/condition preserve at minimum:

- exact-solution coverage;
- generated and unique terminal count;
- productive/sink slot counts;
- total learned recurrent updates;
- Stage-A frontier width;
- live nonterminal population by Stage-B slot;
- activated parent depth by Stage-B slot;
- visible candidate count by Stage-B slot;
- score observations by Stage-B slot;
- total/max Stage-B score observations;
- selected visible rank where applicable;
- selected global score rank as post-decision evaluation-only telemetry;
- selected parent path by Stage-B slot;
- runtime seed/world index;
- checkpoint parameter count/fingerprint.

The independent auditor reconstructs work identity, visibility bounds, paired effects and non-inferiority from raw vectors/telemetry.

## Claims boundary

A positive confirmation would support only the narrow statement that, within this controlled fixed-population hypothesis-search regime, learned adaptive work allocation can survive a bounded candidate-score visibility channel: K16 remains useful relative to a matched answer-blind bounded control and retains near-global capability under the frozen five-percentage-point non-inferiority margin.

It would not establish:

- physical decentralization or distributed-machine execution;
- arbitrary communication graphs/locality robustness;
- AGI or arbitrary-task generalization;
- optimal communication complexity;
- per-FLOP/per-joule superiority;
- scaling to 20K/100K runtime workers;
- universal sufficiency of K8/K16.

## Stop rule

After the first admitted 512-world confirmation matrix is inspected:

- do not change checkpoints;
- do not change K values;
- do not change the 0.05 non-inferiority margin;
- do not change 63/96 slot split;
- do not change total 159 slots / 2,544 learned updates;
- do not change L256;
- do not change depth/reliability;
- do not change bounded sampling or strict score-visibility semantics;
- do not add conditions;
- do not rerun alternate confirmation namespaces to seek a favorable outcome.

The first valid admitted confirmation matrix is final Gate-5 v0 evidence.
