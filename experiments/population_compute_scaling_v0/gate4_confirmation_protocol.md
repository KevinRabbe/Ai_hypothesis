# Gate-4 v0 — adaptive activation confirmation protocol

## Status

**FROZEN BEFORE ANY GATE-4 CONFIRMATION WORLD IS GENERATED OR INSPECTED.**

This confirmation protocol starts from the fully qualified Gate-4 A2 development evidence head:

`caf4ca3ff74191b4005a29e7df0242852ab5411a`

The development run returned:

`G4_A2_ROBUST_ADAPTIVE_ACTIVATION_BENEFIT`

which is the only development outcome permitted by the frozen Gate-4 protocol to open a separately versioned confirmation test.

## Confirmation question

On an untouched world namespace, does the same frozen learned score-driven adaptive scheduler again improve exact search coverage over both:

1. the matched static generation-synchronous schedule; and
2. the same dynamic queue mechanics using answer-blind hash priority;

while learned parameters, information, latent capacity, active neural width and total learned recurrent work remain fixed?

## Frozen model/checkpoints

Reuse the exact three Gate-3 v1 / Gate-4 development checkpoints:

- C0 SHA-256 `e63748a1182472d03c540f9123c3442ce44e130519e2176681034568826db590`, fingerprint `e4f9990f08e85393a155637cfa50b5831d779770447ec3716fc9e67946992afc`;
- C1 SHA-256 `8ce384627f5543fa4fe78498f9990f8214cf2a4afed0c9b734e86761ad13c989`, fingerprint `2a57984f755ee1404fd828c2df36a5efff8f13a541b7d7c93a891014fbf4897c`;
- C2 SHA-256 `103826bbd9451b965eced5134d1674cb8e893f5d3b378cb828312ffdb6fc9a37`, fingerprint `8afaf956f200f41ea914eafdd1b5f151dd303cae1552165c74e12bb8c945af02`.

Learned parameter count remains exactly **19,649/checkpoint**.

No training or fine-tuning is permitted.

## Frozen task and information

Per confirmation world:

- hidden binary path depth: `8`;
- noisy-hint reliability: `0.70`;
- public/hidden separation unchanged from development;
- runtime may inspect only public noisy hints and its own candidate neural states/scores;
- hidden answer is available only to the evaluator after search execution.

Confirmation uses exactly **512 untouched worlds** from a deterministic namespace disjoint from Gate-4 development and all Gate-3 namespaces.

No confirmation world may be generated, executed or inspected during protocol qualification.

## Frozen latent population and active work

Every condition uses:

- latent reserve capacity: **L256**;
- persistent candidate-specific recurrent state;
- two active child lanes per productive parent expansion;
- eight recurrent updates per generated child;
- exactly **159 scheduled parent-expansion slots/world**;
- exactly **2,544 learned recurrent updates/world**.

If a condition cannot spend a scheduled slot productively, matched sink work must preserve the exact recurrent-update total and may not affect search state.

## Frozen scheduler conditions

The confirmation matrix reuses the three development schedulers unchanged:

### `adaptive_score`

At each slot choose the highest-ranked currently live nonterminal parent by the already-frozen quantized neural score and deterministic answer-blind tie break, expand it, return nonterminal children to L256 reserve, and record terminal children.

### `static_generation`

Use the unchanged matched static schedule:

- first 127 parent slots construct the complete depth-7 frontier;
- remaining 32 slots expand the top 32 depth-7 parents by the same frozen score/tie-break rule.

### `adaptive_hash`

Use the same one-parent-at-a-time dynamic queue mechanics as `adaptive_score`, but activation priority is the already-frozen deterministic SHA-based answer-blind ordering rather than neural score.

No scheduler semantics may change after confirmation data exposure.

## Confirmation matrix

For each of the three frozen checkpoints evaluate all three schedulers on the same 512 confirmation worlds:

`3 checkpoints × 3 scheduler conditions = 9 cells`.

Evaluation batch size: **64 worlds**.

Use **4,000 deterministic paired bootstrap samples**.

## Frozen paired comparisons

For each checkpoint reconstruct from raw per-world exact-coverage vectors:

1. `adaptive_score - static_generation` — primary adaptive-allocation confirmation effect;
2. `adaptive_score - adaptive_hash` — learned-routing confirmation control;
3. `static_generation - adaptive_hash` — secondary descriptive scheduler comparison.

## Confirmation acceptance

Assign final outcome:

`GATE4_CONFIRMED_ADAPTIVE_ACTIVATION_BENEFIT`

**only if all of the following hold:**

1. independent artifact audit accepts the artifact with no errors;
2. all checkpoint identities/fingerprints/parameter counts match the frozen values;
3. all information, L256 population and 159-slot / 2,544-update work invariants pass;
4. for **every checkpoint C0/C1/C2**, paired-bootstrap 95% CI low is strictly `> 0` for `adaptive_score - static_generation`;
5. for **every checkpoint C0/C1/C2**, paired-bootstrap 95% CI low is strictly `> 0` for `adaptive_score - adaptive_hash`.

The secondary `static_generation - adaptive_hash` comparison is not an acceptance gate.

If artifact/mechanical invariants fail, classify the evidence as invalid rather than positive or negative.

If the artifact is valid but either preregistered benefit fails to clear zero in any checkpoint, assign:

`GATE4_CONFIRMATION_NOT_ESTABLISHED`.

There is no post-result retuning, subset selection or additional checkpoint rescue path in v0.

## Required telemetry

Per world/condition preserve at minimum:

- exact-solution coverage;
- generated terminal count and unique terminal count;
- productive slot count;
- sink slot count;
- total learned recurrent updates;
- maximum live nonterminal population;
- mean live nonterminal population;
- activated parent-depth histogram;
- terminal-generation slot indices;
- runtime seed and world index;
- parameter count and fingerprint.

The independent auditor reconstructs paired effects and work identity from raw vectors/telemetry.

## Claims boundary

A positive confirmation would support only the narrow statement that, within this controlled fixed-population hypothesis-search regime, a frozen learned hypothesis score can route a fixed active neural-work budget more effectively than both a matched static breadth schedule and an answer-blind dynamic-priority control.

It would not establish:

- AGI or arbitrary-task generalization;
- globally optimal scheduling;
- superiority over all possible serial/replay/search algorithms;
- per-FLOP/per-joule superiority;
- that adaptive score routing always helps;
- unconstrained population or compute scaling.

## Stop rule

After the first admitted 512-world confirmation matrix is inspected:

- do not change checkpoints;
- do not change L256;
- do not change 159 slots / 2,544 recurrent updates;
- do not change scheduler semantics;
- do not change depth/reliability;
- do not add conditions;
- do not rerun alternate confirmation namespaces to seek a favorable outcome.

The first valid admitted confirmation matrix is final evidence for Gate-4 v0.
