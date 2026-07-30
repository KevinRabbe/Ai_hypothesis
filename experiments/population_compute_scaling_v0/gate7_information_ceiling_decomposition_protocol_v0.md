# Gate-7 information-ceiling decomposition v0 — frozen protocol

## Status

**DATA-FROZEN DIAGNOSTIC PROTOCOL — EXECUTION CLOSED.**

Exact qualified routing-continuation result head:

`4591dae55cada819e848ae7f929d5e8f2b8805d6`

Bound continuation evidence:

- result SHA-256: `4921ea99b44156f08271d6fb2b2e0bcba98ef6a646ed0aaf040762d47aa03b36`;
- audit SHA-256: `92f52a9e7fad3cb5d8962a9127a0cd7140656a0a8f03cfba08fe7cd5376a03fd`;
- manifest SHA-256: `ee9dcefbaf5efe9a75b20d407cb1a4f47ff0b04bbdce4613f2539b76af2c8cca`.

The completed continuation established that learned routing remains viable through N131072 while absolute global coverage declines. The next question is not whether K must grow beyond 512. It is whether the declining coverage is primarily:

1. an unavoidable consequence of the public noisy-hint information budget and fixed 128 terminal parent attempts; or
2. a recoverable learned scorer representation/ranking deficit.

This branch opens no execution path and contains no result values from the future diagnostic.

## Why this diagnostic precedes communication experiments

Gate-7 Stage A constructs the complete binary parent frontier. For population N, frontier index `i` is exactly the depth-`log2(N)` binary parent path `i`. Therefore the hidden terminal parent exists exactly once in every complete frontier.

Stage B evaluates both binary terminal children for every selected parent. Consequently terminal coverage fails only when the hidden parent is absent from the selected parent set. It is not caused by choosing the wrong terminal action after selecting the correct parent.

Under the frozen task:

- every prefix bit has one public hint;
- each hint independently equals the hidden bit with probability 0.70;
- hidden paths are uniform;
- exactly 128 parents are activated in the primary condition.

For this model, the Bayes-optimal public parent ranking is Hamming distance from the public hint prefix. All candidates in one Hamming shell have equal posterior probability, so ties must be resolved by an answer-independent public deterministic tie stream.

Before introducing worker relays, recycling, specialization or communication topology, we must determine whether the existing model is already close to this task-specific Bayes ranking ceiling.

## Analytic primary ceiling

For frontier depth `d = log2(N)`, the hidden parent’s mismatch count is distributed as:

`H ~ Binomial(d, 0.30)`.

A top-M Bayes ranker admits complete Hamming shells in ascending distance and an `M / shell_size` fraction of the boundary shell under public answer-independent tie-breaking.

The exact expected top-128 parent-coverage values frozen before diagnostic execution are:

| Population | Frontier depth | Exact Bayes top-128 expectation |
| ---: | ---: | ---: |
| 16,384 | 14 | 0.1725810781290399 |
| 32,768 | 15 | 0.12944371790375186 |
| 65,536 | 16 | 0.09386607328230152 |
| 131,072 | 17 | 0.06627595867880422 |

The declining ceiling follows from increased hidden-path entropy under fixed hint reliability and fixed terminal attempts. It is not fitted from the observed continuation result.

## Frozen populations and checkpoints

All three exact transition checkpoints are retained:

- T0 SHA `be1b18e22a1f713b48de8934aca1c0302f1255342684158c6ae31a7c3618a719`, fingerprint `0be5c26d1056da1bf12a53be5ba5e6d1cadb7815eb117e2a77db269391c1c5aa`;
- T1 SHA `a9b712736f440168ccd86ee070b8d301be0eb834b9d17418d40c0175207cadfb`, fingerprint `b9685382992cb2f94454b6faa3675d458f236e7b0c7d8399bea256bedcb02e46`;
- T2 SHA `cd7869511d07679cb1ad430743cad031299a575413e8a853ef9116d545c7480a`, fingerprint `1bca7012c7350c3b3fe8c9790a65c36eea8fcd8bef3e123034817ff78673a7bb`.

The fixed population ladder is:

```text
N16384
N32768
N65536
N131072
```

No population may be omitted based on an earlier diagnostic outcome.

## Fresh evidence standard

A future separately admitted execution must use:

- 512 fresh worlds per population;
- exactly eight physical batches of 64 worlds;
- a fresh information-ceiling hidden namespace;
- a fresh hint namespace;
- a fresh runtime namespace;
- a separate public answer-independent tie namespace;
- a separate paired-bootstrap namespace;
- 10,000 deterministic paired-bootstrap samples;
- the same fixed 19,649 learned parameters;
- no training or checkpoint selection.

The completed routing-continuation worlds may not be reused as fresh evidence.

## Frozen ranker matrix

Every checkpoint/population must evaluate all three rankers over the same complete frontier:

1. `learned_score_rank` — the checkpoint’s learned frontier scores;
2. `bayes_hint_likelihood_rank` — exact Hamming-distance likelihood with public deterministic tie-breaking;
3. `public_hash_rank` — answer-independent public hash order with no hint or learned score.

Hidden parent identity is used only after all public rankings are constructed, solely to compute rank and coverage.

## Frozen attempt curve

Every ranker reports top-M hidden-parent coverage for the complete fixed curve:

```text
M = 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024
```

M128 is the primary preregistered comparison because it matches the completed Gate-7 Stage-B parent budget. The larger M values are descriptive decomposition points. They do not authorize a post-result increase in Stage-B budget or a rescue execution.

## Required measurements

For every checkpoint/population/ranker, the result must preserve:

- exact hidden-parent rank for all 512 worlds;
- top-M coverage vector for every fixed M;
- mean and quantiles of hidden-parent rank;
- mean reciprocal rank;
- mean `log2(rank + 1)`;
- complete public ranking identity/checksum;
- learned parameter count and fingerprint where applicable.

For primary M128, 10,000 paired bootstraps must report:

- learned minus Bayes coverage interval;
- learned minus hash coverage interval;
- Bayes minus hash coverage interval.

## Frozen two-point near-ceiling margin

The absolute non-inferiority margin is:

`0.02`.

A checkpoint/population is **near ceiling** only when:

1. learned-minus-Bayes 95% CI low is strictly greater than `-0.02`;
2. learned-minus-hash 95% CI low is strictly positive;
3. Bayes-minus-hash 95% CI low is strictly positive.

A checkpoint/population has a **clear scorer gap** only when:

1. learned-minus-Bayes 95% CI high is strictly below `-0.02`;
2. Bayes-minus-hash 95% CI low is strictly positive.

Intervals crossing the margin are neither silently passed nor silently failed.

## Frozen campaign outcomes

### `G7_INFORMATION_CEILING_DOMINANT`

All twelve checkpoint/population rows satisfy the near-ceiling rule.

Interpretation: the declining Gate-7 success rate is primarily explained by the benchmark’s public information and fixed-attempt ceiling. The next protocol must normalize information difficulty or explicitly vary runtime attempt/communication budgets before claiming population capability scaling.

### `G7_SCORER_REPRESENTATION_GAP`

All twelve rows satisfy the clear-scorer-gap rule.

Interpretation: recoverable public information exists, but the learned recurrent scorer does not rank it adequately. Scorer representation/calibration becomes the next target before communication.

### `G7_INFORMATION_AND_SCORER_GAP_MIXED`

At least one row has a clear scorer gap, but not all rows do.

Interpretation: the limitation changes across checkpoint or depth. Any next study must target the exact failing strata rather than introducing a universal coordination mechanism.

### `G7_INFORMATION_CEILING_INCONCLUSIVE`

No row has a clear scorer gap, but one or more rows fail the all-near-ceiling rule because uncertainty crosses the frozen margin or a learned/hash/Bayes/hash contrast is not positive.

Interpretation: no architecture intervention is admitted from this evidence alone.

## Scientific boundaries

The future diagnostic may not:

- retrain or fine-tune any checkpoint;
- select a checkpoint after exposure;
- add a communication, relay, recycling or specialization mechanism;
- change hint reliability;
- adapt M after observing results;
- use hidden answers to construct rankings or resolve ties;
- reuse continuation worlds as fresh evidence;
- claim that the analytic Bayes expectation is an empirical result;
- claim asymptotic behavior outside the frozen population ladder;
- open a second diagnostic or rescue condition from the result branch.

This study determines whether the apparent coordination bottleneck is genuinely architectural or is first a benchmark information-budget effect. Only after that distinction is resolved may a concrete coordination intervention be preregistered.
