# Repaired One-Checkpoint Population Curve v0 — Development Result

## Status

Development-only result. Confirmation remains unopened and this PR is never merged.

Workflow run: `30230714665`

One persisted/reloaded checkpoint with **26,669 learned parameters** and fingerprint:

`55b7772697da2588eb388035181b4ab19a5914ca8432bcedaaa5ce440097275b`

was reused across every relay depth, population size, and communication condition.

Training preserved the #60 mixed-population protocol and added only the two mechanisms localized by #69–#72:

- training-only oracle gate-selection loss, averaged across relay hops, weight 1.0;
- parameter-free softmax-normalized population aggregation.

Training summary:

- 2,000 steps;
- batch size 64;
- 128,000 training worlds;
- initial total loss `2.11305`;
- final relay loss `0.05081`;
- final gate loss `0.01575`;
- final total loss `0.06656`;
- mean last-50 total loss `0.07737`.

Development evaluation used 1,000 matched worlds per relay depth and populations `1 / 4 / 16 / 64 / 256`.

## Raw communicating solve curve

| Active workers | relay-2 | relay-4 | relay-8 |
| ---: | ---: | ---: | ---: |
| 1 | 5.1% | 8.1% | 3.3% |
| 4 | 26.4% | 26.4% | 2.0% |
| 16 | 50.1% | 50.0% | 33.6% |
| 64 | 75.0% | 74.8% | 66.5% |
| 256 | **99.9%** | **99.6%** | **98.9%** |

No-communication exact solve was **0% at every population and every relay depth**.

Endpoint gains from 1 -> 256 active workers:

- relay-2: **+94.8 percentage points**;
- relay-4: **+91.5 points**;
- relay-8: **+95.6 points**.

Communication endpoint advantage at width 256:

- relay-2: **+99.9 points**;
- relay-4: **+99.6 points**;
- relay-8: **+98.9 points**.

## Information availability

The benchmark deliberately controls the first population at which the full relay chain is present.

Information-complete rates:

| Active workers | relay-2 | relay-4 | relay-8 |
| ---: | ---: | ---: | ---: |
| 1 | 0% | 0% | 0% |
| 4 | 25% | 25% | 0% |
| 16 | 50% | 50% | 33.3% |
| 64 | 75% | 75% | 66.7% |
| 256 | 100% | 100% | 100% |

## Solve rate given information complete

This is the critical population-stability diagnostic.

| Active workers | relay-2 | relay-4 | relay-8 |
| ---: | ---: | ---: | ---: |
| 4 | **100%** | **100%** | — |
| 16 | **99.8%** | **99.8%** | **99.70%** |
| 64 | **100%** | **99.73%** | **99.55%** |
| 256 | **99.9%** | **99.6%** | **98.9%** |

Thus the same learned checkpoint remains approximately 99–100% correct once the required information is inside the active population, from the smallest complete population through 256 workers and from 2-hop through 8-hop relay.

This is strong development evidence that the repaired learned computation itself is population-stable across the tested ladder.

## Important benchmark shortcut found during audit

The low-width raw solve rates must **not** be interpreted as successful reasoning on information-incomplete worlds.

The v0 generator places all chain edges randomly inside the selected scope threshold. Therefore the final chain edge can appear in the visible prefix before the whole chain is present. Its value is the answer key.

At one active worker, normalized aggregation has only one candidate and therefore gives it weight 1.0. Exact solve occurs whenever slot 0 happens to contain the final chain edge.

A deterministic generator audit over the exact 1,000 development seeds gives:

- relay-2 final-answer edge visible at width 1: **51/1000 = 5.1%**;
- relay-4: **81/1000 = 8.1%**;
- relay-8: **33/1000 = 3.3%**.

These values match the observed width-1 solve rates **exactly**.

Therefore the width-1 raw baseline contains a structural answer-exposure shortcut. Wider incomplete cohorts contain the same potential exposure, although learned query selectivity suppresses much of it.

This does not invalidate the solve-given-information-complete result, and it does not explain the near-99% width-256 endpoint. But it means the raw v0 population curve is not clean enough to promote as canonical scaling evidence without a benchmark correction.

## Interpretation

The preregistered development outcomes were:

1. raw capability increases and solve-given-complete stays strong -> one fixed checkpoint converts additional runtime population into usable capability once information becomes available;
2. raw rises while complete-conditional solve collapses -> learned population computation is not population-stable;
3. communication does not beat no communication -> shared computation adds no usable capability.

The learned-computation result matches case 1 very strongly: complete-conditional accuracy remains approximately 99–100%, and no-communication remains 0%.

However, the raw curve contains the newly identified answer-exposure shortcut and should be treated as **provisional development evidence**, not final Gate-v0 evidence.

## Next boundary

Create a new relay benchmark version that prevents the final answer value from becoming visible before the declared scope threshold. The simplest deterministic correction is to place the final chain edge at the frontier slot that is guaranteed to lie beyond the previous population point, while retaining the remaining chain-edge randomization.

Then rerun the same repaired one-checkpoint development protocol without changing training hyperparameters or model architecture.

Required outcome before confirmation:

- incomplete-world exact solve should fall to approximately chance / negligible levels;
- raw capability should track information availability cleanly;
- solve-given-information-complete should retain the strong population-stable behavior seen here;
- no-communication should remain at chance/zero.

Do not open confirmation until that benchmark correction is qualified and the development curve is rerun.
