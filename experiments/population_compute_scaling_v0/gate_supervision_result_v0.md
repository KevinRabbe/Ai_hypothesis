# Relay Gate-Supervision Diagnostic v0 — Development Result

## Status

Development-only diagnostic over corrected #64. No confirmation data was opened. The auxiliary oracle labels existed only during training; ordinary held-out inference was unchanged.

Workflow run: `30229746392`

## Frozen protocol

- relay-2;
- active width 16;
- 26,669 learned parameters;
- training seed 901;
- 2,048 optimizer steps;
- batch size 16;
- state width 64 / message width 24;
- threshold-16 information-complete training and held-out worlds;
- 512 held-out worlds;
- ordinary relay BCE plus training-only gate loss;
- gate-loss weight fixed at 1.0 before result inspection.

Auxiliary objective:

`total_loss = relay_BCE + 0.5 * (hop1_gate_CE + oracle_clean_hop2_gate_CE)`

No inference path, aggregation rule, communication rule, readout, or learned parameter was changed.

## Result

Ordinary held-out inference:

- exact solve rate: **99.8047%**;
- bit accuracy: **99.9674%**.

Training losses:

- first total loss: `3.53254`;
- final total loss: `0.010264`;
- mean last-100 total loss: `0.011496`;
- final relay loss: `0.006021`;
- final gate loss: `0.004243`.

### Hop 1 — clean start query

- correct-worker top-1: **100%**;
- mean rank: **1.0 / 16**;
- mean correct logit: `2.5142`;
- mean best-nonmatch logit: `-4.9536`;
- mean margin: **+7.4678**;
- median margin: **+7.4347**.

### Hop 2 — ordinary model-produced query

- correct-worker top-1: **100%**;
- mean rank: **1.0 / 16**;
- mean correct logit: `2.2752`;
- mean best-nonmatch logit: `-5.0341`;
- mean margin: **+7.3093**.

### Hop 2 — oracle-clean query diagnostic

- correct-worker top-1: **100%**;
- mean rank: **1.0 / 16**;
- mean correct logit: `2.4953`;
- mean best-nonmatch logit: `-5.0068`;
- mean margin: **+7.5021**.

### First-hop shared-query fidelity

Model-produced hop-1 shared field versus exact clean next-query representation:

- mean cosine similarity: **0.99746**;
- mean RMSE: **0.04326**;
- mean L2 distance: **0.21193**.

## Frozen interpretation

The preregistered outcomes were:

1. strong gates + strong relay solve -> end-to-end relay credit assignment is the main bottleneck;
2. strong gates + poor relay solve -> aggregation/shared-query/readout remains a separate bottleneck;
3. gate supervision remains weak -> gate parameterization/capacity is the bottleneck.

The observed result is case 1.

The unchanged inference architecture is capable of almost perfectly solving width-16 relay-2 once training teaches the existing gate which worker should win. Strong selectivity also makes the ordinary summed shared field nearly identical to the clean next-query representation, so the earlier query corruption was largely a consequence of weak population selection rather than an independent representational limit.

This does **not** establish the population-scaling hypothesis. It establishes a narrower training result:

> The current shared-weight architecture has sufficient width-16 relay capacity; the previous failure was primarily training/credit assignment, not fixed architectural capacity.

## Next diagnostic

Before changing the canonical training protocol, test the same fixed auxiliary gate objective at independent widths 4, 16, 64 and 256 under the existing equalized-active-worker training-budget diagnostic.

Interpretation:

- strong fixed-width performance through 256 -> the selectivity mechanism scales and the next question is mixed-population training with the auxiliary objective;
- degradation with width despite strong gate supervision -> population aggregation/selectivity capacity still imposes a width-dependent limit;
- gates remain strong but final solve degrades -> inspect shared-field accumulation/readout at the failing width.

No confirmation data or Gate-v0 pass/fail decision is opened by this result.
