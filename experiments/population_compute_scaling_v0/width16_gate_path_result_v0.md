# Width-16 Relay Gate-Path Diagnostic v0 — Development Result

## Status

Development-only diagnostic over corrected #64. No confirmation data was opened and this is not a Gate-v0 population-scaling result.

Workflow run: `30229486275`

Training/evaluation protocol was frozen in PR #68 before the result was inspected:

- relay-2;
- active width 16;
- corrected exact compositional query/value protocol;
- hop-local worker-state reset;
- training seed 901;
- 2,048 optimizer steps;
- batch size 16;
- state width 64;
- message width 24;
- 512 held-out threshold-16 information-complete worlds;
- 26,669 learned parameters.

## Result

Ordinary held-out relay performance:

- exact solve rate: **0.0%**;
- bit accuracy: **58.724%**;
- first loss: `0.72285`;
- best loss: `0.61284`;
- final loss: `0.69983`;
- mean last-100 loss: `0.66920`.

### Hop 1 — clean start query

Correct chain worker gate:

- top-1 rate: **12.30%**;
- mean rank: **7.01 / 16**;
- mean correct logit: `-0.6227`;
- mean best-nonmatch logit: `-0.3839`;
- mean margin: **-0.2387**;
- median margin: **-0.2571**.

### Hop 2 — model-produced query

Correct chain worker gate:

- top-1 rate: **7.81%**;
- mean rank: **8.16 / 16**;
- mean correct logit: `-0.7997`;
- mean best-nonmatch logit: `-0.5150`;
- mean margin: **-0.2847**.

### Hop 2 — oracle-clean intermediate query

Correct chain worker gate:

- top-1 rate: **11.72%**;
- mean rank: **6.97 / 16**;
- mean correct logit: `-0.6241`;
- mean best-nonmatch logit: `-0.3844`;
- mean margin: **-0.2397**.

### First-hop shared-query fidelity

Model-produced hop-1 shared field versus the exact clean intermediate-node query representation:

- mean cosine similarity: **0.3968**;
- mean RMSE: **0.6514**;
- mean L2 distance: **3.1914**.

## Frozen interpretation

The preregistered interpretation was:

- poor hop-1 gate ranking -> gate learning / end-to-end credit assignment failure;
- good hop 1 + good oracle hop 2 but poor model hop 2 -> first-hop query corruption;
- poor oracle hop 2 -> gate does not generalize key/query discrimination at width 16 under relay training;
- strong gate ranks but poor solve -> inspect shared/readout transformation.

The observed result matches the first and third cases.

The gate is already weak at hop 1 from a clean start query, and replacing the model-produced hop-2 query with an oracle-clean intermediate query does not restore strong selection. Query corruption is real and makes model hop 2 worse, but it is not the primary width-16 failure.

## Next diagnostic

Test whether the unchanged inference architecture can solve width-16 relay-2 when training receives an auxiliary, training-only gate-selection objective for the correct worker at hop 1 and under an oracle-clean hop-2 query.

Interpretation:

- strong gates + strong relay solve -> end-to-end relay credit assignment is the main bottleneck;
- strong gates + poor relay solve -> aggregation/shared/readout remains a separate bottleneck;
- gate supervision still fails to produce strong gate ranking -> gate parameterization/capacity is the bottleneck under relay conditions.

The auxiliary oracle signal is diagnostic training information only and is not proposed as part of the final population architecture.
