# Exact-identity fixed-width relay capacity diagnostic v0

Development-only result from temporary PR #67 / workflow run `30224757440`.

## Why this rerun was required

The earlier fixed-width and competitive-aggregation diagnostics were measured before #64's relay protocol was corrected from:

`candidate = tanh(node_projection(value)) -> shared = tanh(candidate)`

to the exact compositional contract:

`candidate = node_projection(value) -> shared = tanh(candidate)`.

The corrected path guarantees that perfect candidate selection produces exactly the same bounded representation used for that node as a fresh query.

## Design

Fresh independent relay-2 checkpoints were trained at fixed widths `4 / 16 / 64 / 256` with batches `64 / 16 / 4 / 1`, keeping 256 active worker states per optimizer batch before the two recurrent rounds.

Every checkpoint used the same architecture and **26,669 learned parameters**. Training used 2,048 development-only steps on threshold-matched information-complete worlds. Evaluation used 512 disjoint threshold-matched held-out worlds per width. No confirmation data was opened.

## Result

| Width | Sparse exact | Sparse bit accuracy | No-comm exact | Mean last-100 loss |
| ---: | ---: | ---: | ---: | ---: |
| 4 | **99.8047%** | 99.9186% | 0% | 0.00451 |
| 16 | **0%** | 58.7240% | 0% | 0.6692 |
| 64 | **0%** | 52.5391% | 0% | 0.6901 |
| 256 | **0%** | 50.9115% | 0% | 0.7014 |

The corrected exact-identity implementation therefore preserves a nearly solved width-4 relay-2 regime but still collapses at width 16 and above.

The inherited mixed-population diagnostic on the same corrected bytes also remained effectively unchanged: solve-given-information-complete was 2.34% at width 4, 0.39% at width 16, and 0% at 64/256.

## Interpretation

The earlier double-`tanh` distortion was a genuine protocol defect but **not the population-width scaling bottleneck**.

The next uncertainty is now narrower than aggregation magnitude or message identity:

> At width 16, does the trained gate fail to select the correct worker, or does hop 1 select adequately but produce a query that corrupts hop-2 selection?

The next diagnostic must inspect the true chain worker's gate logit/rank/margin at:

1. hop 1 from the clean start query;
2. hop 2 from the model-produced shared query;
3. hop 2 from an oracle-clean intermediate-node query used for diagnosis only.

Also measure similarity between the model-produced hop-1 shared field and the exact clean intermediate query representation.

Interpretation is frozen:

- poor hop-1 gate ranking -> gate learning / end-to-end credit assignment failure;
- good hop-1 and good oracle hop-2, but poor model hop-2 -> hop-1 query corruption;
- poor oracle hop-2 -> the learned gate itself does not generalize key/query discrimination at width 16 under relay training;
- strong gate ranks at both hops but poor final solve -> inspect shared/readout transformation next.

Do not add model capacity or another reducer before this decomposition.

No Gate-v0 population-scaling conclusion is claimed.
