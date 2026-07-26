# Fixed-width relay capacity diagnostic v0

Development-only result from temporary PR #65 / workflow run `30223713106`.

## Question

After #64 made relay-2 learnable at fixed width 4, can the same repaired architecture learn relay-2 independently at fixed widths 16, 64 and 256?

Each width used a fresh checkpoint with the same 26,669 learned parameters. Training batches were scaled so each optimizer step processed 256 active worker states before the two recurrent rounds:

- width 16 × batch 16;
- width 64 × batch 4;
- width 256 × batch 1.

Every training and held-out world was information-complete at exactly the tested width. No confirmation data was opened.

## Result

| Width | Sparse exact | Sparse bit accuracy | No-comm exact | Mean last-100 loss |
| ---: | ---: | ---: | ---: | ---: |
| 16 | 0.390625% | 58.19% | 0% | 0.6655 |
| 64 | 0% | 52.70% | 0% | 0.6872 |
| 256 | 0% | 49.90% | 0% | 0.7010 |

The three trained checkpoints retain the same learned parameter count but have independent fingerprints as expected.

## Interpretation

The repaired relay architecture does **not** merely suffer from mixed-width curriculum interference. It already fails when trained independently at fixed width 16, and degrades further toward chance as population size increases.

This localizes the next bottleneck to **population aggregation/selectivity**.

The strongest supporting contrast is the earlier one-hop diagnostic: the same general local gate/shared-field path can retrieve one matching value at width 16 with 95.85% exact accuracy. Failure emerges when a population-wide selection must produce a sufficiently clean shared query to support another selection round.

The current sparse field sums independently sigmoid-gated candidate messages. Residual nonmatching gate mass therefore has a population-dependent accumulation path. A small nonmatch emission that is harmless with three distractors can dominate when tens or hundreds of distractors are active.

## Next diagnostic

Test the smallest parameter-free scale-normalization of the shared aggregation while preserving the same learned gate and message content:

- convert per-worker gate logits into normalized competitive weights across active workers;
- keep communication O(N × message_width);
- keep learned parameter count unchanged;
- rerun fixed-width relay-2 at 4/16/64/256.

If normalized competition restores larger-width learnability, aggregation scale was the blocker. If not, inspect gate discrimination itself at each hop before adding model capacity.

No Gate-v0 population-scaling result is claimed.
