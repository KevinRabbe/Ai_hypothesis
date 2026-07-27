# Width-256 Normalized Transport Diagnostic v0 — Development Result

## Status

Development-only same-checkpoint ablation over corrected #64. No confirmation data was opened. One width-256 gate-supervised checkpoint was trained once and evaluated without weight changes under two reducers.

Workflow run: `30230279663`

## Result

The trained gate itself was highly selective:

- mean correct-worker sigmoid gate: **0.99943**;
- mean total nonmatch sigmoid gate mass: **15.0399**;
- mean nonmatch/correct sigmoid mass ratio: **15.04×**;
- mean correct-worker softmax weight: **0.96723**.

Thus the correct worker is individually almost fully open, but 255 small independent sigmoid leaks collectively outweigh it by about fifteen to one.

### Ordinary sigmoid-sum path

- exact solve: **0.0%**;
- bit accuracy: **51.04%**;
- first-hop shared→clean cosine: **0.33115**;
- RMSE: **0.90263**;
- hop-2 correct-worker top-1: **4.6875%**;
- mean hop-2 rank: **70.62 / 256**.

### Same checkpoint, softmax-normalized path

- exact solve: **0.0%**;
- bit accuracy: **55.09%**;
- first-hop shared→clean cosine: **0.99887**;
- RMSE: **0.02003**;
- hop-2 correct-worker top-1: **100%**;
- mean hop-2 rank: **1.0 / 256**.

Learned parameters remained **26,669** and the checkpoint fingerprint was unchanged during both evaluations.

## Frozen interpretation

The preregistered outcomes were:

1. normalized query + solve restore -> residual message accumulation is primary;
2. normalized query restores but solve does not -> readout is coupled to the old aggregation distribution;
3. normalized query does not restore -> deeper message geometry/representation issue.

The observed result is case 2.

Parameter-free normalization completely repairs the **communication/query transport** failure on the same learned checkpoint: the next-query representation and second-hop worker selection become essentially perfect. Final answer decoding remains poor because the checkpoint's recurrent/readout path was optimized only while the ordinary shared field was corrupted.

This provides direct causal evidence for two distinct failures that appeared sequentially:

1. ordinary relay BCE did not teach strong population selectivity;
2. once selectivity is taught, independent sigmoid message accumulation is not population-size stable.

## Next diagnostic

Train a fresh width-256 relay-2 checkpoint with both mechanisms active during training and inference:

- the fixed training-only gate-selection objective from #69;
- parameter-free softmax-normalized population aggregation.

Keep learned parameter count, local worker architecture, message representation, state-reset policy, task, training seed/budget and held-out protocol otherwise unchanged.

Interpretation:

- width-256 solve becomes strong -> credit assignment + population-normalized transport jointly remove the observed scaling failure;
- gates/query remain strong but solve stays weak -> recurrent pooled-state/readout is independently limiting at width 256;
- gates/query degrade during normalized training -> the two mechanisms interact and require a different optimization formulation.

No Gate-v0 conclusion is claimed.
