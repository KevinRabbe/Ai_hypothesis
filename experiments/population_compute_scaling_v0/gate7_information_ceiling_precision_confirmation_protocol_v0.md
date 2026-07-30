# Gate-7 information-ceiling precision confirmation v0 — protocol

## Status

**DATA-FROZEN PRECISION-CONFIRMATION PROTOCOL — EXECUTION CLOSED.**

Exact qualified decomposition-result head:

`4eb3e50a3ca7898ff81aebebddb7b049ff855df3`

Preceding frozen outcome:

`G7_INFORMATION_CEILING_INCONCLUSIVE`.

This protocol does not reinterpret the preceding result. It defines one independent fresh-evidence study to distinguish information-ceiling non-inferiority from a small recoverable scorer deficit with greater statistical precision.

## Why another diagnostic is necessary

In the validated decomposition, every learned-versus-Bayes M128 point estimate was inside the frozen two-percentage-point margin. The largest deficit was 0.017578125 and the mean absolute deficit was 0.00830078125. However, several cell-wise confidence intervals crossed -0.02, while none established a clear scorer gap with CI high below -0.02.

The appropriate response is not an architecture intervention. It is a higher-precision replication with a preregistered pooled analysis and explicit local-gap guards.

## Frozen matrix

```text
checkpoints = T0, T1, T2
populations = 16384, 32768, 65536, 131072
fresh worlds per population = 2048
checkpoint-world rows = 24576
physical batch size = 64
physical batches per checkpoint/population = 32
primary attempt budget = M128
rankers = learned score, Bayes hint likelihood, public hash
paired bootstrap samples = 20000
learned parameters = 19649
hint reliability = 0.70
near-ceiling margin = 0.02
```

The full M1..M1024 rank curve may be reported descriptively because exact hidden-parent ranks determine every attempt threshold without additional model execution. Only M128 participates in the frozen classifier.

## Freshness and non-intervention

The future study must use new namespaces for:

- hidden paths;
- noisy hints;
- runtime seeds;
- Bayes tie-breaking;
- public-hash ordering;
- bootstrap sampling.

It may not reuse decomposition or continuation worlds as fresh observations.

The study performs no:

- training;
- checkpoint selection;
- parameter change;
- communication step;
- relay;
- recycling;
- specialization;
- topology search;
- recurrent-budget change;
- adaptive attempt exposure;
- M256/M512/M1024 rescue.

## Primary estimand

The primary estimand is the equal-population, equal-checkpoint M128 learned-minus-Bayes coverage difference.

For each population, the same fresh world indices are evaluated at T0, T1 and T2. The bootstrap therefore resamples world indices **within population as clusters shared across all three checkpoints**. Each bootstrap replicate:

1. independently resamples 2,048 world indices within each population;
2. applies each sampled index to the T0/T1/T2 paired vectors for that population;
3. averages the three checkpoint differences inside each population;
4. averages the four population means with equal weight.

This preserves the correlation created by shared worlds and shared Bayes/hash references while preventing large populations or checkpoints from receiving accidental extra weight.

The same clustered procedure is applied to learned-versus-hash and Bayes-versus-hash controls.

## Secondary and guard analyses

The result must also report:

- all twelve cell-wise M128 point differences and paired intervals;
- four population-pooled M128 differences and clustered intervals across T0/T1/T2;
- the global pooled learned-versus-Bayes interval;
- global pooled learned-versus-hash and Bayes-versus-hash intervals;
- complete rank summaries and checksums for all rankers;
- the descriptive full attempt curve;
- exact checkpoint identities and parameter fingerprints.

Population and cell analyses are local-gap guards. They cannot replace or rescue the primary pooled analysis.

## Frozen classifier

### `G7_PRECISION_INFORMATION_CEILING_DOMINANT`

Return this outcome only when all conditions hold:

1. pooled learned-minus-Bayes CI low is greater than -0.02;
2. pooled learned-minus-hash CI low is greater than zero;
3. pooled Bayes-minus-hash CI low is greater than zero;
4. every population-pooled learned-minus-Bayes point estimate is greater than -0.02;
5. no population-pooled or cell-wise comparison establishes a clear scorer gap with CI high below -0.02 and positive Bayes-over-hash evidence.

This outcome establishes that the benchmark information/attempt ceiling is the dominant explanation at the tested scales and precision. It does not prove the scorer is mathematically Bayes-identical.

### `G7_PRECISION_SCORER_REPRESENTATION_GAP`

Return this outcome when:

1. pooled learned-minus-Bayes CI high is below -0.02; and
2. pooled Bayes-minus-hash CI low is greater than zero.

This establishes a recoverable global ranking deficit larger than the frozen margin.

### `G7_PRECISION_INFORMATION_AND_SCORER_GAP_MIXED`

Return this outcome when the pooled global scorer-gap rule does not hold, but at least one population-pooled or cell-wise comparison establishes a clear scorer gap with CI high below -0.02 and positive Bayes-over-hash evidence.

This prevents a strong local failure from being hidden by global averaging.

### `G7_PRECISION_INCONCLUSIVE`

Return this outcome in every remaining case, including:

- pooled uncertainty crossing -0.02;
- failed learned-over-hash or Bayes-over-hash controls;
- a population point estimate outside the margin without a clear local gap;
- incomplete or invalid evidence.

## Interpretation boundary

No communication or coordination intervention is admitted until this precision confirmation resolves.

If the information-ceiling-dominant outcome passes, the current benchmark must be redesigned or normalized so that additional population computation can create measurable headroom before coordination mechanisms are judged.

If a scorer gap passes, the next study targets representation/calibration of the shared scorer while keeping communication unchanged.

If a mixed result passes, the next study isolates the affected populations/checkpoints before architecture changes.

If inconclusive repeats, no result-dependent rescue is allowed. The research record must state that the current task and feasible evidence budget cannot resolve the distinction.

## Protocol-only boundary

This branch may contain only:

- immutable standard-library protocol constants and classifier;
- structural/classifier tests;
- this documentation;
- protocol-only CI.

It contains no Torch import, checkpoint loader, world generator, rank executor, artifact reader, wrapper, result file, communication mechanism or execution-admission flag.
