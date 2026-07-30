# Gate-7 information-ceiling decomposition v0 — result

## Status

**VALID FRESH INFORMATION-CEILING DECOMPOSITION EVIDENCE — THE FROZEN CLASSIFIER RETURNED `G7_INFORMATION_CEILING_INCONCLUSIVE`.**

Exact frozen protocol head:

`3640699f1727886c9ad2e954269fad660dc34370`

Exact admitted execution head:

`161142c1e5552cb9464216c774397def6a4100be`

Exact qualified audit-recovery head:

`b06959ce9d6c7d83bac953ce17a3c3008ad0f306`

Independent recovered audit:

```text
artifact_valid = true
errors = []
campaign_outcome = G7_INFORMATION_CEILING_INCONCLUSIVE
scientific_status = FRESH_GATE7_INFORMATION_CEILING_DECOMPOSITION_EVIDENCE
scientific_rerun = false
```

The scientific artifact was not modified during audit recovery. No training, checkpoint selection, communication intervention, adaptive attempt exposure or continuation-world reuse occurred.

## Immutable evidence identities

```text
result_sha256 = 71a383ced44419f84022738448c460d79a3fb21746f436649e5f14399704f731
rejected_audit_sha256 = 0d9ef5a838810a681afbe23c46b0a98be1a395b31fda83cccc1fdb52376c733e
recovered_audit_sha256 = 86a7dbb774119cca9bcd697978081e0872b41e4e61a3f8b08538e0cc89c8397d
recovery_record_sha256 = ccd4bbd353aba09b8a2d38d155bb9f883b862123bf196693889b515d5452324b
manifest_sha256 = 026f75a76888efe020c57da9d719140169eedd5e024555db20da9590cfea2b45
```

The rejected audit is retained because it documents the post-exposure JSON-object-order defect. Its rejection did not compare or invalidate the scientific rank vectors. The recovered audit canonicalized only temporary `ranks_by_ranker` insertion order and verified the original result hash remained unchanged.

## Frozen design

- exact transition checkpoints T0/T1/T2;
- populations N16384, N32768, N65536 and N131072;
- 512 fresh worlds/checkpoint/population;
- learned-score, Bayes hint-likelihood and public-hash ranks;
- attempt curve M1, M2, M4, M8, M16, M32, M64, M128, M256, M512 and M1024;
- primary attempt budget M128;
- 10,000 deterministic paired-bootstrap samples;
- frozen absolute near-ceiling margin of 0.02;
- fixed 19,649 learned parameters;
- no communication, recycling, relay, specialization or topology intervention.

## Primary M128 matrix

| Population | Checkpoint | Learned | Bayes | Public hash | Learned − Bayes | Learned-vs-Bayes 95% CI |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 16,384 | T0 | 0.173828125 | 0.177734375 | 0.007812500 | -0.003906250 | [-0.0156,+0.0078] |
| 16,384 | T1 | 0.160156250 | 0.177734375 | 0.007812500 | -0.017578125 | [-0.0332,-0.0020] |
| 16,384 | T2 | 0.169921875 | 0.177734375 | 0.007812500 | -0.007812500 | [-0.0234,+0.0078] |
| 32,768 | T0 | 0.132812500 | 0.132812500 | 0.007812500 | +0.000000000 | [-0.0059,+0.0059] |
| 32,768 | T1 | 0.117187500 | 0.132812500 | 0.007812500 | -0.015625000 | [-0.0312,-0.0020] |
| 32,768 | T2 | 0.125000000 | 0.132812500 | 0.007812500 | -0.007812500 | [-0.0234,+0.0078] |
| 65,536 | T0 | 0.080078125 | 0.085937500 | 0.000000000 | -0.005859375 | [-0.0156,+0.0039] |
| 65,536 | T1 | 0.078125000 | 0.085937500 | 0.000000000 | -0.007812500 | [-0.0234,+0.0078] |
| 65,536 | T2 | 0.082031250 | 0.085937500 | 0.000000000 | -0.003906250 | [-0.0156,+0.0078] |
| 131,072 | T0 | 0.066406250 | 0.070312500 | 0.000000000 | -0.003906250 | [-0.0156,+0.0059] |
| 131,072 | T1 | 0.056640625 | 0.070312500 | 0.000000000 | -0.013671875 | [-0.0293,+0.0020] |
| 131,072 | T2 | 0.058593750 | 0.070312500 | 0.000000000 | -0.011718750 | [-0.0273,+0.0020] |

Across the twelve frozen cells, the learned scorer's point deficit from Bayes ranged from 0 to 0.017578125, with a mean absolute deficit of 0.00830078125. Every point estimate remained inside the frozen two-percentage-point near-ceiling margin, and learned performance remained far above the matched public-hash control.

## Why the classifier is inconclusive

The frozen classifier did not classify point estimates alone. `G7_INFORMATION_CEILING_DOMINANT` required every population/checkpoint cell to establish all of the following:

1. learned-minus-Bayes CI low greater than -0.02;
2. learned-minus-hash CI low greater than zero;
3. Bayes-minus-hash CI low greater than zero.

Several learned-minus-Bayes intervals crossed the -0.02 non-inferiority boundary. Conversely, no cell established the frozen clear-scorer-gap condition requiring the learned-minus-Bayes CI high to be below -0.02.

Therefore:

```text
information ceiling dominant = not established
scorer representation gap = not established
mixed ceiling/scorer gap = not established
campaign outcome = G7_INFORMATION_CEILING_INCONCLUSIVE
```

This is an uncertainty result, not evidence that routing, representation or coordination failed.

## Scientific interpretation

### What is established

1. The benchmark's Bayes-optimal M128 coverage declines strongly with population depth under the fixed 70%-reliable hint stream and fixed 128-attempt terminal budget.
2. The learned scorer tracked that Bayes reference closely in point estimate at all twelve checkpoint/population cells.
3. The largest observed learned-to-Bayes point deficit was 1.7578125 percentage points, inside the preregistered two-point margin.
4. Both learned and Bayes ranking strongly outperformed the public-hash control throughout the ladder.
5. Much of the previously observed absolute-coverage decline is therefore consistent with the benchmark's information/attempt ceiling.

### What is not established

The frozen cell-wise confidence rule does not allow the stronger statement that the information ceiling fully explains the decline. The current 512-world strata are not precise enough to distinguish near-Bayes non-inferiority from a small recoverable scorer deficit in every cell.

No communication, recycling, specialization, recurrent scheduling or topology intervention is justified from this result alone. Such an intervention would change the architecture before resolving whether a recoverable scorer gap exists.

## Next admitted research family

The minimum next question is **precision confirmation**, not coordination intervention.

A future protocol may repeat the unchanged learned/Bayes/hash M128 comparison with fresh worlds and greater statistical precision while preserving:

- exact T0/T1/T2 checkpoints;
- exact population ladder;
- fixed 19,649 learned parameters;
- fixed 70% hint reliability;
- fixed M128 primary attempt budget;
- no training or communication intervention.

It must preregister both cell-wise evidence and a population/checkpoint-stratified pooled non-inferiority analysis before generating worlds. It may not reinterpret this result post hoc, reuse these worlds as fresh evidence, or open an architecture intervention as a rescue.

## Audit-recovery boundary

The original independent audit rejected all twelve matrices because JSON serialization sorted nested object keys, while the auditor incorrectly required insertion order equality. Recovery:

- preserved the rejected audit;
- modified no scientific artifact bytes;
- generated no worlds;
- loaded no checkpoint;
- ran no Torch or CUDA code;
- changed no ranks, summaries, confidence intervals or classifier rules;
- returned `artifact_valid=true` and `errors=[]` against the unchanged result SHA-256.
