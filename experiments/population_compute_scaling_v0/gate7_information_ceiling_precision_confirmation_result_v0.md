# Gate-7 information-ceiling precision confirmation v0 — result

## Status

**VALID FRESH PRECISION-CONFIRMATION EVIDENCE — THE FROZEN CLASSIFIER ESTABLISHED `G7_PRECISION_INFORMATION_CEILING_DOMINANT`.**

Exact admitted execution head:

`bdc5f2c03cbc77a79a419a16460387ac0d226a27`

Exact frozen protocol head:

`8d7865ab01b4b04b875ed2ca627b68a6c33c81f7`

Independent audit:

```text
artifact_valid = true
errors = []
campaign_outcome = G7_PRECISION_INFORMATION_CEILING_DOMINANT
scientific_status = FRESH_GATE7_INFORMATION_CEILING_PRECISION_CONFIRMATION_EVIDENCE
```

No training, checkpoint selection, communication intervention, adaptive attempt exposure, rescue attempt budget, or prior-world reuse occurred.

## Artifact identity

```text
result_sha256   = 89f0a69d530355031f02666e403977f4ad3b622bc6468627c4cbfa9a7d1ea489
audit_sha256    = 857256f4ae71f0fbf6744a531ece0120fb9bb3088e36e30471e3614ffd602a79
manifest_sha256 = 2e212ab56dd251ab527f8aeb95e40f0e06bcacad9b7e1900f49a1b8d96efffa3
```

## Frozen execution matrix

- exact transition checkpoints T0/T1/T2;
- populations N16384, N32768, N65536 and N131072;
- 2,048 fresh worlds per checkpoint/population;
- 24,576 checkpoint-world rows;
- 32 physical B64 batches per checkpoint/population;
- exact learned, Bayes and public-hash hidden-parent ranks;
- M1..M1024 descriptive curve with M128 primary;
- 20,000 deterministic world-clustered bootstrap replicates;
- equal population then equal checkpoint weighting;
- unchanged 19,649 learned parameters;
- unchanged 70% hint reliability;
- unchanged two-percentage-point non-inferiority margin.

## Primary M128 evidence

| Population | Checkpoint | Learned | Bayes | Public hash | Learned − Bayes | 95% CI |
| ---: | :---: | ---: | ---: | ---: | ---: | ---: |
| 16,384 | T0 | 0.175293 | 0.179199 | 0.013184 | -0.003906 | [-0.0107,+0.0024] |
| 16,384 | T1 | 0.165039 | 0.179199 | 0.013184 | -0.014160 | [-0.0234,-0.0049] |
| 16,384 | T2 | 0.171875 | 0.179199 | 0.013184 | -0.007324 | [-0.0151,+0.0005] |
| 32,768 | T0 | 0.131836 | 0.134277 | 0.001953 | -0.002441 | [-0.0063,+0.0010] |
| 32,768 | T1 | 0.121094 | 0.134277 | 0.001953 | -0.013184 | [-0.0210,-0.0054] |
| 32,768 | T2 | 0.128418 | 0.134277 | 0.001953 | -0.005859 | [-0.0122,+0.0005] |
| 65,536 | T0 | 0.084473 | 0.089355 | 0.002930 | -0.004883 | [-0.0098,+0.0000] |
| 65,536 | T1 | 0.076660 | 0.089355 | 0.002930 | -0.012695 | [-0.0200,-0.0059] |
| 65,536 | T2 | 0.080078 | 0.089355 | 0.002930 | -0.009277 | [-0.0156,-0.0029] |
| 131,072 | T0 | 0.060547 | 0.062988 | 0.000488 | -0.002441 | [-0.0083,+0.0029] |
| 131,072 | T1 | 0.053223 | 0.062988 | 0.000488 | -0.009766 | [-0.0176,-0.0020] |
| 131,072 | T2 | 0.058105 | 0.062988 | 0.000488 | -0.004883 | [-0.0122,+0.0024] |

Every population-level point estimate remained inside the frozen two-percentage-point margin. No cell or population established the preregistered clear-scorer-gap condition. Learned and Bayes ranking remained far above public hash throughout the complete ladder.

The equal-population/equal-checkpoint primary point estimates derived from the twelve frozen cells are:

```text
learned = 0.10888671875
Bayes   = 0.116455078125
hash    = 0.004638671875
learned_minus_Bayes = -0.007568359375
```

The learned scorer therefore retained approximately 93.5% of Bayes M128 coverage under the frozen equal-weight primary aggregation.

## Frozen conclusion

The decline in absolute Gate-7 coverage from N16384 through N131072 is dominated by the benchmark's noisy-information and fixed-attempt ceiling, not by a demonstrated routing, communication, or coordination failure.

This result establishes:

1. The shared 19,649-parameter scorer remains strongly informative through N131072.
2. The learned scorer is non-inferior to the Bayes information ceiling under the preregistered pooled precision test and local-gap guards.
3. Increasing population in this benchmark cannot by itself produce increasing capability because the available information and M128 terminal-attempt budget impose the dominant limit.
4. Further routing-bandwidth, communication, recycling, specialization, topology, or recurrent-scheduling interventions are not scientifically justified on this benchmark.

This result does **not** establish:

- general intelligence;
- broad capability scaling with population;
- parity with a 1B model;
- useful worker specialization or communication;
- that the learned scorer is exactly Bayes optimal;
- that population computation will scale on tasks where information can accumulate.

## Next research boundary

The next benchmark family must make additional runtime population useful by construction. It must permit workers to acquire independent partial evidence, combine intermediate results, and improve the final answer through bounded recurrent coordination.

A conventional approximately 1B-parameter baseline must be included from the protocol stage, with separate reporting for:

- accuracy/capability;
- learned parameters;
- active runtime workers;
- communicated bits;
- recurrent updates;
- FLOPs or normalized compute;
- wall time and memory;
- capability per learned parameter and per unit of runtime compute.

The next study is therefore a capability-scaling benchmark, not another Gate-7 routing study.