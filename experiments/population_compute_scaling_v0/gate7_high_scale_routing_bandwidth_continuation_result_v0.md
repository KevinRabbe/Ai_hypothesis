# Gate-7 high-scale routing-bandwidth continuation v0 — result

## Status

**VALID FRESH POST-CONFIRMATION CONTINUATION EVIDENCE — THE COMPLETE N16384..N131072 LADDER FINISHED WITHOUT A RESOURCE FRONTIER.**

Exact admitted continuation-execution head:

`19ee6b4e228c56b32a11b11b1c61b35bf640e2c8`

Exact frozen result-interpretation protocol head:

`05eb27e66c7a46dd4646d0ac99ee2f4028d0d131`

Independent audit:

```text
artifact_valid = true
errors = []
campaign_outcome = G7_POST_CONFIRMATION_LADDER_COMPLETE
completed_populations = [16384, 32768, 65536, 131072]
resource_frontier_population = null
```

No training, checkpoint selection, adaptive K exposure, rescue K, second confirmation, second continuation or alternate evidence namespace was used.

## Frozen continuation design

- exact transition checkpoints T0/T1/T2;
- populations N16384, N32768, N65536 and N131072;
- complete K ladder 16, 32, 64, 128, 256 and 512 at every population;
- global score/hash and every K score/hash at every checkpoint/population;
- 512 fresh paired worlds/checkpoint/population;
- eight physical batches of 64 worlds;
- 10,000 deterministic paired-bootstrap samples;
- unchanged five-percentage-point non-inferiority margin;
- 128 terminal Stage-B activations;
- fixed 19,649 learned parameters;
- FP32 with compiler, CUDA graphs and mixed precision disabled.

## Campaign result

| Population N | Global reference viable | Complete passing K set | K_required | K_required / N | Tier outcome |
| ---: | :---: | ---: | ---: | ---: | --- |
| 16,384 | yes | [256, 512] | 256 | 1.5625% | `G7_CONTINUATION_K_REQUIRED` |
| 32,768 | yes | [512] | 512 | 1.5625% | `G7_CONTINUATION_K_REQUIRED` |
| 65,536 | yes | [512] | 512 | 0.78125% | `G7_CONTINUATION_K_REQUIRED` |
| 131,072 | yes | [512] | 512 | 0.390625% | `G7_CONTINUATION_K_REQUIRED` |

Campaign outcome:

`G7_POST_CONFIRMATION_LADDER_COMPLETE`.

The RTX 4060 Ti completed the entire fixed population ladder. There was no CUDA resource frontier.

## N16384

### Global learned reference

| Checkpoint | Global score | Global hash | Delta | 95% CI |
| ---: | ---: | ---: | ---: | ---: |
| T0 | 0.1680 | 0.0098 | +0.1582 | [+0.1250,+0.1914] |
| T1 | 0.1719 | 0.0098 | +0.1621 | [+0.1289,+0.1953] |
| T2 | 0.1777 | 0.0098 | +0.1680 | [+0.1348,+0.2031] |

Stratified pooled delta: `+0.1628`, 95% CI `[+0.1439,+0.1829]`; reference viable.

### Complete K matrix

Each triplet is ordered T0/T1/T2.

| K | Learned scores | Hash scores | learned-vs-hash CI lows | score-vs-global CI lows | Pass |
| ---: | --- | --- | --- | --- | :---: |
| 16 | 0.0527 / 0.0508 / 0.0488 | 0.0098 / 0.0098 / 0.0098 | +0.0234 / +0.0215 / +0.0195 | -0.1504 / -0.1562 / -0.1641 | no |
| 32 | 0.0723 / 0.0645 / 0.0703 | 0.0059 / 0.0059 / 0.0059 | +0.0449 / +0.0371 / +0.0430 | -0.1309 / -0.1426 / -0.1426 | no |
| 64 | 0.1172 / 0.1133 / 0.1172 | 0.0078 / 0.0078 / 0.0078 | +0.0820 / +0.0781 / +0.0820 | -0.0840 / -0.0918 / -0.0938 | no |
| 128 | 0.1270 / 0.1211 / 0.1270 | 0.0078 / 0.0078 / 0.0078 | +0.0898 / +0.0840 / +0.0898 | -0.0684 / -0.0781 / -0.0781 | no |
| 256 | 0.1582 / 0.1562 / 0.1543 | 0.0078 / 0.0078 / 0.0078 | +0.1191 / +0.1172 / +0.1172 | -0.0312 / -0.0391 / -0.0469 | **yes** |
| 512 | 0.1621 / 0.1562 / 0.1777 | 0.0098 / 0.0098 / 0.0098 | +0.1211 / +0.1152 / +0.1348 | -0.0234 / -0.0332 / -0.0156 | **yes** |

`K_required(16384) = 256`.

## N32768

### Global learned reference

| Checkpoint | Global score | Global hash | Delta | 95% CI |
| ---: | ---: | ---: | ---: | ---: |
| T0 | 0.1270 | 0.0039 | +0.1230 | [+0.0938,+0.1523] |
| T1 | 0.1230 | 0.0039 | +0.1191 | [+0.0898,+0.1484] |
| T2 | 0.1211 | 0.0039 | +0.1172 | [+0.0879,+0.1465] |

Stratified pooled delta: `+0.1198`, 95% CI `[+0.1035,+0.1374]`; reference viable.

### Complete K matrix

| K | Learned scores | Hash scores | learned-vs-hash CI lows | score-vs-global CI lows | Pass |
| ---: | --- | --- | --- | --- | :---: |
| 16 | 0.0254 / 0.0312 / 0.0312 | 0.0039 / 0.0039 / 0.0039 | +0.0078 / +0.0137 / +0.0137 | -0.1309 / -0.1230 / -0.1211 | no |
| 32 | 0.0449 / 0.0469 / 0.0430 | 0.0000 / 0.0000 / 0.0000 | +0.0273 / +0.0293 / +0.0273 | -0.1152 / -0.1074 / -0.1074 | no |
| 64 | 0.0664 / 0.0684 / 0.0762 | 0.0000 / 0.0000 / 0.0000 | +0.0449 / +0.0469 / +0.0547 | -0.0938 / -0.0859 / -0.0762 | no |
| 128 | 0.0625 / 0.0488 / 0.0566 | 0.0000 / 0.0000 / 0.0000 | +0.0430 / +0.0312 / +0.0371 | -0.0957 / -0.1016 / -0.0938 | no |
| 256 | 0.0957 / 0.0977 / 0.0918 | 0.0039 / 0.0039 / 0.0039 | +0.0664 / +0.0684 / +0.0625 | -0.0547 / -0.0488 / -0.0547 | no |
| 512 | 0.1230 / 0.1211 / 0.1016 | 0.0039 / 0.0039 / 0.0039 | +0.0898 / +0.0879 / +0.0723 | -0.0234 / -0.0215 / -0.0371 | **yes** |

`K_required(32768) = 512`.

K256 was close but did not satisfy the all-checkpoint frozen criterion: T0 and T2 had score-vs-global CI lows below -0.05.

## N65536

### Global learned reference

| Checkpoint | Global score | Global hash | Delta | 95% CI |
| ---: | ---: | ---: | ---: | ---: |
| T0 | 0.0820 | 0.0000 | +0.0820 | [+0.0586,+0.1074] |
| T1 | 0.0742 | 0.0000 | +0.0742 | [+0.0527,+0.0977] |
| T2 | 0.0762 | 0.0000 | +0.0762 | [+0.0547,+0.0996] |

Stratified pooled delta: `+0.0775`, 95% CI `[+0.0645,+0.0911]`; reference viable.

### Complete K matrix

| K | Learned scores | Hash scores | learned-vs-hash CI lows | score-vs-global CI lows | Pass |
| ---: | --- | --- | --- | --- | :---: |
| 16 | 0.0078 / 0.0078 / 0.0098 | 0.0039 / 0.0039 / 0.0039 | -0.0059 / -0.0059 / -0.0039 | -0.0996 / -0.0918 / -0.0918 | no |
| 32 | 0.0195 / 0.0195 / 0.0195 | 0.0000 / 0.0000 / 0.0000 | +0.0078 / +0.0078 / +0.0078 | -0.0879 / -0.0801 / -0.0820 | no |
| 64 | 0.0332 / 0.0312 / 0.0312 | 0.0000 / 0.0000 / 0.0000 | +0.0176 / +0.0176 / +0.0176 | -0.0762 / -0.0684 / -0.0703 | no |
| 128 | 0.0312 / 0.0352 / 0.0332 | 0.0000 / 0.0000 / 0.0000 | +0.0176 / +0.0195 / +0.0195 | -0.0762 / -0.0645 / -0.0684 | no |
| 256 | 0.0586 / 0.0488 / 0.0449 | 0.0000 / 0.0000 / 0.0000 | +0.0391 / +0.0312 / +0.0273 | -0.0488 / -0.0449 / -0.0547 | no |
| 512 | 0.0645 / 0.0566 / 0.0605 | 0.0000 / 0.0000 / 0.0000 | +0.0449 / +0.0371 / +0.0410 | -0.0371 / -0.0352 / -0.0352 | **yes** |

`K_required(65536) = 512`.

K16 did not establish learned-over-hash value; all larger K values did. K256 missed only the frozen T2 global non-inferiority boundary.

## N131072

### Global learned reference

| Checkpoint | Global score | Global hash | Delta | 95% CI |
| ---: | ---: | ---: | ---: | ---: |
| T0 | 0.0684 | 0.0020 | +0.0664 | [+0.0449,+0.0879] |
| T1 | 0.0664 | 0.0020 | +0.0645 | [+0.0430,+0.0879] |
| T2 | 0.0625 | 0.0020 | +0.0605 | [+0.0391,+0.0820] |

Stratified pooled delta: `+0.0638`, 95% CI `[+0.0514,+0.0762]`; reference viable.

### Complete K matrix

| K | Learned scores | Hash scores | learned-vs-hash CI lows | score-vs-global CI lows | Pass |
| ---: | --- | --- | --- | --- | :---: |
| 16 | 0.0078 / 0.0098 / 0.0098 | 0.0020 / 0.0020 / 0.0020 | -0.0020 / +0.0020 / +0.0000 | -0.0840 / -0.0801 / -0.0762 | no |
| 32 | 0.0215 / 0.0195 / 0.0176 | 0.0039 / 0.0039 / 0.0039 | +0.0039 / +0.0039 / +0.0020 | -0.0703 / -0.0703 / -0.0684 | no |
| 64 | 0.0117 / 0.0137 / 0.0117 | 0.0020 / 0.0020 / 0.0020 | +0.0020 / +0.0039 / +0.0020 | -0.0801 / -0.0762 / -0.0723 | no |
| 128 | 0.0332 / 0.0312 / 0.0332 | 0.0000 / 0.0000 / 0.0000 | +0.0195 / +0.0176 / +0.0195 | -0.0605 / -0.0586 / -0.0527 | no |
| 256 | 0.0488 / 0.0352 / 0.0332 | 0.0000 / 0.0000 / 0.0000 | +0.0312 / +0.0195 / +0.0195 | -0.0469 / -0.0547 / -0.0527 | no |
| 512 | 0.0547 / 0.0508 / 0.0449 | 0.0039 / 0.0039 / 0.0039 | +0.0312 / +0.0293 / +0.0234 | -0.0352 / -0.0352 / -0.0371 | **yes** |

`K_required(131072) = 512`.

## Cross-population interpretation

### What is established

1. The shared 19,649-parameter learned scorer retained statistically positive routing information at every tested population through N131072.
2. At every population, at least one bounded learned-routing budget remained non-inferior to the global learned reference under the frozen five-point margin.
3. A visibility budget of K512 was sufficient at N32768, N65536 and N131072.
4. The required observed fraction fell from 1.5625% at N32768 to 0.390625% at N131072 while the absolute K budget remained 512.
5. The complete population ladder executed on the qualified RTX 4060 Ti implementation without a resource truncation.

### What is not established

This experiment does **not** show that larger populations increased task capability. Absolute global learned scores declined across the confirmed/continuation range:

```text
N8192:   approximately 0.2305..0.2500
N16384:  approximately 0.1680..0.1777
N32768:  approximately 0.1211..0.1270
N65536:  approximately 0.0742..0.0820
N131072: approximately 0.0625..0.0684
```

The global learned-reference advantage over hash remained positive, but weakened with population size:

| N | Stratified pooled global learned-vs-hash delta | 95% CI |
| ---: | ---: | ---: |
| 8,192 confirmation | +0.2246 | [+0.2025,+0.2467] |
| 16,384 | +0.1628 | [+0.1439,+0.1829] |
| 32,768 | +0.1198 | [+0.1035,+0.1374] |
| 65,536 | +0.0775 | [+0.0645,+0.0911] |
| 131,072 | +0.0638 | [+0.0514,+0.0762] |

Therefore the result establishes routing scalability relative to the current global reference, not capability scaling from population alone. It is consistent with useful learned selection surviving while useful workers become increasingly sparse or the current coordination/task representation dilutes signal at high N.

The observed K sequence is descriptive only:

```text
N8192:   K_required = 256
N16384:  K_required = 256
N32768:  K_required = 512
N65536:  K_required = 512
N131072: K_required = 512
```

No asymptotic exponent, monotonic law or interpolation to unobserved populations is claimed.

## Frozen next-question classification

The pre-exposure result protocol selects:

`G7_NEXT_QUESTION_COORDINATION_EFFICIENCY`.

Reason:

- no resource frontier occurred;
- every completed global learned reference was viable;
- every completed population had at least one passing K<=512;
- the full fixed ladder completed.

This opens only the next **question family**: determine whether worker activation, recycling, specialization, sparse communication topology and recurrent scheduling can turn the large routable population into increasing capability or better capability-per-unit-compute.

It does not open a concrete experiment, architecture, training run or evidence namespace. Those remain subject to a separate data-frozen protocol.

## Provenance

- result SHA-256: `4921ea99b44156f08271d6fb2b2e0bcba98ef6a646ed0aaf040762d47aa03b36`
- audit SHA-256: `92f52a9e7fad3cb5d8962a9127a0cd7140656a0a8f03cfba08fe7cd5376a03fd`
- manifest SHA-256: `ee9dcefbaf5efe9a75b20d407cb1a4f47ff0b04bbdce4613f2539b76af2c8cca`
- local output root: `F:\gate7_high_scale_routing_bandwidth_continuation_v0`

## Boundary

This branch records the valid audited result only. A second continuation is closed. No new execution path, training surface, routing rescue, coordination experiment or evidence namespace is introduced here.
