# Gate-9 failure decomposition stage-1 seed-1 result v0

## Status

**IMMUTABLE SECOND-SEED REPLICATION RESULT FOR `single_operator_query_fit`.
SEED 1 FAILED THE FROZEN STAGE-1 THRESHOLDS. STAGES 2–4, GATE-9 V0 SCIENCE,
POPULATION EXECUTION, RETRAINING, CHECKPOINT SELECTION, AND TERMINAL DIAGNOSTIC
CLASSIFICATION REMAIN CLOSED.**

Qualified execution head:

```text
2e1b91d578e7bf9b4c54aa2ee1c120a9ec01b21c
```

Seed-0 result ancestor:

```text
9a36b79453e794886c00c85df67bf0bd8fad7345
```

## Independently reconstructed result

```text
training rows                 1,024
training examples seen        252,928
unique examples               247
final training loss           0.36665672063827515
minimum recorded loss         0.36665672063827515 at step 1,024

evaluation rows               247
full exact correct            58
full exact accuracy           0.23481781376518218
full bit correct              1,699 / 1,976
full bit accuracy             0.8598178137651822
query-only correct            1
query-only accuracy           0.004048582995951417
oracle correct                247
oracle accuracy               1.0
stage passes                  false
```

All 1,024 training rows are contiguous, finite, and follow the exact frozen
linear-warmup/cosine learning-rate schedule. All 247 non-support query bytes
appear exactly once in the evaluation ledger, with exact stored correctness
flags. The fixed-final checkpoint loads using `weights_only=True`, contains 17
finite float32 tensors, and totals exactly 19,649 parameters.

## Replication interpretation

Seed 1 independently reproduces the stage-1 deficiency. The worker did not fit
one fixed affine byte operator even though the complete 247-example evaluation
set was also the complete full-batch training set for all 1,024 steps. The
public-support oracle remained perfect.

The two ordered results are:

```text
seed 0  52 / 247 exact  0.8456477732793523 bit accuracy  failed
seed 1  58 / 247 exact  0.8598178137651822 bit accuracy  failed
```

This reduces the plausibility of a seed-0-only initialization accident, but the
frozen protocol still requires seed 2 before issuing its terminal stage-1
classification. Stage 2 remains closed because advancement required all three
initialization seeds to pass.

```text
seed 2 also fails  -> G9D_BASIC_QUERY_MAPPING_FAILED
seed 2 passes       -> G9D_DIAGNOSTIC_INCONCLUSIVE
```

No hyperparameter extension, retraining, earlier-checkpoint selection, or
architecture change is admitted inside this frozen diagnostic.

## Evidence identities

```text
result JSON SHA-256        80673d996b5075ff09577c5a6fec412c04ff8c9914c932914c4714674f74d5b5
source manifest SHA-256    1694a3a0c23e71b4c830d432234237498666c832b579ce62f2622b007b70fe5f
summary SHA-256            b593c53b1d7d1d6de031ec3e04ea65c1a6ce63121a01acfc21d8fe80866aaa06
training ledger SHA-256    95c0a36f0229b030a84be2e7fb3bc9a2f09a32e55d7827904bea6e46a137e5cf
evaluation ledger SHA-256  418dc7c21f3641baea080d3bdf66103ae8548c929a0d105305120615ecbdc512
checkpoint SHA-256         db7e0189b8d900a71e0410c229241befab2e0d28a81f51accb0bb32c2195a555
git-status SHA-256         e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855
```

The empty `git-status.txt` identity was independently confirmed from the source
machine as an existing zero-byte file with the canonical empty SHA-256.

## Scope

This result slice contains only:

- immutable JSON result record;
- byte-identical source artifact manifest;
- scientific interpretation record.

The exact verifier is supplied by the stacked qualification-prep branch and
runs within the already-registered test suite. This slice contains no model,
trainer, optimizer, checkpoint, JSONL ledger, PowerShell runner, operator
generator, later-stage runtime, scientific-world generator, population runtime,
or diagnostic classifier.
