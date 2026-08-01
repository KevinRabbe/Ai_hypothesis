# Gate-9 failure decomposition stage-1 seed-0 result v0

## Status

**IMMUTABLE FIRST-SEED RESULT FOR `single_operator_query_fit`. SEED 0 FAILED THE
FROZEN STAGE-1 THRESHOLDS. STAGES 2–4, GATE-9 V0 SCIENCE, POPULATION EXECUTION,
RETRAINING, CHECKPOINT SELECTION, AND FINAL DIAGNOSTIC CLASSIFICATION REMAIN
CLOSED.**

Qualified execution head:

```text
2e1b91d578e7bf9b4c54aa2ee1c120a9ec01b21c
```

## Independently reconstructed result

```text
training rows                 1,024
training examples seen        252,928
unique examples               247
final training loss           0.36635908484458923
minimum recorded loss         0.36635908484458923 at step 1,024

evaluation rows               247
full exact correct            52
full exact accuracy           0.21052631578947367
full bit correct              1,671 / 1,976
full bit accuracy             0.8456477732793523
query-only correct            0
query-only accuracy           0.0
oracle correct                247
oracle accuracy               1.0
stage passes                  false
```

All 1,024 training rows are contiguous, finite, and follow the exact frozen
linear-warmup/cosine learning-rate schedule. All 247 non-support query bytes
appear exactly once in the evaluation ledger, with exact stored correctness
flags. The fixed-final checkpoint loads using `weights_only=True`, contains 17
finite float32 tensors, and totals exactly 19,649 parameters.

## Interpretation

This is a substantive failure of the first diagnostic prerequisite, not a
near-threshold miss. The worker did not fit one fixed affine byte operator even
though all 247 evaluated queries were also the complete training set. The
public-support oracle remained perfect, so the operator definition and target
reconstruction were intact.

The result does not yet issue the terminal stage-1 classification. The frozen
protocol requires all three initialization seeds. Because progression requires
all three to pass, stage 2 is already closed; seeds 1 and 2 now serve only as
replication and mixed-outcome resolution:

```text
all three seeds fail  -> G9D_BASIC_QUERY_MAPPING_FAILED
mixed seed outcomes   -> G9D_DIAGNOSTIC_INCONCLUSIVE
```

No hyperparameter extension, retraining, earlier-checkpoint selection, or
architecture change is admitted inside this frozen diagnostic.

## Evidence identities

```text
result JSON Git blob       de694a4610e63b1bea900b6babd0778b15780409
result JSON SHA-256        20188b8d70637b1599f5f603d700512c3d22d50fc5c3d0d1d0a5fa72843c0a80
source manifest SHA-256    8074a4224c4a51f38f869944f57fd0c350a0f164b50627a2f9173352e170dc0e
summary SHA-256            a306fcb06bbd6965553dd8406ffd0aad41259bab45149d39a67c77cab7d143c8
training ledger SHA-256    b74cf27618985dd690c2d50555cf478e6dd06fe5907cbd55e9de7fdf448fcc9c
evaluation ledger SHA-256  98623f9c37f74137722f6d45ef37536064017a429246d45a3820aaee2c9785e1
checkpoint SHA-256         3c2c2bac4036ccd8bc45c5aca8c28fe3b2e470902489907e224ba602bafaa93f
git-status SHA-256         e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855
```

The empty `git-status.txt` identity was independently confirmed from the source
machine as an existing zero-byte file with the canonical empty SHA-256.

## Scope

This result slice contains only:

- immutable JSON result record;
- byte-identical source artifact manifest;
- scientific interpretation record.

The exact verifier is supplied by the stacked qualification-prep branch and is
executed by the registered population-compute regression workflow. This slice
contains no model, trainer, optimizer, checkpoint, JSONL ledger, PowerShell
runner, operator generator, later-stage runtime, scientific-world generator,
population runtime, or diagnostic classifier.
