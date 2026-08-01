# Gate-9 contextual seed-1 replication result v0

## Status

**IMMUTABLE REPLICATION-ONLY RESULT: ORDERED SEED 1 FAILED THE FROZEN
GATE-9 CHECKPOINT-ADMISSION RULE. SCIENTIFIC TEST GENERATION AND EXECUTION
REMAIN CLOSED.**

Stacked seed-0 result head:

```text
3d9e686d158e9a4534fbe55b52d6ba2080419ba8
```

Qualified audit-contract head:

```text
af8dc84cd2884e547b1c40599400bfaf8610ee64
```

Frozen execution head:

```text
bdc1af9bc65b94b01ae3946977686bd90158786f
```

## Independently reconstructed result

The complete uploaded seed-1 artifact set was checked against its recursive
manifest before this record was written.

```text
training rows                 512
training episodes             262,144
final training loss           0.6926665306091309
unique operator-batch hashes  512
unique query-batch hashes     496

validation rows               32,768
full exact correct            143
full exact accuracy           0.004364013671875
full bit accuracy             0.5006904602050781
shuffled-context correct      130
shuffled-context accuracy     0.00396728515625
query-only correct            143
query-only accuracy           0.004364013671875
oracle correct                32,768
oracle accuracy               1.0
```

The validation ledger contains every frozen validation operator exactly once.
The shuffled-context assignment is also a complete derangement with one fixed
nonzero rotation. Every query lies outside the nine public support inputs, and
all stored correctness flags reconstruct exactly from the stored bytes.

The fixed-final checkpoint safely loads with `weights_only=True` and contains
exactly 17 finite float32 tensors totalling 19,649 learned parameters.

The 496 unique query-batch hashes do not indicate repeated training operators.
The qualified audit contract requires all 512 operator-batch hashes to be
unique and records, but does not require, query-batch-hash uniqueness.

## Admission outcome

Seed 1 fails every learned-capability threshold:

```text
required exact accuracy       >= 0.995
observed exact accuracy        0.004364013671875

required bit accuracy         >= 0.999
observed bit accuracy          0.5006904602050781

required full-shuffled delta  > 0.50
observed delta                 0.000396728515625

required full-query delta     > 0.50
observed delta                 0.0

required oracle accuracy      == 1.0
observed oracle accuracy       1.0
```

The exact seed outcome is:

```text
G9_CONTEXTUAL_SEED_CHECKPOINT_ADMISSION_FAILED
```

Full-context accuracy is exactly equal to the faithful query-only control and
only 13 correct episodes above shuffled context. Together with near-random bit
accuracy and perfect oracle performance, this is consistent with another
failure to learn causal use of the public support examples.

## Replication interpretation

Seed 0 already permanently closed Gate-9 v0 checkpoint admission because the
frozen protocol requires all three ordered seeds to pass. Seed 1 now provides a
second independent failure under a different initialization seed.

This weakens the explanation that seed 0 failed only because of an isolated bad
initialization. It does not identify whether the root cause is architecture,
optimization, supervision geometry, training duration, or another frozen
choice. No post-hoc rescue is admitted in v0.

Seed 2 may still run as the final preregistered replication. It cannot reopen
checkpoint admission or authorize scientific local or graph tests.

## Artifact identities

```text
source manifest SHA-256
101cac0769b54200713b9c978fecf0eb3d87d420f50194f20d959c90386d9296

summary SHA-256
2ad40ae3dcabcf9f52c8f2cdd9a0a0d3a8babba6dcab9150440b29474108fc28

training ledger SHA-256
e9250b9ebb180ccf4004c8bd8e45f9f1455a28a99c1f6118e6876d2fbd964927

validation ledger SHA-256
0a5c7ae5aca7b0d230ff58b89a10fc866c7058415e01f4c6fd098ce6379dcb1e

checkpoint SHA-256
7cd592ba9c54f05620e6bf8c41b7e7fea0301e47a89a456a1d19148defe6ab55
```

The manifest binds `git-status.txt` to the canonical SHA-256 of empty bytes.
The uploaded run configuration and Git head bind the exact execution branch,
software versions, seed, protocol head, architecture head and closed scientific
namespaces.

## Closed boundaries

This result slice contains no trainer, optimizer, checkpoint, raw JSONL ledger,
operator generator, scientific assignment key, test-world generator,
population runtime or Gate-9 final classifier.

It performs no retraining, checkpoint selection, world generation, scientific
execution or population execution.

## Next boundary

After this result-only slice qualifies, ordered seed 2 may run from the same
frozen execution head as replication-only evidence. Its artifact must be
preserved and independently audited before interpretation.
