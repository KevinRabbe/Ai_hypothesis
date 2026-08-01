# Gate-9 contextual seed-2 replication result v0

## Status

**IMMUTABLE FINAL REPLICATION-ONLY RESULT. ORDERED SEED 2 FAILED THE FROZEN
CHECKPOINT-ADMISSION RULE. SCIENTIFIC TEST GENERATION AND EXECUTION REMAIN
CLOSED.**

Base seed-1 result head:

```text
71ec79d410f94a669366caa70878a127759b0fd3
```

## Independently reconstructed result

```text
training rows                 512
training episodes             262,144
final training loss           0.693339467048645
unique operator-batch hashes  512
unique query-batch hashes     512

validation rows               32,768
full exact correct            128
full exact accuracy           0.00390625
full bit accuracy             0.5026893615722656
shuffled-context correct      122
shuffled-context accuracy     0.00372314453125
query-only correct            120
query-only accuracy           0.003662109375
oracle correct                32,768
oracle accuracy               1.0
```

The complete validation ledger has unique coverage of all 32,768 frozen
validation operators, a complete shuffled-context derangement, no support-input
queries, and exact correctness flags. The fixed-final checkpoint loads with
`weights_only=True`, contains 17 finite float32 tensors, and totals 19,649
parameters.

## Replication interpretation

Full context improves over query-only by only:

```text
8 / 32,768 = 0.000244140625
```

It improves over shuffled context by only:

```text
6 / 32,768 = 0.00018310546875
```

Bit accuracy remains near chance and the public-support oracle remains perfect.
Seed 2 therefore independently replicates failure to learn causal use of the
support examples. It does not diagnose the root cause.

All three ordered seeds are now complete and all three failed checkpoint
admission. A separate final three-seed result slice must record the frozen
Gate-9 v0 classification. No local or graph scientific tests may be generated.

## Evidence identities

```text
result JSON SHA-256       885239d1dc5ac7fa251f3de2e1b57f9c37885a7fb24a327794b0dc1c7a6881a8
source manifest SHA-256   38e5fde7c706104fa28ee3e3057ebaf308d38723cfc14993cf30db45e2fbacf1
summary SHA-256           960241eef32655e804cd600712e83d99a447e79c638929b98d8932b6a26cdc16
training ledger SHA-256   13a7f19737f110b35cf35c239538c9a19230965f44fa31cb8a4e73b233470279
validation ledger SHA-256 8328ebbcf22760f659d6fc979fd810162ed6d3d9d2c33ae2e86a8b008cf74c53
checkpoint SHA-256        edd305fe5d8d55b12379caf055ceebad260eb1bf024f4c2af375bae885b68643
```

## Scope

This result slice contains only:

- immutable JSON result record;
- byte-identical source artifact manifest;
- scientific interpretation record;
- result-only qualification workflow.

No trainer, optimizer, checkpoint, JSONL ledger, operator generator, scientific
assignment key, test-world generator, population runtime, or final Gate-9
classifier is present.
