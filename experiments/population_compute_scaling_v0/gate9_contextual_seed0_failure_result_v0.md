# Gate-9 contextual seed-0 failure result v0

## Status

**IMMUTABLE RESULT-ONLY RECORD — SEED 0 FAILED THE FROZEN CHECKPOINT-
ADMISSION GATE. GATE-9 V0 SCIENTIFIC TEST GENERATION AND EXECUTION REMAIN
CLOSED.**

Qualified audit head:

```text
af8dc84cd2884e547b1c40599400bfaf8610ee64
```

Qualified execution head:

```text
bdc1af9bc65b94b01ae3946977686bd90158786f
```

Qualified architecture head:

```text
c689cc3f38f6f642916ee1a702d7de7bd0e43b
```

Frozen training-protocol head:

```text
1228c19cbf85da4ab738c3355c58f946cd6a965c
```

## Immutable outcome

```text
G9_CONTEXTUAL_SEED_CHECKPOINT_ADMISSION_FAILED
```

The independent read-only audit reconstructed the complete training and
validation evidence and confirmed:

```text
training rows                 512
training episodes             262,144
final training loss           0.6930734515190125

validation rows               32,768
full exact correct            109
full exact accuracy           0.003326416015625
full bit accuracy             0.50048828125
shuffled-context correct      112
shuffled-context accuracy     0.00341796875
query-only correct            117
query-only accuracy           0.003570556640625
oracle correct                32,768
oracle accuracy               1.0
```

The selected fixed-final checkpoint contains exactly 17 finite float32 tensors
and 19,649 learned parameters.

## Scientific interpretation

The result is a clear failure to learn the frozen contextual affine-operator
task:

- final binary-cross-entropy remained near `ln(2)`;
- output-bit accuracy remained at chance;
- exact-byte accuracy remained near random `1/256`;
- full support context did not outperform shuffled context or the support-free
  query-only control;
- the exact public-support oracle remained perfect, proving the task evidence
  and target reconstruction were valid.

This result does not show that population computation cannot generalize. It
shows that this exact shared 19,649-parameter worker, one-pass training
allocation, optimizer schedule and contextual architecture did not acquire the
required operator-induction capability.

## Gate-9 v0 consequence

The preregistered checkpoint-admission rule requires all three ordered seeds to
pass. Because seed 0 independently failed:

```text
all_seed_admission_still_possible = false
scientific_test_generation_allowed = false
```

Seeds 1 and 2 may still run as preregistered replications or diagnostic
evidence. They cannot reverse the Gate-9 v0 admission consequence, and no local
or graph scientific test may be generated or executed under v0.

## Architecture identity correction

The auditor source originally contained the exact 38-character transcription:

```text
c689cc3f38f6f642916ee1a702d7de7bd0e43b
```

The audit runner corrected it fail-closed to the actual 40-character
architecture branch head:

```text
c689cc3f38f6f642916ee1a702d7de7bd0e43b
```

The correction was allowed only for that exact known typo, was verified against
the remote architecture branch, was recorded in the audit report, and did not
modify the source training artifact.

## Bound evidence

The result record binds:

```text
audit report SHA-256
6f2fbd0289a70dab15d4901ca7733ae49194614ebb9060227b321ed4f387946a

audit manifest SHA-256
f45b0f9b91da0b67e3a0103ae367138eb91faa7307e9e7cad46cda5d9118d766

source checkpoint SHA-256
236f04d4c08e494ee51645750842e68bf18e66004fbf461e51a8d827e7eb1368

source validation ledger SHA-256
e3038554cf424995126c064bb71653aaceab80ad4831f8e8580c06eb3d74efb4

source summary SHA-256
233638bb2869f083e70fa0e17e6fed2a381ac9f4bce8ac3847ab02ffe60f4b88

source manifest SHA-256
a79831a0f25d02e393c22ae5779ec2165df0a671be620b273a3993387dc4736d
```

The committed audit manifest is byte-identical to the uploaded retry3
`manifest.sha256`. It binds the audit report, audit Git head, empty Git status
and audit run configuration.

## Closed boundary

This branch contains only:

- one immutable JSON result record;
- the exact audit manifest;
- this interpretation record;
- result-only qualification CI.

It contains no trainer, optimizer, checkpoint, training ledger, validation
ledger, operator generator, scientific world generator, scientific assignment
key, population runtime or classifier execution.
