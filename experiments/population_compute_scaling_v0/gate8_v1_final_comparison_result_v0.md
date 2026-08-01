# Gate-8 v1 final population-versus-Gemma result v0

## Status

**FINAL IMMUTABLE RESULT-ONLY RECORD — THE PREREGISTERED GATE-8 V1
POPULATION-SCALING AND POPULATION-VERSUS-1B-REFERENCE QUESTIONS ARE CLOSED.**

Exact final-comparison execution head:

```text
474b2590e5e138134bcb993e1d8114c473f0455b
```

Frozen source result heads:

```text
population result   14636d219781381853f81036b96c691b7e6997ee
Gemma result        1d48ecfd623a2fb9e3a2f846a4d1c49d20d8cadc
scientific protocol 6bb89111a47713bea0a23bb1cae662ed5ec56b42
```

This record performs no model loading, inference, population execution, world
generation, checkpoint loading, or training. It preserves the independently
audited output of the qualified data-reader-only finalizer.

## Final classifications

```text
population scaling
G8_POSITIVE_CAPABILITY_SCALING

population versus Gemma 3 1B
G8_POPULATION_EXCEEDS_1B_REFERENCE
```

The unchanged preregistered classifier therefore closes Gate-8 v1 with both
positive capability scaling across the frozen population ladder and a clear
population-organism win over the frozen Gemma 3 1B reference.

## Primary paired result

```text
population mean accuracy       1.0
Gemma mean accuracy            0.003441220238095238
population - Gemma delta       0.9965587797619048
paired 95% CI                  [0.9954427083333335, 0.9975818452380956]
conditions                     21
worlds per condition           512
unique worlds                  10,752
population checkpoint seeds    0, 1, 2
bootstrap samples              20,000
```

Every one of the 21 condition-level lower confidence bounds is strictly above
zero. The population organism scored 1.0 for every checkpoint seed in every
condition. Gemma produced 37 correct strict outputs across the 10,752 worlds.

The paired world-level estimand was frozen before reference inference:

```text
mean(correct_seed0, correct_seed1, correct_seed2) - correct_reference
```

Within a condition, each bootstrap replicate uses the same resampled world
indices for all three population seeds and Gemma. The pooled replicate gives
each condition exact weight `1/21`.

## Condition-level result

The exact 21-row order remains population-major:

```text
(32,4)
(64,4) (64,8)
(128,4) (128,8) (128,16)
(256,4) (256,8) (256,16) (256,32)
(512,4) (512,8) (512,16) (512,32) (512,64)
(1024,4) (1024,8) (1024,16) (1024,32) (1024,64) (1024,128)
```

Population accuracy is 1.0 in every row. Gemma's maximum condition accuracy is
0.017578125. The maximum observed reference prompt length is 9,892 tokens,
below the frozen 24,576-token input limit. The smallest condition-level paired
95% lower bound is 0.970703125.

## Exact final evidence

```text
final summary
A63F1C6C7CB7FACDC71A48E5DF05297CC823017EA342DC052310D36C97394462

condition ledger
276969B304BEEE1EDBEB3979C44A12DB4B256B436E6D82C86FF92DA7CE64F44D

run configuration
8BE990A776CE717291F56498D20D5B242A96FC6C5B9C93C5FF9BBAD7A8830B17

Git head file
DE2C731E2FEDA9EC21AB20FE33DF58AEC380CD4F7E231E2B1B40BCF4B79A1E51

empty Git status
E3B0C44298FC1C149AFBF4C8996FB92427AE41E4649B934CA495991B7852B855

source manifest
BC7FF2CA604C914A2BB610D0089450454C47FBB36EB76954DB28EA898C3CED59
```

The raw comparison ledger and execution outputs remain external. The exact
source manifest is committed, along with the compact result record and a
result-only verifier.

## Source bindings

Population evidence:

```text
summary    6d30d773f11c1155df3346128385da9231610ea05e95937e5acccb5529fca3fe
per-world  45e36bda230440d4fa2342183154b474473498df51917e147a37e0baa81c3323
manifest   8214aa82733a4fab9148a3ea210fd110b0a85f857483f14b15cb53d0f451255d
```

Gemma evidence:

```text
summary       7e7f8002b41d25d6448ecaea6882fa84926b006d95c1aa024a94b774b0b305ab
per-world     dda1009295378f4626b444b016b7aed2ff06c3468dc8b385d64809d4704a4706
prompt index  e238d801743939acaa362455410f85edd6a67d50f189d68a09bf75ffb63c60ab
SQLite ledger 7173853a236b777a02596bce3b61abecef3e61d52df661eb39422325fbb224a1
manifest      3fc0628c0c5fb56901160f35639c708a44cb2501540db9ea5022dce0e374b743
```

## Independent audit

The audit reproduced the final summary and condition-ledger hashes, all 21
condition rows and their frozen order, the 1.0 population accuracy for all three
checkpoint seeds in every condition, the 37-reference-correct pooled accuracy,
the equal-condition pooled delta, both unchanged classifications, the final
manifest entries, and the exact run-config and Git identities.

The uploaded evidence did not separately include `git-status.txt`; the manifest
binds it to the SHA-256 of the empty byte string, consistent with the clean
working-tree precondition and terminal output.

## Interpretation boundary

This result establishes the preregistered outcome for this synthetic,
deterministic distributed-transformation benchmark under its exact frozen
training budget, test namespace, prompt contract, parser, and reference model.
It does not by itself establish broad general intelligence, natural-language
superiority, or superiority to arbitrary 1B models or alternate prompting
protocols.

Gemma is the frozen conventional pretrained reference, not a
training-compute-matched baseline. Efficiency metrics remain separate from the
primary accuracy classifier.

## Closed boundary

This result-only stage contains exactly compact immutable result JSON, the
byte-identical final source manifest, this scientific record, and result-only
qualification CI. It commits no raw JSONL, SQLite database, model, checkpoint,
runner, wrapper, world generator, inference path, or training path.
