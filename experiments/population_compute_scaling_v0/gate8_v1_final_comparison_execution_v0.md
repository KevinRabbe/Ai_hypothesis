# Gate-8 v1 final population-versus-Gemma comparison v0 — execution

## Status

**FROZEN POST-EXPOSURE FINALIZER FOR THE ALREADY COMPLETED POPULATION AND GEMMA EVALUATIONS. NO MODEL, CHECKPOINT, WORLD GENERATOR, TRAINING, POPULATION EXECUTION, OR REFERENCE INFERENCE PATH IS ADMITTED.**

Base: exact qualified Gemma reference-result head:

```text
1d48ecfd623a2fb9e3a2f846a4d1c49d20d8cadc
```

The source predictions are immutable. This slice only verifies their external identities, reconstructs the preregistered paired statistics, and invokes the unchanged five-outcome Gate-8 reference classifier.

## Exact source bindings

Population result:

```text
result head
14636d219781381853f81036b96c691b7e6997ee

summary
6d30d773f11c1155df3346128385da9231610ea05e95937e5acccb5529fca3fe

225,792-row per-world ledger
45e36bda230440d4fa2342183154b474473498df51917e147a37e0baa81c3323

manifest
8214aa82733a4fab9148a3ea210fd110b0a85f857483f14b15cb53d0f451255d
```

Gemma reference result:

```text
result head
1d48ecfd623a2fb9e3a2f846a4d1c49d20d8cadc

summary
7e7f8002b41d25d6448ecaea6882fa84926b006d95c1aa024a94b774b0b305ab

10,752-row per-world ledger
dda1009295378f4626b444b016b7aed2ff06c3468dc8b385d64809d4704a4706

10,752-row prompt index
e238d801743939acaa362455410f85edd6a67d50f189d68a09bf75ffb63c60ab

transactional SQLite ledger
7173853a236b777a02596bce3b61abecef3e61d52df661eb39422325fbb224a1

manifest
3fc0628c0c5fb56901160f35639c708a44cb2501540db9ea5022dce0e374b743
```

Every hash is checked before row parsing. The finalizer rejects altered bytes rather than attempting repair or partial comparison.

## Exact row audit

The population ledger must contain exactly:

```text
21 conditions × 512 worlds × 3 checkpoint seeds × 7 modes
= 225,792 rows
```

Its required ordering is condition, world index, checkpoint seed, then the frozen mode order. The finalizer validates every row schema and identity while extracting only `full` correctness for the comparison.

The reference ledger must contain exactly 10,752 contiguous rows in the same population-major condition and world-index order. Every population/reference pair must agree on world ID and oracle answer. Any duplicate, omission, reordering, schema drift, or correctness inconsistency fails closed.

## Preregistered paired estimand

For one condition and one world index:

```text
mean(correct_seed0, correct_seed1, correct_seed2) - correct_reference
```

The condition point estimate is the mean of those 512 paired world values.

The 95% interval uses 20,000 deterministic empirical world-index bootstrap replicates. Within a condition, one sampled index vector is applied identically to all three population seeds and the reference.

Frozen namespace:

```text
gate8-v1-final-comparison-bootstrap-v0
```

The implementation uses NumPy PCG64, empirical multinomial sampling, and linear 2.5%/97.5% quantiles, matching the already qualified Gate-8 statistical mechanics.

## Pooled comparison

Each condition receives exactly `1/21` weight. Population size, depth, prompt length, runtime, and row count cannot alter the primary weight.

The protocol froze equal condition weighting but did not require artificial correlation between different conditions. Therefore each condition has its own deterministic bootstrap stream. Replicate number `r` is averaged across all 21 condition streams to form pooled replicate `r`:

```text
independent condition streams
same replicate index
exact equal condition weight
```

This coupling rule is data-independent and changes no threshold or classifier branch. Within every condition, all four compared systems remain paired by the same resampled world indices.

## Unchanged classifier

The finalizer invokes `classify_gate8_v1_reference_comparison` from the qualified scientific protocol head:

```text
6bb89111a47713bea0a23bb1cae662ed5ec56b42
```

That wrapper invokes the original Gate-8 classifier unchanged. The noninferiority margin remains `0.05`, all inequalities retain their original strictness, and the only possible outcomes are:

```text
G8_POPULATION_EXCEEDS_1B_REFERENCE
G8_POPULATION_NONINFERIOR_TO_1B_REFERENCE
G8_1B_REFERENCE_SUPERIOR
G8_1B_REFERENCE_MIXED
G8_1B_REFERENCE_COMPARISON_INCONCLUSIVE
```

No observed result is hard-coded into the executor or qualification tests.

## Output

The execution writes only:

```text
run-config.json
git-head.txt
git-status.txt
comparison/gate8-v1-final-comparison-per-condition.jsonl
comparison/gate8-v1-final-comparison-summary.json
manifest.sha256
```

The summary preserves all 21 condition comparisons, the pooled point and interval, the unchanged population-scaling result, the final reference-comparison classification, source identities, bootstrap mechanics, and closed-boundary evidence.

## Closed boundaries

```text
source ledgers read-only              true
world generation performed            false
population execution performed        false
reference model loaded                false
reference inference performed         false
training performed                    false
```

CI compiles the full finalizer and runs synthetic contract tests only. It never downloads or reads the external scientific ledgers and never issues a real scientific classification.
