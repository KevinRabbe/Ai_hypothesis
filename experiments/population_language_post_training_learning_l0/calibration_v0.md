# Post-Training Learning L0 — Calibration Contract v0

Status: **CONTRACT ONLY — NO CALIBRATION OR FINAL RESULT**

This document freezes how the bounded adapter introduced on
`agent/population-language-post-training-learning-l0-adapter-v0` may be
calibrated before any protected final-world execution.

The calibration slice does not train a model, inspect a checkpoint, read final
world labels, start a GPU job, or authorize a final run. It only defines the
candidate grid, validity boundary, qualification thresholds, deterministic
selection rule, and rejection behavior.

## 1. Exact provenance

- source adapter branch:
  `agent/population-language-post-training-learning-l0-adapter-v0`
- source adapter head:
  `508a1021f3724a39023d4a4f7c6918d98f379f5c`
- calibration branch:
  `agent/population-language-post-training-learning-l0-calibration-v0`
- preregistration issue: `#202`
- base model: the frozen 18,967,968-parameter Population Language L0 model
- adapter artifact: declared trainable tensors only
- calibration worlds: `210100`, `210101`, `210102`
- model seeds: `120100`, `120101`, `120102`
- protected final worlds: `220100`, `220101`, `220102`

The model/calibration-world pairs are fixed in listed order:

1. `(120100, 210100)`
2. `(120101, 210101)`
3. `(120102, 210102)`

Final worlds and final labels are forbidden during calibration.

## 2. Candidate grid

The grid is the full Cartesian product:

- adapter rank: `1`, `2`, `4`, `6`
- learning rate: `0.001`, `0.003`, `0.01`
- adaptation updates: `32`, `64`, `128`, `256`

This produces exactly:

- 48 candidates;
- 3 fixed seed pairs per candidate;
- 144 result rows.

Locked grid-manifest SHA-256:

```text
84264fbb475259ca224c01cee81700a62c6baf7e73017a0510fc5cbc6c036874
```

Candidate order is rank, then learning rate, then update count, all in the
listed order. Result rows must follow candidate order and then fixed seed-pair
order. Missing, duplicate, extra, or reordered rows invalidate the calibration
run.

## 3. Adapter budgets

The existing adapter contract remains authoritative.

| Rank | Trainable parameters | Raw FP32 bytes |
|---:|---:|---:|
| 1 | 33,456 | 133,824 |
| 2 | 62,800 | 251,200 |
| 4 | 121,488 | 485,952 |
| 6 | 180,176 | 720,704 |

Every candidate remains within:

- at most 189,680 trainable adaptation parameters;
- at most 1,048,576 persisted bytes;
- declared trainable tensors only;
- immutable base-model parameters.

## 4. Locked optimization contract

Every candidate uses:

```text
optimizer                 AdamW
betas                     (0.9, 0.999)
epsilon                   1e-8
weight decay              0
gradient-norm clip        1.0
microbatch size           8
learning-rate schedule    constant
worker count              32
precision                 CUDA BF16 autocast with FP32 weights
early stopping            forbidden
```

The 64 adaptation examples are consumed in deterministic ordinal order,
restarting from ordinal zero when the sequence repeats. Each update consumes
one microbatch, so the candidate presentation counts are:

| Updates | Example presentations |
|---:|---:|
| 32 | 256 |
| 64 | 512 |
| 128 | 1,024 |
| 256 | 2,048 |

All remain below the protocol maximum of 4,096 presentations.

The adapter initialization seed depends only on the model seed:

```text
adapter_seed = 700000 + model_seed
```

It must not depend on calibration-world or final-world identity.

## 5. Calibration measurements

For each candidate and fixed seed pair, record:

- baseline direct-holdout accuracy on 64 unseen depth-1 examples;
- immediate post-adaptation direct-holdout accuracy;
- fresh-process post-restart direct-holdout accuracy;
- baseline composition accuracy on 512 depth-2 examples;
- immediate post-adaptation composition accuracy;
- fresh-process post-restart composition accuracy;
- canonical base checkpoint hash before and after adaptation;
- strict adapter-artifact hash and byte count;
- original Population Language L0 path bitwise-identity result;
- all optimizer, seed, provenance, and anti-oracle fields required by the
  executable contract.

Calibration composition uses only the calibration split. The depth-3 validation
and depth-4 test splits belong to protected final worlds and may not be loaded.

## 6. Candidate qualification

A candidate qualifies only when all three fixed seed-pair rows are valid and:

1. direct-holdout gain is strictly positive for every seed pair;
2. depth-2 composition gain is strictly positive for every seed pair;
3. mean depth-2 composition gain is at least `0.01`;
4. the base checkpoint hash is unchanged;
5. immediate versus fresh-process restart accuracy drift is at most `0.001`;
6. the original L0 path is bitwise identical;
7. the strict artifact, provenance, and anti-oracle boundaries pass.

There is no early stopping and no candidate-specific threshold adjustment.

## 7. Deterministic selection

Qualified candidates are ranked by this exact tie-break sequence:

1. larger minimum composition gain;
2. larger mean composition gain;
3. larger minimum direct-holdout gain;
4. larger mean direct-holdout gain;
5. fewer trainable adaptation parameters;
6. fewer adaptation updates;
7. lower learning rate;
8. lexicographically smaller candidate identifier.

No other metric or judgment may alter the order after calibration results are
visible.

## 8. Rejection behavior

When no candidate qualifies, the calibration conclusion is exactly:

```text
POST_TRAINING_LEARNING_L0_CALIBRATION_REJECTS_ADAPTER_CANDIDATE
```

In that state:

- no candidate is selected;
- protected final execution is not eligible;
- the thresholds may not be weakened retrospectively;
- a new adapter or prospective protocol version is required.

When a candidate qualifies, calibration marks it only as:

```text
FINAL_EXECUTION_ELIGIBLE_AFTER_SEPARATE_EXPLICIT_AUTHORIZATION
```

Selection is not operational authorization. A protected final-world run still
requires a separate explicit owner instruction.

## 9. Anti-oracle and neural boundary

Calibration is invalid if any of the following occurs:

- a final world or final label is loaded;
- the world seed is provided to adaptation;
- affine rule parameters are provided to adaptation;
- the world generator is imported by model runtime;
- symbolic rule fitting is used;
- symbolic execution replaces model logits;
- raw adaptation examples are persisted in the artifact;
- external retrieval is enabled during evaluation;
- the base checkpoint changes;
- a restart claim is made without a fresh process;
- result rows are missing, duplicated, extra, or reordered.

The model receives adaptation examples through the declared token interface and
learns through gradients only. Model logits remain authoritative.

## 10. Scope of this slice

This slice contains only:

- the executable calibration/selection contract;
- tests using synthetic result rows;
- this protocol document;
- CI qualification.

It contains no calibration runner, checkpoint loader, optimizer execution,
final-world evaluator, result artifact, or GPU launch path. The active reference
training run remains untouched.
