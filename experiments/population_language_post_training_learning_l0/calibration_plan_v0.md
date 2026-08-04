# Post-Training Learning L0 — Calibration Plan v0

Status: **immutable plan only; calibration and final execution are not authorized**

## Purpose

This slice binds a previously verified Population Language L0 reference output to the already frozen Post-Training Learning L0 calibration grid.

It creates a reviewable, hash-pinned plan before any calibration execution. The plan contains exact checkpoint identities, candidate order, model/calibration-world pairing, and expected result-row order, but no final-world material and no execution authorization.

## Required verified input

Plan construction accepts only a `ReferenceOutputManifest` produced by the strict verifier in PR #211.

The manifest must report:

```text
diagnosis = POPULATION_LANGUAGE_L0_REFERENCE_RUN_VALID
post_training_base_eligible = true
```

It must also contain exactly three population checkpoint records in preregistered model-seed order:

```text
120100
120101
120102
```

Each record binds:

- one absolute checkpoint path;
- positive bounded file size;
- exact checkpoint-file SHA-256;
- exact canonical model-state SHA-256.

A missing, reordered, duplicated-path, malformed, or ineligible reference binding fails closed.

## Frozen calibration contents

The plan records the exact calibration contract:

```text
ranks          = 1, 2, 4, 6
learning rates = 0.001, 0.003, 0.01
updates        = 32, 64, 128, 256
candidates     = 48
seed pairs     = 3
result rows    = 144
```

Candidate order remains:

```text
rank
→ learning rate
→ update count
```

Each candidate record includes:

- candidate identifier;
- rank;
- learning rate;
- update count;
- exact trainable adapter parameters;
- exact persisted FP32 bytes;
- exact adaptation-example presentations.

The plan also stores:

- calibration grid SHA-256;
- exact model-seed/calibration-world-seed pairs;
- calibration-world fingerprints;
- all 144 expected `(candidate_id, model_seed, calibration_world_seed)` result keys in frozen order.

## Explicit absence of final-world material

The exact plan schema contains no field for:

- final-world seeds;
- final-world fingerprints;
- final labels;
- final validation or test rows;
- retention labels;
- a selected candidate;
- a final conclusion.

Any additional field changes the exact schema and is rejected.

## Non-authorization boundary

Every valid plan contains:

```text
calibration_authorized = false
final_execution_authorized = false
```

Changing either value to `true` invalidates the artifact.

This deliberately separates three decisions:

```text
verified reference output
→ immutable calibration plan
→ later explicit calibration authorization
```

A valid plan is therefore necessary preparation but never sufficient authority to start GPU work.

## Persistence contract

The plan is canonical UTF-8 JSON with exact insertion order and a maximum size of 256 KiB.

It is written with create-once `xb` semantics, flushed, fsynced, and returned with its complete SHA-256. Loading requires the expected SHA-256 and rejects:

- symbolic links;
- non-regular files;
- empty or oversized files;
- duplicate JSON keys;
- malformed UTF-8 or JSON;
- schema drift;
- hash mismatch;
- candidate, pair, checkpoint, or fingerprint drift;
- authorization drift.

## Qualification fixtures

CI constructs a synthetic eligible `ReferenceOutputManifest` with three explicit checkpoint records. No checkpoint file is opened and no active output is inspected.

Qualification proves:

- exact 48-candidate and 144-row order;
- exact three checkpoint and seed-pair bindings;
- create-once hash-pinned save/load;
- absence of final-world fields;
- rejection of ineligible and reordered manifests;
- rejection of relative result roots;
- rejection of calibration or final authorization;
- rejection of injected final-world fields;
- rejection of duplicate checkpoint paths, grid drift, and file tampering.

## Explicit exclusions

This slice performs no:

- access to the active reference-training directory;
- checkpoint loading;
- calibration-world model execution;
- adapter optimization;
- final-world, validation, test, or retention access;
- CUDA launch;
- candidate selection;
- result generation;
- operational authorization;
- merge.

The next execution slice may consume a reviewed plan only after a separate explicit authorization boundary is defined and approved.
