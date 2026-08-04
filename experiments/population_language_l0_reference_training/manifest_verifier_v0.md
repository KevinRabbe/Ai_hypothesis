# Population Language L0 — Reference Manifest Verifier v0

Status: **read-only verifier implementation; no active output access or scientific execution**

## Purpose

This slice defines the exact verification boundary for the completed Population Language L0 reference-training output.

It exists so that after the active run is finished, one explicitly named output directory can be checked without trusting filenames, copied summaries, or manually selected checkpoints.

The verifier does not search for runs and does not choose among output directories. The caller must supply:

- one absolute output-root path;
- the exact 40-character execution head expected for that run.

## Expected immutable inventory

A complete reference output contains exactly 17 regular files:

```text
run-start.json
summary.json

seed-120100.json
seed-120101.json
seed-120102.json

progress/transformer-seed-120100.json
progress/population-seed-120100.json
progress/transformer-seed-120101.json
progress/population-seed-120101.json
progress/transformer-seed-120102.json
progress/population-seed-120102.json

checkpoints/transformer-seed-120100.pt
checkpoints/population-seed-120100.pt
checkpoints/transformer-seed-120101.pt
checkpoints/population-seed-120101.pt
checkpoints/transformer-seed-120102.pt
checkpoints/population-seed-120102.pt
```

Missing files, additional files, symbolic links, unsupported filesystem entries, malformed JSON, duplicate JSON keys, empty files, oversized files, and inventory changes during verification fail closed.

## Recomputed evidence

The verifier does not trust the conclusion strings in `summary.json` by themselves. It recomputes:

- the exact training contract;
- the locked training-schedule SHA-256;
- the first-256 train, validation, and test dataset fingerprints;
- reference-run validity with `classify(seed_rows)`;
- the population worker-scaling conclusion;
- the fixed scientific boundary fields.

It also requires:

- exact reference version, branch, base head, and caller-pinned execution head;
- exact seed order `120100, 120101, 120102`;
- valid CUDA/BF16 execution evidence;
- positive dataset-cache timing and resident-byte evidence;
- equality between every `seed-*.json` object and its corresponding summary row;
- exact `COMPLETE` progress records at optimizer step 4096;
- equality of progress hashes and curves with the final trained-model rows.

## Checkpoint linkage

For all six checkpoint files, the verifier checks:

- the exact expected relative path;
- the bounded regular-file contract;
- the complete file SHA-256 against the trained-model row.

For the three population checkpoints that may become immutable bases for Post-Training Learning L0, it additionally invokes the strict checkpoint contract from PR #209. That contract safely decodes the exact checkpoint on CPU and verifies:

- reference-training version;
- model identity;
- preregistered seed;
- completed optimizer step 4096;
- exact state tensor names, order, shapes, dtypes, layout, contiguity, and finiteness;
- exact canonical model-state SHA-256.

The returned manifest records only those three fully validated population checkpoint identities.

## Eligibility boundary

`post_training_base_eligible` is true only when the recomputed reference diagnosis is:

```text
POPULATION_LANGUAGE_L0_REFERENCE_RUN_VALID
```

This field does not authorize calibration. It only states that the completed reference output satisfies the already frozen reference-run validity contract and therefore contains checkpoint candidates that may be considered later.

## Qualification fixtures

CI uses a fully synthetic temporary 17-file output tree. It does not read any local or active reference output.

The fixture exercises:

- exact inventory validation;
- recomputation of reference validity and population scaling;
- per-seed, progress, and checkpoint-hash linkage;
- rejection of additional files;
- rejection of seed-artifact drift;
- rejection of checkpoint tampering;
- rejection of an incorrect execution head;
- rejection of a conclusion string that disagrees with recomputation.

The strict population checkpoint decoder is already independently qualified with production-shaped checkpoint fixtures in PR #209. Manifest tests replace that expensive decoder with a controlled boundary stub while still proving that each expected seed, path, file hash, and canonical hash is passed into it exactly.

## Explicit exclusions

This slice performs no:

- access to `F:\population_language_l0_reference_training_v0` or any other active output;
- output-directory discovery;
- modification of reference artifacts;
- model training or evaluation;
- Post-Training Learning adapter optimization;
- calibration-world access;
- final-world, validation, test, or retention-label access;
- CUDA launch;
- candidate selection;
- operational authorization;
- merge.

After the reference run finishes, verification remains a separate explicit step. The verifier must be run against the completed directory before any checkpoint is supplied to calibration tooling.
