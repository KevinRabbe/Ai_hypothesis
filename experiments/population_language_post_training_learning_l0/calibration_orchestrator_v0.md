# Post-Training Learning L0 calibration orchestrator v0

## Status

Preparation and verification only.

This slice does not execute calibration, load a checkpoint, create an authorization, access a final world, select a candidate operationally, or authorize final evaluation.

## Purpose

The calibration contract defines 48 candidates across three model/calibration-world pairs, producing exactly 144 result rows. The immutable calibration plan binds those candidates to the three verified population checkpoints.

This slice adds the missing operational boundary between that plan and a future authorized calibration run:

1. validate one separately supplied calibration-only authorization;
2. materialize the exact 144 work items in frozen order;
3. bind every work item to its checkpoint, candidate, world, schedule, paths, and SHA-256;
4. collect the observed rows without silently discarding invalid or negative evidence;
5. verify the completed bundle independently;
6. recompute candidate selection or rejection;
7. preserve the requirement for a second explicit authorization before final-world execution.

## Exact source stack

- source calibration-plan head: `780c19c0c8e63dafe6e7c74bbfe3d579129e53fe`
- calibration grid: 48 candidates
- calibration pairs: 3
- expected rows: 144
- worker count: 32
- update counts: 32, 64, 128, 256

## Authorization boundary

The orchestrator does not construct or save an authorization.

It only accepts a separately supplied canonical JSON artifact with the exact fields:

- authorization version;
- calibration-only scope;
- source calibration-plan head;
- exact calibration-plan SHA-256;
- exact reference-summary SHA-256;
- exact result root;
- 256-bit authorization identifier;
- exact operator acknowledgement;
- `calibration_authorized = true`;
- `final_execution_authorized = false`.

The required acknowledgement is:

> I explicitly authorize the frozen 144-row calibration only; final-world execution remains unauthorized.

A changed plan hash, result root, reference summary, scope, acknowledgement, or final-execution flag is rejected.

## Run manifest

A valid authorization permits construction of one immutable run manifest. The manifest contains exactly 144 deterministic work items in the order declared by the calibration contract.

Each work item binds:

- ordinal;
- candidate identity, rank, learning rate, and updates;
- model seed and calibration-world seed;
- checkpoint path, file hash, and canonical-state hash;
- adapter initialization seed;
- exact adaptation-schedule SHA-256;
- adapter artifact path;
- fresh-process request and result paths;
- result-row path;
- expected result key;
- work-item SHA-256.

The schedule hashes are independently recomputed from the frozen deterministic microbatch order and checked against the already qualified execution-primitives hashes.

The run manifest is canonical JSON, create-once, fsynced, hash-addressed, and bounded to 2 MiB.

## Result bundle

The result bundle records all 144 observed rows in exact work-item order. It does not contain a trusted scientific conclusion.

Every record binds:

- ordinal;
- work-item SHA-256;
- raw observed calibration result row.

Invalid rows are not silently omitted. The bundle therefore preserves valid selection evidence, valid rejection evidence, and structurally complete but scientifically invalid evidence.

The result bundle is canonical JSON, create-once, fsynced, hash-addressed, and bounded to 16 MiB. Each row is bounded to 64 KiB.

## Independent verification

The verifier independently checks:

- canonical plan, authorization, manifest, and bundle provenance;
- exact schema and key order;
- exact 144-row identity and ordering;
- work-item hashes;
- reference-summary binding;
- checkpoint canonical-state binding for every row;
- the complete existing calibration row contract;
- calibration validity;
- candidate qualification and deterministic tie-breaking.

The verifier recomputes one of three outcomes:

1. valid calibration selecting a candidate;
2. valid calibration rejecting all candidates;
3. invalid calibration with no selection.

Even a successful selection produces only:

`FINAL_EXECUTION_ELIGIBLE_AFTER_SEPARATE_EXPLICIT_AUTHORIZATION`

The verification artifact always contains:

- `calibration_was_authorized = true`;
- `final_execution_authorized = false`.

## Protected boundaries

This slice performs no:

- discovery or access of the active reference output;
- checkpoint loading;
- model construction;
- adapter training;
- calibration-world model execution;
- final-world, validation, test, or retention access;
- CUDA launch;
- authorization creation;
- final-evaluation authorization;
- merge.

CI uses only synthetic manifest metadata and synthetic result rows. No active output path or protected label is used.

## Qualification criteria

The slice qualifies only when:

- the full existing reference and Post-Training Learning preparation stack remains green;
- authorization is exact, plan-bound, and calibration-only;
- 144 work items are deterministic and hash-stable;
- result order and work-item linkage fail closed;
- valid selection and valid rejection are both preserved;
- checkpoint provenance failure becomes an invalid run;
- canonical provenance hashes are mandatory;
- create-once persistence is enforced;
- the exact four-file scope is preserved.
