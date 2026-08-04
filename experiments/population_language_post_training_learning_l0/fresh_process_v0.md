# Post-Training Learning L0 — Fresh-Process Persistence Harness v0

Status: **harness-only; no calibration, protected final evaluation, or scientific result**

## Purpose

This slice proves one narrow operational claim before the real experiment:

> A separately started Python process can load one exact immutable Population Language L0 checkpoint plus one exact tensor-only adapter artifact and produce neural outputs without inheriting parent-process model state.

This is not yet the protocol's persistence result. The probe accepts no labels, computes no accuracy, and does not access calibration, final, validation, test, or retention splits.

## Exact source stack

- checkpoint-contract head: `0b43d2cfedcaaf92a9905750ba3cac809645bebd`
- base checkpoint loader: strict seed + file SHA-256 + canonical state SHA-256
- adapter artifact: strict tensor-only format from the execution-primitives slice
- execution device: CPU only

## Parent/child boundary

The parent writes one create-once bounded request artifact containing:

```text
version
request nonce
parent PID
explicit checkpoint path, seed, file hash, canonical hash
explicit adapter path, file hash, rank
worker count
explicit adaptation-path input IDs
create-once result path
```

The parent launches:

```text
python -m ai_hypothesis.population_language.post_training_learning_l0_fresh_process \
  --child ABSOLUTE_REQUEST_PATH
```

The child process:

1. validates the exact request schema and bounded canonical JSON;
2. records its independent PID and a child-created start nonce;
3. hash-validates and safely loads the exact CPU base checkpoint;
4. hash-validates and loads the exact tensor-only adapter artifact;
5. reconstructs a fresh `BoundedPopulationAdapter`;
6. runs the supplied label-free neural probe;
7. verifies that the immutable base canonical hash did not change;
8. writes one create-once bounded result artifact;
9. exits.

The parent then verifies the request hash, request nonce, parent and child PIDs, child start nonce, checkpoint identities, unchanged base hash, adapter identity, input fingerprint, output token bounds, and full-logits SHA-256.

## Label and oracle boundary

The request schema has no field for:

- targets;
- expected answers;
- world seeds;
- affine rule parameters;
- calibration metrics;
- final metrics;
- retention metrics;
- selection results.

Only explicit input token IDs are accepted. They must match the exact adaptation-task prefix shape and end at `<answer>`. This proves artifact reconstruction and neural execution but cannot produce or leak a scientific conclusion.

## Process isolation evidence

A valid result requires:

- a positive parent PID equal to the process that launched the probe;
- a positive child PID different from the parent PID;
- a fresh 256-bit child start nonce generated inside the child;
- a child-reported request SHA-256 equal to the parent's pinned request artifact;
- a result file that did not exist before launch and was created with exclusive `xb` semantics.

No Python object, model instance, optimizer, worker state, or raw adaptation example is passed through process memory. The only cross-process state is the explicit request, checkpoint file, adapter artifact, and result file.

## Bounded execution

```text
request JSON      <= 32 KiB
result JSON       <= 64 KiB
probe batch       <= 8
probe sequence    <= 8 tokens
worker count      <= 256
subprocess timeout <= 600 seconds
```

The parent hides CUDA devices from the child. The checkpoint remains CPU-resident for this qualification probe.

## Qualification fixture

CI creates only temporary synthetic production-shaped artifacts:

- a freshly initialized 18,967,968-parameter population checkpoint;
- a rank-1 tensor-only adapter with one bounded nonzero bias entry;
- one exact five-token adaptation prefix.

The test then launches a real child Python process and verifies successful reconstruction. Additional tests prove fail-closed behavior for request-file tampering, wrong adapter hashes, pre-existing result files, relative paths, invalid adaptation prefixes, and attempts to add labels or result fields to the request.

## Explicit exclusions

This slice performs no:

- active reference-output discovery or access;
- actual trained checkpoint loading;
- adapter optimization;
- calibration-world execution;
- final-world, validation, test, or retention access;
- accuracy calculation;
- paired bootstrap;
- candidate selection;
- CUDA execution;
- final-run authorization.

A later runner may reuse this process boundary only after the completed checkpoint manifest is reviewed and the corresponding execution is explicitly authorized.
