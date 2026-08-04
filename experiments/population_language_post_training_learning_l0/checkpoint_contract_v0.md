# Post-Training Learning L0 — Reference Checkpoint Contract v0

Status: **implementation-only; no checkpoint discovery, loading of the active run, calibration, or final result**

## Purpose

This slice defines the exact trust boundary for loading the immutable Population Language L0 base checkpoint after the active reference run has completed and its manifest has been reviewed.

The checkpoint loader is deliberately unusable without all three externally supplied identities:

- the preregistered model seed;
- the exact checkpoint-file SHA-256;
- the exact canonical model-state SHA-256.

It does not search directories, choose a checkpoint, infer a seed, trust a filename, or accept an unpinned artifact.

## Exact source stack

- execution-primitives head: `821449afe7381d4becc9c43dc456632b66b8f034`
- reference training implementation: `population-language-l0-reference-training-v0`
- base model: `population`
- exact learned parameters: `18,967,968`
- exact completed optimizer step: `4,096`

## Accepted checkpoint envelope

The safely decoded top-level object must be a plain dictionary with this exact key order:

```text
version
model
seed
optimizer_step
state_dict
```

The metadata must match:

```text
version        = population-language-l0-reference-training-v0
model          = population
seed           = one explicitly requested preregistered model seed
optimizer_step = 4096
```

The loader uses `torch.load(..., map_location="cpu", weights_only=True)` only after verifying the complete file SHA-256.

## State-dictionary contract

The state dictionary must match a freshly constructed production-shaped `PopulationLanguageOrganism` exactly:

- exact tensor names;
- exact insertion order;
- no missing or extra tensors;
- exact tensor shapes;
- exact dtypes;
- CPU placement;
- strided and contiguous layout;
- finite floating-point values;
- exact raw byte count;
- strict `load_state_dict` success;
- exact canonical post-load state SHA-256.

The raw FP32 model state is exactly:

```text
18,967,968 × 4 bytes = 75,871,872 bytes
```

The complete serialized checkpoint is bounded at 96 MiB. Empty files, oversized files, directories, symbolic links, malformed archives, changing files, and hash mismatches fail closed.

## RNG and device boundary

Fresh model construction occurs inside an isolated CPU RNG fork so checkpoint validation does not perturb the caller's RNG state.

This slice performs no CUDA operation. A successfully loaded model remains on CPU and in evaluation mode. Later code may explicitly move the already validated model to an execution device.

## Manifest dependency

This contract does not invent the checkpoint hashes. After the active reference training run completes, a separate manifest-verification step must establish the exact checkpoint path, file SHA-256, canonical state SHA-256, seed, training version, and completion status.

Only those reviewed values may be supplied to this loader.

## Explicit exclusions

This slice contains no:

- active output-directory access;
- checkpoint discovery;
- selection between checkpoint candidates;
- calibration-world execution;
- adapter training;
- final-world or retention-label access;
- subprocess restart harness;
- GPU launch;
- result publication;
- authorization to run calibration or final evaluation.

## Qualification evidence

The tests use a newly initialized synthetic production-shaped checkpoint written inside a temporary directory. They do not read any active or historical experiment output.

Qualification covers:

- exact metadata and size constants;
- full production-shaped save/load round trip;
- file and canonical hash pinning;
- rejection of missing, extra, reordered, wrong-dtype, and nonfinite tensors;
- rejection of malformed payloads, wrong metadata, wrong seeds, directories, and invalid path types;
- CPU-only loading and evaluation mode;
- preservation of the caller's CPU RNG state.
