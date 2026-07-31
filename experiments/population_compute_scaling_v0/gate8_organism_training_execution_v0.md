# Gate-8 organism training execution v0

## Status

**FROZEN TRAINING AND DEVELOPMENT-VALIDATION EXECUTION PATH — SCIENTIFIC TEST WORLDS AND THE 1B REFERENCE REMAIN CLOSED.**

Exact qualified protocol head:

`869791e5b44089f9c79447b8ae212ce830f8496a`

This stage implements the already-frozen three-seed training experiment. CI exercises only contract-world mechanics and wrapper preflight; it does not consume any training or validation world.

## Execution components

### Local-label mechanics

`gate8_organism_training.py`:

- accepts only `contract`, `train` and `validation` public worlds;
- rejects `test` and `demonstration` before world validation;
- topologically derives carrier/symbol state for every node;
- creates one exact transition example per edge;
- never reads a world truth object;
- collates all edges from a world batch onto one device;
- computes the frozen message, answer and activity losses;
- reports exact edge correctness and 256-code coverage.

### Development runtime

`gate8_organism_development_runtime.py` is a full-mode mirror of the qualified deterministic runtime. It adds only `train` and `validation` split admission and retains:

```text
root seed code 0
synchronous next-round mailboxes
activity_logit >= 0
argmax 256-code messages
argmax 16-symbol terminal answer
public depth round cap
8 bits per delivered message
```

It reads no truth and exposes no ablation. Contract regressions require identical prediction, target reachability, rounds, recurrent updates, delivered messages and communicated bits versus the qualified runtime.

### GPU runner

`train_gate8_organism.py`:

1. enforces seed 0, 1 or 2;
2. requires CUDA device zero;
3. enforces `PYTHONHASHSEED=seed` and `CUBLAS_WORKSPACE_CONFIG=:4096:8`;
4. disables TF32, autocast and nondeterministic algorithms;
5. initializes one exact 19,649-parameter organism from the run seed;
6. executes exactly 1,024 AdamW steps over 262,144 on-demand training worlds;
7. writes checkpoints at steps 256, 512, 768 and 1,024;
8. validates each checkpoint over the frozen 3,072 development worlds;
9. applies the frozen selector and admission classifier;
10. writes complete telemetry and checkpoint identities.

The runner caches the validation worlds after the first candidate so all four candidates see byte-equivalent generated objects and no validation-generation timing difference can affect selection.

## Training batches

For optimizer step `s` in `1..1024`:

```text
global world start = (s - 1) * 256
global world end   = s * 256, exclusive
```

Each global index resolves through the frozen six-condition round-robin schedule. Every generated world contributes all of its edges to one local-transition batch. There is no gradient accumulation, dropped world, repeated world, adaptive batch size or curriculum.

Per-step telemetry records:

- exact global world range;
- world and edge counts;
- learning rate;
- total, message, answer and activity losses;
- pre-clip gradient norm;
- CUDA-synchronized step duration;
- cumulative inbox- and target-code coverage.

## Candidate validation

At each frozen checkpoint step, the model is placed in evaluation mode. For every training-regime condition:

- exactly 512 `validation` worlds with indices 0..511 are run through complete trees;
- target reachability and exact answer correctness are counted;
- recurrent updates, messages and communicated bits are recorded;
- all edges are separately evaluated for message, answer and activity accuracy and loss;
- inbox and target message-code sets are accumulated.

The model returns to training mode only after the candidate record is complete.

No depth above 16 and no scientific-test world is generated.

## Checkpoint artifacts

Each candidate checkpoint contains:

```text
experiment version
qualified protocol head
training seed
optimizer step
learned parameter count
CPU state dictionary
```

The runner records SHA-256 for every candidate. After all four candidates exist, the frozen selector chooses one and copies it byte-for-byte to `selected-checkpoint.pt`; the copy must retain the candidate SHA-256.

A non-admitted selected checkpoint is still preserved as negative training evidence, but it cannot open scientific execution.

## Result artifact

`gate8-organism-training-result.json` contains:

- source, protocol, runtime and architecture heads;
- Python, Torch, CUDA, cuDNN, GPU and deterministic environment identity;
- complete optimizer settings;
- all 1,024 telemetry rows;
- cumulative training code coverage;
- all four checkpoint hashes;
- all validation condition rows and local metrics;
- selected checkpoint step, hash and frozen admission decision;
- explicit boundary flags.

Required flags:

```text
training_performed = true
validation_performed = true
scientific_test_worlds_generated = false
reference_tokenizer_loaded = false
reference_model_weights_loaded = false
reference_inference_performed = false
```

## Windows admission wrapper

`train_gate8_organism.ps1` requires:

- exact execution branch;
- clean Git worktree;
- output outside the repository;
- unused output path;
- CUDA preflight;
- explicit seed 0, 1 or 2.

It writes Git and run provenance before execution. On success it validates the result structure and creates a recursive SHA-256 manifest covering all checkpoints, selected checkpoint, result, telemetry and provenance. On failure it preserves the partial output for diagnosis.

## CI boundary

Linux CI:

- installs CPU Torch;
- compiles execution modules and runner;
- derives labels only from contract worlds;
- performs one optimizer step only on contract worlds;
- proves development-runtime equivalence only on contract worlds;
- verifies test/demonstration rejection;
- never calls the full runner.

Windows CI invokes the wrapper only with `GATE8_ORGANISM_TRAINING_WRAPPER_SMOKE=1`, which exits before Torch, CUDA, worlds, optimizer or checkpoint creation.

## Closed boundaries

The execution path contains no:

- scientific-test world generation;
- demonstration generation;
- 1B tokenizer or model load;
- reference-model inference;
- population-scaling classifier;
- 1B comparison classifier;
- post-hoc rescue, threshold change or checkpoint addition.

A successful admitted training result is necessary but not sufficient for scientific execution. Exact 1B weight binding and the joint execution/audit protocol remain separate future stages.
