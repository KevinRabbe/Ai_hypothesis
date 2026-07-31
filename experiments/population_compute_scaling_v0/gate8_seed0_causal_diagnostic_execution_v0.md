# Gate-8 seed-0 causal diagnostic execution v0

## Status

This slice implements the fully qualified protocol at
`0fa9ec48c31b36c90d58da827139457fd812b98c`. It does not change the original
seed-0 non-admission, admit seeds 1 or 2, generate scientific-test worlds, or
load the Gemma tokenizer or model.

## Required source artifacts

The guarded wrapper accepts exactly these three immutable files:

```text
selected-checkpoint.pt
4aca6bfde7fa82cd2c1fec3613c4cc59303788616f352e8d419c90662d7b9a1b

gate8-organism-training-result.json
5f477022ac45a80d8b05b112b3485e4112519ddef78e0bb23990a090e0cc92e2

manifest.sha256
bb814b9ebb5116f0a13ff2ce130c5ad8e32ed4bd80453ddc167143b6cbf0bb8d
```

Both PowerShell and Python verify all three SHA-256 identities. Python loads the
checkpoint with `weights_only=True`, requires the original step-1024 seed-0
payload, loads its state dictionary strictly into the qualified architecture,
and verifies 15 tensors totaling exactly 19,649 parameters.

## Runtime interventions

A separate diagnostic runtime mirrors the qualified development scheduler. It
admits only `contract`, `train`, and `validation` public worlds and never reads
truth. The runner evaluates truth only after each runtime call to score the
unchanged validation set.

Four modes are executed from one immutable model copy:

```text
baseline
forced_active
message_low4_decode
forced_active_message_low4_decode
```

The interventions do not alter topology, worker order, hidden-state update,
message logits, message argmax, round horizon, or the source checkpoint.

Before any optimization, the runner must exactly reproduce the frozen seed-0
message, answer, activity, mean-target, minimum-target, and root-invariance
metrics. Any mismatch aborts the diagnostic.

## Independent optimization probes

Two separate model copies start from the same original checkpoint.

### Head-only

Only the exact 9,009 parameters under these prefixes are trainable:

```text
message_head.
activity_head.
answer_head.
```

It runs 256 steps on fresh global training addresses `[262144,327680)` and
preserves checkpoints at 64, 128, 192, and 256.

### Full resume

All 19,649 parameters are trainable. It runs 512 steps on disjoint fresh global
training addresses `[327680,458752)` and preserves checkpoints at 128, 256, 384,
and 512.

The learning rate follows the preregistered cosine schedule from `3e-4` at step
1 to exactly `3e-5` at step 512.

Each of the eight checkpoints is evaluated on:

- all 3,072 unchanged development-validation worlds;
- all local validation edges;
- the exhaustive 2,048-case non-root target-incoming root-symbol invariance
  table.

No checkpoint selection or early stopping is performed.

## Result

The result contains:

- exact source identities and Git head;
- deterministic CUDA environment;
- baseline reproduction evidence;
- all four runtime probes and per-condition accounting;
- all 768 training telemetry rows;
- all eight checkpoint files, hashes, and validation metrics;
- the five preregistered boolean findings;
- explicit closed-boundary flags.

The fixed completion status is:

```text
G8_SEED0_CAUSAL_DIAGNOSTIC_COMPLETE
```

This status is not an admission outcome.

## Guarded local execution

From the exact qualified execution branch, with a clean working tree:

```powershell
.\scripts\diagnose_gate8_seed0.ps1 `
    -CheckpointPath "F:\gate8_organism_training_seed0_v0\training\selected-checkpoint.pt" `
    -SourceResultPath "F:\gate8_organism_training_seed0_v0\training\gate8-organism-training-result.json" `
    -SourceManifestPath "F:\gate8_organism_training_seed0_v0\manifest.sha256" `
    -OutputRoot "F:\gate8_seed0_causal_diagnostic_v0"
```

The output directory must not already exist and must be outside the repository.

## Qualification boundary

CI compiles the full runner but never invokes it. Linux runs only contract-world
runtime regressions and finite synthetic invariance checks. Windows executes the
wrapper smoke path before source artifacts, Torch, CUDA, world generation,
optimization, or checkpoint writes.
