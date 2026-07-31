# Gate-8 v1 factorized-message replication execution

## Status

This slice admits **only seeds 1 and 2** after the independently qualified seed-0 admission result on:

```text
f259620f7d3beab2f886c76271c753e9ebf96dc9
```

Seed 0 remains bound to its original qualified execution and permanent result. This slice cannot rerun it.

## Frozen scientific stack

Replication reuses the exact qualified stack:

```text
protocol      a33dc123d090268a531d112251ea3ab53cb50062
runtime       333d88ac4fc52f1651741fba224e0b4605feedd3
architecture  c3ab64008c816fa1eb6f9d6f8f0a1a99ed195ec8
```

No architecture, message semantics, training condition, world allocation, optimizer, learning-rate schedule, loss, checkpoint set, validation range, selector, admission threshold or artifact hash procedure changes.

## Replication run

Each seed independently executes:

```text
262,144 training worlds
256 worlds per batch
1,024 optimizer steps
checkpoints at 256, 512, 768 and 1,024
validation indices 512..1,023
512 validation worlds per condition
```

Training conditions remain:

```text
(32,4), (64,4), (64,8), (128,4), (128,8), (128,16)
```

The runner uses the existing qualified seed-0 runner only for frozen helper mechanics and provenance constants. It constructs a fresh model and optimizer for each replication seed and generates worlds under that exact seed.

## Entry boundaries

The direct Python runner and PowerShell wrapper both admit only:

```text
seed 1
seed 2
```

They reject seed 0 and all other integers before CUDA execution.

The wrapper additionally requires:

- the exact replication branch;
- a clean Git working tree;
- an output directory outside the repository that does not exist yet;
- deterministic CUDA configuration;
- the exact completed result and selected-checkpoint artifact contract.

## Closed boundaries

Replication remains development-only:

```text
scientific-test worlds generated  false
reference tokenizer loaded        false
reference model weights loaded    false
reference inference performed     false
```

No depth-32/64/128 scientific result is produced here. Scientific evaluation remains closed until both replication seeds are audited and permanently recorded.

## Qualification boundary

CI may:

- compile the replication runner;
- load it without CUDA execution;
- prove seed 0 and invalid seeds fail before Torch/model/world work;
- inspect exact frozen source bindings and artifact mechanics;
- smoke the Windows wrapper for seeds 1 and 2 before Torch/CUDA/world generation;
- prove the exact five-file slice.

CI may not invoke the full replication runner or generate any train, validation, test or demonstration world.

## Required external artifacts per seed

After each run, preserve and upload:

```text
training/gate8-factorized-organism-training-result.json
training/selected-checkpoint.pt
manifest.sha256
```

Both seeds must be audited independently before any scientific-test execution path opens.
