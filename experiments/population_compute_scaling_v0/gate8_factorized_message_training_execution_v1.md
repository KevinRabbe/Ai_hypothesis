# Gate-8 factorized-message training execution v1

## Status

**PRE-EXECUTION IMPLEMENTATION. ONLY SEED 0 MAY BE RUN. SEEDS 1/2, SCIENTIFIC-TEST WORLDS, AND THE 1B REFERENCE REMAIN CLOSED.**

Base: exact qualified v1 factorized training-protocol head:

`a33dc123d090268a531d112251ea3ab53cb50062`

Qualified runtime head:

`333d88ac4fc52f1651741fba224e0b4605feedd3`

Qualified architecture head:

`c3ab64008c816fa1eb6f9d6f8f0a1a99ed195ec8`

Fixed learned-parameter budget:

`19,649`

## Execution components

This slice contains seven files:

1. factorized local-label, batch, loss, and metric mechanics;
2. deterministic development runtime for contract/train/validation worlds;
3. deterministic CUDA training runner;
4. guarded Windows seed-0 wrapper;
5. contract-only CPU-Torch regressions;
6. this execution record;
7. Linux semantic qualification plus Windows pre-execution smoke.

## Local labels

Every public training-tree edge is labelled without reading stored truth.

The root node starts with:

```text
carrier = 0
symbol  = public root symbol
code    = public root symbol
```

For each reachable edge carrying primitive transform `T`:

```text
input carrier, input symbol = decode(source mailbox)
target carrier              = input carrier + 1 mod 16
target symbol               = T(input symbol)
target code                 = encode(target carrier, target symbol)
```

The target code becomes the public-state input of every child edge. The process continues until all population edges have one local example in worker-index order.

The mechanics module rejects `test` and `demonstration` before world validation and contains no world generator, optimizer, checkpoint writer, CUDA policy, truth read, or reference-model path.

## Factorized model surface

A training batch contains only:

```text
inbox code
transform ID
carrier target
symbol target
recomposed message target
```

The recomposed message target must equal:

```text
carrier target * 16 + symbol target
```

The model receives only inbox code, transform ID, and one shared initial hidden state. Its outputs are exactly:

```text
hidden          [edges, 65]
carrier logits  [edges, 16]
symbol logits   [edges, 16]
```

There is no root feature, role flag, activity output, 256-way joint output, or separate answer output.

## Objective

```text
carrier loss = cross entropy(carrier logits, carrier target)
symbol loss  = cross entropy(symbol logits, symbol target)
total loss   = carrier loss + symbol loss
```

Exact-message accuracy requires both predicted components to be correct on the same edge.

## Development runtime

The development runtime is a full-mode semantic mirror of the qualified v1 contract runtime. It admits only:

```text
contract
train
validation
```

It seeds the root mailbox with the public root symbol, executes synchronous rounds, delivers every scheduled factorized message, and reads the terminal answer from the target worker's symbol prediction. It contains no truth read, oracle, primitive-transform application, causal ablation, checkpoint loader, or reference-model path.

Contract-world regressions require prediction and accounting equality with the qualified runtime.

## Frozen CUDA run

The runner:

- admits seed 0 only;
- requires CUDA;
- requires `PYTHONHASHSEED=0`;
- requires `CUBLAS_WORKSPACE_CONFIG=:4096:8`;
- enables deterministic Torch algorithms;
- disables TF32 and autocast;
- constructs exactly 19,649 float32 learned parameters;
- trains exactly 262,144 worlds in 1,024 batches of 256 worlds;
- writes checkpoints at steps 256, 512, 768, and 1,024;
- never stops early;
- preserves all four checkpoints;
- validates every candidate on 3,072 fresh development worlds;
- selects and classifies only after all candidates finish.

## Fresh validation

Validation cache addresses are exactly:

```text
split       = validation
seed        = current training seed
conditions  = six training-regime conditions
indices     = 512..1023 inclusive
worlds      = 512 per condition
```

The runner rejects any validation-index drift from `tuple(range(512, 1024))`.

The v0 development worlds at indices `0..511` are not reused for checkpoint selection.

## Candidate evidence

Every candidate records:

```text
six full-tree target accuracies
full-tree reachability and resource accounting
exact-message accuracy
carrier accuracy
symbol accuracy
carrier and symbol losses
inbox-code coverage
target-code coverage
target-carrier coverage
target-symbol coverage
checkpoint path and SHA-256
```

The runner writes complete step telemetry, all candidate details, the selected checkpoint, environment provenance, and a final result JSON.

The PowerShell wrapper writes a recursive SHA-256 manifest over the complete output root.

## Seed boundary

Both the PowerShell wrapper and direct Python runner admit only seed 0 in this slice. Seeds 1 and 2 require a later separately qualified execution update after seed-0 admission and artifact audit.

## CI boundary

CI may:

- import the qualified architecture and runtime;
- generate public `contract` worlds;
- derive local contract labels;
- execute one CPU optimizer step;
- compare development and qualified runtime semantics;
- parse the runner and wrapper;
- smoke the Windows wrapper before Torch/CUDA/world/optimizer/checkpoint operations.

CI may not invoke the full runner or generate `train`, `validation`, `test`, or `demonstration` worlds.

## Output status

A completed seed-0 run ends with exactly one of:

```text
G8_V1_TRAINING_CHECKPOINT_ADMITTED
G8_V1_TRAINING_CHECKPOINT_NOT_ADMITTED
```

Neither outcome is a scientific Gate-8 capability result. An admitted checkpoint only opens the later frozen scientific-evaluation stage.
