# Gate-8 factorized-message training protocol v1

## Status

**DATA-FROZEN PRE-EXECUTION PROTOCOL. TRAINING EXECUTION, CHECKPOINT WRITES, SEEDS 1/2, SCIENTIFIC-TEST WORLDS, AND THE 1B REFERENCE REMAIN CLOSED.**

Base: exact qualified v1 deterministic runtime head:

`333d88ac4fc52f1651741fba224e0b4605feedd3`

Qualified v1 architecture head:

`c3ab64008c816fa1eb6f9d6f8f0a1a99ed195ec8`

Fixed learned-parameter budget:

`19,649`

## Purpose

The seed-0 v0 diagnostic established that the organism communicated the answer more accurately than its independent answer head reported it, while carrier transport was nearly solved and the monolithic 256-way objective left material symbol errors.

V1 therefore changes only the supervision required by the qualified repaired architecture. It does not change the benchmark, training-condition matrix, training-world count, optimizer-step count, checkpoint cadence, or learned-parameter budget.

## Controlled comparison

The following values remain identical to v0:

```text
training seeds             = 0, 1, 2
training worlds per seed   = 262,144
worlds per optimizer batch = 256
optimizer steps            = 1,024
checkpoint steps           = 256, 512, 768, 1,024
training conditions        = (32,4), (64,4), (64,8),
                             (128,4), (128,8), (128,16)
optimizer                  = AdamW
peak learning rate         = 3e-3
minimum learning rate      = 3e-5
warmup                     = 64 steps
betas                      = (0.9, 0.95)
weight decay               = 1e-4
gradient clip norm         = 1.0
parameter dtype            = float32
autocast                   = disabled
TF32                       = disabled
deterministic algorithms   = required
```

The exact round-robin training allocation remains:

```text
(32,4)   43,691 worlds
(64,4)   43,691 worlds
(64,8)   43,691 worlds
(128,4)  43,691 worlds
(128,8)  43,690 worlds
(128,16) 43,690 worlds
----------------------
          262,144 worlds
```

Reusing the same training schedule isolates the architecture/objective repair. Training data are development data and do not provide an unbiased scientific result.

## Fresh validation boundary

The v0 run and causal diagnosis examined validation indices `0..511` for each of the six training conditions. V1 must not reuse those worlds for checkpoint selection.

V1 validation is frozen to:

```text
namespace          = validation
world indices      = 512..1023 inclusive
worlds/condition   = 512
conditions         = six training-regime conditions
validation worlds  = 3,072 per checkpoint
```

No depth 32, 64, or 128 world may enter training or checkpoint selection. Scientific-test worlds remain unopened.

## Exact local target

Every training edge receives one eight-bit inbox code:

```text
high four bits = carrier
low four bits  = current symbol
```

For local primitive transform `T`, the exact target is:

```text
target carrier = (input carrier + 1) mod 16
target symbol  = T(input symbol)
target code    = (target carrier << 4) | target symbol
```

The root mailbox is already the qualified v1 runtime message:

```text
carrier = 0
symbol  = public root symbol
code    = public root symbol
```

There is no root-specific local-label exception.

Each of the eight canonical primitive transforms induces a bijection over all 256 input codes. Across the complete transition algebra there are exactly 2,048 `(transform, inbox code)` cases.

## Frozen objective

The v1 model exposes two heads and receives two equal-weight losses:

```text
carrier cross-entropy weight = 1.0
symbol cross-entropy weight  = 1.0
```

The following v0 losses do not exist:

```text
joint 256-way message cross-entropy
independent answer cross-entropy
activity binary cross-entropy
```

The terminal answer is not separately supervised because the qualified runtime defines it as the target worker's symbol-head argmax—the same symbol used in the emitted message.

## Validation metrics

Every candidate checkpoint is evaluated on complete validation trees through the qualified deterministic v1 runtime and records:

```text
six condition target accuracies
mean target accuracy
minimum target accuracy
exact message accuracy
carrier accuracy
symbol accuracy
validation loss
inbox-code coverage
target-code coverage
target-carrier coverage
target-symbol coverage
```

Exact message accuracy requires both predicted components to be correct for the same local example.

## Candidate selection

All four checkpoints must be written and validated. No early stopping is permitted.

Selection is the lexicographic maximum of:

```text
1. mean target accuracy
2. minimum condition target accuracy
3. exact message accuracy
4. symbol accuracy
5. carrier accuracy
6. negative validation loss
7. negative step number
```

The final field selects the earliest checkpoint on an exact tie.

## Admission

The selected checkpoint is admitted only if all guards pass inclusively:

```text
every condition target accuracy >= 0.990
exact message accuracy          >= 0.995
carrier accuracy                >= 0.995
symbol accuracy                 >= 0.995
inbox-code coverage             == 256
target-code coverage            == 256
target-carrier coverage         == 16
target-symbol coverage          == 16
```

Outcomes are exactly:

```text
G8_V1_TRAINING_CHECKPOINT_ADMITTED
G8_V1_TRAINING_CHECKPOINT_NOT_ADMITTED
```

Admission is a development checkpoint decision. It is not a Gate-8 scientific capability result.

## Seed order

Seed 0 is executed first. Seeds 1 and 2 remain blocked until seed 0 is admitted and its artifacts are audited. A non-admitted seed 0 stops the v1 training sequence and requires a separately preregistered diagnosis.

## Closed boundaries

This protocol contains no Torch import, model construction, world generation, optimizer execution, backward pass, checkpoint read/write, CUDA operation, tokenizer/model loading, scientific-test generation, or reference inference.

The next stage after qualification is a separate guarded v1 training-execution slice. CI may exercise protocol mechanics and contract-only smoke data, but may not generate a `train`, `validation`, or `test` world.
