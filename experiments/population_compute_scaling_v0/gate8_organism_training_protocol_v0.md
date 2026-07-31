# Gate-8 organism training protocol v0

## Status

**DATA FROZEN — TRAINING EXECUTION, CHECKPOINT WRITES, SCIENTIFIC TEST WORLDS AND THE 1B REFERENCE REMAIN CLOSED.**

Exact qualified deterministic runtime head:

`1a2be148411bc71ba35fda12b035b724f06ec166`

This protocol freezes how the exact 19,649-parameter organism may be trained and admitted. It does not execute an optimizer or create a checkpoint.

## Scientific purpose

The organism must learn a shared local rule:

```text
received symbol + local primitive transform -> transformed symbol
```

At inference, longer capability emerges only by composing that same learned rule across more workers through the already-qualified synchronous mailbox runtime.

Training is dense and local rather than an opaque terminal-only optimization. Every edge in every training world supplies one transition example. This makes the learned primitive behavior auditable and reduces the risk that a checkpoint succeeds through an accidental global shortcut.

## Complete 8-bit message semantics

A message is one integer from 0 through 255:

```text
bits 7..4 = carrier phase, 0..15
bits 3..0 = current symbol, 0..15
code      = carrier * 16 + symbol
```

For a non-root edge:

```text
input carrier = inbox_code // 16
input symbol  = inbox_code % 16
output symbol = local primitive transform(input symbol)
output carrier= (input carrier + 1) mod 16
output code   = output carrier * 16 + output symbol
```

For a root-source edge:

```text
inbox code    = frozen root seed code 0
input symbol  = public root symbol
input carrier = public root symbol
```

The carrier is deterministic phase redundancy. It contributes no answer information and never depends on path truth, population, depth or test performance. Its purpose is to exercise the complete 256-code embedding and output space rather than retaining 240 structurally dead message codes.

Because every primitive transform is a symbol bijection and carrier increment is a carrier bijection, each primitive induces a bijection over all 256 message codes.

## Edge-level labels

Training labels are derived from the synthetic training world's public rooted tree and frozen primitive transformations:

- `inbox_code`: exact code carried by the source node, or root seed code 0;
- `message_target`: exact transformed output code;
- `answer_target`: low four bits of the output code;
- `activity_target`: one.

Labels are generated for every edge, including distractor branches. The model therefore learns general transformation transport, not path membership. It receives no truth-path indicator.

The three frozen losses are:

```text
1.0 * message cross-entropy over 256 codes
1.0 * answer cross-entropy over 16 symbols
0.1 * activity binary cross-entropy, target one
```

## Frozen training schedule

Three independent runs:

```text
training seeds = 0, 1, 2
worlds/seed    = 262,144
world batch    = 256 worlds
optimizer steps= 1,024
```

Training conditions, in exact round-robin order:

```text
(32,4)
(64,4)
(64,8)
(128,4)
(128,8)
(128,16)
```

Global world index `i` selects:

```text
condition       = conditions[i mod 6]
condition index = floor(i / 6)
```

Exact worlds per condition:

| Population | Depth | Worlds per seed |
| ---: | ---: | ---: |
| 32 | 4 | 43,691 |
| 64 | 4 | 43,691 |
| 64 | 8 | 43,691 |
| 128 | 4 | 43,691 |
| 128 | 8 | 43,690 |
| 128 | 16 | 43,690 |

Every world is generated on demand from the `train` namespace of its training seed. No training world is reused within a seed.

## Frozen optimizer

```text
optimizer                 = AdamW
initial learning rate     = 3.0e-3
minimum learning rate     = 3.0e-5
warmup                    = 64 optimizer steps, linear
post-warmup schedule      = cosine decay through step 1,024
betas                      = (0.9, 0.95)
epsilon                    = 1.0e-8
weight decay               = 1.0e-4
global gradient clip       = 1.0
parameter and compute dtype= float32
autocast                   = disabled
TF32                       = disabled
deterministic algorithms   = enabled
```

The runner must seed Python, Torch CPU and every CUDA device from the run seed. CUDA deterministic workspace configuration is mandatory. No adaptive batch size, gradient accumulation, loss reweighting, curriculum, rescue phase or early stopping is permitted.

## Frozen development validation

Checkpoint selection uses only:

```text
split             = validation
seed              = matching training seed
conditions        = the same six training-regime conditions
world indices     = 0..511 per condition
worlds/condition  = 512
runtime           = qualified deterministic full runtime
```

Depths 32, 64 and 128 remain completely unseen during training and checkpoint selection. The scientific `test` namespace remains closed.

The runner must also report edge-level message accuracy, activity accuracy, inbox-code coverage and target-code coverage over the same validation material.

## Candidate checkpoints

Exactly four candidates are written:

```text
step 256
step 512
step 768
step 1,024
```

All four are trained regardless of intermediate performance. The selected candidate maximizes, in order:

1. mean target accuracy across the six validation conditions;
2. minimum target accuracy across those conditions;
3. message accuracy;
4. activity accuracy;
5. negative validation loss;
6. earliest step.

No scientific-test metric may influence selection.

## Training admission gate

The selected checkpoint is admitted only if all conditions hold:

```text
minimum target accuracy in every validation condition >= 0.99
message accuracy                                  >= 0.995
activity accuracy                                 >= 0.99
observed inbox-code coverage                      = 256 / 256
observed target-code coverage                     = 256 / 256
```

Frozen outcomes:

```text
G8_TRAINING_CHECKPOINT_ADMITTED
G8_TRAINING_CHECKPOINT_NOT_ADMITTED
```

A non-admitted run does not open scientific execution. The result is recorded rather than rescued post hoc.

## Required training artifact

A later execution stage must bind at minimum:

- exact source head and run seed;
- Python, Torch, CUDA, GPU and driver identity;
- deterministic settings;
- every optimizer hyperparameter;
- every batch/world address;
- loss and gradient telemetry per optimizer step;
- all four candidate state dictionaries and SHA-256 values;
- complete validation rows and code coverage;
- deterministic selected-step decision;
- exact learned-parameter count 19,649;
- explicit confirmation that no test or reference-model path was touched.

## Closed boundaries

This protocol contains no:

- Torch import;
- world generator invocation;
- optimizer object or optimizer step;
- backward pass;
- checkpoint write;
- train or validation execution;
- scientific-test world;
- 1B tokenizer, model weight or inference path.

The next slice may implement the training execution path exactly as frozen here. Any change to message semantics, supervision, optimizer, schedule, validation or admission thresholds requires a new pre-exposure protocol version.
