# Gate-9 contextual worker training protocol v0

## Status

**DATA-FROZEN THREE-SEED TRAINING AND CHECKPOINT-ADMISSION PROTOCOL — NO
OPERATOR GENERATOR, OPTIMIZER INSTANCE, TRAINING EXECUTION, CHECKPOINT,
SCIENTIFIC TEST WORLD OR RESULT IS ADMITTED.**

Qualified architecture head:

```text
c689cc3f38f6f6f642916ee1a702d7de7bd0e43b
```

This protocol freezes all training choices before any Gate-9 checkpoint exists.
No validation-dependent hyperparameter change, best-step selection, seed
selection or retraining is permitted under v0.

## Exact data allocation

Every checkpoint seed consumes every training operator exactly once:

```text
training operator range start  0
unique operators               262,144
episodes per seed              262,144
query episodes per operator    1
batch size                     512
optimizer steps                512
```

The three checkpoint seeds use initialization seeds:

```text
900900
900901
900902
```

Each seed has its own invertible affine permutation of the complete training
operator range and its own balanced query mapping. The three order
multipliers are odd and therefore invertible modulo `2^18`.

Every query is chosen from the 247 bytes outside the nine support inputs.
Across a complete seed run, query counts differ by at most one.

## Exact validation allocation

All checkpoints evaluate the same disjoint validation set:

```text
validation range start  2^32
unique operators        32,768
episodes                32,768
batch size              512
batches                  64
```

The validation order is one fixed invertible affine permutation. Validation
queries use one fixed balanced mapping over the same 247 novel bytes.

No local-science operator beginning at `2^40`, graph-science operator beginning
at `2^48`, or scientific graph assignment key may be accessed during training
or validation.

## Optimizer and numerical contract

```text
optimizer             AdamW
base learning rate    1.0e-3
minimum learning rate 1.0e-4
warmup                16 steps, linear
remaining schedule    cosine decay through step 512
betas                  (0.9, 0.95)
epsilon                1.0e-8
weight decay           1.0e-4
global gradient clip   1.0
loss                   BCE-with-logits, mean over batch and eight bits
precision              float32
AMP                    disabled
TF32                   disabled
compile                disabled
deterministic mode     required
data-loader workers    0
```

The exact execution runtime is frozen to Python `3.11.9`, Torch
`2.9.1+cu130`, NumPy `2.3.5` and CUDA.

## Fixed checkpoint rule

Only the step-512 model state is eligible. There is no best-validation
checkpoint selection and no earlier rescue checkpoint.

The selected checkpoint contains exactly the 17 architecture state tensors,
all finite float32 values, and exactly 19,649 learned parameters. Optimizer
state is not included in the selected checkpoint.

Required metadata includes:

```text
experiment and architecture identity
training-protocol identity
checkpoint seed and initialization seed
step 512
262,144 training episodes
19,649 learned parameters
17 state tensors
exact state dictionary
```

Every checkpoint receives an immutable SHA-256 identity. The three identities
must be distinct.

## Validation modes and admission

Every final checkpoint evaluates:

### `full`

Nine public support pairs plus the novel query through the qualified worker.

### `shuffled_context`

A deterministic derangement assigns another validation operator's complete,
valid support set to the current query and answer. Query, target and marginal
support distributions remain fixed.

### `query_only`

The architecture's qualified `forward_query_only` path supplies no support rows.

### `oracle`

Exact public-support reconstruction. Accuracy must be exactly `1.0` or the
validation artifact is invalid.

One seed passes admission only when:

```text
full exact-byte accuracy       >= 0.995
full bit accuracy              >= 0.999
full - shuffled-context exact  >  0.50
full - query-only exact        >  0.50
oracle accuracy                == 1.0
```

All three ordered seeds must pass. Otherwise the frozen outcome is:

```text
G9_CONTEXTUAL_CHECKPOINT_ADMISSION_FAILED
```

If all three pass:

```text
G9_CONTEXTUAL_CHECKPOINTS_ADMITTED
```

A failed seed cannot be discarded, repeated with another initialization, given
more episodes or rescued with altered hyperparameters under this protocol.

## Required execution evidence

The future training artifact must preserve per seed:

- exact software, CUDA and determinism state;
- complete episode-order and query-allocation hashes;
- one row per optimizer step with loss, learning rate and gradient norm;
- exact training operator coverage and uniqueness;
- exact validation predictions for every mode;
- byte and bit accuracies;
- final checkpoint tensor names, shapes, dtypes and finiteness;
- checkpoint SHA-256;
- explicit zero overlap with validation and scientific ranges;
- no scientific assignment key access;
- final immutable admission outcome.

The independent auditor must rebuild all allocation, query, schedule,
validation and admission calculations without importing the trainer.

## Closed boundary

This branch contains only standard-library constants, deterministic allocation
functions, the learning-rate formula, evidence schemas, admission classifier,
synthetic tests, documentation and protocol-only CI.

It imports no Torch or NumPy, constructs no operator, instantiates no optimizer,
executes no training step, serializes no checkpoint, reads no artifact and
opens no scientific namespace.

## Next admission boundary

A separate stacked execution slice may implement the exact trainer, validation
controls, transactional progress evidence and fixed-final checkpoint writer.
CI may smoke and structurally qualify that code, but it may not run the full
three-seed training or emit scientific checkpoint evidence.
