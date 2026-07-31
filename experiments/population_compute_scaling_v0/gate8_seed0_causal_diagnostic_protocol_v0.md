# Gate-8 seed-0 causal diagnostic protocol v0

## Status

**Preregistered and execution-closed.**

This protocol starts from the fully qualified seed-0 non-admission result at
`70e7e40149f9259d36b0e37ab17fc8c30370201e`. It does not alter or retroactively
reinterpret that result. Seeds 1 and 2, scientific-test worlds, the Gemma model,
and reference inference remain closed.

## Exact source artifact

```text
checkpoint SHA-256
4aca6bfde7fa82cd2c1fec3613c4cc59303788616f352e8d419c90662d7b9a1b

result JSON SHA-256
5f477022ac45a80d8b05b112b3485e4112519ddef78e0bb23990a090e0cc92e2

source manifest SHA-256
bb814b9ebb5116f0a13ff2ce130c5ad8e32ed4bd80453ddc167143b6cbf0bb8d
```

The model remains the exact 19,649-parameter shared Gate-8 organism.

## Question

The seed-0 checkpoint learned activity and carrier transport nearly exactly but
remained below admission on message-symbol accuracy, answer readout, and
complete-tree target accuracy. The diagnostic separates five claims rather than
forcing one exclusive explanation:

```text
activity_gate_material
answer_head_material
frozen_core_linearly_sufficient
continued_optimization_effective
core_interference_persists
```

Each finding is an independently classified boolean.

## Frozen validation surface

Every evaluation uses the unchanged seed-0 `validation` namespace:

```text
(P=32,  D=4)
(P=64,  D=4)
(P=64,  D=8)
(P=128, D=4)
(P=128, D=8)
(P=128, D=16)
```

There are exactly 512 worlds per condition, 3,072 total. No depth 32, 64, or
128 world is admitted.

## Probe 1 — runtime interventions

The original checkpoint is evaluated without parameter updates in four modes:

1. `baseline` — learned activity gate and answer head;
2. `forced_active` — every scheduled worker emits its message;
3. `message_low4_decode` — terminal answer is decoded from the low four bits of
   the terminal message argmax while retaining learned activity;
4. `forced_active_message_low4_decode` — both interventions together.

Topology, synchronous rounds, worker order, checkpoint parameters, validation
worlds, and message argmax remain unchanged.

A runtime intervention is material when mean target accuracy rises by at least
`0.02` absolute over baseline on the same 3,072 worlds.

## Probe 2 — frozen-core head retraining

All parameters except these exact prefixes remain frozen:

```text
message_head.
activity_head.
answer_head.
```

Training uses 256 batches of 256 fresh seed-0 training worlds:

```text
world indices [262,144, 327,680)
```

Optimizer:

```text
AdamW
learning rate 1e-3 constant
betas (0.9, 0.95)
epsilon 1e-8
weight decay 0
gradient clip 1.0
float32, deterministic CUDA, no TF32
```

Checkpoints are fixed at steps 64, 128, 192, and 256. The final checkpoint is
classified as `frozen_core_linearly_sufficient` only when all are true:

```text
message accuracy          >= 0.995
answer accuracy           >= 0.99
activity accuracy         >= 0.999
message root invariance   >= 0.99
answer root invariance    >= 0.99
```

This asks whether the frozen hidden representation already contains a nearly
exact linearly decodable solution.

## Probe 3 — full-model continuation

A separate copy of the original selected checkpoint resumes with every one of
the 19,649 parameters trainable. It never starts from the head-only result.

Training uses 512 batches of 256 additional fresh seed-0 training worlds:

```text
world indices [327,680, 458,752)
```

Optimizer:

```text
AdamW
learning rate cosine 3e-4 -> 3e-5, no warmup
betas (0.9, 0.95)
epsilon 1e-8
weight decay 1e-4
gradient clip 1.0
float32, deterministic CUDA, no TF32
```

Checkpoints are fixed at steps 128, 256, 384, and 512.

`continued_optimization_effective` is true only when the final checkpoint gains
both:

```text
message accuracy      >= baseline + 0.03
mean target accuracy  >= baseline + 0.10
```

The fixed baselines are `0.9167085535386029` and `0.4036458333333333`.

## Probe 4 — irrelevant-root invariance

For every non-root finite local transition, the root-symbol input is swept over
all 16 values while every relevant input is held fixed. Message and answer
predictions are scored invariant only when all 16 root-symbol variants produce
the same argmax.

`core_interference_persists` is true when either final full-resume invariance is
below `0.95`.

## Interpretation

The findings are intentionally non-exclusive:

- activity and answer-readout interventions may both help;
- frozen heads may be insufficient while full-model continuation remains useful;
- continued optimization may improve accuracy while irrelevant-feature
  interference persists;
- no result from this diagnostic can admit the original checkpoint.

A later architecture or training amendment must cite the complete finding vector
and receive a separate preregistration before execution.

## Closed boundaries

This protocol adds no Torch import, checkpoint loader, optimizer execution,
world generator, scientific-test access, seed-1/2 training, tokenizer/model load,
or reference inference. Those capabilities require a separately qualified
execution slice.
