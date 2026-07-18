# Step 1 — 10M Reference Run

## Purpose

This is the first neural experiment in Step 1.

It answers one question before any shrinking begins:

> Can the selected architecture family learn the five Step 1 benchmark families reliably under a comfortably capable parameter budget?

The reference configuration contains approximately **10.15 million trainable parameters**.

This run is not intended to identify the minimum useful unit and is not evidence that 10M is an optimal size.

## Architecture

Frozen configuration:

```text
input:                32 x 16 feature sequence
model width:          256
residual blocks:      11
attention heads:      8
feed-forward width:   1280
output:                11 answerable-label logits + 1 uncertainty logit
trainable parameters: approximately 10.15M
```

Configuration file:

```text
configs/step01/reference_10m.json
```

## Before the real run

Run the existing benchmark tests:

```bash
python -m unittest discover -s tests -v
```

Then run the neural pipeline smoke test:

```bash
python -m ai_hypothesis.step01.train_reference --smoke-test
```

The smoke test uses a much smaller neural configuration and only a few training steps. It verifies the data adapter, forward pass, loss, optimizer, checkpointing, evaluation, deterministic-baseline comparison, and result writing. It is not a research result.

## Full reference run

From the repository root:

```bash
python -m ai_hypothesis.step01.train_reference --device cuda
```

The default full configuration uses:

```text
training samples:       100,000
validation samples:      20,000
test samples:            20,000
batch size:                 256
maximum training steps:    5,000
validation interval:         250 steps
early-stopping patience:       8 validation checks
optimizer:                 AdamW
learning rate:             3e-4
weight decay:               0.01
```

These are initial controlled reference settings. If the 10M model fails to learn, the first response is to diagnose architecture, benchmark, optimization, or training-budget failure rather than immediately concluding that the hypothesis failed.

## Output

Default output directory:

```text
results/step01/reference_10m/seed_1/
```

Expected files:

```text
best.pt
result.json
```

`result.json` records:

- actual trainable parameter count;
- best validation step;
- validation history;
- overall test accuracy;
- macro task accuracy;
- accuracy by task;
- accuracy by difficulty;
- accuracy by task/difficulty pair;
- uncertainty precision and recall;
- invalid-output rate;
- deterministic baseline results;
- inference latency and batched throughput;
- training duration;
- checkpoint size;
- Git revision when available.

## Reference success gate

Do not begin the minimum-size search until the reference run demonstrates that the architecture can learn the intended benchmark.

The first analysis should check:

1. whether all five task families are learned rather than only the easiest tasks;
2. whether hard cases improve over the applicable deterministic baselines;
3. whether ambiguous cases produce useful abstention behavior;
4. whether invalid task-specific outputs are rare;
5. whether any task family remains near chance despite the 10M capacity.

If the reference is healthy, the next experiment should choose a substantially smaller but still informative size rather than jumping directly to the expected-collapse extreme.
