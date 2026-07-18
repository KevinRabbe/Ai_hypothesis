# Step 1 — 1M Compression Checkpoint

## Purpose

Test whether the useful learned behavior demonstrated by the approximately 10M reference unit survives an approximately 10x reduction in trainable parameters.

This is the first compression checkpoint after the successful 10M reference run.

## Architecture

```text
d_model:             128
block_count:           5
attention_heads:       4
feed_forward_width:  512
sequence_length:      32
feature_width:        16
```

Expected trainable parameter count with the current implementation:

```text
999,436
```

The benchmark version, data counts, optimizer family, learning rate, weight decay, evaluation interval, early stopping policy, and uncertainty threshold remain unchanged from the 10M reference experiment.

## Comparison target

Reference result from the first 10M run:

```text
parameter_count:          10,148,108
best_step:                4,750
test_accuracy:            0.9437
test_macro_task_accuracy: 0.9437
```

The 1M experiment should answer:

1. How much aggregate accuracy is retained after approximately 10x compression?
2. Which task families degrade first?
3. Does uncertainty handling degrade before ordinary classification?
4. Does the smaller unit still provide value relative to deterministic baselines?
5. How much inference throughput and resource efficiency improve?

## Run

```powershell
cd F:\AI_hypothesis
.\.venv\Scripts\Activate.ps1
python -m ai_hypothesis.step01.train_reference --config configs/step01/checkpoint_1m.json --device cuda
```

## Interpretation

- If performance remains close to the 10M reference, the next checkpoint should move substantially lower, likely toward approximately 100K parameters.
- If performance declines but remains clearly useful, the next checkpoint should be chosen to bracket the developing collapse region.
- If performance collapses unexpectedly, the interval between approximately 1M and 10M should be investigated before moving lower.

No conclusion about population scaling should be drawn from this experiment alone.
