# Step 1 Checkpoint — ~100K Parameters

## Purpose

This checkpoint tests whether the Step 1 learned processing unit retains most of its benchmark capability after another approximately 10× reduction from the ~1M checkpoint.

The first two measured points were:

```text
10,148,108 parameters -> 94.37% test accuracy
   999,436 parameters -> 94.53% test accuracy
```

The ~1M result showed no measurable capability loss relative to the 10M reference in the first seed-1 run. The next useful experiment is therefore an aggressive reduction toward ~100K rather than spending compute on intermediate sizes first.

## Architecture

```text
d_model:             64
block_count:          2
attention_heads:       4
feed_forward_width:  240
sequence_length:      32
feature_width:        16
```

Expected trainable parameter count with the current Step 1 implementation:

```text
99,884
```

The training runner will record the instantiated parameter count as the source of truth.

## Controlled comparison

Keep the following unchanged from the 10M and 1M runs:

- benchmark version;
- architecture family;
- train/validation/test distributions;
- dataset counts;
- optimizer family;
- learning rate;
- weight decay;
- batch size;
- maximum training steps;
- validation interval;
- early stopping policy;
- uncertainty threshold;
- evaluation and deterministic-baseline pipeline.

Only the documented structural dimensions of the neural unit change.

## Research question

> Does an approximately 100K-parameter unit still retain substantial useful learned capability on the Step 1 benchmark, or does the first major capability decline appear between ~100K and ~1M parameters?

## Interpretation

Possible outcomes:

- **Near 1M performance:** the collapse boundary is still below ~100K; continue shrinking aggressively.
- **Moderate degradation but clearly useful:** ~100K may be near the beginning of the useful/collapse transition; test denser sizes around this region later.
- **Sharp collapse:** the informative boundary lies between ~100K and ~1M; bracket that interval.
- **Optimization failure:** distinguish training instability from true capacity collapse before interpreting the size boundary.

A single seed is sufficient for this coarse search checkpoint. Additional seeds should be concentrated around the eventual boundary rather than spent on sizes that are obviously capable or obviously collapsed.

## Run

```powershell
cd F:\AI_hypothesis
git pull
.\.venv\Scripts\Activate.ps1
python -m ai_hypothesis.step01.train_reference --config configs/step01/checkpoint_100k.json --device cuda
```

Expected result directory:

```text
results/step01/checkpoint_100k/seed_1/
```
