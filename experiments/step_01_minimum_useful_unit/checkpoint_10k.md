# Step 1 Checkpoint — ~10K Parameters

## Purpose

Test whether the Step 1 neural unit remains useful after another approximately 10× parameter reduction from the ~100K checkpoint.

This checkpoint is intentionally aggressive because the ~100K model retained nearly all of the 10M reference capability on the first seed.

## Architecture

```text
model width:         32
blocks:               1
attention heads:      4
feed-forward width:  64
sequence length:     32
feature width:       16
```

Expected trainable parameter count with the current implementation:

```text
10,572
```

The training runner records the instantiated model's actual parameter count and that recorded value remains the source of truth.

## Controlled comparison

Keep unchanged from the 10M, 1M, and 100K runs:

- benchmark version;
- train/validation/test sample counts;
- seed policy;
- optimizer family;
- learning rate;
- weight decay;
- batch size;
- maximum training steps;
- validation interval;
- early stopping policy;
- uncertainty threshold;
- evaluation and deterministic-baseline pipeline.

## Current first-seed curve

```text
10,148,108 params -> 94.37% test accuracy
   999,436 params -> 94.53% test accuracy
    99,884 params -> 94.025% test accuracy
    10,572 params -> unknown
```

The differences among the first three runs are single-seed observations and must not yet be interpreted as statistically significant ranking differences.

## Interpretation targets

### If ~10K remains close to ~94%

The minimum useful-unit boundary is likely below 10K for this benchmark. Continue downward if mechanically valid configurations exist, or begin a denser low-parameter architecture search.

### If ~10K drops but remains clearly useful

The interval between ~10K and ~100K becomes the first likely capacity-transition region. Add intermediate sizes such as ~30K before making conclusions.

### If ~10K collapses

The useful/noise boundary is bracketed between ~10K and ~100K. Run intermediate checkpoints and repeat boundary configurations across additional seeds.

## Run

```powershell
cd F:\AI_hypothesis
git pull
.\.venv\Scripts\Activate.ps1
python -m ai_hypothesis.step01.train_reference --config configs/step01/checkpoint_10k.json --device cuda
```
