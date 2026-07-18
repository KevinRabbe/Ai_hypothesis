# Step 1 Multi-Seed Confirmation Sweep

## Purpose

Confirm whether the apparent 25K, 50K, 75K, and 100K worker-size tradeoffs are reproducible across random seeds before freezing candidate worker architectures or moving into population experiments.

The current single-seed results suggest:

- ~25K may be the smallest candidate that still reaches the benchmark's full-quality region with enough training.
- ~50K may provide the strongest size-versus-time-to-quality tradeoff.
- ~75K may have a slightly higher observed capability ceiling, but the difference is too small to interpret from one seed.
- ~100K is retained as a larger reference and is given the same 15,000-step ceiling for a fairer convergence comparison.

These are hypotheses to test, not conclusions.

## Candidate sizes

- 25,356 parameters
- 50,268 parameters
- 74,836 parameters
- 99,884 parameters

Each candidate remains a homogeneous worker architecture. This experiment does not mix worker sizes inside one model.

## Seeds

Default seeds:

- 1
- 2
- 3
- 4
- 5

## Execution

All runs execute strictly sequentially. Only one training subprocess is active at a time.

The default order is interleaved by seed:

1. Seed 1: 25K, 50K, 75K, 100K
2. Seed 2: 25K, 50K, 75K, 100K
3. Seed 3: 25K, 50K, 75K, 100K
4. Seed 4: 25K, 50K, 75K, 100K
5. Seed 5: 25K, 50K, 75K, 100K

Completed result files are reused by default. Pass `--rerun-completed` only when an intentional full rerun is required.

## Command

```powershell
python -m ai_hypothesis.step01.run_multiseed_confirmation --device cuda
```

## Training protocol

All four sizes use the same Step 1 benchmark and training protocol, with:

- 100,000 training samples
- 20,000 validation samples
- 20,000 test samples
- batch size 256
- AdamW
- learning rate 0.0003
- weight decay 0.01
- maximum 15,000 training steps
- validation every 250 steps
- early-stopping patience 60 evaluations
- gradient clipping at 1.0
- uncertainty threshold 0.5

Only architecture size and random seed vary.

## Quality thresholds

The confirmation summary records the first validation step at or above:

- 90%
- 92%
- 93%
- 93.82%, the original 100K best-validation reference target

## Summary

The combined summary is written to:

```text
results/step01/confirmation_25k_50k_75k_100k/summary.json
```

For each size it reports aggregate statistics across completed seeds for:

- best validation score
- test accuracy
- best training step
- training duration
- first-step-to-quality thresholds

The aggregate statistics include mean, sample standard deviation, minimum, and maximum. Threshold summaries also record how many seeds reached each quality target.

## Decision use

The confirmation sweep should be used to decide whether the observed size tiers are reproducible and which worker sizes deserve to enter fixed-total-budget population experiments.

No candidate should be declared superior solely from a single seed or a sub-percentage-point difference without considering variance, convergence behavior, and inference cost.
