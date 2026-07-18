# Sequential 25K-75K Sweet-Spot Sweep

This sweep explores the region between the 10K and 100K checkpoints while treating training time as part of the optimization problem.

## Execution order

The experiments run strictly one after another:

1. ~25K parameters: 25,356 actual trainable parameters
2. ~50K parameters: 50,268 actual trainable parameters
3. ~75K parameters: 74,836 actual trainable parameters

No experiments run in parallel. The next process starts only when the previous process exits successfully. If one run fails, the sweep stops immediately.

All three configurations use two Transformer encoder blocks to reduce architecture-shape confounding inside this mid-size sweep. They retain the same benchmark, seed, optimizer family, learning rate, batch size, dataset sizes, evaluation interval, and uncertainty threshold.

Each experiment has a maximum budget of 15,000 training steps. Validation is recorded every 250 steps, and the best validation checkpoint is restored for test evaluation.

## Run

From the repository root:

```powershell
.\.venv\Scripts\Activate.ps1
python -m ai_hypothesis.step01.run_sequential_sweep --device cuda
```

The default order is:

```text
configs/step01/checkpoint_25k_extended_15k.json
configs/step01/checkpoint_50k_extended_15k.json
configs/step01/checkpoint_75k_extended_15k.json
```

Completed result files are skipped by default, making the command safe to restart after an interruption. Use `--rerun-completed` only when an intentional repeat is required.

## Combined summary

After all runs complete, the runner writes:

```text
results/step01/sweep_25k_50k_75k/summary.json
```

For each model, the summary records:

- actual parameter count;
- best validation step;
- best validation score;
- frozen test accuracy;
- training wall-clock duration;
- first validation checkpoint reaching 90%;
- first validation checkpoint reaching 92%;
- first validation checkpoint reaching 93%;
- first validation checkpoint reaching the 100K model's current best-validation target of 93.82%.

The purpose is to estimate the Pareto region between parameter count, training time, and achieved capability. A model is not preferred merely because it is smaller or because it trains faster; the useful region is where further parameter reduction begins to require disproportionate training compute or causes an irrecoverable capability loss.
