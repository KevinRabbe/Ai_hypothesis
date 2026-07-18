# Step 1 Experiment Protocol v0

## Purpose

This protocol defines how the first minimum-useful-unit size sweep should be executed so that results remain comparable and reproducible.

The objective is to locate a capability-collapse region, not to maximize benchmark scores through per-size hand tuning.

## Software principle

Use the smallest practical implementation stack.

Initial implementation target:

- Python;
- PyTorch for model definition, training, and GPU execution;
- Python standard library for configuration and result serialization where practical;
- optional lightweight system-metrics dependency only if required for CPU/RAM sampling.

Do not introduce distributed-training frameworks, experiment-management platforms, or complex abstraction layers for Step 1 unless measurements demonstrate a concrete need.

## Experiment identity

Every run receives a unique experiment identifier derived from or associated with:

```text
benchmark version
architecture version
configuration name
actual parameter count
random seed
code revision
```

Every saved result must include the Git commit SHA when available.

## Configuration format

Use one human-readable configuration file per architecture size or generate configurations deterministically from one sweep definition.

Preferred initial format: JSON.

Minimum configuration fields:

```text
experiment_name
architecture_version
benchmark_version
seed
model_width
block_count
attention_heads
feed_forward_width
input_length
input_feature_width
batch_size
learning_rate
weight_decay
max_training_steps
early_stopping_patience
training_seed_range
validation_seed_range
test_seed_range
```

The actual trainable parameter count is calculated from the instantiated model and stored in the result. Configuration names must never be treated as the source of truth for parameter count.

## Dataset generation

The benchmark is procedural and deterministic for a fixed benchmark version and seed.

Rules:

1. Training, validation, and test use disjoint generator-seed ranges.
2. Test seeds are frozen before the first evaluated model run.
3. Difficulty distributions are fixed for comparisons within one benchmark version.
4. Any generator bug fix that changes examples requires a new benchmark version.
5. The generator must expose the clean latent truth needed to verify labels and deterministic baselines.

The first implementation may generate samples on demand rather than storing a large static dataset, provided deterministic replay is guaranteed.

## Split policy

Initial policy:

```text
train:       generated from train-only seeds
validation:  generated from validation-only seeds
test:        generated from test-only seeds
```

The exact seed ranges and sample counts are defined in code/config before training starts and included in result metadata.

No test example may be used for model selection or hyperparameter tuning.

## Training fairness

All size variants use the same:

- benchmark version;
- training-data distribution;
- validation distribution;
- test set;
- loss definitions;
- optimizer family;
- evaluation code.

A fixed training-step budget alone is not sufficient because different sizes may converge at different rates.

Therefore:

1. define a generous maximum training budget;
2. use validation-based early stopping;
3. record the step of best validation performance;
4. restore the best validation checkpoint for final test evaluation;
5. record total training steps and wall-clock training time.

If the smallest units fail to improve while larger units converge, additional targeted runs may test whether the failure is optimization-related. These runs must be labeled separately and may not silently replace the original controlled sweep.

## Optimizer policy

Start with one simple optimizer configuration across the coarse sweep.

Initial default candidate:

```text
AdamW
```

Do not tune a separate optimizer configuration for every size during the first sweep.

If optimization becomes unstable at extreme sizes, document the failure first. A second controlled sweep may then test a revised shared optimization policy.

## Random-seed policy

The coarse sweep is intended to locate the collapse region efficiently.

Initial policy:

```text
coarse sweep:          minimum 3 training seeds per size
boundary confirmation: minimum 5 training seeds per size
```

If variance remains high near the boundary, increase repeats there rather than multiplying runs at obviously strong or obviously collapsed sizes.

Report mean, median, standard deviation, and individual run results.

## Parameter-size sweep

Start with coarse targets such as:

```text
10M
3M
1M
300K
100K
30K
10K
3K
```

The architecture configuration generator should choose valid structural dimensions near each target.

Record both:

```text
target parameter count
actual trainable parameter count
```

Do not force an invalid architecture merely to hit an exact round number.

After the first sweep, identify the transition region and generate additional sizes between the nearest clearly useful and clearly collapsed configurations.

## Evaluation order

For each architecture size:

1. instantiate model;
2. count parameters;
3. train with controlled protocol;
4. select best checkpoint by validation performance;
5. evaluate frozen test set;
6. run deterministic baselines on the same test examples;
7. run latency and throughput measurements;
8. save compact result record;
9. save checkpoint reference/checksum if retained.

## Resource measurement

Step 1 resource telemetry should be deliberately low overhead.

Record exactly:

- parameter count;
- serialized checkpoint size;
- training wall-clock duration;
- inference wall-clock duration;
- samples processed;
- batch size;
- device type.

Sample approximately rather than continuously:

- CPU utilization;
- RAM usage;
- GPU utilization where available;
- VRAM usage where available.

A sampling interval around one second is sufficient for long-running training measurements.

For short inference microbenchmarks, rely primarily on synchronized wall-clock timing and framework memory statistics rather than high-frequency system polling.

Do not add fine-grained telemetry that materially changes the workload being measured.

## Inference performance tests

Measure two different questions separately.

### Single-unit latency

How long does one unit take to process one small input or a minimal batch?

### Batched throughput

How many unit-input evaluations per second can the hardware process as batch width increases?

Suggested batch widths where memory allows:

```text
1
4
16
64
256
1024
```

These measurements do not yet represent the final population runtime. They only establish whether the architecture family can use batched hardware efficiently.

## Collapse classification

After the coarse sweep, classify each size region as one of:

```text
CLEARLY_USEFUL
WEAK_BUT_USEFUL
BOUNDARY_UNCERTAIN
MOSTLY_NOISE
DETERMINISTICALLY_DOMINATED
OPTIMIZATION_FAILURE
```

Classification must be based on benchmark results and deterministic baselines, not parameter count alone.

## Step 1 exit package

Step 1 should produce:

1. benchmark generator and frozen benchmark version;
2. deterministic baselines;
3. scalable unit implementation;
4. size-sweep configurations;
5. per-run result records;
6. aggregate comparison table;
7. resource measurements;
8. identified useful/collapse interval;
9. candidate sizes to carry into Step 2;
10. documented failures and unresolved questions.

## No premature conclusions

Step 1 must not claim that the population architecture is superior.

It only determines whether sufficiently small learned units exist to make population-scaling experiments worth running.
