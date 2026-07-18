# Step 1 — Minimum Useful Neural Unit

## Objective

Find the smallest version of one identical neural processing architecture that still performs a useful local learned transformation.

This step does **not** attempt to build the final population model. It establishes the lower useful size boundary that later population experiments depend on.

## Core question

> As the same architecture family is progressively reduced in parameter count, where does useful learned signal collapse into mostly noise or become less valuable than deterministic logic?

## Experimental rule

Change unit size while holding the rest of the experiment as controlled as reasonably possible:

- same task definitions;
- same input/output contract;
- same training-data distribution;
- same evaluation split;
- same optimizer family and training methodology unless scaling requires a documented change;
- same measurement pipeline.

## Candidate size sweep

Initial coarse checkpoints:

- 10M;
- 3M;
- 1M;
- 300K;
- 100K;
- 30K;
- 10K;
- 3K.

The purpose is to bracket the collapse boundary, not to force every size to work.

Once the boundary is roughly located, run a denser sweep around it.

## Step 1 design decisions

The v0 experiment now defines:

1. **Architecture family** — a compact residual sequence encoder / tiny Transformer-style encoder operating on standardized feature sequences rather than raw language tokens.
2. **Input contract** — up to 32 elements with 16 floating-point features per element plus a validity mask.
3. **Output contract** — compact structured logits with explicit uncertainty/abstention support; no free-form language generation.
4. **Benchmark families** — noisy local pattern detection, fuzzy change detection, local conflict detection, local relationship extraction, and sparse relevant-signal detection.
5. **Dataset strategy** — deterministic procedural generation with hidden latent truth and disjoint train/validation/test seed ranges.
6. **Training policy** — controlled size sweep with shared objectives, validation-based early stopping, and best-checkpoint evaluation.
7. **Seed policy** — at least three training seeds in the coarse sweep and at least five around the suspected boundary.
8. **Performance policy** — measure end-to-end latency, batch throughput, memory use, and low-overhead system resource samples.

Detailed specifications:

- [`architecture_v0.md`](architecture_v0.md)
- [`protocol_v0.md`](protocol_v0.md)
- [`../../benchmarks/step_01_benchmark_v0.md`](../../benchmarks/step_01_benchmark_v0.md)

## Why the worker does not process raw text in Step 1

Vocabulary embeddings and natural-language decoding can require more parameters than the tiny worker sizes being investigated.

Step 1 therefore isolates the learned processing unit itself. Future shared front-ends may convert text, images, audio, or other modalities into the compact representation consumed by the population.

This is analogous to supplying standardized raw materials to many small factories rather than rebuilding the entire supply chain inside every worker.

## Deterministic baseline rule

Every benchmark task must document whether a deterministic algorithm can solve it.

If deterministic logic solves a task equally well or better at lower cost, that task is not evidence that a neural unit is necessary. It may remain as a control.

The important boundary is the smallest unit that still adds useful learned behavior where rigid deterministic methods are insufficient or less robust.

## Metrics

### Quality

- accuracy where applicable;
- precision;
- recall;
- F1 where applicable;
- calibration or abstention quality;
- invalid-output rate;
- useful information retained.

### Noise

- false-positive rate;
- false-negative rate;
- output instability across equivalent inputs;
- performance on distractor-heavy inputs;
- variance across training seeds.

### Performance

- actual parameter count;
- model/checkpoint size;
- single-unit inference latency;
- batched throughput;
- CPU utilization;
- GPU utilization where available;
- RAM;
- VRAM where available;
- data-transfer overhead where measurable.

## Collapse classifications

Each tested size should eventually be classified as one of:

```text
CLEARLY_USEFUL
WEAK_BUT_USEFUL
BOUNDARY_UNCERTAIN
MOSTLY_NOISE
DETERMINISTICALLY_DOMINATED
OPTIMIZATION_FAILURE
```

A unit is not considered useful merely because it performs slightly above random chance.

## Success criteria

Step 1 succeeds when the results identify:

1. at least one unit-size range that produces clearly useful learned signal;
2. at least one smaller range where performance clearly collapses, becomes mostly noise, or loses value relative to deterministic logic;
3. enough evidence to choose candidate unit sizes for Step 2 population-scaling experiments.

The result does not need to be one exact parameter count. A useful interval is sufficient for the first pass.

## Failure is informative

Possible valid outcomes include:

- the minimum useful unit is much larger than expected;
- the tested architecture does not scale down gracefully;
- the benchmark does not contain tasks where learning adds value;
- tiny units remain useful but are inefficient to batch;
- deterministic methods dominate the selected task family.

Each outcome should update the hypothesis rather than being hidden.

## Implementation order

The next work should proceed in this order:

1. implement the procedural benchmark generator;
2. implement and validate deterministic baselines;
3. freeze benchmark v0 test seeds and difficulty distributions;
4. implement the scalable neural-unit architecture;
5. implement parameter-count/config generation;
6. train one comfortably large reference configuration;
7. confirm every intended task family is learnable;
8. run the coarse size sweep;
9. locate and confirm the collapse boundary.

No population coordination code is needed until Step 1 establishes which unit sizes are worth scaling.
