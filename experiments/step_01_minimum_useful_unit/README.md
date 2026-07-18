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

Initial checkpoints are placeholders and may be adjusted after the architecture is selected:

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

## Benchmark properties

The first benchmark suite should test small local transformations rather than full language generation.

Tasks should include a mix of:

1. transformations that deterministic logic can solve, to establish a baseline;
2. transformations with noisy, fuzzy, incomplete, or ambiguous inputs where learned processing may add value;
3. transformations where a unit can explicitly return an uncertain/unknown state instead of guessing.

Potential task families:

- relevance detection;
- anomaly detection;
- local relationship extraction;
- fuzzy change detection;
- conflict detection under noisy representations;
- local pattern classification;
- compact signal extraction from small text or feature inputs.

The exact benchmark dataset must be defined before training starts.

## Output contract

Prefer a very small structured output space for Step 1.

Example conceptual statuses:

- `SIGNAL`;
- `NO_SIGNAL`;
- `UNCERTAIN`.

Task-specific outputs may use compact class IDs or fixed-size vectors.

Avoid evaluating tiny units through verbose natural-language generation because vocabulary and decoding overhead would dominate the question being studied.

## Metrics

Record at minimum:

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
- performance on distractor-heavy inputs.

### Performance

- parameter count;
- model size;
- single-unit inference latency;
- batched throughput;
- CPU utilization;
- GPU utilization;
- RAM;
- VRAM;
- data-transfer overhead where measurable.

## Deterministic baseline

Every benchmark task must document whether a deterministic algorithm can solve it.

If deterministic logic solves a task equally well or better at lower cost, that task should not be used as evidence that the neural unit is necessary.

It may remain useful as a control.

## Success criteria

Step 1 succeeds when the results identify:

1. at least one unit-size range that produces clearly useful learned signal;
2. at least one smaller range where performance clearly collapses, becomes mostly noise, or loses value relative to deterministic logic;
3. enough evidence to choose candidate unit sizes for Step 2 population-scaling experiments.

The result does not need to be a single exact parameter count. A useful interval is sufficient for the first pass.

## Failure is informative

Possible valid outcomes include:

- the minimum useful unit is much larger than expected;
- the tested architecture does not scale down gracefully;
- the benchmark does not contain tasks where learning adds value;
- tiny units remain useful but are inefficient to batch;
- deterministic methods dominate the selected task family.

Each outcome should update the hypothesis rather than being hidden.

## Before implementation

Still to define:

1. the first architecture family;
2. the exact benchmark tasks;
3. training-data generation or collection;
4. train/validation/test split policy;
5. size-scaling method;
6. hardware measurement method;
7. experiment configuration format;
8. reproducibility requirements and random-seed policy.

Those definitions are the next planning work before training code is written.
