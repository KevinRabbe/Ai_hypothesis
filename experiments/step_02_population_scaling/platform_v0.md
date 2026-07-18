# Population Experiment Runtime v0

## Purpose

The first population program is a research runtime, not a polished end-user application. Its job is to connect the information produced by many homogeneous neural workers into one inspectable experimental pipeline.

The runtime must let us answer two different questions cleanly:

1. What useful information exists anywhere inside the population?
2. How much of that information can the aggregation system actually recover into the final answer?

The platform should therefore preserve worker-level evidence long enough to measure aggregation failures instead of exposing only the final prediction.

## Core rule

One population run contains one worker architecture and one worker size.

A population may contain many independently weighted workers, but they all use the same tensor shapes and model structure. Different worker sizes are separate population configurations and are never mixed inside one population.

## v0 execution pipeline

```text
Experiment Plan
      |
      v
Population Registry
      |
      v
Homogeneous Worker Bank
      |
      v
Population Executor
      |
      v
Worker Evidence Matrix
      |
      +--------------------------+
      |                          |
      v                          v
Evidence Aggregator        Diagnostic Analyzer
      |                          |
      v                          |
Final Prediction                |
      |                          |
      +-------------+------------+
                    |
                    v
              Evaluator
                    |
                    v
          Result + Telemetry Store
```

## Main components

### 1. Experiment Plan

A versioned experiment description defines exactly what will run.

Minimum fields:

- benchmark version;
- worker architecture version;
- worker parameter count;
- total population parameter target;
- actual worker count;
- worker training seeds or checkpoint identifiers;
- population width under evaluation;
- activation policy;
- aggregation policy;
- baseline configuration;
- dataset seeds;
- hardware/device selection;
- code revision.

The actual worker count must be derived from the real measured parameter count rather than the rounded labels `25K`, `50K`, or `75K`.

### 2. Population Registry

The registry owns the identity and metadata of every worker in one homogeneous population.

Each worker has:

- stable worker id;
- architecture id;
- checkpoint id;
- training seed;
- parameter count;
- optional reliability statistics derived only from training/validation data.

The registry must reject heterogeneous worker shapes inside the same population.

### 3. Homogeneous Worker Bank

The worker bank stores the independently learned workers that form the population.

v0 requirements:

- same architecture for every worker;
- independent learned weights;
- deterministic ordering;
- deterministic subset selection for population-width experiments;
- support for multiple deterministic permutations;
- no adaptive routing in the first population baseline.

The execution interface must not assume that workers will always be executed one at a time. The backend should be replaceable later by a more efficient vectorized/batched implementation without changing the evidence or experiment contracts.

### 4. Population Executor

The executor receives a benchmark batch and a selected worker subset.

For Step 2A all selected workers receive the same input. The executor produces a tensor-shaped worker evidence matrix rather than immediately reducing outputs to one answer.

Conceptual shape:

```text
[sample, worker, evidence_field]
```

For the current Step 1 task family, the raw learned output remains:

- 11 non-uncertain label logits;
- 1 uncertainty logit.

The executor converts these outputs into the versioned evidence contract before aggregation.

### 5. Worker Evidence Matrix

The evidence matrix is the central connection point between neural computation and aggregation.

For each sample and worker, retain at least:

- worker id;
- label support distribution;
- uncertainty;
- task-invalid probability mass;
- support strength;
- reliability weight if enabled;
- provenance back to the exact worker/checkpoint.

The runtime must preserve continuous evidence. A worker output must not be collapsed prematurely into a single categorical vote.

### 6. Evidence Aggregator

The aggregator combines population information into a final prediction while preserving rare strong signals.

The first aggregator should expose separate channels rather than one destructive average:

#### Consensus channel

Measures broad population support for each candidate label.

#### Strong-evidence channel

Preserves maximum and top-k support for each candidate label so a small minority of workers can retain decisive evidence.

#### Conflict channel

Measures incompatible high-confidence support between competing labels.

#### Uncertainty channel

Combines individual uncertainty, population disagreement, invalid-label mass, and unresolved conflict.

The final decision rule may consume these channels, but the channels themselves must remain available for diagnostics.

Majority vote is implemented only as a baseline comparator.

### 7. Diagnostic Analyzer

The diagnostic analyzer determines whether failures come from the workers or from aggregation.

Required metrics include:

- oracle-any-correct coverage;
- oracle-top-k coverage;
- unique finding rate;
- minority-rescue opportunities;
- minority-rescue rate;
- minority-suppression rate;
- majority-harm rate;
- evidence-utilization gap;
- population disagreement;
- invalid-label mass;
- uncertainty calibration.

A large gap between oracle coverage and final accuracy is evidence that useful information exists inside the population but the aggregator is failing to recover it.

### 8. Evaluator

The evaluator applies the same frozen benchmark rules used for controlled comparisons.

It reports:

- final task quality;
- per-task quality;
- per-difficulty quality;
- uncertainty behavior;
- deterministic-baseline comparison;
- majority-vote comparison;
- dense-model comparison where applicable;
- population diagnostics;
- inference latency;
- throughput;
- coordination overhead;
- memory use.

### 9. Result and Telemetry Store

Every run writes a compact machine-readable summary plus enough structured evidence for later analysis.

Recommended layout:

```text
results/step02/<experiment_id>/
├── result.json
├── experiment_plan.json
├── population_manifest.json
├── metrics.json
├── telemetry.json
└── evidence/
```

Full per-worker evidence can become very large. v0 should therefore separate:

- always-retained aggregate metrics;
- compact per-sample diagnostic tensors;
- optionally retained full worker evidence for selected diagnostic subsets.

The result format must record exactly what evidence was retained so comparisons remain reproducible.

## First supported experiment families

### A. Population width scaling

For one confirmed worker size:

```text
W = 1, 4, 16, 64, 256
```

Use deterministic nested subsets and multiple deterministic worker permutations.

Goal: measure whether additional workers add useful information before coordination overhead dominates.

### B. Worker granularity x total population budget

Candidate grid:

```text
                  Worker size
Total budget      25K      50K      75K
-----------------------------------------
~1M               run      run      run
~5M               run      run      run
~10M              run      run      run
```

Each population remains homogeneous.

Add dense baselines near:

- 1M;
- 5M;
- 10M.

This tests whether the optimal worker granularity changes as total learned capacity scales.

### C. Later active-compute comparison

After adaptive activation exists, compare a partially active large population against a conventional dense model with approximately equal active neural compute.

Example:

```text
5M total population at 25% active
vs
~1.25M dense model
```

This is not part of the first implementation because routing would confound the initial population result.

## v0 non-goals

The first platform does not include:

- mixed worker sizes in one population;
- adaptive worker activation;
- semantic routing;
- large-input partitioning;
- hierarchical multi-stage aggregation;
- SSD/RAM/VRAM weight streaming;
- a polished desktop GUI;
- real-world text/image workloads.

These are later layers built only after the basic population evidence flow is understood.

## CLI-first interface

The first usable interface should be a CLI so experiments remain reproducible and automatable.

Conceptual commands:

```text
python -m ai_hypothesis.step02.run_population --plan <plan.json>
python -m ai_hypothesis.step02.run_grid --grid <grid.json>
python -m ai_hypothesis.step02.summarize --results <directory>
```

A later local dashboard should read the same result artifacts rather than owning experimental logic.

## Architectural boundary

The platform should be divided into stable contracts:

```text
Experiment Plan
      -> Population Registry
      -> Executor
      -> Evidence Contract
      -> Aggregator
      -> Evaluator
      -> Result Contract
```

Execution strategy, aggregation algorithms, and UI can evolve independently as long as they obey these contracts.

## First implementation target

The minimum useful platform slice is:

1. load a homogeneous worker bank;
2. select a deterministic worker subset;
3. run all selected workers on the same benchmark batch;
4. materialize the worker evidence matrix;
5. aggregate through consensus + strong-evidence + conflict + uncertainty channels;
6. produce a final prediction;
7. compare against individual-worker, majority-vote, deterministic, and dense baselines;
8. write reproducible result and diagnostic artifacts.

Only after this pipeline works should the project optimize population execution or add dynamic activation.
