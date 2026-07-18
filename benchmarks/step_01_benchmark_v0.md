# Step 1 Benchmark v0 — Minimum Useful Neural Unit

## Objective

Locate the parameter range where one learned unit transitions from:

```text
useful learned signal
→ weak but still useful signal
→ mostly noise
```

The benchmark must not reward language generation or large memorized knowledge stores. It should test small local transformations where learning can be useful and where deterministic baselines can also be measured.

## Common representation

Every task is encoded into the Step 1 standard input shape:

```text
32 elements × 16 floating-point features
```

Unused elements are masked.

Each task uses the same model backbone and compact structured output mechanism.

The generator must be deterministic for a fixed seed and version.

## Task family A — Noisy local pattern detection

### Goal

Detect whether a target local pattern is present in a noisy feature field.

### Example concepts

- short line-like motif;
- corner-like motif;
- repeated local sequence;
- sparse target among distractors.

### Difficulty controls

- additive noise;
- missing elements;
- distractor density;
- partial occlusion;
- target strength.

### Outputs

```text
SIGNAL
NO_SIGNAL
UNCERTAIN
```

### Deterministic baselines

- fixed template correlation;
- thresholded nearest-template distance.

This task is useful only if learned processing remains competitive under perturbations where rigid templates degrade.

## Task family B — Fuzzy change detection

### Goal

Given two compact local states, determine whether a meaningful change occurred rather than reacting to harmless noise.

### Example concepts

- stable state plus measurement jitter;
- gradual drift;
- abrupt change;
- incomplete second observation.

### Outputs

```text
NO_CHANGE
CHANGE
UNCERTAIN
```

### Deterministic baselines

- absolute-difference threshold;
- moving statistical threshold;
- simple distance metric.

The learned unit must show value on cases where one fixed threshold is insufficient across the data distribution.

## Task family C — Local conflict detection

### Goal

Detect whether two noisy local observations are compatible, contradictory, or too incomplete to decide.

### Outputs

```text
COMPATIBLE
CONFLICT
UNCERTAIN
```

### Difficulty controls

- observation noise;
- partial evidence;
- distractor features;
- conflicting evidence strength;
- missing discriminating features.

### Deterministic baselines

- exact comparison on clean latent state;
- distance or rule threshold on observed state.

The clean latent-state baseline represents an upper bound that is not available to the model at inference time.

## Task family D — Local relationship extraction

### Goal

Infer one local relationship from a compact set of noisy observations.

Candidate relations:

```text
A_BEFORE_B
B_BEFORE_A
SAME_TIME
UNCERTAIN
```

or an equivalent spatial relation set.

The first version should use one relation domain only to avoid mixing unrelated difficulty sources.

### Deterministic baselines

- direct rule on clean latent variables;
- rule on noisy observed variables.

The learned unit is valuable only if it can infer the underlying relation more robustly than the simple observed-variable rule.

## Task family E — Sparse relevant-signal detection

### Goal

Determine whether a small input contains information relevant to a target condition when most elements are distractors.

### Difficulty controls

- proportion of distractors;
- target signal strength;
- target position;
- near-miss distractors;
- incomplete target pattern.

### Outputs

```text
RELEVANT
NOT_RELEVANT
UNCERTAIN
```

### Deterministic baselines

- nearest-pattern similarity;
- fixed feature rule.

This task is intended to become an early proxy for later large-input worker allocation.

## Why multiple task families

A unit size should not be declared useful because it memorizes one trivial mapping.

The first sweep should report performance separately for each family and also calculate a compact cross-task summary.

The purpose is to identify whether shrinking causes:

- all learned transformations to collapse together;
- some transformations to survive at much smaller sizes;
- uncertainty handling to collapse before classification;
- specific tasks to become better handled by deterministic logic.

## Dataset generation

The v0 benchmark should use procedurally generated synthetic data so that:

- arbitrary amounts of training data can be produced;
- train, validation, and test examples can be generated from disjoint seeds;
- hidden latent truth is available for exact scoring;
- noise and ambiguity can be controlled precisely;
- no unit can gain an advantage from memorizing a finite test corpus.

Each example must store or reproducibly regenerate:

```text
task family
input tensor
validity mask
target label
ambiguity / uncertainty target
difficulty parameters
generator seed
```

## Split policy

Initial proposal:

```text
train seeds:       dedicated range A
validation seeds:  dedicated range B
test seeds:        dedicated range C
```

No seed may appear in more than one split.

The test generator configuration must be frozen before the first evaluated training run.

## Difficulty strata

Every task should include at least:

```text
easy
medium
hard
ambiguous
```

The ambiguous stratum contains examples where available evidence is intentionally insufficient or near the decision boundary. Correct abstention should be rewarded.

Results must be reported by stratum rather than only as one average.

## Core metrics

Per task and size:

- accuracy;
- precision;
- recall;
- F1 where applicable;
- abstention precision;
- abstention recall;
- false-positive rate;
- false-negative rate;
- equivalent-input stability;
- distractor robustness;
- performance by difficulty stratum.

## Useful-signal criterion

A unit is not considered useful merely because it scores above random chance.

A candidate useful region should show:

1. reproducible performance above trivial and simple deterministic baselines on at least one learned-value task;
2. non-catastrophic behavior on held-out seeds;
3. useful abstention or uncertainty behavior rather than confident random output;
4. stable output under semantically equivalent perturbations;
5. inference cost low enough to remain relevant to later population execution.

The exact acceptance threshold should be chosen after observing the large-model reference results but before interpreting the small-model sweep.

## Noise-collapse indicators

Evidence that a unit has become too small includes:

- accuracy approaching chance;
- sharp increase in confident false positives;
- failure to distinguish ambiguous examples from answerable examples;
- high output instability under equivalent inputs;
- inconsistent results across random seeds;
- little or no improvement with additional training;
- deterministic baselines dominating every tested transformation.

## First execution order

1. Implement the generator and deterministic baselines.
2. Validate labels and ambiguity rules independently of the neural model.
3. Train a comfortably large reference configuration.
4. Confirm that the architecture can learn every intended task family.
5. Run the coarse size sweep.
6. Locate the approximate collapse region.
7. Run a denser size sweep around that region.
8. Repeat boundary configurations with additional random seeds.

## What this benchmark does not establish

Step 1 does not prove that a population of units is superior to a dense model.

It only identifies which unit sizes are worth carrying into Step 2.
