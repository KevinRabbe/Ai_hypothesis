# Step 2 — Population Scaling

## Goal

Determine whether increasing the number of architecturally identical learned neural processing units produces additional useful information before coordination and execution overhead dominate.

Step 2 begins only after Step 1 identifies a statistically credible viable worker-size region. The current Step 1 confirmation sweep is expected to inform the exact worker sizes used here; this document intentionally does not hard-code a winner before those results are complete.

## Core hypothesis

A population of independently weighted but architecturally identical workers can produce more useful evidence than a single worker, while remaining efficient enough to batch on a GPU.

The population is not a collection of autonomous agents. Workers do not hold conversations, plan independently, or vote on a final answer. Each worker performs the same compact learned transformation and emits a structured evidence packet.

The population layer combines evidence, not votes.

## Step 2A — Width scaling

Use one primary worker architecture selected from the Step 1 confirmation results.

Test population widths such as:

- 1 worker;
- 4 workers;
- 16 workers;
- 64 workers;
- 256 workers;
- larger widths only when hardware and results justify them.

Initial execution uses 100% of the configured workers. Adaptive activation is deliberately excluded from Step 2A so that worker-count effects are not confounded with routing quality.

Questions:

1. Does useful output quality improve as width increases?
2. Does uncertainty calibration improve?
3. Does evidence coverage improve?
4. Do unique or rare useful signals survive aggregation?
5. At what width do gains saturate?
6. At what width does coordination or execution overhead erase the benefit?

## Step 2B — Fixed-budget worker-size organization study

After Step 2A proves that population width itself adds useful signal, compare homogeneous populations built from the Step 1 candidate worker sizes under an approximately equal total learned-parameter budget.

Illustrative example using a 5M total worker-parameter budget:

- approximately 197 × 25K workers;
- approximately 99 × 50K workers;
- approximately 66 × 75K workers.

Actual counts must use the measured trainable parameter count of each frozen architecture and should minimize total-budget mismatch.

Each population remains homogeneous internally. A 25K-worker population contains only 25K workers; a 50K-worker population contains only 50K workers; a 75K-worker population contains only 75K workers. Mixed worker shapes are explicitly out of scope because they would weaken uniform batching and complicate scheduling.

This Step 2B experiment is an early population-organization study. The broader dense-versus-population fixed-budget competition remains a later research stage.

## Non-goals

Step 2 does not yet attempt to prove:

- adaptive worker allocation;
- dynamic CPU/GPU scheduling;
- large-document partitioning;
- hierarchical recursive aggregation;
- dense-model superiority or inferiority;
- scaling to approximately 1B total parameters.

Those are separate hypotheses and should not be mixed into the first population test.

## Required controls

Population results must be compared against:

- a single worker;
- majority vote as a deliberately naive control, not the primary decision rule;
- mean-logit or mean-probability ensembling;
- the evidence-preserving population reducer defined for Step 2;
- deterministic task baselines already used in Step 1 where applicable.

## Success criteria

Step 2 is promising if increasing population width produces reproducible gains in at least one of:

- final task quality;
- uncertainty quality;
- error detection;
- unique evidence recall;
- robustness;

and those gains remain meaningful after end-to-end execution and aggregation costs are included.

## Failure conditions

Step 2 should be considered negative or redirected if:

- population width gives no reproducible gain over one worker;
- gains are explained entirely by naive ensembling and do not justify the architecture;
- rare evidence is systematically lost during aggregation;
- worker outputs are too correlated to add useful information;
- batching or aggregation overhead dominates useful neural compute;
- larger populations become less reliable without a compensating advantage.

A negative result remains scientifically useful because it identifies a boundary of the population hypothesis.
