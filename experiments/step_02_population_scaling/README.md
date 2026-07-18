# Step 2 — Population Scaling

## Goal

Determine whether increasing the number of architecturally identical learned neural processing units produces additional useful information before coordination and execution overhead dominate, and determine how worker granularity interacts with total learned capacity.

Step 2 begins only after Step 1 identifies a statistically credible viable worker-size region. The current Step 1 confirmation sweep is expected to inform the exact worker sizes used here; this document intentionally does not hard-code a winner before those results are complete.

## Core hypothesis

A population of independently weighted but architecturally identical workers can produce more useful evidence than a single worker, while remaining efficient enough to batch on a GPU.

The population is not a collection of autonomous agents. Workers do not hold conversations, plan independently, or vote on a final answer. Each worker performs the same compact learned transformation and emits structured evidence.

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

## Step 2B — Worker size × total population budget

After Step 2A establishes how width behaves, compare homogeneous populations built from the Step 1 candidate worker sizes across multiple fixed total learned-parameter budgets.

Initial grid:

| Total worker-parameter budget | ~25K workers | ~50K workers | ~75K workers | Dense control |
|---|---:|---:|---:|---:|
| ~1M | measured count | measured count | measured count | ~1M |
| ~5M | measured count | measured count | measured count | ~5M |
| ~10M | measured count | measured count | measured count | ~10M |

The rounded worker names are only labels. Actual worker counts must be calculated from the measured trainable parameter count of each frozen architecture and should minimize total-budget mismatch.

For example, a nominal 5M budget would contain more 25K workers than 50K workers and more 50K workers than 75K workers. The exact counts are derived programmatically from the real checkpoint architecture.

Each population remains homogeneous internally:

- a 25K-worker population contains only 25K workers;
- a 50K-worker population contains only 50K workers;
- a 75K-worker population contains only 75K workers.

Mixed worker shapes are explicitly out of scope because they weaken uniform batching, fragment execution, and add scheduler complexity.

The primary question is no longer only "which population wins at 5M?" It is:

> How does the optimal worker granularity change as total learned capacity scales?

Possible outcomes include:

- one worker size remains best across all total budgets;
- smaller workers become more competitive as population size grows;
- larger workers dominate at low or high total capacity;
- population gains saturate at different total budgets for different worker sizes;
- population-quality curves cross as the total learned budget increases.

## Dense controls

At every total-budget level, include a conventional dense model with approximately the same trainable parameter budget where a fair architecture can be defined.

The dense controls answer a different question from the population-to-population comparison:

> Under approximately equal total learned capacity, does splitting capacity into a homogeneous population help or hurt compared with keeping capacity in one network?

A dense model winning at one budget does not prove that splitting is universally harmful. Likewise, a small-worker population winning at one budget does not prove that smaller workers are universally better. The full scaling surface must be measured.

These dense controls are early experimental baselines. Strong architectural claims require normalized training compute, data exposure, hardware, inference cost, and repeated seeds.

## Runtime v0

The first Step 2 program is the Population Experiment Runtime.

Current implementation target:

```text
homogeneous checkpoint bank
        ↓
stacked worker execution
        ↓
worker evidence matrix
        ↓
evidence-preserving reducer
        ↓
final prediction + diagnostics
        ↓
streaming evaluator
        ↓
reproducible result JSON
```

The runtime must reject mixed worker architectures before execution.

The initial evaluator reports:

- single-worker accuracy distribution;
- majority vote as a naive control;
- mean-logit ensemble;
- mean-probability ensemble;
- evidence-preserving reducer;
- oracle-any-correct coverage;
- all-wrong rate;
- minority-rescue opportunities and rescue rate;
- minority suppression;
- majority harm;
- evidence-utilization gap;
- disagreement, uncertainty, and invalid-label diagnostics.

Aggregation thresholds are developed on validation data and frozen before formal test-set evaluation.

## Non-goals

Step 2 does not yet attempt to prove:

- adaptive worker allocation;
- partial worker activation;
- dynamic CPU/GPU scheduling;
- large-document partitioning;
- hierarchical recursive aggregation;
- scaling to approximately 1B total parameters.

Those are separate hypotheses and should not be mixed into the first population tests.

## Required controls

Population results must be compared against:

- a single worker;
- majority vote as a deliberately naive control, not the primary decision rule;
- mean-logit ensembling;
- mean-probability ensembling;
- the evidence-preserving population reducer defined for Step 2;
- deterministic task baselines already used in Step 1 where applicable;
- a same-total-parameter dense control for the multi-budget organization study.

## Success criteria

Step 2 is promising if increasing population width or changing population organization produces reproducible gains in at least one of:

- final task quality;
- uncertainty quality;
- error detection;
- unique evidence recall;
- robustness;
- quality per unit of end-to-end compute;

and those gains remain meaningful after execution and aggregation costs are included.

## Failure conditions

Step 2 should be considered negative or redirected if:

- population width gives no reproducible gain over one worker;
- gains are explained entirely by ordinary ensembling and do not justify the architecture;
- rare evidence is systematically lost during aggregation;
- worker outputs are too correlated to add useful information;
- batching or aggregation overhead dominates useful neural compute;
- larger populations become less reliable without a compensating advantage;
- dense controls consistently dominate under genuinely normalized end-to-end budgets.

A negative result remains scientifically useful because it identifies a boundary of the population hypothesis.
