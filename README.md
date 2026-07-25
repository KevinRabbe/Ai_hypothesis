# AI Hypothesis — Population Model Research

This repository investigates whether a fixed neural parameter and hardware budget can be used more effectively by organizing learned computation as a dynamically allocated population of many identical small neural processing units instead of one conventional dense network.

## Core hypothesis

The goal is **not** to build thousands of autonomous agents.

The goal is to use the smallest **practical** identical learned worker that can perform useful local information transformations at useful latency and cost, then test whether populations of those workers can improve capability and resource efficiency.

A unit may be individually weak. Useful system-level capability is expected to come from:

- distributing large inputs across many units;
- allocating more units where uncertainty or information density is high;
- preserving rare but decisive evidence rather than using majority voting;
- recursively combining local findings into higher-level representations;
- using deterministic algorithms for routing, exact logic, validation, arithmetic, coordinates, scheduling, and other tasks that do not require learned prediction;
- batching identical neural units efficiently on GPU hardware;
- using the CPU for orchestration, evidence management, deterministic logic, and resource scheduling.

The long-term reference budget is approximately **1 billion total learned parameters**, but scale is not a current milestone. The project first has to demonstrate an advantage at small scale.

## Central research question

> At a fixed hardware and parameter budget, can a population of practical small learned workers produce more useful capability per unit of active compute than a conventional fixed dense model?

The current immediate sub-question is narrower:

> When additional workers contain correct minority evidence that the population mean suppresses, can the runtime identify and use that evidence without trusting noisy minority outliers?

## Core principles

1. Workers contribute **evidence and transformations, not votes**.
2. More workers are allocated to obtain missing or discriminating information, not merely to repeat the same opinion.
3. Rare evidence must never be averaged away because it is held by a minority of workers.
4. Deterministic logic should replace neural prediction wherever the task can be solved reliably without learning.
5. End-to-end cost matters: routing, memory movement, batching, aggregation, and synchronization count against the architecture.
6. If organization costs more than the compute it saves, that configuration fails.
7. The population size is dynamic. Total parameters represent available capacity, not mandatory compute for every task.
8. Worker shrinking is useful only while it improves the practical system. Parameter count is not an objective by itself.
9. Consumer hardware is a first-class target. Inefficiency should not be hidden behind datacenter-scale compute.
10. Established engineering facts should be reused rather than re-proven; experiments are reserved for architecture-specific uncertainty.
11. Compiler optimization is a separate experimental variable and must not be silently mixed into neural-architecture results.

## Research sequence

The project proceeds through narrow research gates. A gate exists only when its answer depends on this architecture rather than on an already established engineering fact.

The active milestone is **Step 2A: Population Evidence Utilization**.

Current order:

1. keep the 50K reference worker frozen;
2. determine whether strong minority evidence can be rescued safely;
3. find the useful population-width region;
4. compare 25K / 50K / 75K homogeneous populations under equal total budgets;
5. measure compiler impact as a separate variable;
6. test adaptive allocation;
7. move to real workloads;
8. scale further only if a small-scale advantage survives.

See:

- [`docs/hypothesis.md`](docs/hypothesis.md)
- [`docs/research_questions.md`](docs/research_questions.md)
- [`docs/roadmap.md`](docs/roadmap.md)
- [`experiments/step_02_population_scaling/README.md`](experiments/step_02_population_scaling/README.md)
- [`experiments/step_02_population_scaling/minority_rescue_v0.md`](experiments/step_02_population_scaling/minority_rescue_v0.md)
- [`benchmarks/step_01_benchmark_v0.md`](benchmarks/step_01_benchmark_v0.md)

## Current status

**Step 1 worker shrinking is closed for the current phase.** The project already established a practical small-worker region and Step 2 uses the ~50K architecture as the frozen reference worker.

The latest Step 2A validation checkpoint uses 16 independently trained 50K workers. On 20,000 validation samples:

- one-worker oracle-any-correct coverage was 94.825%;
- 16-worker oracle-any-correct coverage reached 97.770%;
- majority accuracy was 95.340%;
- reducer-v0 accuracy was 95.260%;
- 463 samples contained a true minority opportunity;
- reducer-v0 rescued 19 of those opportunities.

The population therefore already contains additional useful information. The active bottleneck is **turning that information into final capability without creating more errors by trusting noisy minority evidence**.
