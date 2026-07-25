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

The current immediate empirical sub-question is narrower:

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
11. Compiler optimization is a separate systems variable and must not be silently mixed into neural-architecture results.
12. **Final architecture from day 1; minimal implementation from day 1.** Scale should improve or split implementations behind stable contracts rather than repeatedly redesigning the system.
13. **Construction does not wait for empirical results.** Record uncertainty, freeze a provisional assumption behind a stable contract, continue independent construction, and run the deferred experiment when its result becomes decision-relevant.

## Population scaling and the information bottleneck

At sufficiently large population sizes, neural execution may stop being the dominant scaling constraint.

A large worker population can generate observations, hypotheses, failures, contradictions, and candidate evidence in parallel. The shared runtime must still preserve, deduplicate, connect, verify, route, summarize, and exploit the useful subset.

The project therefore distinguishes:

- **exploration throughput** — how many meaningfully different evidence-producing attempts are executed per unit wall-clock time;
- **knowledge integration bandwidth** — how much useful evidence becomes persistent, connected, actionable shared knowledge per unit wall-clock time without losing decisive minority information.

The useful population limit may be reached when marginal workers generate useful information faster than the integration path can absorb it, even if additional neural compute remains available.

This is a future scaling hypothesis, not a claim that the current 16-worker experiment is integration-bound. The present empirical uncertainty remains evidence utilization.

The intended scaling strategy is hierarchical: keep raw/local results close to their Work Threads, propagate only meaningful knowledge changes upward, and preserve references to the original evidence rather than broadcasting all information to all workers.

## Research and construction proceed independently

Research gates define the order in which empirical uncertainties should be resolved **when their answers are needed**. They are not construction blockers.

The research sequence remains:

1. minority-evidence utilization;
2. useful population-width region;
3. fixed-budget worker organization;
4. population versus dense baselines;
5. adaptive allocation;
6. knowledge-integration bandwidth when volume justifies measuring it;
7. real workloads;
8. larger-scale validation only after useful behavior exists.

Construction proceeds continuously in parallel. When an experiment is deferred, its exact question and provisional assumption are recorded, while implementation continues through stable interfaces. A later result may replace a reducer, policy, threshold, storage strategy, or scheduler heuristic without stopping unrelated work or changing the architecture.

See [`docs/construction_policy.md`](docs/construction_policy.md) for the explicit non-blocking development rule.

See:

- [`docs/hypothesis.md`](docs/hypothesis.md)
- [`docs/runtime_architecture.md`](docs/runtime_architecture.md)
- [`docs/architecture_contracts.md`](docs/architecture_contracts.md)
- [`docs/construction_policy.md`](docs/construction_policy.md)
- [`docs/research_questions.md`](docs/research_questions.md)
- [`docs/roadmap.md`](docs/roadmap.md)
- [`experiments/step_02_population_scaling/README.md`](experiments/step_02_population_scaling/README.md)
- [`experiments/step_02_population_scaling/minority_rescue_v0.md`](experiments/step_02_population_scaling/minority_rescue_v0.md)
- [`benchmarks/step_01_benchmark_v0.md`](benchmarks/step_01_benchmark_v0.md)
- [`benchmarks/large_scope_relevance_v0.md`](benchmarks/large_scope_relevance_v0.md) — constructed controlled scope benchmark; no Gate 5 result is claimed.

## Current status

**Step 1 worker shrinking is closed for the current phase.** The project already established a practical small-worker region and Step 2 uses the ~50K architecture as the frozen reference worker.

The latest Step 2A validation checkpoint uses 16 independently trained 50K workers. On 20,000 validation samples:

- one-worker oracle-any-correct coverage was 94.825%;
- 16-worker oracle-any-correct coverage reached 97.770%;
- majority accuracy was 95.340%;
- reducer-v0 accuracy was 95.260%;
- 463 samples contained a true minority opportunity;
- reducer-v0 rescued 19 of those opportunities.

The population therefore already contains additional useful information. The minority-rescue diagnostic remains a deferred empirical test until its answer is needed for a reducer decision; it does **not** block construction of the stable runtime.