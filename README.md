# AI Hypothesis — Population Model Research

This repository investigates whether a fixed neural parameter and hardware budget can be used more effectively by organizing learned computation as a dynamically allocated population of many identical tiny neural processing units instead of one conventional dense network.

## Core hypothesis

The goal is **not** to build thousands of autonomous agents.

The goal is to find the smallest identical learned neural unit that can still perform a useful local information transformation beyond what deterministic logic can do, then test whether dynamically allocating more of those units to difficult, noisy, ambiguous, or information-dense regions can improve capability and resource efficiency.

A unit may be individually weak. Useful system-level capability is expected to come from:

- distributing large inputs across many units;
- allocating more units where uncertainty or information density is high;
- preserving rare but decisive evidence rather than using majority voting;
- recursively combining local findings into higher-level representations;
- using deterministic algorithms for routing, exact logic, validation, arithmetic, coordinates, scheduling, and other tasks that do not require learned prediction;
- batching identical neural units efficiently on GPU hardware;
- using the CPU for orchestration, evidence management, deterministic logic, and resource scheduling.

The long-term reference budget is approximately **1 billion total learned parameters**, but the first experiments will use much smaller fixed budgets. Example worker counts such as 10,000 are illustrative only; the optimal unit size and population width must be discovered empirically and will depend on the available hardware.

## Central research question

> At a fixed hardware and parameter budget, what is the smallest learned neural processing unit that still produces useful signal beyond deterministic logic, and can a dynamically allocated population of those units outperform a conventional fixed dense model in useful capability per unit of active compute?

## Core principles

1. Workers contribute **evidence and transformations, not votes**.
2. More workers are allocated to obtain missing or discriminating information, not merely to repeat the same opinion.
3. Rare evidence must never be averaged away because it is held by a minority of workers.
4. Deterministic logic should replace neural prediction wherever the task can be solved reliably without learning.
5. End-to-end cost matters: routing, memory movement, batching, aggregation, and synchronization count against the architecture.
6. If organization costs more than the compute it saves, that configuration fails.
7. The population size is dynamic. Total parameters represent available capacity, not mandatory compute for every task.
8. The smallest useful unit is an empirical result, not a predetermined parameter count.
9. Consumer hardware is a first-class target. Inefficiency should not be hidden behind datacenter-scale compute.

## Research sequence

The project proceeds through research gates. Each stage should answer one major uncertainty before the next architectural commitment is made.

The immediate milestone is **Step 1: Minimum Useful Neural Unit**.

See:

- [`docs/hypothesis.md`](docs/hypothesis.md)
- [`docs/research_questions.md`](docs/research_questions.md)
- [`docs/roadmap.md`](docs/roadmap.md)
- [`experiments/step_01_minimum_useful_unit/README.md`](experiments/step_01_minimum_useful_unit/README.md)
- [`experiments/step_01_minimum_useful_unit/architecture_v0.md`](experiments/step_01_minimum_useful_unit/architecture_v0.md)
- [`experiments/step_01_minimum_useful_unit/protocol_v0.md`](experiments/step_01_minimum_useful_unit/protocol_v0.md)
- [`benchmarks/step_01_benchmark_v0.md`](benchmarks/step_01_benchmark_v0.md)

## Current status

**Step 1 experiment design.** The v0 architecture family, benchmark concept, and controlled experiment protocol are now defined. The next work is implementation: procedural benchmark generators and deterministic baselines first, followed by the scalable neural-unit implementation and the first large reference training run.
