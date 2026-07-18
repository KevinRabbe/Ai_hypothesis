# Population Model Hypothesis

## Problem statement

Conventional dense neural networks generally apply a largely fixed model structure to every request. A simple task and a difficult task may use the same parameter set even when the simple task requires only a small fraction of the available learned capability.

This project explores a different organization of learned computation.

Instead of asking one large dense network to process all information, divide a fixed total parameter budget into many **architecturally identical tiny neural processing units**. The units are not autonomous agents and do not need to solve complete tasks independently. A unit only needs to perform a useful local learned transformation.

Examples of useful local transformations may include:

- detecting a fuzzy pattern;
- identifying relevance;
- extracting a relationship;
- recognizing an anomaly;
- identifying ambiguity or conflict;
- transforming a compact local representation into another useful representation.

The runtime can allocate additional identical units when a region, subproblem, or evidence gap requires more learned processing.

## Factorio analogy

Think of each neural unit as the smallest useful subfactory.

A subfactory does not build the entire rocket. It performs one useful production step. When more throughput is needed, more identical subfactories are assigned to the production line.

The research question is therefore not "how many tiny models can we create?" It is:

> How small can a learned processing unit become before it stops producing useful signal, and how efficiently can additional identical units be allocated when more processing capacity is needed?

## Fixed-budget framing

A long-term reference experiment may use approximately 1 billion total learned parameters.

Possible organizations are illustrative, not predetermined:

- 1 × 1B
- 10 × 100M
- 100 × 10M
- 1,000 × 1M
- 10,000 × 100K

The project does **not** assume that any of these is optimal. The useful unit size may be much larger or smaller, and the optimum depends on batching efficiency, hardware, memory movement, coordination cost, and task type.

## Dynamic allocation

The total population represents available neural capacity.

A task may activate only a small subset:

- easy local transformation → few units;
- noisy or ambiguous region → more units;
- large divisible input → many units working on different regions;
- missing discriminating evidence → targeted additional units;
- deterministic task → zero additional neural units when ordinary logic can solve it.

The system should scale compute by need rather than activating the entire parameter population for every request.

## Evidence, not voting

Worker count is not truth.

If 8,000 units do not observe a decisive fact and 200 units process the only region containing that fact, the smaller group may hold the important evidence.

Therefore aggregation must preserve:

- source provenance;
- local observations;
- unique findings;
- contradictions;
- evidence coverage;
- evidence strength.

Population disagreement may indicate uncertainty or a need for more investigation, but majority vote must not be the final decision rule.

## Deterministic computation boundary

Learned computation should be used where learning adds value.

Deterministic algorithms should handle tasks such as:

- arithmetic;
- exact comparisons;
- sorting;
- coordinates;
- deduplication;
- state transitions;
- routing rules;
- permissions;
- queue management;
- resource limits;
- schema validation.

The research should explicitly identify the lower boundary where a neural unit becomes either:

1. mostly noise; or
2. so simple that deterministic logic performs the same transformation more reliably and cheaply.

## Hardware hypothesis

The architecture should treat CPU and GPU as complementary resources.

The CPU is suited to orchestration, deterministic logic, task graphs, evidence tracking, queues, scheduling, and data preparation.

The GPU is suited to batched execution of many identical neural units through efficient matrix operations.

The architecture succeeds only if batching and resource scheduling keep coordination overhead below the value gained from sparse, parallel, and adaptive computation.

## Primary hypothesis

> For at least some workloads, a fixed neural parameter budget can produce more useful information per unit of active compute when organized as a dynamically allocated population of identical small learned processing units than when organized as one fixed dense network.

## Null / failure outcomes

The hypothesis should be considered unsupported for a tested configuration when any of the following dominate:

- individual units become too weak and produce mostly noise;
- additional workers add correlated errors rather than useful information;
- routing and aggregation cost exceeds saved neural compute;
- GPU utilization collapses because workloads are too granular;
- memory movement dominates execution;
- hierarchical compression loses decisive information;
- a conventional dense model reaches better end-to-end quality under the same resource budget.

Negative results are useful: they define the boundary of the architecture and narrow the search for the viable operating region.
