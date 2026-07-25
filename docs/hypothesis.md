# Population Model Hypothesis

## Problem statement

Conventional dense neural networks generally apply a largely fixed model structure to every request. A simple task and a difficult task may use the same parameter set even when the simple task requires only a small fraction of the available learned capability.

This project explores a different organization of learned computation.

Instead of asking one large dense network to process all information, divide a fixed total parameter budget into many **architecturally identical tiny neural processing units**. The units are not autonomous agents and do not need to solve complete tasks independently. A unit only needs to perform a useful local learned transformation.

Workers should use the same architecture but independently learned weights. Homogeneous architecture makes workers interchangeable and efficiently batchable; independent weights create different error surfaces, sensitivities, and candidate approaches.

Examples of useful local transformations may include:

- detecting a fuzzy pattern;
- identifying relevance;
- extracting a relationship;
- recognizing an anomaly;
- identifying ambiguity or conflict;
- transforming a compact local representation into another useful representation.

The runtime can allocate additional identical units when a region, subproblem, or evidence gap requires more learned processing.

## Central mechanism

The population is not expected to become useful merely because worker count is large.

The mechanism being tested is:

> **more genuinely different possibilities explored -> more useful evidence generated -> more important uncertainty removed -> stronger system-level knowledge and decisions**

Different workers may:

- inspect different source regions;
- test different hypotheses;
- use different decompositions;
- search for supporting evidence;
- search for counterexamples;
- independently replicate a result;
- discover that an apparently promising path fails.

A failed attempt is not automatically wasted. It is useful when it rules out a possibility, identifies a condition under which a hypothesis fails, exposes a contradiction, or prevents the population from repeating the same dead end.

The system therefore optimizes for **useful information produced by the population**, not for every worker being individually correct.

## Factorio analogy

Think of each neural unit as the smallest useful subfactory.

A subfactory does not build the entire rocket. It performs one useful production step. When more throughput is needed, more identical subfactories are assigned to the production line.

The analogy is incomplete in one important way: research work can branch. Thousands of workers may test different paths rather than simply duplicating the same transformation. The runtime must preserve the useful outputs of those branches and redirect resources as the search space becomes better understood.

The research question is therefore not "how many tiny models can we create?" It is:

> What practical worker granularity, population organization, and information-integration architecture produces the most useful system-level knowledge under real resource constraints?

## Fixed-budget framing

A long-term reference experiment may use approximately 1 billion total learned parameters.

Possible organizations are illustrative, not predetermined:

- 1 × 1B
- 10 × 100M
- 100 × 10M
- 1,000 × 1M
- 10,000 × 100K

The project does **not** assume that any of these is optimal. The useful unit size may be much larger or smaller, and the optimum depends on worker capability, diversity, batching efficiency, hardware, memory movement, coordination cost, knowledge-integration cost, and task type.

Very large populations are a later scaling possibility rather than a current target. A million small workers would be useless if their combined information cannot be organized and exploited.

## Dynamic allocation

The total population represents available neural capacity.

A task may activate only a small subset:

- easy local transformation → few units;
- noisy or ambiguous region → more units;
- large divisible input → many units working on different regions;
- missing discriminating evidence → targeted additional units;
- important hypothesis → independent challenge and verification attempts;
- deterministic task → zero additional neural units when ordinary logic can solve it.

The system should scale compute by need rather than activating the entire parameter population for every request.

The scheduler controls at least four useful dimensions:

- **width** — number of independent attempts;
- **depth** — number of bounded passes on a persistent Work Thread;
- **scope** — which information or regions are inspected;
- **purpose** — explore, progress, challenge, verify, or synthesize.

## Persistent work, disposable workers

Workers do not own long-lived goals or unbounded loops.

Persistent Work Threads hold objectives, evidence, hypotheses, failures, open questions, dependencies, and progress. Workers temporarily execute bounded attempts against those threads.

This allows the runtime to:

- pause work safely;
- rotate a stuck worker while retaining useful progress;
- fork competing approaches;
- merge compatible findings;
- resume work after interruption;
- reassign compute as priorities change.

The intended invariant is:

> **Workers are disposable; useful work is not.**

## Exploration and exploitation

The population should not place all compute behind the current best-known solution.

Some capacity should progress promising work while another share continues to discover alternatives, challenge assumptions, and inspect under-covered possibilities.

The balance changes with evidence:

- early uncertainty favors broad exploration;
- promising branches receive more progression compute;
- important claims receive challenge and independent verification;
- synthesis increases when the remaining search space becomes small;
- a nonzero exploration budget protects against premature convergence.

The first scheduler should use explicit deterministic rules plus structured random exploration rather than another learned controller.

## Evidence, not voting

Worker count is not truth.

If 8,000 units do not observe a decisive fact and 200 units process the only region containing that fact, the smaller group may hold the important evidence.

Therefore aggregation must preserve:

- source provenance;
- local observations;
- unique findings;
- contradictions;
- evidence coverage;
- evidence strength;
- failed and falsified hypotheses where they constrain future search.

Population disagreement may indicate uncertainty or a need for more investigation, but majority vote must not be the final decision rule.

Discovery and verification intentionally use opposite redundancy policies:

> **Diversity for discovery; independent redundancy for verification.**

## Knowledge state

Claims, evidence, and accepted knowledge must remain distinct.

A worker may propose a hypothesis or record an observation. That does not automatically become a global fact.

The runtime should preserve a durable event/history layer from which it can derive current:

- Work Threads;
- hypotheses;
- supporting and contradicting evidence;
- verification status;
- coverage;
- unresolved questions;
- accepted knowledge.

Summaries may compress this state for efficient use, but the original evidence must remain recoverable by stable provenance references.

## Information generation and knowledge integration

At sufficiently large populations, worker compute may not be the dominant scaling constraint.

Many workers can generate candidate information in parallel. The system must still:

- preserve useful observations;
- deduplicate repeated findings;
- connect related evidence;
- detect contradictions;
- verify important claims;
- route changes to affected Work Threads;
- summarize without losing decisive minority evidence;
- turn accumulated evidence into actionable knowledge.

This motivates two separate throughput concepts.

### Exploration throughput

How many meaningfully different evidence-producing attempts can the population execute per unit wall-clock time?

### Knowledge integration bandwidth

How much useful evidence can be converted into persistent, connected, actionable shared knowledge per unit wall-clock time without losing decisive information?

The useful population limit may be reached when marginal workers generate useful information faster than the integration path can absorb it. At that point an integration backlog grows even though more neural compute remains available.

This is a future scaling hypothesis. It is not assumed to be the bottleneck at the current 16-worker scale.

## Hierarchical integration

A large population should not send every raw output to one global reducer.

Prefer a hierarchy:

```text
worker attempts
    -> local/thread evidence
    -> branch/topic integration
    -> cross-topic integration
    -> global knowledge changes
```

Only information that materially changes a higher-level state needs to propagate upward. Raw details can remain local or archived while staying addressable by reference.

The same worker architecture may execute integration Work Threads. The project does not assume a separate large global integration model.

The core scaling rule is:

> **Massive global state must not force massive local context.**

No worker, scheduler decision, or integration step should have to read the complete system merely because the total population is large.

## Knowledge deltas

Information movement should emphasize changes rather than complete retransmission.

Examples:

- a hypothesis was invalidated;
- a new contradiction appeared;
- a source region became covered;
- a replication succeeded or failed;
- an unresolved question became the highest-value discriminator;
- one thread solved a dependency of several others.

The historical record can be large. Active context should remain bounded and purpose-specific.

## Deterministic computation boundary

Learned computation should be used where learning adds value.

Deterministic algorithms should handle tasks such as:

- arithmetic;
- exact comparisons;
- sorting;
- coordinates;
- identity and provenance;
- deduplication;
- state transitions;
- routing rules;
- permissions;
- queue management;
- resource limits;
- schema validation;
- durable event ordering;
- scheduler budgets.

The research should explicitly identify the lower boundary where a neural unit becomes either:

1. mostly noise; or
2. so simple that deterministic logic performs the same transformation more reliably and cheaply.

Worker shrinking is already closed for the current phase; this boundary is reopened only if later population results provide a concrete reason.

## Hardware hypothesis

The architecture should treat CPU and GPU as complementary resources.

The CPU is suited to orchestration, deterministic logic, Work Thread projections, evidence tracking, queues, scheduling, persistence, and data preparation.

The GPU is suited to batched execution of many identical neural units through efficient matrix operations.

The architecture succeeds only if batching, memory movement, evidence transport, and knowledge integration remain cheaper than the value gained from sparse, parallel, and adaptive learned computation.

## Primary hypothesis

> For at least some workloads, a fixed neural parameter and hardware budget can produce more useful information and system-level capability per unit of active compute when organized as a dynamically allocated population of independently weighted, architecturally identical small learned processing units than when organized as one fixed dense network.

A stronger long-term form is also testable:

> Population scale remains useful only while additional possibility coverage produces useful evidence that the information-integration architecture can preserve, verify, connect, and exploit at acceptable cost.

## Null / failure outcomes

The hypothesis should be considered unsupported for a tested configuration when any of the following dominate:

- individual units become too weak and produce mostly noise;
- additional workers add correlated errors rather than useful possibilities;
- additional workers mostly duplicate existing work;
- rare decisive evidence cannot be distinguished from harmful outliers;
- routing and aggregation cost exceeds saved neural compute;
- GPU utilization collapses because workloads are too granular;
- memory movement dominates execution;
- useful evidence is generated faster than it can be integrated;
- integration backlog grows without bound;
- hierarchical compression loses decisive information or provenance;
- verification cannot keep up with consequential claims;
- a conventional dense model reaches better end-to-end quality under the same resource budget.

Negative results are useful: they define the boundary of the architecture and narrow the search for the viable operating region.
