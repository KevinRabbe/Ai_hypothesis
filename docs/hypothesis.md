# Neural Population Compute Hypothesis

## Problem statement

Conventional neural models usually bind learned parameters, runtime state, and computation structure tightly together. A model with more stored parameters generally has more learned capacity, while additional test-time computation is often limited to longer sequences, more generated tokens, or repeated passes through essentially the same architecture.

This project asks whether those quantities can be separated more aggressively.

The target is one computational organism with:

- a fixed learned parameter set;
- a potentially very large population of temporary neural worker states;
- shared/reused learned update machinery;
- sparse information flow between workers;
- recurrent computation;
- dynamic activation according to problem difficulty.

The individual worker is not intended to be a complete model or autonomous agent. It may know almost nothing. Useful capability may exist mainly in the collective dynamics.

## Primary hypothesis

> **For at least some workloads, a fixed learned-parameter budget can produce increasing system-level capability as additional population computation is made available through many weak runtime neural states, recurrent updates, and bounded communication.**

The immediate question is intentionally narrower than proving superiority over a conventional 1B model:

> **At fixed learned parameters, does capability rise when population compute rises?**

If not, the architecture does not earn larger-scale investment.

## What an artificial worker is

A worker is best treated as a temporary computational state, not a miniature AI.

A minimal worker may contain:

- local input/context;
- a bounded hidden state;
- a shared neural update function;
- a bounded output/message;
- optional local history represented only in its state.

The learned update machinery may be identical across thousands of worker states.

This means:

> **runtime population size != learned parameter count**

A system may contain 100,000 active or available worker states while still reusing a much smaller fixed set of learned weights.

## Why ants are relevant but not prescriptive

Ant colonies demonstrate that strong collective behavior does not require every component to possess a global model of the problem.

The useful lesson is:

> **weak local processors can participate in strong global behavior while each processor sees and communicates only a small amount of information.**

The project does not attempt to copy pheromones, castes, nests, or biological ant algorithms literally.

Artificial systems can use mechanisms biology does not have, including exact numeric messages, learned routing, differentiable state updates, GPU batching, deterministic algorithms, and explicit memory structures.

## Central scaling idea

The architecture tries to turn additional runtime resources into additional capability without adding learned weights.

Conceptually:

```text
fixed learned weights
        |
        v
shared neural update machinery
        |
        +---- worker state 1
        +---- worker state 2
        +---- worker state 3
        +---- ...
        +---- worker state N

worker states <-> bounded communication
        |
        v
population-level computation
```

A difficult problem may activate more states or more recurrent rounds than an easy problem.

The system therefore has at least three independent resource axes:

1. **learned parameters** — stored learned machinery;
2. **runtime population state** — temporary memory/capacity distributed across workers;
3. **population compute** — worker updates, recurrent rounds, communication, and deterministic orchestration.

The research must measure all three rather than treating runtime state or communication as free.

## Why shared weights are now the default

Earlier project work explored homogeneous small workers with independently trained weights. That remains useful evidence about how small a local learned transformation can become, but independent checkpoints make total learned capacity grow with population size.

That is not the cleanest test of the new question.

The primary population-compute experiment therefore uses **shared/reused learned machinery** so the learned parameter count and exact parameter fingerprint can remain identical while population size changes.

Independent specialist parameters may be revisited later only if a fixed-budget comparison justifies them.

## Communication hypothesis

The likely limiting problem is not raw multiplication throughput alone. It is information transport.

A population of 100,000 workers cannot afford to send every worker state to every other worker.

The architecture should therefore prefer:

- bounded local reads/writes;
- small shared signals;
- sparse routing;
- decaying/replaceable working state;
- hierarchy only when flat communication stops scaling;
- summaries or deltas instead of complete histories.

The core systems rule is:

> **Do not solve the large-population communication problem by moving all information. Design the computation so most information never needs to move.**

## Population state as computation

Information may be represented not only in explicit messages, but also in:

- which workers are active;
- worker hidden states;
- local specialization that emerges during a problem;
- signal strengths;
- activation/recruitment patterns;
- competing or inhibited state trajectories;
- topology of temporary connections;
- recurrent persistence over time.

The population is therefore not merely a set of processors connected by plumbing. The evolving population state may itself be part of the representation.

## Recurrent computation

The same learned parameters can be reused repeatedly.

This gives a second way to increase capability without increasing stored parameters:

```text
population state t0
      -> shared update
population state t1
      -> shared update
population state t2
      -> ...
```

The experiment must distinguish:

- wider population state;
- deeper recurrent processing;
- total worker-update budget.

A serial compute control is required so a population gain is not automatically interpreted as evidence for width when the same benefit could come from simply spending more recurrent FLOPs.

## First benchmark requirement

The first benchmark should force information integration across local processors.

The initial family is **collective relay**:

- a world contains local key/value relations and distractors;
- a query starts from one key;
- the answer requires following a multi-hop chain;
- required links are distributed across worker-local contexts;
- no worker receives the complete chain;
- communication and recurrent depth are required to propagate intermediate state.

This benchmark is intentionally synthetic. It tests the computational substrate before language, world knowledge, or broad benchmark memorization can obscure the result.

## Controls

A population result is uninterpretable without controls.

### No communication

Run the same number of local workers without inter-worker information flow.

This measures independent coverage/attempt benefits.

### Sparse communication

Run the primary architecture with bounded shared information flow.

This measures whether population interaction adds capability beyond isolated workers.

### Serial compute control

Spend a matched worker-update budget through a small number of recurrent states.

This measures whether the important resource is genuinely population state/organization or simply additional test-time compute.

## Fixed-parameter experimental invariant

Every point on one population-scaling curve must share:

- exact learned parameter count;
- exact parameter fingerprint;
- model architecture;
- training data/procedure;
- benchmark worlds;
- decoder;
- hardware;
- compiler/execution mode.

Changing any of these creates a different experiment.

Compiler optimization remains a separate systems variable.

## Primary measurement

The primary graph is:

> **capability vs population compute at fixed learned parameters**

Secondary measurements include:

- solve rate by task difficulty;
- active/available worker count;
- recurrent rounds;
- total worker updates;
- communication messages/scalars/bytes;
- peak runtime worker-state memory;
- wall time and device time;
- utilization/batching efficiency.

A population architecture is not successful merely because a larger count can be executed.

## Falsification rule

The architecture must be allowed to fail.

Only two communication designs are preregistered before a kill/redirect decision:

1. a minimal sparse shared-signal design;
2. one hierarchical-summary design if the first fails despite correct mechanics/training.

If capability remains essentially flat as population compute increases under both variants, while compute and communication costs rise, the weak-unit population-compute hypothesis is considered unsupported for this benchmark family.

The response is to stop or redirect, not to add unlimited routing complexity until a graph appears.

## Positive result interpretation

A positive population curve would establish only that:

> **fixed learned machinery can convert additional population runtime state/compute into additional capability on the tested task.**

It would not yet prove:

- superiority over a dense model;
- superiority per FLOP;
- superiority per joule;
- language intelligence;
- general intelligence;
- scaling to 1B learned parameters;
- scaling to 100K workers.

Those are later questions.

## Long-term 1B reference

Approximately 1 billion learned parameters remains a useful long-term reference budget because it is large enough to compare with practical small language-model regimes while remaining potentially local-hardware relevant.

The intended question is not how to split 1B parameters into 10,000 independent models.

It is:

> **How much effective computational intelligence can one fixed ~1B-parameter learned system produce when it can instantiate, route, and recurrently update a very large population of weak neural states?**

Population size may eventually reach thousands or tens of thousands if the measured scaling curve and local hardware behavior justify it.

## Hardware boundary

Initial research is single-machine only.

Do not spend experiments simulating geographic or multi-machine latency tolerance. Distributed hardware is not a current research question.

The relevant local measurements are:

- batching;
- CPU/GPU orchestration;
- memory behavior;
- communication/state movement;
- activation sparsity;
- recurrent depth;
- compiler effect as a separate variable.

## Relationship to previous worker-size work

The previous worker-size experiments remain useful background:

- ~10K parameters became difficult;
- ~25K was the smallest strong candidate region;
- ~50K was a practical reference worker;
- larger 75K/100K workers provided reference points.

Those results answered a different question: whether a tiny learned network can perform a useful local transformation.

The new primary objective is not to keep shrinking that worker.

The new objective is to test whether **population computation itself scales intelligence at fixed learned parameters**.

## Research value

The project treats negative results as progress.

If the scaling graph fails, we have learned that this is not the mechanism to pursue under the tested conditions.

If it succeeds, the next work is to identify **why** it succeeds, how long the scaling persists, what communication structure is actually necessary, and whether the advantage survives fair compute/resource comparisons against simpler recurrent or dense alternatives.
