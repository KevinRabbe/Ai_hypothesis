# Evolutionary Organism Optimization — Preserved Future Direction

## Status

**Preserved high-potential direction. Not an active research gate.**

Current project order is Gate 0 completed positive, Gate 1 completed positive on the target GPU, and Gate 2 active. Evolutionary organism optimization remains deferred until the population substrate demonstrates an organization-specific capability/resource phenotype worth optimizing.

This document preserves the unique future direction originally developed on historical PR #85 without making that PR an active implementation dependency.

## Core hypothesis

> **A population-compute organism may reach a higher capability ceiling faster or more reliably if training maintains and evolves multiple candidate organisms instead of relying on one gradient-training trajectory to discover the best shared neural machinery.**

The important distinction is that evolution acts **between candidate organisms**, not between thousands of separately weighted runtime workers.

The deployed organism remains:

- one shared learned parameter set;
- many weak temporary runtime states;
- shared/reused neural update machinery;
- recurrent population computation;
- bounded communication;
- runtime population size independent of learned parameter count.

More runtime workers must never silently mean more learned parameters.

## Two nested search processes

The long-term architecture separates two different search problems.

### Runtime search — problem space

One trained organism spends reusable population compute on:

- source regions;
- hypotheses;
- branches;
- evidence;
- verification paths;
- local transformations;
- alternative decompositions.

The runtime population searches the problem.

### Evolutionary search — organism space

Training maintains multiple candidate shared-weight organisms and searches for better machinery:

- better scaling with runtime compute;
- better communication behavior;
- better persistent-state utilization;
- better generalization;
- better robustness;
- better dynamic-compute behavior;
- better capability/resource frontiers.

Compactly:

```text
evolution searches organism space
              ↓
selected shared-weight organism
              ↓
runtime population searches problem space
```

## Why evolution could help

A single gradient run follows one path through initialization, curriculum, optimizer state, parameter space and training history.

An evolutionary archive can preserve multiple trajectories simultaneously. That matters if a useful population-compute trait is rare, initially mediocre on average metrics, or located behind a training path that ordinary checkpoint selection tends to discard.

The future system should therefore test whether evolutionary lineage search can:

- preserve rare useful traits;
- escape local training optima;
- accumulate improvements across generations;
- discover high-capability organisms sooner;
- directly select for population-level behavior rather than single-worker score.

This remains a hypothesis, not an established result.

## Preferred first form: hybrid neuroevolution

Do not start by randomly evolving every neural weight from scratch.

The preferred first experiment is:

```text
within each lineage:
    gradient-based learning

between lineages:
    selection
    + controlled mutation
    + inheritance
    + diversity preservation
```

A descendant may inherit a trained parent checkpoint, receive a controlled mutation, and continue gradient training.

Gradient descent remains responsible for local parameter improvement. Evolution searches over trajectories and retained organism states.

## Candidate evolutionary loop

A minimal future loop is:

```text
1. maintain K organism lineages
2. train each for a bounded gradient budget
3. evaluate all lineages under the same frozen development protocol
4. record a multi-dimensional phenotype
5. retain Pareto-strong and behaviorally novel lineages
6. select parents
7. create controlled descendants
8. continue bounded gradient training
9. repeat
```

Every generation must preserve immutable provenance:

- organism/lineage ID;
- parent ID or parent IDs;
- exact checkpoint hash;
- architecture/protocol version;
- mutation description;
- training data/curriculum version;
- optimizer configuration;
- training-compute budget;
- evaluation-compute budget;
- phenotype metrics;
- confirmation/test exposure state.

The evolutionary layer must be as auditable as the current frozen Gate protocols.

## What should evolve first

Keep the deployed organism architecture fixed for the first causal experiment.

Preferred initial mutation/search channels:

1. **learned parameter state** — small perturbations or inherited checkpoints followed by recovery training;
2. **training curriculum** — task mixture, difficulty progression, population-width curriculum and recurrent-depth curriculum;
3. **objective mixtures** — task loss, communication/gating supervision and robustness objectives;
4. **training process** — optimizer settings, learning-rate schedules, mutation strength and bounded lineage training duration;
5. **learned communication/gating components** already inside the frozen organism contract.

Defer arbitrary topology evolution, specialized worker types and unrestricted architecture search. Those would confound the first evolutionary comparison.

## System-level phenotype

An organism should be selected for the behavior of the complete population-compute system, not for the standalone performance of one weak runtime state.

The phenotype should remain a vector rather than one opaque scalar.

### Capability dimensions

- exact solve/accuracy on frozen workloads;
- population scaling curve;
- recurrent-compute scaling curve;
- performance under incomplete information;
- held-out difficulty generalization;
- robustness across workload families.

### Resource dimensions

- learned worker updates;
- latency;
- throughput;
- work/span behavior;
- peak runtime-state memory;
- communication volume;
- synchronization cost;
- capability under matched practical compute budgets.

### Population-behavior dimensions

- marginal value of additional runtime states;
- usefulness of persistent distributed state;
- locality retention;
- communication efficiency;
- robustness to noisy/unhelpful states;
- dynamic-activation potential.

A Pareto frontier is the default representation. Do not collapse all dimensions into one arbitrary fitness number unless later evidence justifies it.

## Preserve niches and lineage diversity

Pure winner-take-all selection risks homogenizing the archive around one local optimum.

A serious evolutionary system should preserve:

- Pareto elites;
- behaviorally novel lineages;
- niche specialists;
- a small random-survivor fraction;
- mutated descendants;
- occasional fresh independent seeds.

Possible diversity descriptors include:

- shape of the population scaling curve;
- workload families solved;
- communication statistics;
- failure modes;
- sensitivity to incomplete source scope;
- preferred compute depth;
- resource profile;
- learned representation/output behavior where measurable.

The training-time analogue of permanent structured exploration is a non-zero evolutionary exploration budget.

## Training compute is not deployed parameter budget

Evolution may spend substantial offline search compute across many candidate organisms.

That does not change the deployed parameter constraint:

```text
candidate A: P learned parameters
candidate B: P learned parameters
candidate C: P learned parameters
...
selected deployed organism: P learned parameters
```

However, evolutionary training must not be allowed to win merely by spending more total optimization compute.

A fair causal comparison must record and eventually match total training compute.

## First clean evolutionary research question

> **Under matched total training compute and the same fixed organism architecture and learned-parameter count, does maintaining an evolutionary archive of gradient-trained lineages discover organisms with a better held-out population-compute capability/resource frontier than independent ordinary training runs?**

Freeze for the comparison:

- organism architecture;
- learned parameter count;
- aggregate training compute;
- training-data access;
- development/confirmation split policy;
- evaluation workloads;
- runtime population ladder;
- resource-measurement protocol.

### Baseline

Independent ordinary gradient-training runs with standard checkpoint selection.

### Evolutionary condition

The same aggregate training budget divided across inherited/mutated lineages with multi-objective selection and diversity preservation.

Primary outcomes:

- best untouched-confirmation organism frontier;
- distribution of confirmation frontiers across repeated runs;
- compute required to first reach predefined development capability levels;
- best-so-far capability versus cumulative training compute.

This tests both **ceiling** and **discovery speed**.

## Capability-discovery speed

Useful development curves include:

```text
best held-out development capability
vs
cumulative training compute
```

After protocol freeze, confirmation reports can additionally compare the selected organism against the training compute spent before selection.

Potential descriptive metrics:

- compute-to-threshold;
- wall-clock-to-threshold;
- best frontier after fixed training compute;
- area under the best-so-far capability curve;
- number of meaningfully distinct useful lineages discovered.

Do not tune confirmation thresholds after seeing confirmation data.

## Later extension: evolving challenge environments

Only after the basic evolutionary-organism hypothesis earns continuation, a more open-ended system may coevolve development challenges/curricula with organism lineages.

Conceptually:

```text
organism archive
      ↕
development challenge search
```

As organisms master current development tasks, challenge search may identify new valid environments that expose weaknesses.

Hard boundaries:

- challenge evolution may influence development/training only;
- frozen confirmation/test data remain external and untouched;
- novelty is not itself capability;
- generated challenges must remain valid tasks rather than exploit benchmark bugs;
- test data must never become evolutionary feedback.

This is deliberately deferred because it carries large Goodhart and leakage risks.

## Other deferred coevolution targets

Only after a clean positive evolutionary result consider separately:

- scheduler heuristics;
- dynamic-compute stopping policies;
- evidence-integration behavior;
- local memory/update rules;
- communication topology;
- challenge generators.

Do not coevolve all of these simultaneously. Causal interpretation would collapse.

## Crossover is optional

Direct neural-parameter crossover is not required for the core hypothesis and may be destructive because independently trained networks can implement similar functions with incompatible internal parameter permutations.

Start with:

- checkpoint inheritance;
- controlled mutation;
- gradient adaptation;
- lineage selection.

Test crossover only after a concrete representation/alignment method provides a reason to expect useful inheritance.

## Failure modes

### Homogenization

All lineages converge to effectively the same organism.

Mitigation: novelty/niche preservation and fresh-seed injection.

### Fitness overfitting

Lineages exploit development benchmarks without becoming generally better.

Mitigation: multiple workload families, strict split discipline and untouched confirmation.

### Extra-compute confound

Evolution wins only because it receives more optimization work.

Mitigation: matched aggregate training-compute controls and compute-to-threshold reporting.

### Scalar-fitness Goodharting

A single score rewards a pathological shortcut.

Mitigation: multi-objective/Pareto phenotype and explicit causal controls.

### Rare-lineage loss

A currently weak but uniquely useful lineage is eliminated too early.

Mitigation: niches, archive diversity and permanent exploration budget.

### Architecture confounding

Architecture, parameter count, curriculum, scheduler and evolutionary policy all change together.

Mitigation: freeze the organism architecture in the first experiment.

### Runaway cost

Evolution multiplies training cost beyond practical value.

Mitigation: bounded lineage count, bounded per-generation gradient budget, explicit compute accounting and development-only early pruning rules.

## Activation condition relative to current gates

Evolutionary organism optimization should not interrupt Gate 2.

Activate it only when the project has evidence that there is a meaningful population-organization phenotype worth optimizing — for example a confirmed Gate-2 organization-specific capability/resource advantage or another equally strong substrate result.

Until then, the direction remains preserved but inactive.

## Long-term synthesis

The highest-ceiling version of the project has nested adaptive timescales:

```text
EVOLUTIONARY TIMESCALE
candidate shared-weight organisms
compete / mutate / inherit / preserve niches
              ↓
TRAINED ORGANISM
one shared learned genome
              ↓
RUNTIME TIMESCALE
many weak neural states dynamically activated
              ↓
PROBLEM SEARCH TIMESCALE
states process local information / hypotheses / evidence
              ↓
PERSISTENT OR EXTERNAL STATE WHEN JUSTIFIED
useful discoveries survive beyond one local computation
```

In compact form:

> **Evolution improves the machinery; population computation reuses that machinery; runtime control directs compute; durable information mechanisms are added only when measured workloads require them.**

This is a preserved future hypothesis, not a proven result.
