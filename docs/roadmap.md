# Research Roadmap

The primary research objective is now:

> **Determine whether capability scales with population computation while learned parameters remain fixed.**

The project does not proceed to a 1B-scale implementation merely because it is technically possible. Each gate must earn the next one.

## Completed background — tiny local learned transformations

Earlier work established that useful local neural transformations survive in a very small parameter regime.

Observed reference region:

- ~10K parameters: difficult;
- ~25K: smallest strong candidate region;
- ~50K: practical reference;
- ~75K / ~100K: larger references.

That work remains useful, but worker shrinking is no longer the primary goal.

## Gate 0 — Fixed-parameter population scaling — ACTIVE

### Question

> With one frozen learned model, does increasing runtime population computation increase capability?

### Frozen invariants

Across one compared scaling curve keep identical:

- learned parameter count;
- exact parameter fingerprint/checkpoint;
- shared update architecture;
- training data and procedure;
- benchmark worlds;
- decoder;
- hardware;
- compiler/execution mode.

Compiler behavior is recorded but not changed inside the neural comparison.

### Variable resources

Measure and vary only:

- available/active worker states;
- recurrent population rounds;
- communication mode/budget;
- total worker updates.

Runtime state and activation memory are resources and must be recorded.

## Gate 0A — Executable scientific contract — IN PROGRESS

Build the model-independent experiment boundary before training the neural architecture.

Required:

- fixed parameter identity validation;
- deterministic benchmark generation;
- population condition schema;
- exact worker-update accounting;
- communication accounting;
- runtime-state memory accounting;
- explicit no-communication and serial controls;
- frozen development population counts;
- negative-result semantics.

Development counts:

- 1;
- 4;
- 16;
- 64;
- 256.

Current benchmark family: **collective relay**.

The task distributes a multi-hop key/value chain across local contexts among distractors. No worker receives the complete path.

Exit criterion:

- benchmark worlds and accounting are deterministic, validated, and independent of later model tuning.

## Gate 0B — Minimal shared-weight population

Build the smallest architecture that can express the hypothesis.

Allowed:

- one shared neural update cell;
- one bounded state per runtime worker;
- one bounded shared signal;
- recurrent rounds;
- deterministic orchestration;
- homogeneous batching.

Not allowed yet:

- independently trained worker checkpoints;
- specialized worker classes;
- learned global router;
- external knowledge database;
- large language model components;
- hierarchical communication;
- compiler-specific optimization;
- multi-machine execution.

The architecture should be deliberately simple enough that a negative result is interpretable.

Exit criterion:

- model trains and executes correctly;
- parameter count is independent of runtime population size;
- exact parameter fingerprint remains identical across the scaling curve;
- collective-relay outputs are produced through population state rather than an oracle/deterministic resolver.

## Gate 0C — Sparse shared communication v0

Primary communication design:

> `sparse_shared_v0`

Workers exchange only a bounded signal/state through the minimal shared communication mechanism.

Run the development scaling curve at 1, 4, 16, 64, 256 workers.

For each population point record:

- solve rate by difficulty;
- worker updates;
- recurrent rounds;
- messages/signals;
- communicated scalar values/bytes;
- peak worker-state bytes;
- wall time;
- device time where measurable;
- utilization/batching telemetry where measurable.

Matched controls:

1. `no_communication`;
2. `serial_control` with matched worker-update budget.

Exit criterion:

- development results are mechanically valid and sufficient to freeze the neural/training configuration for confirmation;
- no scientific pass claim is made from tuning data alone.

## Gate 0D — Frozen confirmation

Before confirmation:

- freeze architecture;
- freeze training recipe;
- freeze benchmark generation/splits;
- freeze population sizes;
- freeze interpretation thresholds;
- freeze communication budget.

Confirmation uses untouched worlds and at least three independent training seeds.

Provisional per-curve minimum effect, to be frozen before trained development curves are inspected:

- 256-worker solve rate at least +5 percentage points over 1 worker on at least two nontrivial difficulty tiers;
- at least 3/4 adjacent population steps non-decreasing within 1 percentage-point tolerance;
- communicating 256-worker endpoint at least +5 points over matched no-communication on at least one multi-hop tier;
- exact learned parameter identity across every compared point.

These are minimum continuation thresholds, not claims of optimality.

## Gate 0E — One allowed communication rescue

Only if `sparse_shared_v0` fails despite correct mechanics/training, test one additional preregistered communication structure:

> `hierarchical_summary_v0`

This may add:

- bounded local groups;
- bounded group summaries;
- sparse promotion between groups.

It may not change the benchmark objective, learned-parameter budget, or turn workers into complete agents.

No third, fourth, or tenth communication redesign is allowed before a research reset.

## Gate 0 decision

### Continue

Continue if fixed learned machinery shows reproducible positive capability scaling with additional population computation.

### Stop / redirect

Stop or redirect if both communication variants produce essentially flat capability while worker updates and communication increase.

A negative result is successful research because it eliminates this architecture path.

### Bottleneck result

If capability scales but communication/memory/runtime costs grow faster than useful capability, continue only into the information-transport problem. Do not respond by making each worker larger unless evidence requires it.

# Gate 1 — Scaling frontier

Activate only after Gate 0 passes.

Question:

> How long does the positive population-compute curve persist?

Candidate counts, subject to local hardware and batching viability:

- 1,024;
- 4,096;
- 16,384;
- higher counts up to the practical single-machine limit.

Measure marginal capability gain per:

- worker update;
- communicated byte/scalar;
- wall-clock second;
- peak activation/state memory.

Exit criterion:

- identify the useful population-compute region and the first meaningful saturation/bottleneck.

# Gate 2 — Population state vs ordinary extra compute

Question:

> Is population organization useful, or is the result explained by simply spending more recurrent compute?

Compare under matched end-to-end budgets:

- wide population;
- serial/recurrent control;
- conventional recurrent baseline;
- dense baseline where appropriate.

Normalize:

- learned parameters;
- training data;
- worker-update/FLOP budget as closely as practical;
- hardware;
- benchmark worlds.

Exit criterion:

- demonstrate a workload where population organization gives a real capability/resource advantage, or accept that ordinary recurrent/dense compute is better.

# Gate 3 — Dynamic activation

Activate only if fixed-width population organization is useful.

Question:

> Can the system spend population compute according to difficulty without losing the fixed-parameter advantage?

Vary dynamically:

- active workers;
- recurrent depth;
- inspected scope;
- communication budget.

Start with deterministic allocation. Do not add a learned scheduler until traces show a concrete limitation.

Compare adaptive execution against matched fixed-width execution.

Exit criterion:

- harder tasks reliably consume more population compute and improve the capability/resource frontier.

# Gate 4 — Information transport and integration bandwidth

Activate when measurements show communication or integration becoming the limiting resource.

Question:

> How much useful information can the population move and integrate before communication dominates neural computation?

Measure:

- message rate;
- useful-message fraction where measurable;
- communicated bytes/scalars;
- synchronization cost;
- local-to-shared state promotion;
- backlog depth/age if persistent integration is used;
- compression/summary loss;
- rare decisive information retention;
- memory movement.

Preferred progression:

1. bounded shared signal;
2. local neighborhoods/groups;
3. hierarchical summaries;
4. only then more complex routing if measured need exists.

Never broadcast complete global state to all workers merely because it is easy to implement.

Exit criterion:

- demonstrate bounded information movement across the useful population range, or establish communication as the scaling limit.

# Gate 5 — Richer learned workloads

Only after the synthetic substrate works.

Candidate workload families:

- large-document analysis;
- image-region understanding;
- codebase analysis;
- event/log streams;
- anomaly detection;
- multi-stage reasoning/planning;
- research-like hypothesis exploration.

The goal is not benchmark accumulation. Select workloads where distributed weak processing has a plausible structural advantage.

Exit criterion:

- identify at least one non-toy workload where the population-compute mechanism survives real learned ambiguity and still improves the practical resource/capability frontier.

# Gate 6 — Toward the ~1B learned-parameter reference

Only after smaller experiments establish a real advantage.

The goal is not:

> split 1B parameters into thousands of small independent models.

The goal is:

> **use a fixed ~1B learned system as shared machinery for a potentially very large dynamic population of weak neural states.**

At this stage investigate:

- how much of the 1B budget belongs in shared worker machinery;
- whether a sparse specialist parameter bank adds value under a fixed total budget;
- population-state dimensionality;
- memory systems;
- topology/routing;
- dynamic activation;
- compiler optimization as a separate experimental variable.

Exit criterion:

- compare against strong parameter-matched baselines and report the complete cost frontier honestly.

# Systems variables that are not research claims

Profile or optimize these only when needed:

- GPU batching/vectorization;
- CPU/GPU task placement;
- memory layout;
- queueing/backpressure;
- tensor fusion;
- compilation;
- quantization;
- checkpoint/state serialization;
- storage/indexing;
- observability.

Compiler mode must remain a separate variable from neural architecture.

# Scope exclusions

Current research is local to one PC.

Do not spend experiments on:

- geographic distribution;
- multi-machine latency tolerance;
- datacenter networking;
- distributed consensus.

Those are established engineering domains and not current uncertainties.

# Global stop conditions

Stop or redirect a path when reproducible evidence shows that:

- capability does not rise with population compute at fixed learned parameters;
- communication adds no useful capability beyond independent workers;
- population gains are fully dominated by a simpler serial/recurrent control;
- runtime state or communication cost destroys the practical advantage;
- batching/utilization collapses at the required granularity;
- scaling disappears on untouched confirmation worlds;
- richer workloads do not preserve the synthetic advantage;
- strong parameter/resource-matched dense or recurrent baselines are consistently better.

A negative result removes uncertainty and prevents unnecessary engineering work.
