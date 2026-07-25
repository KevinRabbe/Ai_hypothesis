# Population Runtime Architecture

## Purpose

This document defines the current target architecture for turning a population of small learned workers into one coherent system.

The design is intentionally smaller than a conventional multi-agent platform. Workers are not autonomous agents, do not own long-lived goals, and do not communicate with each other directly. The runtime owns work, memory, evidence, scheduling, verification, and resource allocation.

The core causal hypothesis is:

> More genuinely different possibilities explored -> more useful evidence generated -> more important uncertainty removed -> stronger system-level knowledge and decisions.

The architecture succeeds only if the system can preserve and exploit the additional information faster than population growth creates unusable coordination and integration load.

## Architecture freeze rule

The project adopts the following implementation rule:

> Final architecture from day 1; minimal implementation from day 1.

The long-lived component roles and semantic contracts are fixed early. Scale may change storage engines, indexing, batching, partitioning, caching, integration fan-out, scheduling heuristics, and deployment topology, but those changes should happen behind the existing boundaries whenever possible.

The stable contracts are defined in [`architecture_contracts.md`](architecture_contracts.md).

A new top-level component, worker communication mode, scheduler dimension, or persistence model requires measured evidence that the existing contracts cannot express or efficiently remove a real bottleneck. Hypothetical future scale is not sufficient reason to redesign the architecture.

## Core invariants

1. **Identical architecture, independently learned weights.** Workers remain homogeneous enough for efficient batching and interchangeable execution, while independent training/initialization provides functional diversity.
2. **Workers are disposable; work is persistent.** A worker may stop, fail, rotate, or be replaced without destroying useful progress.
3. **Work Threads own loops.** No worker owns an unbounded reasoning loop. The runtime repeatedly assigns bounded attempts to persistent work.
4. **Workers produce evidence, not votes.** Population count is not truth.
5. **Claims are not facts.** Hypotheses, observations, contradictions, verification results, and accepted knowledge remain distinct states.
6. **Failure is information.** A failed attempt is valuable when it rules out a possibility, exposes a contradiction, identifies a boundary condition, or prevents repeated work.
7. **Diversity for discovery; redundancy for verification.** Duplicate exploration is usually wasteful, while independent replication is intentionally valuable during verification.
8. **Global cooperation, local competition.** Workers do not compete to appear correct. Runtime allocation favors attempts that make useful marginal contributions to the shared search.
9. **Deterministic computation stays deterministic.** Exact routing, IDs, arithmetic, deduplication, state transitions, budgets, persistence, and scheduling are ordinary software unless a learned component proves necessary.
10. **Massive global state, bounded local context.** No worker, scheduler decision, or integration step should require the whole system state merely because the total population is large.
11. **Preserve provenance through compression.** Summaries may be lossy for convenience, but decisive source evidence must remain recoverable by reference.
12. **Integrate knowledge changes, not complete worker histories.** The runtime should propagate compact deltas that materially change what is known.

## Minimal system

The first serious runtime should contain only five major pieces:

1. **Worker Bank** - frozen homogeneous worker architecture with independently learned checkpoints.
2. **Research Ledger** - append-only durable record of work and knowledge-changing events.
3. **State Projector** - derives current Work Threads, evidence, hypotheses, coverage, contradictions, and status from the ledger.
4. **Scheduler v0** - deterministic priority policy plus a permanent structured-random exploration budget.
5. **Worker Runtime** - creates bounded worker attempts, batches neural execution, provides purpose-specific context views, and commits results back to the ledger.

Reducer and verification logic are part of the evidence/integration path rather than separate autonomous agents.

## Research Ledger

The Research Ledger is the canonical persistent source of truth.

Separate blackboard, handoff, failure-memory, checkpoint-history, and progress-history databases are unnecessary in the initial design. They are projections of the same event history.

Example event types:

- `THREAD_CREATED`
- `ATTEMPT_STARTED`
- `OBSERVATION_RECORDED`
- `EVIDENCE_ADDED`
- `HYPOTHESIS_PROPOSED`
- `HYPOTHESIS_REJECTED`
- `CONTRADICTION_FOUND`
- `VERIFICATION_PASSED`
- `VERIFICATION_FAILED`
- `THREAD_FORKED`
- `THREAD_MERGED`
- `THREAD_PAUSED`
- `THREAD_COMPLETED`

The ledger should preserve immutable IDs and source references so summaries can always lead back to the underlying evidence.

For the first local implementation, a simple durable local store is preferred over distributed infrastructure. Scale-specific storage systems are introduced only when measurements require them.

## Work Threads

A Work Thread represents persistent cognitive work independently of whichever worker currently executes it.

A projected thread should contain at least:

- objective;
- purpose;
- current status;
- source references;
- relevant evidence;
- active and rejected hypotheses;
- contradictions;
- open questions;
- failed approaches;
- dependencies;
- parent/child relationships;
- compute spent;
- last meaningful progress;
- priority metadata.

A worker receives a bounded view of the thread, performs one bounded attempt, and returns structured changes.

### Pause and handoff

Stopping a worker must not destroy the thread.

A safe pause is:

1. finish or roll back the current atomic external action;
2. flush new evidence and side-effect records;
3. commit the attempt result;
4. release temporary resources and ownership;
5. leave the thread resumable.

A different worker may later reconstruct the current Thread View and continue.

The system preserves useful conclusions and failures, but it does not need to preserve the exact hidden neural trajectory that caused a worker to become stuck.

This gives the desired property:

> Thread continuity + worker discontinuity.

## Bounded looping

Looping belongs to Work Threads, not workers.

After each bounded attempt, the scheduler chooses among actions such as:

- `CONTINUE` - another pass on the same line of work;
- `ROTATE_WORKER` - preserve thread state but use a fresh independently weighted worker;
- `ADD_WIDTH` - add independent attempts;
- `FORK` - pursue multiple plausible branches;
- `CHALLENGE` - search specifically for counterevidence;
- `VERIFY` - independently replicate or check a claim;
- `PAUSE` - preserve work without spending current compute;
- `COMPLETE` - close the thread.

Stagnation is a runtime signal. Repeated attempts that add no novel evidence, eliminate no possibilities, resolve no questions, and reduce no important uncertainty should lose priority or trigger worker rotation rather than consuming an unbounded loop.

## Purpose-specific worker views

Different worker roles do not require different model architectures. The same worker can receive different context views.

### `EXPLORE`

Goal: increase possibility coverage.

Expose the objective, relevant source material, coverage metadata, and constraints while hiding dominant conclusions when independence is important.

### `PROGRESS`

Goal: advance the strongest current branch.

Expose the current hypothesis, supporting evidence, unresolved dependencies, and next blocking questions.

### `CHALLENGE`

Goal: attack an important hypothesis.

Expose the claim and supporting evidence, then ask for counterexamples, violated assumptions, missing cases, or alternative explanations.

### `VERIFY`

Goal: independently reproduce or invalidate a claim.

Expose the claim and source references but avoid worker identity, popularity, or vote counts that could bias the attempt.

### `SYNTHESIZE`

Goal: compress already established evidence into a coherent higher-level representation while retaining provenance references.

These purposes may coexist at the same time. They are not rigid global phases.

## Scheduler v0

The scheduler is the executive-control layer, but it should not begin as another learned model.

Scheduler v0 should use explicit metadata and simple policies.

Useful factors include:

- objective importance;
- unresolved uncertainty;
- contradiction severity;
- missing coverage;
- estimated novelty;
- downstream dependency impact;
- recent progress rate;
- verification need;
- starvation/age;
- estimated cost.

The scheduler should reserve a permanent exploration share so a currently dominant approach cannot absorb all compute.

Exploration should be **structured random**, not uniform random. Prefer under-covered regions, under-tested hypotheses, alternative decompositions, or under-used workers, then randomize among plausible choices.

As evidence becomes stronger, allocation can move from broad exploration toward progression, challenge, verification, and synthesis without reducing discovery to exactly zero unless the problem is genuinely closed.

## Local competition without leaderboards

Workers do not need explicit scores, rank displays, or awareness of competition.

The useful selection pressure is resource allocation:

- attempts that repeatedly produce duplicated, irrelevant, or non-progressing output receive less future compute;
- attempts/workers that produce novel evidence, important contradictions, efficient eliminations, or successful independent verification receive more relevant opportunities;
- a nonzero exploration probability prevents premature elimination of unusual workers or approaches.

The scheduler should optimize marginal contribution, not individual worker accuracy alone.

## Information generation versus knowledge integration

At large populations, neural execution may stop being the dominant scaling problem.

A population can generate candidate observations, hypotheses, failures, and evidence approximately in parallel. The shared system must still preserve, connect, deduplicate, verify, route, and exploit the useful subset.

Define two distinct throughput concepts:

### Exploration throughput

How many meaningfully different possibilities or evidence-producing attempts can be executed per unit wall-clock time?

### Knowledge integration bandwidth

How much useful evidence can be successfully incorporated into persistent, connected, actionable shared knowledge per unit wall-clock time without losing decisive minority information?

The useful population limit is reached when adding workers increases useful information production faster than the runtime can preserve and exploit that information. At that point the system may accumulate an integration backlog even though more neural compute remains available.

This is a population-scaling failure, not a worker-compute failure.

## Integration hierarchy

The architecture should not route every raw worker output to one global reducer.

Prefer progressive integration:

```text
worker attempts
    -> local/thread evidence
    -> topic or branch integration
    -> cross-topic integration
    -> global knowledge changes
```

The same worker architecture may perform integration Work Threads; a separate large integration model is not assumed.

Most output should remain local or archived when it does not change higher-level knowledge.

A result propagates farther when it is, for example:

- genuinely novel;
- a contradiction to an important accepted hypothesis;
- a new boundary condition;
- a successful or failed independent replication that changes confidence materially;
- relevant to multiple downstream threads;
- a large reduction in important uncertainty.

## Knowledge deltas

Workers and downstream integrators should consume compact changes whenever possible.

Instead of repeatedly transmitting a complete research state, communicate events such as:

- evidence `E91` was invalidated;
- hypothesis `H17` gained a new independent contradiction;
- region `R42` is now covered;
- test `T51` became the highest-value discriminator;
- thread `W8` was closed because another branch solved its objective.

Large source data and historical evidence remain addressable through stable references rather than being copied into every context.

## Compression rule

The system is allowed to compress summaries aggressively only when the underlying evidence remains recoverable.

A summary may say:

> H17 has substantial support but remains unresolved because of contradiction E882.

It must still retain references that allow a worker to retrieve the evidence, source region, experiment, or attempt that produced `E882`.

This protects rare decisive evidence from being averaged or summarized away.

## Scaling metrics

Population experiments should distinguish at least:

- raw attempt throughput;
- distinct/novel attempt rate;
- raw evidence production rate;
- unique useful evidence rate;
- knowledge-changing event rate;
- evidence duplication rate;
- evidence rejection/error rate;
- integration latency;
- integration backlog depth/age;
- knowledge integration bandwidth;
- rare-evidence retention;
- provenance recoverability;
- compute spent per useful uncertainty reduction;
- scheduler overhead;
- memory and communication volume.

No single scalar is assumed to capture the full system frontier.

## Width, depth, scope, and purpose

The scheduler ultimately controls four high-level allocation dimensions:

- **width** - how many independent attempts are allocated;
- **depth** - how many bounded passes a Work Thread receives;
- **scope** - which source information or regions are examined;
- **purpose** - whether the attempt explores, progresses, challenges, verifies, or synthesizes.

Additional control dimensions should not be added until a concrete workload demonstrates that these four are insufficient.

## Scaling rule

Do not scale the population merely because more workers can be executed.

Scale while marginal workers continue to generate useful information that the integration path can absorb at acceptable cost.

A larger population fails when any of the following dominate:

- additional workers mostly duplicate existing work;
- correlated workers stop adding meaningful possibility coverage;
- useful evidence is produced faster than it can be integrated;
- integration backlog grows without bound;
- hierarchical summaries lose decisive evidence;
- verification cannot keep pace with important claims;
- scheduler/integration overhead dominates useful work;
- final capability stops improving under equal end-to-end budgets.

## Relationship to the current Step 2A experiment

The large-scale integration architecture is not the active experiment yet.

Step 2A asks a smaller prerequisite question: the current 16-worker population already contains additional correct minority information, but reducer-v0 often fails to convert it into the final class. The project must first prove that rare useful evidence can be distinguished from noisy minority outliers.

That experiment is directly relevant to future integration because a hierarchical large-population runtime is useless if compression or aggregation discards the rare decisive result.

Therefore:

1. finish the minority-rescue diagnostic;
2. freeze the evidence-utilization mechanism if it works, or repair the evidence representation if it does not;
3. measure useful population width and information-production characteristics;
4. build the minimal persistent runtime only when the next experiment requires it;
5. introduce hierarchical integration only when measured evidence volume begins to justify it.

The project should always prefer the smallest mechanism that removes the next demonstrated bottleneck.
