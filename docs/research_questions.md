# Research Questions

## Primary question

At a fixed hardware and parameter budget, can a population of practical small learned workers produce more useful capability per unit of active compute than a conventional fixed dense model?

The population mechanism being tested is:

> **more genuinely different possibilities explored -> more useful evidence generated -> more important uncertainty removed -> stronger system-level knowledge and decisions**

The project does **not** treat established engineering capabilities as research questions. Batching, multi-device execution, information tiling, coordinate/source tracking, deterministic routing, durable persistence, compiler fusion, and CPU/GPU task placement are implementation tools whose costs are profiled when used. Experiments are reserved for uncertainties whose answers depend on this population architecture.

## Closed exploration — worker shrinking

The project has already pushed the worker architecture into the small-parameter regime. Further shrinking is not a goal by itself.

For the current phase:

- ~50K is the frozen reference Worker v1 used by Step 2A;
- ~25K / ~50K / ~75K remain useful worker-size candidates for later equal-budget organization comparisons;
- smaller sizes are revisited only if population results provide a concrete reason.

The relevant worker question is practical capability, not minimum parameter count: useful output, diversity, latency, throughput, memory, and total system cost must all remain acceptable.

## RQ1 — Evidence utilization and minority rescue — ACTIVE

When additional workers contain correct evidence that population mean evidence suppresses, can the runtime identify and use that evidence without trusting noisy minority outliers?

Measure separately:

- whether the correct alternative is present in the population;
- whether inference-visible evidence proposes it as a candidate;
- whether a gate can distinguish useful minority candidates from harmful outliers;
- rescued errors;
- harmed correct decisions;
- net accuracy/capability gain;
- evidence provenance retention.

The current 16-worker checkpoint already establishes a substantial oracle-any-correct gap, so this question comes before further population expansion.

This is also a prerequisite for future large-population integration: hierarchical aggregation cannot scale safely if rare decisive evidence is destroyed at small width.

## RQ2 — Useful population-width region

With Worker v1 and an evidence-utilization mechanism held fixed, how does useful information and final output change as population width increases?

Candidate widths include:

- 1;
- 4;
- 16;
- 64;
- 256 only when smaller widths justify it.

Measure:

- final task quality;
- oracle-any-correct coverage;
- unique findings;
- minority/rare evidence opportunities;
- functional correlation and shared failure modes;
- duplicate-attempt/evidence rate where measurable;
- raw evidence production rate;
- useful/knowledge-changing evidence rate;
- aggregation/integration latency;
- latency and throughput;
- end-to-end coordination cost.

Worker diversity is measured here as a population property rather than treated as a separate research program.

The purpose is not to reach the largest possible width. It is to locate the region where additional workers still add enough useful possibility coverage to justify their full system cost.

## RQ3 — Fixed-budget worker organization

Under an equal total learned-parameter and hardware budget, which homogeneous worker size gives the best system-level result?

Initial candidates:

- ~25K workers;
- ~50K workers;
- ~75K workers.

Compare populations with as closely matched total parameter budgets as practical. The objective is to find the practical organization sweet spot, not the smallest individual worker.

Measure both individual-worker quality and population properties such as unique evidence, functional diversity, evidence utilization, and integration cost.

## RQ4 — Population versus dense baseline

Under normalized training data, parameter budget, hardware, runtime budget, and evaluation data, does the best population organization outperform a conventional dense baseline on any meaningful capability/resource frontier?

Measure:

- final task quality;
- useful information extracted;
- evidence recall;
- unique useful evidence;
- latency;
- throughput;
- RAM and VRAM;
- active learned parameters;
- end-to-end coordination/integration overhead;
- compute and energy efficiency where measurable.

This is the decisive test of the central small-scale hypothesis.

## RQ5 — Adaptive allocation

Once a useful fixed-width population exists, can the runtime allocate width, depth, scope, and purpose dynamically so that the same end-to-end budget produces more useful information?

The baseline scheduler should remain simple and inspectable:

- deterministic priorities for importance, uncertainty, contradiction, missing coverage, progress, verification need, and cost;
- a permanent structured-random exploration budget to preserve alternative possibilities;
- bounded Work Thread attempts rather than worker-owned infinite loops;
- worker rotation when progress stagnates;
- diversity during discovery and intentional redundancy during verification.

Candidate triggers for additional work include:

- unresolved contradiction;
- missing discriminating evidence;
- low signal quality;
- disagreement;
- local information density;
- failure of an earlier processing round;
- under-covered regions or hypotheses;
- consequential claims requiring independent verification.

Compare adaptive execution against fixed-width execution under equal end-to-end budgets.

A learned scheduler is **not** a prerequisite. It becomes justified only if accumulated traces show a concrete limitation of the deterministic baseline.

## RQ6 — Knowledge integration bandwidth — SCALE-DEPENDENT

When the population produces substantial useful information in parallel, how much of that information can the runtime preserve, connect, verify, route, and exploit before integration becomes the dominant scaling bottleneck?

This question should not become an active engineering program at the current 16-worker scale unless measurements show an actual bottleneck.

Distinguish:

- **exploration throughput** — meaningfully different evidence-producing attempts per unit wall-clock time;
- **knowledge integration bandwidth** — useful evidence incorporated into persistent, connected, actionable shared knowledge per unit wall-clock time without losing decisive minority information.

Measure:

- raw evidence production rate;
- unique useful evidence production rate;
- knowledge-changing event rate;
- duplication rate;
- evidence rejection/error rate;
- integration latency;
- integration queue/backlog depth and age;
- rare-evidence retention;
- provenance recoverability;
- summary/compression loss;
- verification throughput;
- communication and memory volume;
- compute spent per useful uncertainty reduction.

The key scaling failure condition is:

> marginal workers generate useful information faster than the integration path can absorb it, so the backlog grows or decisive evidence is lost even though additional neural compute remains available.

If that limit appears, test the smallest sufficient hierarchical integration design:

```text
worker attempts
    -> local/thread evidence
    -> branch/topic integration
    -> cross-topic integration
    -> global knowledge changes
```

Do not assume a separate large integrator model. The same homogeneous workers may execute integration Work Threads if learned integration is required.

## RQ7 — Real-workload usefulness

Which real workloads actually benefit from the architecture?

Candidates include:

- large-document analysis;
- image-region understanding;
- logs and event streams;
- codebase analysis;
- anomaly discovery;
- multi-stage planning;
- research-like search where many plausible approaches can be explored and falsified.

Information partitioning, coordinates/source pointers, overlap, zoom-in, persistent Work Threads, and hierarchical reduction are not discoveries by themselves. They are implementation techniques.

What must be measured is whether the complete population system:

- explores useful possibilities that a single-stream baseline misses;
- preserves decisive evidence and failed-path information;
- converts that evidence into better decisions or knowledge;
- keeps active local context bounded as total global work grows;
- maintains a useful capability/resource frontier after integration costs are counted.

## RQ8 — Scaling after advantage

Only if a smaller system demonstrates a reproducible advantage: does that advantage survive as total learned capacity, active population, workload size, and information volume increase?

Scaling toward approximately 1B total learned parameters is a later reference point, not an assumption and not a current milestone. Larger populations are also conceptually possible, but worker count has no value by itself.

Scale only while:

- marginal workers still add meaningful possibility coverage;
- the extra coverage produces useful evidence;
- the evidence-utilization path preserves rare decisive findings;
- knowledge integration keeps up without unbounded backlog;
- end-to-end capability continues to improve at acceptable cost.

## Systems ablations that are not independent research questions

These are implementation/profiling axes. Measure them when the architecture needs them; do not create standalone existence experiments.

### Compiler/execution mode

Compare the same workers, workload, population policy, and hardware under execution modes such as:

- ordinary eager/runtime orchestration;
- batching/vectorization;
- compiled/fused execution;
- compiled memory/scheduling improvements where available.

Compiler gains must not be attributed to the neural architecture itself.

### Persistent runtime

When required, use the smallest durable runtime that supports:

- Work Thread persistence;
- append-only evidence/work events;
- pause/resume/handoff;
- fork/merge relationships;
- provenance;
- bounded queues and backpressure.

A distributed database or multi-machine runtime is not justified until local measurements require one.

## Measurements that apply across questions

These are required instrumentation, not separate research questions:

- neural execution time;
- routing and aggregation time;
- batching efficiency;
- memory transfers;
- synchronization/idle time;
- CPU/GPU utilization;
- RAM/VRAM;
- communication volume inside the local machine;
- compiler mode;
- evidence provenance and loss;
- evidence production and integration rates;
- integration backlog depth/age when applicable;
- duplicate versus unique useful work;
- compute per useful uncertainty reduction when measurable.

A configuration fails when organization, integration, memory movement, or latency makes the workers impractical even if raw worker throughput or raw accuracy improves.
