# Research Roadmap

The roadmap contains only uncertainties that still require architecture-specific evidence. Established engineering capabilities are reused and profiled when needed rather than re-proven as standalone experiments.

The project follows one additional rule:

> **Build the smallest mechanism that removes the next demonstrated bottleneck. Preserve interfaces that make later scaling possible, but do not implement scale infrastructure before measurements require it.**

## Completed foundation — practical small-worker regime

The initial worker-size sweep established that useful learned transformations survive deep into the small-parameter regime. Worker shrinking is closed for the current phase.

Current decision:

- ~50K is frozen as Worker v1 for Step 2A;
- ~25K / ~50K / ~75K remain candidates for a later equal-budget organization comparison;
- smaller workers are revisited only if later results create a concrete reason.

The project optimizes for a **practical worker**, not minimum parameter count. Capability, functional diversity, latency, throughput, memory, and total system cost all matter.

## Gate 1 — Minority evidence utilization — ACTIVE

Goal: determine whether the population can convert rare correct evidence into final capability without trusting noisy minority outliers.

Current evidence at width 16 already shows a material oracle-any-correct gap. The immediate problem is therefore utilization rather than discovery of additional population signal.

Tasks:

- keep the 16 frozen 50K workers unchanged;
- propose strong protected non-primary candidates without using truth;
- measure how many primary errors become recoverable candidates;
- fit a tiny rescue gate on development validation data only;
- select a threshold under an explicit harm budget;
- evaluate once on untouched validation confirmation data;
- keep the frozen test set unopened.

Exit criterion:

- either demonstrate reproducible net rescue under an acceptable harm rate;
- or identify whether the failure is candidate proposal or inability to distinguish useful minority evidence from noise.

See [`experiments/step_02_population_scaling/minority_rescue_v0.md`](../experiments/step_02_population_scaling/minority_rescue_v0.md).

This gate remains narrow. Do not mix persistent-runtime, scheduler, or large-scale integration changes into the rescue experiment.

## Gate 2 — Useful population-width region

Goal: with Worker v1 and the evidence-utilization mechanism frozen, determine where additional workers stop providing enough **new useful information** to justify their full end-to-end cost.

Candidate widths:

- 1;
- 4;
- 16;
- 64;
- 256 only if the smaller widths justify it.

Measure:

- final quality;
- oracle-any-correct coverage;
- unique/rare evidence;
- functional correlation;
- duplicate versus unique findings;
- raw evidence production rate where measurable;
- useful/knowledge-changing evidence rate where measurable;
- aggregation/integration latency;
- latency and throughput;
- end-to-end organization cost.

Exit criterion:

- identify the useful width region and where marginal gains saturate, become redundant, or become impractical to integrate.

Do not interpret the largest runnable width as the best width.

## Gate 3 — Fixed-budget worker organization

Goal: identify the practical worker-size/population-width sweet spot under equal total learned-parameter budgets.

Initial homogeneous candidates:

- populations of ~25K workers;
- populations of ~50K workers;
- populations of ~75K workers.

Keep training data, hardware, evaluation data, and total learned budget as comparable as practical.

Measure system-level value, not only individual-worker accuracy:

- final quality;
- useful evidence coverage;
- unique contributions;
- functional correlation;
- evidence utilization;
- execution and integration cost.

Exit criterion:

- identify whether one worker granularity gives a reproducible capability/resource/information advantage.

## Gate 4 — Population versus dense baseline

Goal: test the central small-scale hypothesis directly.

Compare the best population configuration against conventional dense baselines under normalized:

- parameter budget;
- training data;
- hardware;
- runtime budget;
- evaluation set.

Measure quality, useful information extracted, evidence recall, latency, throughput, memory, active parameters, coordination/integration cost, and compute efficiency.

Exit criterion:

- establish whether a useful population advantage exists on any meaningful capability/resource frontier.

A consistently better dense baseline is a valid negative result and a stop/redirect signal.

## Minimal runtime foundation — BUILD ONLY WHEN NEEDED

A persistent runtime becomes useful when experiments move from one-shot fixed-width evaluation to long-lived Work Threads and adaptive allocation.

Build only five major pieces:

1. **Worker Bank** — homogeneous architecture, independently learned checkpoints;
2. **Research Ledger** — append-only durable events and provenance;
3. **State Projector** — current Work Threads, evidence, hypotheses, coverage, contradictions, and dependencies;
4. **Scheduler v0** — deterministic priorities plus permanent structured-random exploration;
5. **Worker Runtime** — bounded attempts, purpose-specific context views, batching, pause/resume, and commit of results.

Do not build separate blackboard, handoff, failure-memory, progress-history, and checkpoint databases. They are views of the same Research Ledger.

Do not build a learned scheduler until real traces demonstrate a limitation of deterministic allocation.

Do not build distributed storage or multi-machine coordination until local measurements require it.

The semantic architecture is frozen before implementation scale grows. See [`docs/runtime_architecture.md`](runtime_architecture.md) and [`docs/architecture_contracts.md`](architecture_contracts.md).

## Gate 5 — Adaptive allocation

Goal: determine whether the runtime can spend learned computation where it removes the most important uncertainty rather than using fixed width everywhere.

The scheduler controls four primary dimensions:

- **width** — number of independent attempts;
- **depth** — repeated bounded attempts on persistent Work Threads;
- **scope** — which source regions or information are examined;
- **purpose** — `EXPLORE`, `PROGRESS`, `CHALLENGE`, `VERIFY`, or `SYNTHESIZE`.

Construction note: [`benchmarks/large_scope_relevance_v0.md`](../benchmarks/large_scope_relevance_v0.md) implements and mechanically qualifies the controlled scope-only versus diverse-weight benchmark, including an exact shared-base width-1 control and paired diverse-minus-same statistics. This does **not** activate Gate 5: the frozen Worker-v1 development result is still pending, and no adaptive-allocation result is claimed.

Scheduler v0 should:

- start from simple deterministic priorities;
- reserve a nonzero structured-random exploration budget;
- prefer under-covered possibilities during discovery;
- intentionally duplicate important claims during verification;
- rotate workers when progress stagnates rather than preserving an unbounded worker-owned loop;
- preserve work even when workers are stopped or reassigned.

Candidate triggers for additional work include:

- unresolved contradiction;
- missing discriminating evidence;
- low signal quality;
- disagreement;
- local information density;
- failure of an earlier processing round;
- under-covered hypothesis/source region;
- high-impact claim requiring replication.

Compare adaptive execution against fixed-width execution under equal end-to-end budgets.

Exit criterion:

- show a reproducible workload where adaptive allocation improves the practical information/capability frontier.

## Gate 6 — Knowledge integration bandwidth — ACTIVATE ONLY WHEN VOLUME JUSTIFIES IT

Goal: determine when useful information production from the worker population begins to exceed the runtime's ability to preserve, connect, verify, route, summarize, and exploit it.

This is expected to be a potential **large-population scaling bottleneck**, not an assumed current bottleneck.

Distinguish:

- **exploration throughput** — meaningfully different evidence-producing attempts per unit wall-clock time;
- **knowledge integration bandwidth** — useful evidence successfully incorporated into persistent, connected, actionable shared knowledge per unit wall-clock time without losing rare decisive information.

Measure:

- raw evidence production rate;
- unique useful evidence rate;
- knowledge-changing event rate;
- duplicate/irrelevant output rate;
- integration latency;
- integration backlog depth and age;
- verification throughput;
- rare-evidence retention;
- provenance recoverability;
- summary/compression loss;
- memory and communication volume;
- scheduler/integration compute.

Trigger for this gate:

- evidence volume or backlog measurements show that adding useful workers is becoming limited by information handling rather than worker execution.

If triggered, test the smallest sufficient hierarchy:

```text
worker attempts
    -> local/thread evidence
    -> branch/topic integration
    -> cross-topic integration
    -> global knowledge changes
```

Important constraints:

- propagate **knowledge-changing deltas**, not complete histories;
- keep raw evidence recoverable through stable provenance references;
- do not broadcast all global knowledge to all workers;
- allow the same homogeneous workers to execute integration Work Threads before introducing specialized integration models.

Exit criterion:

- identify the useful information-volume frontier and demonstrate an integration architecture that keeps backlog bounded and preserves decisive evidence at the tested scale;
- or establish that knowledge integration prevents further useful population scaling.

## Gate 7 — Real workloads

Goal: determine where the architecture is genuinely useful outside the synthetic benchmark.

Candidates:

- large-document analysis;
- image-region understanding;
- logs and event streams;
- codebase analysis;
- anomaly detection;
- multi-stage planning;
- research-like search with many plausible approaches and falsifiable hypotheses.

Tiling, source coordinates, overlap, zoom-in, deterministic routing, persistent Work Threads, and hierarchical reduction are implementation techniques, not standalone discoveries.

Measure whether the complete system:

- discovers useful possibilities a single-stream baseline misses;
- preserves both successful and failed evidence-producing attempts;
- avoids premature convergence by maintaining exploration while progressing strong branches;
- keeps local context bounded as global work grows;
- preserves rare decisive evidence through integration;
- improves useful capability/resource trade-offs after full information-handling cost is counted.

Exit criterion:

- identify at least one real workload class where the architecture provides a practical advantage.

## Gate 8 — Scale only after advantage

Only after a small-scale advantage is established, test whether it survives larger total learned capacity, active population, workload size, and information volume.

Scaling toward approximately 1B total learned parameters is a later reference point. It is not a current target and must not hide weak small-scale results behind more compute.

Larger populations are conceptually possible, but worker count is never the objective.

Scale only while:

- marginal workers add meaningful possibility coverage;
- that coverage generates useful evidence;
- evidence utilization preserves minority discoveries;
- integration bandwidth keeps up without unbounded backlog;
- end-to-end capability continues to improve at acceptable cost.

The useful population limit may therefore be determined by **information integration**, not by the maximum number of workers that can technically be executed.

## Systems ablations that are not research gates

Implement/profile these when required:

- homogeneous GPU batching/vectorization;
- CPU/GPU task placement;
- deterministic routing and exact logic;
- bounded queues and backpressure;
- memory-transfer minimization;
- source/coordinate tracking;
- input partitioning and overlap;
- local durable persistence;
- hierarchical data structures;
- observability/dashboard metrics;
- compiler modes and graph/fusion optimizations;
- quantization when memory/throughput measurements justify it.

Compiler behavior remains a **separate systems variable**. Compare the same neural workers and population policy under execution modes rather than treating compiler gains as neural-architecture gains.

The existence of these techniques does not need to be re-proven. Only their measured effect on this implementation matters.

# Global stop conditions

Stop or redirect a path when evidence shows that:

- additional population signal cannot be converted into usable capability;
- correct minority evidence cannot be distinguished from harmful outliers well enough to matter;
- population width adds no meaningful possibility coverage or useful information;
- duplicate/correlated work dominates additional width;
- latency makes the workers impractical;
- organization or memory movement dominates useful compute;
- useful evidence is generated faster than it can be integrated and the backlog cannot be bounded economically;
- hierarchical summaries lose decisive evidence or provenance;
- verification cannot keep pace with consequential claims;
- equal-budget dense baselines are consistently better;
- apparent gains disappear on untouched confirmation or real workloads.

A negative result is valuable because it removes an uncertainty and prevents unnecessary engineering work.