# Research Roadmap

The roadmap contains only uncertainties that still require architecture-specific evidence. Established engineering capabilities are reused and profiled when needed rather than re-proven as standalone experiments.

## Completed foundation — practical small-worker regime

The initial worker-size sweep established that useful learned transformations survive deep into the small-parameter regime. Worker shrinking is closed for the current phase.

Current decision:

- ~50K is frozen as Worker v1 for Step 2A;
- ~25K / ~50K / ~75K remain candidates for a later equal-budget organization comparison;
- smaller workers are revisited only if later results create a concrete reason.

The project optimizes for a **practical worker**, not minimum parameter count. Capability, latency, throughput, memory, and total system cost all matter.

## Gate 1 — Minority evidence utilization — ACTIVE

Goal: determine whether the population can convert rare correct evidence into final capability without trusting noisy minority outliers.

Current evidence at width 16 already shows a material oracle-any-correct gap. The immediate problem is therefore utilization rather than discovery of additional population signal.

Tasks:

- keep the 16 frozen 50K workers unchanged;
- propose strong protected non-primary candidates without using truth;
- measure how many primary errors become recoverable candidates;
- calibrate a tiny rescue gate on development validation data only;
- select a threshold under an explicit harm budget;
- evaluate once on untouched validation confirmation data;
- keep the frozen test set unopened.

Exit criterion:

- either demonstrate reproducible net rescue under an acceptable harm rate;
- or identify whether the failure is candidate proposal or inability to distinguish useful minority evidence from noise.

See [`experiments/step_02_population_scaling/minority_rescue_v0.md`](../experiments/step_02_population_scaling/minority_rescue_v0.md).

## Gate 2 — Useful population-width region

Goal: with Worker v1 and the evidence-utilization mechanism frozen, determine where additional workers stop providing enough value.

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
- latency and throughput;
- end-to-end organization cost.

Exit criterion:

- identify the useful width region and where marginal gains saturate or become impractical.

## Gate 3 — Fixed-budget worker organization

Goal: identify the practical worker-size/population-width sweet spot under equal total learned-parameter budgets.

Initial homogeneous candidates:

- populations of ~25K workers;
- populations of ~50K workers;
- populations of ~75K workers.

Keep training data, hardware, evaluation data, and total learned budget as comparable as practical.

Exit criterion:

- identify whether one worker granularity gives a reproducible capability/resource advantage.

## Gate 4 — Population versus dense baseline

Goal: test the central hypothesis directly.

Compare the best population configuration against conventional dense baselines under normalized:

- parameter budget;
- training data;
- hardware;
- runtime budget;
- evaluation set.

Measure quality, latency, throughput, memory, active parameters, coordination cost, and compute efficiency.

Exit criterion:

- establish whether a useful population advantage exists on any meaningful capability/resource frontier.

A consistently better dense baseline is a valid negative result and a stop/redirect signal.

## Gate 5 — Compiler ablation

Goal: measure how much execution compilation improves the already-defined population architecture.

Compiler behavior is a **separate experimental variable**. Compare the same neural workers and population policy under execution modes such as:

- eager/runtime orchestration;
- vectorized/batched execution;
- compiled/fused execution;
- compiled memory/scheduling improvements where available.

Exit criterion:

- quantify compiler contribution without attributing it to the neural architecture.

## Gate 6 — Adaptive worker allocation

Goal: allocate more learned processing only where it improves the quality/resource trade-off.

Tasks:

- start from a small active population;
- use inference-visible uncertainty, contradiction, missing-evidence, or information-density signals;
- request additional workers for discriminating information rather than repeated voting;
- compare against fixed-width execution under equal end-to-end budgets.

Exit criterion:

- show a reproducible workload where adaptive allocation improves the practical frontier.

## Gate 7 — Real workloads

Goal: determine where the architecture is genuinely useful outside the synthetic benchmark.

Candidates:

- large-document analysis;
- image-region understanding;
- logs and event streams;
- codebase analysis;
- anomaly detection;
- multi-stage planning.

Tiling, source coordinates, overlap, zoom-in, deterministic routing, and hierarchical reduction are implementation techniques, not standalone discoveries. Measure whether the complete system preserves decisive evidence and improves useful capability/resource trade-offs.

Exit criterion:

- identify at least one real workload class where the architecture provides a practical advantage.

## Gate 8 — Scale only after advantage

Only after a small-scale advantage is established, test whether it survives larger total learned capacity and larger workloads.

Scaling toward approximately 1B total learned parameters is a later hypothesis. It is not a current target and must not hide weak small-scale results behind more compute.

## Engineering work that is not a research gate

Implement these when required and profile their actual cost:

- homogeneous GPU batching/vectorization;
- CPU/GPU task placement;
- deterministic routing and exact logic;
- bounded queues and backpressure;
- memory-transfer minimization;
- source/coordinate tracking;
- input partitioning and overlap;
- hierarchical data structures;
- local-machine scheduling;
- observability/dashboard metrics.

The existence of these techniques does not need to be re-proven. Only their measured effect on this implementation matters.

# Global stop conditions

Stop or redirect a path when evidence shows that:

- additional population signal cannot be converted into usable capability;
- correct minority evidence cannot be distinguished from harmful outliers well enough to matter;
- latency makes the workers impractical;
- organization or memory movement dominates useful compute;
- population width adds no meaningful information;
- equal-budget dense baselines are consistently better;
- apparent gains disappear on untouched confirmation or real workloads.

A negative result is valuable because it removes an uncertainty and prevents unnecessary engineering work.
