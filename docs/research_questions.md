# Research Questions

## Primary question

At a fixed hardware and parameter budget, can a population of practical small learned workers produce more useful capability per unit of active compute than a conventional fixed dense model?

The project does **not** treat established engineering capabilities as research questions. Batching, multi-device execution, information tiling, coordinate/source tracking, deterministic routing, compiler fusion, and CPU/GPU task placement are implementation tools whose costs are profiled when used. Experiments are reserved for uncertainties whose answers depend on this population architecture.

## Closed exploration — worker shrinking

The project has already pushed the worker architecture into the small-parameter regime. Further shrinking is not a goal by itself.

For the current phase:

- ~50K is the frozen reference Worker v1 used by Step 2A;
- ~25K / ~50K / ~75K remain useful worker-size candidates for later equal-budget organization comparisons;
- smaller sizes are revisited only if population results provide a concrete reason.

The relevant worker question is practical capability, not minimum parameter count: useful output, latency, throughput, memory, and total system cost must all remain acceptable.

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

## RQ2 — Useful population-width region

With Worker v1 and an evidence-utilization mechanism held fixed, how does useful output change as population width increases?

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
- latency and throughput;
- end-to-end coordination cost.

Worker diversity is measured here as a population property rather than treated as a separate research program.

## RQ3 — Fixed-budget worker organization

Under an equal total learned-parameter and hardware budget, which homogeneous worker size gives the best system-level result?

Initial candidates:

- ~25K workers;
- ~50K workers;
- ~75K workers.

Compare populations with as closely matched total parameter budgets as practical. The objective is to find the practical organization sweet spot, not the smallest individual worker.

## RQ4 — Population versus dense baseline

Under normalized training data, parameter budget, hardware, runtime budget, and evaluation data, does the best population organization outperform a conventional dense baseline on any meaningful capability/resource frontier?

Measure:

- final task quality;
- useful information extracted;
- evidence recall;
- latency;
- throughput;
- RAM and VRAM;
- active learned parameters;
- end-to-end coordination overhead;
- compute and energy efficiency where measurable.

This is the decisive test of the central hypothesis.

## RQ5 — Compiler effect — SEPARATE VARIABLE

How much does compilation/execution planning improve the already-defined population architecture?

Compare the same workers, workload, population policy, and hardware under execution modes such as:

- ordinary eager/runtime orchestration;
- batching/vectorization;
- compiled/fused execution;
- compiled memory/scheduling improvements where available.

Compiler gains must not be attributed to the neural architecture itself.

## RQ6 — Adaptive worker allocation

Once a useful fixed-width population exists, can the runtime begin with a small allocation and add workers only when additional learned processing is likely to help?

Candidate triggers include:

- unresolved contradiction;
- missing discriminating evidence;
- low signal quality;
- disagreement;
- local information density;
- failure of an earlier processing round.

Compare adaptive execution against fixed-width execution under equal end-to-end budgets.

## RQ7 — Real-workload usefulness

Which real workloads actually benefit from the architecture?

Candidates include:

- large-document analysis;
- image-region understanding;
- logs and event streams;
- codebase analysis;
- anomaly discovery;
- multi-stage planning.

Information partitioning, coordinates/source pointers, overlap, zoom-in, and hierarchical reduction are not treated as discoveries by themselves. They are implementation techniques. What must be measured is whether the complete population system preserves decisive cross-region evidence and provides a useful capability/resource advantage on the workload.

## RQ8 — Scaling after advantage

Only if a smaller system demonstrates a reproducible advantage: does that advantage survive as total learned capacity and workload scale increase?

Scaling toward approximately 1B total learned parameters is a later hypothesis, not an assumption and not a current milestone.

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
- evidence provenance and loss.

A configuration fails when organization overhead or latency makes the workers impractical even if raw accuracy improves.
