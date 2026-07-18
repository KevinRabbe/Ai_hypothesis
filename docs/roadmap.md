# Research Roadmap

The project is organized as a sequence of research gates. A later stage should not be treated as settled until the earlier uncertainty it depends on has been measured.

## Step 1 — Minimum Useful Neural Unit

Goal: find the smallest identical learned processing unit that still performs a useful local transformation.

Tasks:

- define a compact benchmark suite;
- define one architecture family that can be scaled down cleanly;
- train size variants under controlled conditions;
- compare useful signal against noise;
- establish deterministic baselines;
- measure inference cost and batch throughput.

Exit criterion:

- identify at least one viable unit-size region;
- identify a lower region where shrinking clearly stops being useful;
- record uncertainty if the boundary is not yet clear.

## Step 2 — Population Scaling

Goal: determine whether multiple viable units provide additional useful information.

Tasks:

- evaluate 1, 4, 16, 64, 256, and larger worker widths where practical;
- measure evidence coverage and unique findings;
- reject majority-vote aggregation as the primary decision rule;
- measure coordination overhead end to end.

Exit criterion:

- establish whether increasing width improves useful output before overhead dominates.

## Step 3 — Correlated Independent Weights

Goal: find a useful diversity range for architecturally identical units.

Tasks:

- vary initialization and training stochasticity;
- keep architecture and overall training distribution constant;
- measure functional correlation and shared failure modes;
- test whether diversity improves evidence discovery or uncertainty detection.

Exit criterion:

- identify configurations that avoid both clone collapse and incoherent divergence.

## Step 4 — Efficient Batched Execution

Goal: execute many selected units as efficient GPU batches.

Tasks:

- benchmark batched matrix execution;
- measure kernel-launch and scheduling overhead;
- measure weight gathering and memory-transfer cost;
- compare against equivalent dense execution.

Exit criterion:

- demonstrate a worker granularity where useful compute dominates organization overhead.

## Step 5 — Adaptive Worker Allocation

Goal: dynamically add workers only where additional learned processing is useful.

Tasks:

- start tasks with a small worker allocation;
- define uncertainty, conflict, and missing-evidence signals;
- allocate additional workers to discriminating evidence rather than repeated voting;
- compare adaptive allocation against fixed-width baselines.

Exit criterion:

- show a workload where adaptive allocation improves quality/resource trade-offs.

## Step 6 — Large Information Partitioning

Goal: process information larger than one unit should receive at once.

Tasks:

- divide large text, image, log, or structured inputs;
- preserve source references;
- introduce overlap or semantic boundaries;
- detect boundary uncertainty;
- trigger targeted cross-boundary processing.

Exit criterion:

- recover distributed relevant information without requiring one unit to read the full source.

## Step 7 — Hierarchical Recombination

Goal: combine many local outputs without recreating a single-context bottleneck.

Tasks:

- recursively reduce local outputs;
- preserve provenance at every level;
- support zoom-back to original evidence;
- measure information loss at each aggregation stage.

Exit criterion:

- reconstruct global task state while retaining decisive local evidence.

## Step 8 — Deterministic Decision and Logic Layer

Goal: minimize unnecessary neural prediction.

Tasks:

- classify operations as learned versus deterministic;
- implement exact routing, state, arithmetic, coordinate, validation, and policy operations in code;
- measure neural calls avoided;
- verify that deterministic substitution does not reduce task capability.

Exit criterion:

- establish a clear learned/deterministic boundary for the tested workloads.

## Step 9 — Heterogeneous CPU/GPU Scheduler

Goal: coordinate consumer CPU and GPU resources efficiently.

Tasks:

- profile orchestration and neural execution separately;
- assign CPU-friendly and GPU-friendly work empirically;
- batch GPU work while keeping the CPU productive;
- add bounded queues and backpressure;
- avoid hidden background work.

Exit criterion:

- stable end-to-end execution with measured CPU, GPU, RAM, and VRAM usage.

## Step 10 — Fixed-Budget Competition

Goal: compare population organizations against dense baselines.

Example fixed total budget:

- 1 × 100M;
- 10 × 10M;
- 100 × 1M;
- 1,000 × 100K.

Actual configurations will depend on Step 1 results.

Compare under normalized:

- training data;
- parameter budget;
- hardware;
- runtime budget;
- evaluation set.

Exit criterion:

- identify whether a measurable sweet spot exists and where further splitting becomes harmful.

## Step 11 — Scale Toward ~1B Total Parameters

Goal: test whether observed advantages survive larger total capacity.

Candidate progression:

- 10M;
- 50M;
- 100M;
- 250M;
- 500M;
- approximately 1B.

Each scale is a new experiment, not an assumption.

Exit criterion:

- determine whether the population advantage persists, disappears, or changes with scale.

## Step 12 — Real-World Workloads

Goal: evaluate workloads where divisible information and adaptive processing may matter.

Candidates:

- large-document analysis;
- image understanding with adaptive regional processing;
- log and event-stream analysis;
- codebase analysis;
- anomaly detection;
- planning with recursively generated subproblems.

Exit criterion:

- identify the workload classes where the architecture is genuinely useful.

## Step 13 — Runtime and Interface

Goal: expose resource allocation and execution behavior transparently.

Interface metrics may include:

- active and idle units;
- workers allocated per task or region;
- CPU and GPU utilization;
- RAM and VRAM;
- neural execution time;
- coordination time;
- batch sizes;
- evidence discovered;
- additional worker requests;
- total task duration.

Exit criterion:

- the architecture can be inspected and optimized based on real measured behavior rather than assumptions.

# Global stop conditions

At any stage, the project should be willing to stop or redirect if evidence shows that:

- useful signal collapses before units become small enough to offer practical advantages;
- communication overhead dominates;
- memory movement prevents efficient batching;
- worker diversity does not provide useful additional information;
- hierarchical aggregation loses too much evidence;
- dense baselines are consistently better under equal end-to-end budgets.

A negative result is still a valid research result because it identifies where the hypothesis fails.
