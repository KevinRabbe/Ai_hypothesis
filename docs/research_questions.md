# Research Questions

## Primary question

At a fixed hardware and parameter budget, what is the smallest learned neural processing unit that still produces useful signal beyond deterministic logic, and can dynamically allocating populations of such units improve capability or resource efficiency compared with a conventional fixed dense model?

## RQ1 — Minimum useful unit

How does useful signal change as the identical unit architecture is progressively reduced in parameter count?

Measure:

- task accuracy;
- precision and recall;
- invalid or unusable outputs;
- consistency;
- useful information extracted;
- noise rate;
- inference latency;
- memory use;
- batch throughput.

Key boundary:

> At what size does further shrinking stop producing a useful learned transformation?

## RQ2 — Learned value versus deterministic logic

For which transformations does a learned unit outperform or complement ordinary deterministic algorithms?

A neural unit is not justified when a deterministic implementation is more reliable, cheaper, and equally capable.

## RQ3 — Population scaling

For a viable unit size, how does system quality change as more identical units are allocated?

Test widths such as:

- 1;
- 4;
- 16;
- 64;
- 256;
- larger widths when hardware and results justify them.

The objective is not majority voting. Measure whether additional units increase evidence coverage, unique findings, ambiguity resolution, or error detection.

## RQ4 — Correlation and diversity

How similar should independently weighted units be?

Too much functional correlation may create many copies of the same failure mode. Too little correlation may produce incompatible or noisy outputs.

Measure functional diversity under controlled changes to:

- initialization;
- sample order;
- dropout or masking;
- optimization trajectory.

All compared units should retain the same architecture and overall training distribution.

## RQ5 — Adaptive worker allocation

Can the runtime begin with a small worker allocation and add workers only when more learned processing is useful?

Possible triggers include:

- low signal quality;
- unresolved contradiction;
- missing discriminating evidence;
- high local information density;
- high disagreement;
- boundary uncertainty;
- failure of a prior prediction.

Compare adaptive allocation against fixed-width execution under equal end-to-end resource budgets.

## RQ6 — Information partitioning

Can large information spaces be divided across tiny units without losing important cross-boundary relationships?

Study:

- overlapping partitions;
- semantic boundaries;
- cross-boundary reprocessing;
- source pointers;
- targeted zoom-in on uncertain regions.

Candidate modalities include text, images, logs, and structured state.

## RQ7 — Hierarchical recombination

How can thousands of local outputs be recursively reduced without recreating a single-context bottleneck?

The aggregation system must preserve access to original evidence so compressed representations can be challenged and expanded.

## RQ8 — Evidence preservation

How should the runtime preserve rare but decisive evidence?

A minority of workers may observe critical evidence that most workers never received.

Measure:

- unique finding recall;
- evidence provenance retention;
- contradiction preservation;
- failure rate from premature aggregation or averaging.

## RQ9 — Coordination overhead

At what granularity do routing, weight gathering, data movement, synchronization, and aggregation become more expensive than the neural computation saved?

End-to-end measurements must include:

- neural execution;
- routing;
- batching;
- memory transfers;
- aggregation;
- scheduling;
- idle time.

A configuration fails if organization overhead dominates useful compute.

## RQ10 — GPU batch efficiency

Can large numbers of identical small units be executed as efficient batched matrix operations rather than thousands of tiny launches?

Measure utilization and throughput as unit size and batch width change.

## RQ11 — CPU/GPU scheduling

Which operations belong on CPU versus GPU for consumer hardware?

Initial hypothesis:

CPU:

- scheduling;
- deterministic logic;
- evidence graphs;
- queues;
- partitioning;
- aggregation;
- resource control.

GPU:

- batched learned transformations;
- training forward/backward passes;
- large parallel worker batches.

The allocation should be decided by profiling, not by fixed assumptions.

## RQ12 — Fixed-budget comparison

Under equal or carefully normalized parameter, training-data, hardware, and runtime budgets, how do different organizations compare?

Example comparison:

- one dense network;
- few larger units;
- medium population;
- large population of smaller units.

Metrics:

- final task quality;
- useful information extracted;
- evidence recall;
- latency;
- throughput;
- RAM and VRAM;
- active learned parameters;
- coordination overhead;
- compute efficiency.

## RQ13 — Scaling toward 1B total parameters

If a smaller experiment demonstrates a measurable advantage, does that advantage persist as the total learned parameter population grows toward approximately 1 billion parameters?

Scaling is a separate hypothesis and must be retested rather than assumed.

## RQ14 — Workload dependence

Which workloads benefit from this architecture?

Likely candidates to test include workloads where information is divisible and only some regions deserve expensive processing:

- large document analysis;
- image-region understanding;
- logs and event streams;
- codebase analysis;
- anomaly discovery;
- multi-stage planning.

The architecture does not need to outperform dense models on every task to be useful.
