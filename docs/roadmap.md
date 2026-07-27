# Research Roadmap

The primary research objective is:

> **Determine whether a fixed learned-parameter system can gain useful capability and resource efficiency by reusing the same learned machinery across a dynamic population of weak runtime states.**

The project does not scale worker count or learned parameters merely because it is technically possible. Each gate must remove a real architecture-specific uncertainty and earn the next one.

## Completed background — tiny local learned transformations

Earlier work established that useful local neural transformations survive in a very small parameter regime.

Observed reference region:

- ~10K parameters: difficult;
- ~25K: smallest strong candidate region;
- ~50K: practical reference;
- ~75K / ~100K: larger references.

That work remains useful background, but worker shrinking is no longer the primary objective.

# Gate 0 — Fixed-parameter population scaling — **COMPLETED POSITIVE**

## Question

> With learned parameters held fixed, can additional reusable runtime neural computation and additional available distributed source scope reproducibly produce additional capability?

## Canonical system

The confirmed system uses:

- one shared learned relay model per independently trained seed;
- **26,669 learned parameters** independent of runtime population size;
- weak runtime worker states carrying local key/value records;
- one bounded recurrent shared query/value representation;
- hop-local worker-state reset;
- parameter-free normalized competition over active worker messages;
- training-only gate-selection supervision;
- no oracle chain information at inference.

Canonical protocol:

- experiment: `population-compute-relay-training-v1`;
- protocol: `relay-protocol-v1-normalized-gate-supervised`;
- benchmark: `collective-relay-v1-answer-frontier`;
- runtime populations: `1 / 4 / 16 / 64 / 256`;
- relay depths: `2 / 4 / 8`;
- matched communication and no-communication conditions.

## Frozen confirmation result

Confirmation used untouched worlds and new training seeds `1 / 2 / 3` under the rule frozen before the confirmation split was opened.

All **3 / 3** seeds passed independently.

Across the three seeds, mean 256-worker exact solve was:

- relay-2: **99.63%**;
- relay-4: **99.33%**;
- relay-8: **98.27%**.

Every seed passed all three relay tiers. All four adjacent population steps were non-decreasing under the frozen tolerance.

Critical controls:

- exact solve given incomplete information was **0%** everywhere incomplete worlds existed;
- no-communication exact solve was **0%** in all 45 seed × relay-depth × population conditions;
- every population/control point inside one seed reused exactly the same checkpoint and learned-parameter count;
- the three confirmation seeds had distinct independently trained checkpoint fingerprints.

Canonical evidence:

- [`../experiments/population_compute_scaling_v0/relay_v1_confirmation_result_v0.md`](../experiments/population_compute_scaling_v0/relay_v1_confirmation_result_v0.md)
- [`../experiments/population_compute_scaling_v0/confirmation_gate_v1.json`](../experiments/population_compute_scaling_v0/confirmation_gate_v1.json)
- [`../experiments/population_compute_scaling_v0/confirmation_protocol_v1.md`](../experiments/population_compute_scaling_v0/confirmation_protocol_v1.md)

## Hard interpretation boundary — serial equivalence

The same repaired relay function has already been qualified under an exactly matched serial schedule.

For arbitrary fixed weights, parallel normalized execution and a one-live-state serial execution produce the same mathematical result within floating-point tolerance while using the same `N × relay_hops` learned worker updates.

Therefore Gate 0 establishes:

> **fixed-parameter runtime-compute/source-scope scaling on the controlled relay task.**

It does **not** establish:

- extra function-level capability from simultaneous wide state at equal learned work;
- better capability per worker update/FLOP than the serial schedule;
- superiority over dense/recurrent baselines;
- a real-workload advantage;
- that 1K+ or 100K runtime states remain useful.

This serial result is not a caveat to hide. It removes an important uncertainty and defines the next gate.

## Gate 0 communication-rescue branch — NOT ACTIVATED

The originally allowed hierarchical rescue is unnecessary for Gate 0. The normalized v1 communication path passed frozen confirmation.

Do not add further communication architectures until a later measured bottleneck requires them.

# Gate 1 — Work/span resource frontier — **ACTIVE**

## Question

> **Does parallel population execution provide a useful practical resource frontier over the exactly equivalent serial execution?**

The current relay function is an unusually clean systems experiment because the neural function and total learned work can be held constant while only execution organization changes.

## Frozen comparison

Where possible, use the same checkpoint, benchmark worlds, output decoder, learned worker-update count, and relay rounds for both schedules.

Compare:

### Parallel normalized

For `N` active records and `H` relay hops:

- total learned worker updates: `N × H`;
- peak live learned states: `N`;
- critical learned dependency depth: approximately `H` population rounds plus reductions;
- records within a hop are batch/parallel candidates.

### Serial normalized

- total learned worker updates: the same `N × H`;
- peak live learned states: `1`;
- records are time-multiplexed through one learned state;
- the deterministic normalized reducer is updated online;
- critical learned execution depth grows with the serial record schedule.

The two schedules must remain output-equivalent within the existing tolerance. Any capability difference is a correctness failure, not a positive result.

## Primary hardware

Run the decisive resource comparison on the actual local consumer GPU/hardware target, not only on GitHub-hosted CPU runners.

Record exact:

- CPU;
- GPU;
- VRAM;
- PyTorch version;
- driver/CUDA runtime;
- execution backend;
- compiler mode;
- batch/world size.

GitHub CPU runs remain useful for mechanics and reproducibility but are not the primary practical performance claim.

## Required metrics

For every population size and schedule record:

- exact output-equivalence status;
- total learned worker updates;
- estimated/measured FLOPs where reliable;
- sequential span / recurrent dependency rounds;
- median and tail latency after warm-up;
- throughput;
- device execution time where measurable;
- CPU orchestration time;
- peak activation/state memory;
- GPU peak allocated/reserved memory;
- communicated scalars/bytes;
- host↔device transfer where measurable;
- synchronization count/cost where measurable;
- batching/utilization telemetry where available.

Derived measures should remain transparent:

- speedup = `serial latency / parallel latency`;
- parallel efficiency relative to available hardware;
- extra peak memory per unit speedup;
- communication/memory cost per solved world.

Do not collapse the frontier into one arbitrary scalar score.

## Population points

Start with the already-qualified functional ladder:

- 1;
- 4;
- 16;
- 64;
- 256.

Only add larger counts during Gate 1 if required to locate the first systems saturation point. Do not enlarge the benchmark merely to produce a bigger worker number.

## Compiler/runtime ablation

Compiler behavior is a separate systems variable.

First establish an eager reference. Then, if profiling justifies it, compare the same function/schedule under clearly separated modes such as:

- eager;
- `torch.compile` default;
- reduced-overhead / graph-oriented execution where supported;
- custom kernels only after profiling identifies launch/fusion overhead as material.

Never report compiler speedup as neural-architecture capability gain.

## Gate 1 exit criteria

### Continue

Continue if parallel width provides a meaningful practical frontier, for example lower latency or higher throughput under the same learned work while memory and communication remain acceptable.

The result does not need to be perfect linear speedup. Substantial useful speedup can justify population organization even though the function is serializable.

### Negative / redirect

Treat Gate 1 as negative for the current relay organization if:

- wide execution is no faster on the target hardware;
- synchronization/memory movement erases batching gains;
- peak state memory grows too quickly for the achieved speedup;
- ordinary serial/recurrent execution dominates the practical frontier.

A negative Gate 1 does not invalidate Gate 0. It means the confirmed capability scaling is better implemented serially for this workload.

# Gate 2 — Organization-specific capability/resource advantage

Activate only if Gate 1 shows a useful execution frontier or if a workload has a structural reason to need persistent distributed state.

## Question

> **Can population organization improve capability under a fixed practical budget on a workload where locality, persistent distributed state, or parallel exploration matters?**

Gate 0 relay is intentionally too reducible to answer this question: its normalized population reduction is exactly serializable.

The next workload should therefore require at least one of:

- persistent local state that should not be collapsed after each hop;
- multiple simultaneously live hypotheses/branches;
- local context that is expensive to move globally;
- parallel search where more possibilities can be examined under the same latency budget;
- spatial/source locality where different weak states operate on different regions;
- asynchronous evidence production and later verification/integration.

Candidate reusable substrates already in the repository include:

- large-scope relevance;
- persistent Work Threads / Research Ledger;
- scope-region allocation;
- evidence/knowledge integration;
- deterministic Scheduler v0.

## Controls

Compare at matched practical budgets:

- wide population;
- best serial/recurrent implementation;
- fixed-width recurrent baseline;
- dense baseline where appropriate.

Match or report explicitly:

- learned parameters;
- training data;
- wall-clock/latency budget;
- hardware;
- total learned work/FLOPs;
- peak memory;
- information inspected;
- communication volume.

## Exit criterion

- identify at least one workload where population organization moves the capability/resource frontier;
- or accept that a simpler serial/recurrent organization is better for the tested class.

# Gate 3 — Larger population frontier

Activate only after Gate 1/2 justify larger populations.

## Question

> How far can the useful population frontier extend before selectivity, batching, communication, or state memory saturates?

Candidate counts:

- 1,024;
- 4,096;
- 16,384;
- higher counts only while the measured frontier remains useful.

Do not assume every workload needs the same population size.

Measure marginal value per:

- worker update;
- wall-clock second;
- communicated byte/scalar;
- peak state memory;
- useful evidence/capability gained.

Exit criterion:

- identify the first meaningful saturation/bottleneck and the useful count region on target hardware.

# Gate 4 — Dynamic activation

Activate only after fixed-width population execution has a useful frontier.

## Question

> Can the system spend population compute according to task difficulty and uncertainty instead of always activating the full population?

Dynamically vary:

- active workers;
- recurrent depth;
- inspected scope;
- communication budget;
- purpose (`EXPLORE`, `PROGRESS`, `CHALLENGE`, `VERIFY`, `SYNTHESIZE`) when persistent runtime work is used.

Start with deterministic allocation. Do not add a learned scheduler until real traces show a concrete limitation.

Compare adaptive execution against matched fixed-width execution.

Exit criterion:

- harder/uncertain tasks reliably consume more population compute and improve the practical capability/resource frontier.

# Gate 5 — Information transport and integration bandwidth

Activate when measurements show information movement or integration becoming the limiting resource.

## Question

> How much useful information can the population move, preserve, and integrate before communication dominates neural computation?

Measure:

- message/evidence production rate;
- useful-message/evidence fraction where measurable;
- communicated bytes/scalars;
- synchronization cost;
- memory movement;
- local-to-shared promotion rate;
- integration backlog depth/age if persistent runtime is used;
- verification throughput;
- compression/summary loss;
- rare decisive information retention;
- provenance recoverability.

Preferred progression:

1. bounded shared signal;
2. local neighborhoods/groups;
3. hierarchical summaries;
4. only then more complex routing if measurements demand it.

Never broadcast complete global state to all workers merely because it is easy to implement.

Exit criterion:

- demonstrate bounded information movement across the useful population range;
- or establish information transport as the scaling limit.

# Gate 6 — Richer learned workloads

Only after the synthetic substrate and resource frontier justify continuing.

Candidate workload families:

- large-document analysis;
- image-region understanding;
- codebase analysis;
- event/log streams;
- anomaly detection;
- multi-stage reasoning/planning;
- research-like hypothesis exploration.

Select workloads where distributed weak processing has a plausible structural advantage. Do not accumulate benchmarks for their own sake.

Exit criterion:

- identify at least one non-toy workload where the fixed-parameter population mechanism survives real learned ambiguity and still improves the practical resource/capability frontier.

# Gate 7 — Toward the ~1B learned-parameter reference

Only after smaller experiments establish a real advantage.

The goal is not:

> split 1B parameters into thousands of small independent models.

The goal is:

> **use a fixed ~1B learned system as shared machinery for a potentially very large dynamic population of weak neural states.**

At this stage investigate:

- how much of the learned budget belongs in shared worker machinery;
- whether a sparse specialist parameter bank adds value under a fixed total budget;
- population-state dimensionality;
- memory systems;
- topology/routing;
- dynamic activation;
- compiler/runtime optimization as a separate systems variable.

Exit criterion:

- compare against strong parameter/resource-matched baselines and report the complete frontier honestly.

# Systems variables that are not research claims

Profile or optimize only when measurements require them:

- homogeneous GPU batching/vectorization;
- CPU/GPU task placement;
- memory layout;
- queueing/backpressure;
- kernel launch/fusion overhead;
- compilation;
- quantization;
- checkpoint/state serialization;
- storage/indexing;
- observability.

Compiler mode remains a separate variable from neural architecture.

# Scope exclusions

Current research is local to one PC.

Do not spend research gates on:

- geographic distribution;
- multi-machine latency tolerance;
- datacenter networking;
- distributed consensus.

Those are established engineering domains and not current uncertainties.

# Global stop / redirect conditions

Stop or redirect a path when reproducible evidence shows that:

- additional runtime compute cannot produce useful capability at fixed learned parameters;
- communication adds no useful function beyond isolated workers where communication is required;
- a simpler serial/recurrent implementation dominates the practical resource frontier;
- parallel population width provides no useful latency/throughput advantage on target hardware;
- runtime state or communication cost destroys the practical advantage;
- batching/utilization collapses at the required granularity;
- larger populations add no meaningful capability/evidence or only duplicate work;
- information integration cannot keep pace economically;
- richer workloads do not preserve the synthetic advantage;
- strong parameter/resource-matched dense or recurrent baselines are consistently better.

A negative result removes uncertainty and prevents unnecessary engineering work.
