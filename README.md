# AI Hypothesis — Neural Population Compute Research

This repository investigates whether a **fixed learned-parameter budget** can be used more effectively when the same learned machinery is reused across a large, dynamic population of weak neural processing states instead of following one fixed dense computation path.

## Core hypothesis

The goal is **not** to build thousands of autonomous agents or miniature ChatGPTs.

The target is one computational organism:

- one fixed learned parameter set;
- many temporary runtime worker states;
- shared/reused neural update machinery;
- sparse bounded communication;
- recurrent population updates;
- dynamic activation so harder problems can spend more runtime computation than easy ones.

An individual worker may know almost nothing and may be useless as a standalone model. Useful system behavior may emerge from how many weak states are organized, what local information they hold, and how information moves between them.

Ant colonies are inspiration for weak-local-unit / strong-population behavior, not a biological implementation blueprint.

The long-term reference budget is approximately **1 billion learned parameters**, but scale is not a goal by itself. Every larger experiment must first earn its cost at small scale.

## Gate v0 result — positive

The first fixed-parameter population-compute gate is now complete on the synthetic `collective-relay-v1-answer-frontier` benchmark.

Frozen confirmation used:

- one **26,669-parameter** shared model per training seed;
- new independent training seeds `1 / 2 / 3`;
- runtime populations `1 / 4 / 16 / 64 / 256`;
- relay depths `2 / 4 / 8`;
- 1,000 untouched confirmation worlds per relay depth;
- canonical normalized sparse communication;
- matched no-communication control;
- a pass rule and 3/3 cross-seed aggregation frozen before confirmation was opened.

All **3 / 3** seeds passed. In fact every seed passed all three relay tiers and all four adjacent population steps were non-decreasing under the frozen tolerance.

Across the three seeds, mean 256-worker exact solve was:

- relay-2: **99.63%**;
- relay-4: **99.33%**;
- relay-8: **98.27%**.

Critical controls remained clean:

- exact solve given incomplete information was **0% everywhere incomplete worlds existed**;
- no-communication exact solve was **0% in all 45 seed × relay-depth × population conditions**;
- every population/control point inside one seed used exactly the same learned-parameter count and checkpoint fingerprint;
- the three confirmation seeds had distinct independently trained checkpoint fingerprints.

See:

- [`experiments/population_compute_scaling_v0/relay_v1_confirmation_result_v0.md`](experiments/population_compute_scaling_v0/relay_v1_confirmation_result_v0.md)
- [`experiments/population_compute_scaling_v0/confirmation_protocol_v1.md`](experiments/population_compute_scaling_v0/confirmation_protocol_v1.md)

The supported statement is deliberately narrow:

> **With learned parameters fixed, additional reusable runtime neural computation plus additional available distributed source scope can reproducibly produce additional capability on the tested synthetic relay task. Bounded recurrent communication is required.**

## Serial-control result — equally important

The repaired relay function was also qualified under an equal-work serial schedule.

For arbitrary fixed weights, the parallel normalized population execution and a one-live-state serial execution produce the same final shared representation/logits within floating-point tolerance across relay-2/4/8 and populations `1 / 4 / 16 / 64 / 256`, while using the same `N × relay_hops` learned worker updates.

Therefore Gate v0 does **not** establish that simultaneous wide population state is intrinsically more capable per learned update than an equal-work serial implementation.

See [`experiments/population_compute_scaling_v0/serial_schedule_equivalence_result_v0.md`](experiments/population_compute_scaling_v0/serial_schedule_equivalence_result_v0.md).

## Current primary research question

Gate v0 has answered the first question positively for one controlled task:

> Can fixed learned parameters exploit additional runtime computation/source scope to gain capability?

The next architecture-specific question is stronger and more useful:

> **Can many weak shared-weight runtime states produce a better practical capability/resource frontier than an equivalent serial/recurrent execution?**

The project now measures population organization using the language of parallel algorithms:

- **work** — total learned worker updates / useful neural operations;
- **span/depth** — the sequential dependency path that limits latency;
- wall-clock latency and throughput;
- peak activation/state memory;
- communication scalars/bytes and memory movement;
- hardware utilization;
- capability under matched work, matched wall-clock, or matched hardware budgets.

Parallel width is valuable only if it converts available hardware into a useful frontier rather than merely re-expressing serial work with more resident state.

## What stays fixed in serious comparisons

Across one scientific comparison, freeze as appropriate:

- learned parameter count;
- exact checkpoint/fingerprint;
- neural update architecture;
- benchmark examples;
- output decoding;
- hardware;
- compiler/execution mode unless it is the explicit systems variable.

Runtime state, memory movement, communication traffic, synchronization, FLOPs, latency, and scheduling overhead are **not free**. They belong in the result.

## Core principles

1. **Workers are weak processing elements, not complete agents.**
2. **Parameter count and runtime population are separate quantities.** More workers must not silently mean more learned weights.
3. **Population state may carry computation, but its value must be measured against serial/equivalent alternatives.**
4. **Communication stays sparse and bounded.** Do not solve scaling by broadcasting all worker state to all workers.
5. **Recurrent/test-time compute is a first-class resource.** Shared weights may be reused when a problem deserves more computation.
6. **Dynamic activation is preferred to mandatory full-population execution.**
7. **Deterministic algorithms replace learned computation when exact logic is sufficient.**
8. **All systems costs count.** Memory, communication, synchronization, routing, batching, and latency belong in the architecture frontier.
9. **Consumer hardware is a first-class target.**
10. **Compiler optimization is a separate systems variable.** Compiler gains are never reported as neural-architecture gains.
11. **Negative results are valuable.** A failed frontier removes an architectural uncertainty.
12. **Do not scale worker count merely because it is technically possible.** Larger populations must be justified by measured marginal value.

## Current research sequence

The immediate sequence is now:

1. **Gate 1 — work/span resource frontier:** benchmark parallel normalized versus exactly equivalent serial execution on real local hardware under matched work;
2. **Gate 2 — organization-specific capability:** only if justified, introduce a workload where persistent distributed state, locality, adaptive allocation, or parallel exploration can matter beyond a commutative serializable reduction;
3. **Gate 3 — larger populations:** test 1K+ runtime states only after Gate 1/2 demonstrate a useful frontier;
4. **Gate 4 — fixed-budget dense/recurrent baselines and richer workloads;**
5. **scale toward larger learned budgets only if the advantage survives.**

See [`docs/roadmap.md`](docs/roadmap.md) for the detailed gates.

## Previous work and reusable infrastructure

Earlier experiments established that useful learned local transformations can survive in very small networks. The previous Step 1 sweep found a practical region around roughly 25K–100K parameters per independently trained unit, with ~50K used as a reference in later experiments.

That result remains useful background, but **finding the smallest individually useful worker is no longer the primary objective**.

The repository also contains a persistent Research Ledger/runtime, scheduler, evidence/knowledge integration, partitioning, large-scope benchmarks, and qualified indexed execution paths. Those remain reusable infrastructure for later adaptive-population experiments.

## Existing documentation

See:

- [`docs/hypothesis.md`](docs/hypothesis.md)
- [`docs/roadmap.md`](docs/roadmap.md)
- [`docs/runtime_architecture.md`](docs/runtime_architecture.md)
- [`docs/architecture_contracts.md`](docs/architecture_contracts.md)
- [`docs/construction_policy.md`](docs/construction_policy.md)
- [`docs/research_questions.md`](docs/research_questions.md)
- [`docs/evolutionary_organism_direction.md`](docs/evolutionary_organism_direction.md) — preserved future direction for hybrid gradient/evolutionary optimization of whole shared-weight organisms.
- [`benchmarks/large_scope_relevance_v0.md`](benchmarks/large_scope_relevance_v0.md)
- [`benchmarks/large_scope_runtime_bridge_v0.md`](benchmarks/large_scope_runtime_bridge_v0.md)

## Current status

**Gate v0 is complete and positive.**

What is proven is fixed-parameter runtime-compute/source-scope scaling on one controlled synthetic distributed task.

What remains open—and is now the main research target—is whether the population organization itself provides a useful **capability / work / span / memory / communication / latency** advantage over simpler alternatives.

The project therefore does **not** jump directly to thousands of runtime states. Gate 1 measures the resource frontier first.
