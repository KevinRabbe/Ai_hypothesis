# AI Hypothesis — Neural Population Compute Research

This repository investigates whether a **fixed learned-parameter budget** can produce more capability when the same learned machinery is reused across a large, dynamic population of weak neural processing states instead of relying on one fixed dense computation path.

## Core hypothesis

The goal is **not** to build thousands of autonomous agents or thousands of miniature ChatGPTs.

The target is one computational organism:

- one fixed learned parameter set;
- many temporary runtime worker states;
- shared/reused neural update machinery;
- sparse bounded communication;
- recurrent population updates;
- dynamic activation so hard problems can spend more population compute than easy ones.

An individual worker may know almost nothing and may be useless as a standalone model. The research question is whether useful intelligence can exist primarily in the **organization and interaction of the population**.

Ant colonies are an existence proof that weak local units can participate in strong collective behavior. They are inspiration, not a biological blueprint.

The long-term reference budget is approximately **1 billion learned parameters**, but the immediate experiment is deliberately small enough to falsify the architecture before scaling toward that budget.

## Primary research question

> **With learned parameters held fixed, does increasing population computation produce increasing system-level capability?**

The primary graph is:

> **capability vs population compute at fixed learned parameters**

If that curve remains flat after the preregistered communication variants are implemented correctly, the population-compute path is stopped or redirected rather than expanded indefinitely.

## What changes and what stays fixed

Across one scaling curve, keep fixed:

- learned parameter count;
- exact parameter fingerprint/checkpoint;
- neural update architecture;
- training data and procedure;
- benchmark examples;
- output decoding;
- hardware;
- compiler/execution mode.

Vary only architecture-specific runtime quantities such as:

- active/available worker states;
- recurrent rounds;
- communication budget/topology;
- total worker updates.

Runtime state, activation memory, communication traffic, latency, and FLOPs are **not free**. They are measured as part of the architecture.

## Core principles

1. **Workers are weak processing elements, not complete agents.**
2. **Parameter count and runtime population are separate quantities.** More workers must not silently mean more learned weights.
3. **Population state can carry information.** Which workers are active, their local states, and bounded shared signals may be part of the computation.
4. **Communication must remain sparse and bounded.** The architecture should not solve scaling by broadcasting every worker state to every other worker.
5. **Recurrent compute is a first-class resource.** The same weights may be reused repeatedly when a problem needs more depth.
6. **Dynamic activation is preferred to mandatory full-population execution.**
7. **Deterministic algorithms should replace learned computation where exact logic is sufficient.**
8. **All systems costs count.** Memory movement, communication, synchronization, routing, batching, and latency belong in the result.
9. **Consumer hardware is a first-class target.** Initial experiments remain local to one PC.
10. **Compiler optimization is a separate systems variable.** Compiler gains must not be reported as neural-architecture gains.
11. **Negative results are valuable.** A failed scaling curve removes an architectural uncertainty.

## Current research sequence

The new primary gate is documented in:

- [`experiments/population_compute_scaling_v0/README.md`](experiments/population_compute_scaling_v0/README.md)
- [`docs/roadmap.md`](docs/roadmap.md)
- [`docs/hypothesis.md`](docs/hypothesis.md)

The first benchmark family is a synthetic **collective relay** task. Multi-hop information is distributed across local worker contexts so no single worker receives the complete solution path. This lets the project separately measure scope, communication, recurrent depth, worker updates, and runtime-state cost before moving to richer workloads.

Development population points are initially:

- 1;
- 4;
- 16;
- 64;
- 256.

Larger counts are attempted only if the small-scale curve justifies them.

## Required controls

Every serious population curve must include:

- **no communication** — extra local workers without inter-worker information flow;
- **sparse communication** — the primary population architecture;
- **serial compute control** — matched worker-update budget spent through a small number of recurrent states.

These controls separate population effects from simple extra FLOPs or extra input coverage.

## Previous work

Earlier experiments established that useful learned local transformations can survive in very small networks. The previous Step 1 sweep found a practical region around roughly 25K–100K parameters per independently trained unit, with ~50K used as a reference in later experiments.

That result remains useful background, but **finding the smallest individually useful worker is no longer the primary objective**.

The repository also contains a substantial persistent runtime, evidence/integration, partitioning, and large-scope benchmark stack. Those components are retained as reusable infrastructure. They do not count as evidence for the new fixed-parameter population-scaling hypothesis until exercised under the new scientific contract.

## Existing architecture documentation

See:

- [`docs/runtime_architecture.md`](docs/runtime_architecture.md)
- [`docs/architecture_contracts.md`](docs/architecture_contracts.md)
- [`docs/construction_policy.md`](docs/construction_policy.md)
- [`docs/research_questions.md`](docs/research_questions.md)
- [`benchmarks/step_01_benchmark_v0.md`](benchmarks/step_01_benchmark_v0.md)
- [`benchmarks/large_scope_relevance_v0.md`](benchmarks/large_scope_relevance_v0.md)
- [`benchmarks/large_scope_runtime_bridge_v0.md`](benchmarks/large_scope_runtime_bridge_v0.md)
- [`benchmarks/large_scope_result_audit_v0.md`](benchmarks/large_scope_result_audit_v0.md)

Those documents describe prior research and reusable system infrastructure. Where they conflict with the fixed-parameter population-compute gate, the new gate is authoritative for the current research objective.

## Current status

**Gate 0A is active.**

Implemented in the current slice:

- fixed parameter identity/accounting contract;
- frozen development population sizes;
- communication-mode/control vocabulary;
- preregistered per-curve minimum-effect rules;
- deterministic collective-relay world generation;
- explicit negative-result semantics.

Next implementation boundary:

> build the smallest shared-weight recurrent worker-state model that can run the collective-relay benchmark without introducing specialized workers, a learned router, external memory, or compiler-specific optimization.
