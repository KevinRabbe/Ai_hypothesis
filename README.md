# AI Hypothesis — Population Intelligence 300M

This repository investigates whether a compact neural system can become substantially more capable by organizing computation as a recurrent, dynamically allocated population rather than relying on one fixed dense forward pass.

## Locked primary objective

> Build and rigorously evaluate the smartest population-based language model that can be produced within an approximately **300 million learned-parameter budget**.

"Smartest" means maximizing demonstrated capability in:

- reasoning and compositional generalization;
- code understanding, generation, execution-guided repair, and verification;
- adaptive test-time computation;
- persistent post-training skill acquisition;
- memory use without confusing retrieval with learning;
- uncertainty recognition, counterexample search, and self-correction;
- long-horizon planning and learning in deterministic interactive environments.

The project is not primarily optimizing for the cheapest runtime, the largest worker count, or deployment on microcontrollers. Those may become separate downstream engineering projects after the intelligence architecture is scientifically qualified.

See [`docs/project_charter.md`](docs/project_charter.md) for the change-controlled project objective and scope boundaries.

## Core hypothesis

A fixed learned-parameter budget may produce more useful intelligence when it is organized around:

- a shared recurrent neural core;
- many concurrent worker states that can develop different hypotheses;
- sparse and hierarchical communication;
- adaptive allocation of workers and reasoning rounds;
- explicit verification and revision;
- distinct temporary, episodic, semantic, and persistent learned memory;
- deterministic tools for exact operations;
- bounded post-training adaptation that survives restart without modifying the immutable base.

Workers are not thousands of autonomous chat agents. They are neural processing states inside one model and may share the same learned parameters. Additional workers are useful only when they add causally distinct information, computation, or error correction.

## Evaluation modes

Every major result should distinguish three capability modes:

1. **One-pass, closed-book model capability** — no retrieval, tools, persistent adaptation, or extra recursive compute.
2. **Recursive population capability** — the same learned parameters with additional workers, recurrent rounds, search, and verification.
3. **Full-system capability** — the same model with declared retrieval, deterministic tools, and qualified persistent memory or adapters.

Reports should include total learned parameters, active parameters, worker count, recurrent rounds, inference FLOPs, retrieved bytes, tool calls, latency, memory use, and coordination overhead. This prevents external memory or unlimited test-time compute from being misreported as model intelligence.

## Fixed research sequence

The current evidence ladder is:

1. finish and verify Population Language L0;
2. prove bounded persistent post-training learning;
3. isolate recurrent depth, worker diversity, adaptive compute, verification, memory, and routing mechanisms at small scale;
4. integrate qualified mechanisms in an approximately 50M model;
5. train and evaluate an approximately 100M language-and-code model;
6. freeze and train the strongest justified approximately 300M architecture;
7. test long-horizon continual learning in deterministic interactive worlds;
8. only then test scaling beyond 300M;
9. treat ultra-cheap edge deployment as a separate later project.

Later gates must not silently replace earlier scientific questions or redefine success after results are visible.

## Scientific discipline

- Preregister decisive experiments before protected result access.
- Keep final evaluation worlds and labels unavailable during calibration.
- Compare against matched dense and ablation baselines.
- Count routing, memory movement, synchronization, retrieval, and verification costs.
- Preserve negative and null results.
- Separate memory, retrieval, adaptation, and base-weight learning.
- Do not weaken thresholds after observing results.
- Do not claim that a mechanism scales until it is retested at the next model size.

## Project documents

- [`docs/project_charter.md`](docs/project_charter.md) — locked objective, scope, and change control
- [`docs/hypothesis.md`](docs/hypothesis.md) — primary and subordinate scientific hypotheses
- [`docs/research_questions.md`](docs/research_questions.md) — current research questions and measurements
- [`docs/roadmap.md`](docs/roadmap.md) — ordered experimental gates toward the 300M model
- [`experiments/population_language_post_training_learning_l0/protocol_v0.md`](experiments/population_language_post_training_learning_l0/protocol_v0.md) — preregistered persistent-learning protocol

## Current status

The active evidence line is Population Language L0 and its bounded Post-Training Learning L0 extension. The later 50M, 100M, and 300M architectures are intentionally not frozen yet; they must be selected from qualified small-scale evidence rather than preference.
