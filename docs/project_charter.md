# Population Intelligence 300M — Project Charter v0

Status: **LOCKED PROJECT DIRECTION**

This document records the project objective and prevents later implementation choices, interesting side projects, or observed results from silently redefining the research program.

## 1. Primary objective

> Build and rigorously evaluate the smartest population-based language model that can be produced within an approximately **300 million learned-parameter budget**.

The 300M target is the main integration scale. Smaller models are experimental instruments used to discover and reject mechanisms before the expensive 300M training stage. Models larger than 300M are later scaling tests, not the current objective.

The model should maximize demonstrated capability in:

- compositional reasoning and generalization;
- code reasoning and execution-guided repair;
- adaptive allocation of inference computation;
- persistent post-training learning;
- memory use and memory consolidation;
- uncertainty recognition and counterexample discovery;
- verification, revision, and recovery from failed plans;
- long-horizon operation in deterministic interactive environments.

## 2. What counts as success

A successful 300M result must be supported by controlled comparisons rather than an attractive demonstration.

At minimum, evaluation must distinguish:

1. **One-pass, closed-book capability** — no retrieval, tools, persistent adaptation, or extra recurrent computation.
2. **Recursive population capability** — the same learned parameters with declared workers, rounds, search, and verification.
3. **Full-system capability** — the same model with declared retrieval, deterministic tools, and qualified persistent memory or adapters.

Every major comparison must report, where applicable:

- total learned parameters;
- active learned parameters;
- trainable post-training parameters;
- worker count and communication topology;
- recurrent reasoning rounds;
- inference and training FLOPs;
- generated reasoning tokens;
- retrieved bytes and memory writes;
- tool or verifier calls;
- latency, throughput, RAM, and VRAM;
- routing, synchronization, and coordination overhead.

The strongest result is not necessarily the fastest model. The primary optimization target is capability under a transparent approximately 300M learned-parameter budget. Efficiency remains an important measured constraint and tie-breaker.

## 3. Core architecture direction

The project studies a single neural system with population-structured computation. Workers are not separate conversational agents.

The current architectural direction is:

- a shared learned lexical and reasoning core;
- multiple recurrent worker states;
- worker-state diversity and private deliberation;
- sparse or hierarchical communication;
- adaptive worker and reasoning-round allocation;
- candidate generation, verification, and revision;
- distinct temporary, episodic, semantic, and persistent learned memory;
- bounded post-training adapters with an immutable base model;
- deterministic implementations for exact operations where neural prediction adds no value.

These are research hypotheses, not protected conclusions. Individual mechanisms may be removed when controlled evidence rejects them. Removing a failed mechanism does not change the project objective.

## 4. Non-goals and protected scope boundaries

The following are explicitly not primary goals of the current project:

- running the model on dozens of microcontrollers;
- minimizing cost at the expense of capability;
- maximizing worker count for its own sake;
- building thousands of autonomous agents;
- claiming AGI from benchmark performance;
- solving robotics before virtual interactive learning is understood;
- adding multimodality before language, code, memory, and continual learning are qualified;
- scaling beyond 300M before the 300M mechanism set is justified.

A later **Population Edge Runtime** project may investigate quantization, flash-backed parameters, sparse deployment, microcontroller clusters, edge accelerators, and minimum cost per inference. It may reuse a qualified checkpoint or architecture, but edge-hardware constraints must not dictate the present intelligence research.

## 5. Ordered evidence ladder

The protected high-level order is:

1. qualify Population Language L0 evidence;
2. qualify bounded persistent Post-Training Learning L0;
3. isolate candidate intelligence mechanisms at small scale;
4. integrate qualified mechanisms near 50M parameters;
5. establish realistic language and code behavior near 100M parameters;
6. freeze and train the strongest justified approximately 300M architecture;
7. evaluate long-horizon learning in deterministic interactive worlds;
8. test transfer beyond 300M only after the 300M result is understood;
9. begin a separate edge-runtime project only when a qualified model is worth deploying.

Individual gates may require additional diagnostic sub-gates. They may not be skipped merely because a later experiment is more exciting.

## 6. Scientific invariants

The following rules apply across the roadmap:

- Decisive protocols are frozen before protected result access.
- Calibration data, final evaluation data, and retention data remain separated.
- Final labels may not influence architecture or hyperparameter selection.
- Raw examples, retrieval stores, and learned adapters are reported separately.
- Base-model changes and post-training changes are reported separately.
- A restart-survival claim requires evaluation in a fresh process from a persisted artifact.
- Negative, null, contradictory, and failed-mechanism results remain part of the record.
- Comparisons use matched data, parameter, hardware, and compute conditions wherever technically possible.
- Organization costs count: routing, synchronization, memory movement, aggregation, and verifier execution are not free.
- No threshold may be weakened after the corresponding protected result becomes visible.
- No scaling claim transfers automatically from 19M to 50M, 100M, 300M, or larger scales.

## 7. Roadmap change control

The project objective, 300M integration target, high-level gate order, and edge-runtime boundary may not be changed by an ordinary implementation PR.

A material change requires a dedicated charter-version proposal that includes:

1. the exact current text being changed;
2. the proposed replacement;
3. the scientific or operational evidence requiring the change;
4. effects on existing preregistrations and result comparability;
5. migration or archival treatment for the previous plan;
6. explicit owner approval before merge.

Existing preregistered protocols remain immutable historical records. A new protocol version may supersede one prospectively, but an old protocol must never be rewritten to match observed results.

Minor wording corrections and new diagnostic details may be added without changing the objective, success definition, or gate ordering. When uncertain, treat the change as material and use a charter-version proposal.

## 8. Decision rule for new ideas

A new idea enters the main roadmap only when it plausibly improves the smartest approximately 300M model through at least one of:

- capability per learned parameter;
- capability per training token;
- capability per inference FLOP;
- generalization or robustness;
- persistent learning with bounded forgetting;
- reliable verification and correction;
- scalable training or experimental throughput that enables better science.

An idea that only reduces deployment cost belongs to the later edge-runtime project unless it independently improves the intelligence objective.

### External-model training acceleration

External models may be used as practical training tools for the later 50M, 100M, and 300M stages. They may generate examples, propose repairs, critique failed attempts, or supply alternative verified solutions when doing so shortens the end-to-end time required to train the next model.

This is an engineering acceleration rule, not a new research objective or mandatory scientific gate. It does not require a dedicated experiment, a fixed percentage-improvement threshold, or a separate capability claim. Use it when it is practically faster; stop using it when teacher generation, verification, or integration makes the training process slower or less reliable.

Teacher outputs are candidate training evidence, not authority. Exact tests, deterministic verifiers, compilation, execution, or environment outcomes should decide whether generated material is accepted. Data provenance should identify the external source and verification method, but ordinary use of teacher-generated data does not change the roadmap.

This rule applies prospectively. It must not alter an active or frozen run, its training distribution, its comparison boundary, or its preregistered interpretation.

## 9. Current interpretation

The present small-scale Population Language and Post-Training Learning experiments are mechanism probes. They do not yet establish the final 300M architecture, general language competence, or scaling behavior.

The project will earn those claims gate by gate.
