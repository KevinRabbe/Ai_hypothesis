# Population Intelligence 300M Research Roadmap

This roadmap is governed by [`project_charter.md`](project_charter.md). Its purpose is to reach the strongest scientifically justified approximately 300M population model without allowing attractive side projects, hardware constraints, or visible results to redefine the objective.

A later gate may begin only when the uncertainty it depends on has been reduced enough to justify the additional cost. Diagnostic work may run in parallel when it cannot contaminate protected results, but the main evidence order remains fixed.

## Status vocabulary

- **QUALIFIED** — implementation and evidence satisfy the frozen gate contract.
- **ACTIVE** — implementation or protected execution is in progress.
- **PREREGISTERED** — protocol is frozen before protected result access.
- **PREPARATION ONLY** — tooling may be built, but protected results may not be accessed.
- **FUTURE** — not authorized as an active scientific result.

## Global evaluation contract

Every major model result should distinguish:

1. **One-pass, closed-book capability**;
2. **Recursive population capability** with declared workers, rounds, search, and verification;
3. **Full-system capability** with declared retrieval, tools, and persistent memory or adapters.

Report, where applicable:

- total, active, and trainable learned parameters;
- worker count, topology, and communication rounds;
- recurrent depth and generated reasoning tokens;
- training and inference FLOPs;
- retrieved bytes and persistent writes;
- verifier or tool calls;
- latency, throughput, RAM, and VRAM;
- routing, synchronization, aggregation, and memory-movement cost.

No later result may hide additional memory or test-time computation inside the approximately 300M parameter claim.

---

## Gate 0 — Project charter and evidence discipline

Status: **QUALIFIED BY DOCUMENTATION PR WHEN MERGED**

Goal: lock the primary objective, scientific invariants, scope boundaries, and change-control process.

Required outputs:

- approximately 300M primary integration target;
- explicit definition of "smartest";
- separation of one-pass, recursive, and full-system capability;
- fixed high-level evidence ladder;
- edge-runtime and robotics boundaries;
- prospective-only protocol changes;
- explicit owner approval for material charter changes.

Exit criterion:

- repository entry points, hypothesis, research questions, and roadmap agree on the same objective.

---

## Gate 1 — Population Language L0 reference evidence

Status: **ACTIVE EVIDENCE LINE**

Goal: establish the smallest clean result about whether a fixed population-language checkpoint benefits from additional worker states.

Required work:

- preserve the frozen protocol, seeds, splits, checkpoints, and protected evaluation boundary;
- complete any already-authorized reference execution without interference;
- verify artifacts, manifests, hashes, and exact implementation provenance;
- evaluate the locked worker-count conditions;
- compare the population model with its matched dense baseline;
- publish only the claim supported by the frozen metrics;
- qualify a result verifier before treating the run as final evidence.

Exit criterion:

- a reproducible, independently verifiable statement about fixed-checkpoint worker scaling;
- no claim about general intelligence, continual learning, or 300M scaling.

Stop or redirect conditions:

- artifact provenance cannot be established;
- protected data boundaries were crossed;
- worker gains disappear under the locked evaluation;
- coordination cost dominates any capability gain.

---

## Gate 2 — Bounded Post-Training Learning L0

Status: **PREREGISTERED / IMPLEMENTATION PREPARATION**

Goal: prove that a frozen small checkpoint can acquire a genuinely new reusable rule from limited verified examples and preserve it as a bounded artifact.

Required capabilities:

- immutable base checkpoint;
- adaptation through examples and gradients only;
- bounded declared trainable tensors;
- no symbolic rule fitting or alternate answer path;
- direct holdout improvement;
- unseen compositional generalization;
- fresh-process persistence after context reset and restart;
- bounded retention loss on the original task;
- strict artifact schema, hashing, and create-once persistence;
- calibration worlds separated from final worlds.

Required sequence:

1. qualify the production adapter implementation;
2. publish and qualify the calibration contract;
3. qualify exact checkpoint loading and provenance;
4. qualify a true subprocess restart harness;
5. run calibration only under explicit authorization;
6. freeze the selected candidate before final-world access;
7. run final evaluation only under explicit authorization;
8. publish positive, negative, or rejection evidence exactly as observed.

Exit criterion:

- a reproducible answer to whether bounded neural post-training learning succeeds under the frozen acquisition, composition, persistence, and retention thresholds.

A calibration rejection is a valid gate result and must block unauthorized final evaluation.

---

## Gate 3 — Small-scale intelligence mechanism laboratory

Status: **FUTURE, AFTER GATES 1–2 EVIDENCE**

Goal: isolate the mechanisms that may justify inclusion in larger models while experiments remain inexpensive enough for real ablations.

The existing approximately 19M model or a comparably small successor should be used as a mechanism probe. Each decisive mechanism requires its own preregistered comparison rather than one combined architecture change.

### Gate 3A — Recurrent latent depth

Test fixed checkpoints at controlled reasoning rounds such as 1, 2, 4, 8, 16, and 32.

Exit criterion:

- identify where additional rounds improve capability, saturate, or destabilize;
- report gain per additional inference FLOP.

### Gate 3B — Worker diversity and private deliberation

Compare unrestricted shared workers with private-before-public computation, differentiated information access, diversity objectives, temporary roles, and adversarial counterexample workers.

Exit criterion:

- demonstrate causally useful independent error correction rather than cosmetic hidden-state differences.

### Gate 3C — Adaptive computation

Train or calibrate a controller for worker count, rounds, candidate generation, and verification depth.

Exit criterion:

- adaptive allocation improves quality-resource trade-offs over strong fixed schedules.

### Gate 3D — Verification-guided repair

Use exact deterministic evidence for code, transformations, and environment transitions.

Exit criterion:

- generate-test-diagnose-revise outperforms one-pass generation and equal-cost independent resampling.

### Gate 3E — Memory versus learning separation

Compare context replay, raw retrieval, compressed memory, persistent adapters, and combined systems.

Exit criterion:

- identify which mechanism is responsible for recall, procedure acquisition, transfer, persistence, and forgetting.

### Gate 3F — Conditional memory

Reallocate a controlled fraction of learned parameters from dense recurrent computation to sparse addressable memory.

Exit criterion:

- determine whether conditional memory improves more than factual recall at matched total parameters and active compute.

### Gate 3G — Hierarchical communication

Compare flat sparse routing with local groups, group summaries, shared blackboards, and hierarchical exchange.

Exit criterion:

- preserve decisive evidence while reducing bandwidth and consensus collapse.

### Gate 3H — Scale-stable parameterization

Derive and test scaling rules for width, workers, router dimension, top-k, recurrence, message magnitude, learning rates, adapter rank, and memory capacity.

Exit criterion:

- small proxy settings transfer predictably enough to justify a 50M integration run.

Global Gate 3 rule:

- a mechanism enters the larger architecture only when its isolated benefit survives matched compute accounting and a meaningful ablation.

---

## Gate 4 — Approximately 50M architecture integration

Status: **FUTURE**

Goal: combine only the mechanisms qualified at small scale and determine whether their interactions remain beneficial.

Candidate components may include:

- shared recurrent population core;
- diversity-preserving worker states;
- adaptive workers and rounds;
- hierarchical communication;
- verifier-guided revision;
- explicit memory hierarchy;
- bounded persistent adapters;
- conditional memory or sparse specialists.

Required comparisons:

- 50M dense transformer baseline;
- recurrent dense baseline;
- population model without each major mechanism;
- equal-compute resampling baseline;
- retrieval-only and adaptation-only baselines.

Required workloads:

- procedural compositional reasoning;
- algorithmic state tracking;
- small code generation and repair;
- multiple sequential learned rules;
- retention after several adaptations.

Exit criterion:

- select one coherent architecture family for realistic language-and-code training;
- reject mechanisms whose interaction cost exceeds their isolated value.

---

## Gate 5 — Approximately 100M language-and-code model

Status: **FUTURE**

Goal: establish that the architecture works beyond synthetic tasks and can support meaningful language, code, verification, and continual-learning behavior.

Training curriculum should be staged and measured:

1. language representation;
2. deterministic transformations and composition;
3. algorithms and state tracking;
4. code prediction;
5. compilation and execution feedback;
6. program repair;
7. long-horizon decomposition;
8. tool use;
9. sequential post-training skill acquisition.

Required evaluations:

- closed-book language and code baselines;
- recursive reasoning scaling;
- adaptive compute behavior;
- verified code success;
- memory-versus-learning decomposition;
- repeated adaptation and forgetting;
- worker specialization and causal ablation;
- training stability and hardware efficiency.

Exit criterion:

- evidence strong enough to freeze a 300M design rather than merely scaling the current model by preference.

---

## Gate 6 — Strongest approximately 300M population model

Status: **PRIMARY FUTURE INTEGRATION GATE**

Goal: build the smartest scientifically justified approximately 300M population language model.

### Gate 6A — Architecture freeze

Before the major training run, freeze:

- parameter allocation;
- worker and routing design;
- recurrence and adaptive-compute policy;
- memory and adapter boundaries;
- verifier integration;
- tokenizer and context contract;
- training curriculum and data mixture;
- optimizer and scaling parameterization;
- checkpoint, retention, and evaluation protocols;
- matched baseline definitions;
- protected final suites.

### Gate 6B — Training and predeclared checkpoints

Train with create-once artifacts, exact provenance, bounded failure recovery, and predefined evaluation checkpoints.

Do not redesign the architecture based on protected final metrics.

### Gate 6C — Three-mode evaluation

Evaluate separately:

1. one-pass, closed-book 300M capability;
2. recursive population capability;
3. full-system capability with declared memory, tools, and adapters.

Compare against matched 300M baselines and selected larger public or local baselines with transparent differences in data and compute.

### Gate 6D — Continual-learning and verifier stress tests

Test:

- several sequential new skills;
- restart persistence;
- selective updating of changing facts and procedures;
- poisoning or contradiction rejection;
- retention after repeated adaptations;
- verifier failure and adversarial plausible solutions;
- ability to stop spending compute when no improvement is likely.

Exit criterion:

- a reproducible capability profile identifying where the 300M population model is stronger, equal, or weaker than dense and recurrent alternatives;
- exact accounting for the mechanisms responsible;
- no automatic claim that results transfer to larger scales.

---

## Gate 7 — Deterministic interactive world learning

Status: **LATER INTEGRATION BENCHMARK**

Goal: test whether the qualified model can combine observation, memory, planning, action, consequence prediction, recovery, and continual learning over thousands of steps.

Progression:

### Level 1 — Structured environment state

Provide controlled state such as location, visible entities, dialogue, inventory, party, goals, and available actions.

Purpose: isolate planning, memory, and learning from visual perception.

### Level 2 — Screen plus extracted text

Provide screenshots and dialogue text without hidden state.

Purpose: add visual grounding and partial observability.

### Level 3 — Pixels and controller only

Provide frames and controller actions.

Purpose: evaluate the full perception-action loop after the cognitive components are already understood.

### Level 4 — Randomized or procedural RPG worlds

Randomize maps, items, NPC roles, rules, prerequisites, and objectives.

Purpose: prevent walkthrough memorization from being mistaken for online learning.

Required persistence test:

- terminate the process;
- clear temporary context and worker state;
- reload the immutable base plus declared persistent artifacts;
- test use of earlier discoveries in novel situations;
- measure retention in unrelated worlds.

Exit criterion:

- demonstrate which parts of long-horizon performance arise from episodic memory, procedural learning, generalization, planning, and persistent adaptation.

Physical robotics remains outside scope until this virtual test exposes a specific limitation that embodiment is required to study.

---

## Gate 8 — Scaling beyond 300M

Status: **FUTURE AFTER GATE 6**

Goal: determine whether the qualified 300M mechanisms transfer toward approximately 1B and larger models.

Required order:

1. short controlled scaling pilot;
2. verify optimization and routing stability;
3. retest worker diversity and recurrence;
4. retest memory allocation and expert balance;
5. compare predicted versus observed scaling;
6. authorize a larger full run only after the pilot passes.

Exit criterion:

- evidence-based decision to scale, redesign, or stop.

A successful 300M model makes larger scaling scientifically easier; it does not make it automatic or operationally cheap.

---

## Separate later project — Population Edge Runtime

Status: **NOT A CURRENT ROADMAP PRIORITY**

Potential goal: preserve as much capability as practical while minimizing deployment cost, power, active memory, and latency.

Possible topics:

- quantization and distillation;
- flash-backed weights or conditional memory;
- sparse activation;
- hierarchical partitioning across devices;
- compressed communication;
- microcontrollers, CPUs, NPUs, or inexpensive accelerators;
- fault tolerance and energy measurement.

This project may begin after a qualified checkpoint is worth deploying. It must not constrain the intelligence architecture merely to make a many-chip demonstration easier.

---

## Global stop and redirect conditions

At any gate, stop, narrow, or redirect a mechanism when evidence shows that:

- additional workers add correlated errors rather than useful computation;
- recurrence amplifies mistakes or creates loops;
- diversity objectives produce noise instead of correction;
- routing or communication overhead dominates;
- memory retrieval is being mislabeled as learning;
- persistent adaptation cannot generalize or preserve retention;
- learned verification accepts persuasive incorrect work;
- conditional memory or specialists reduce usable reasoning capacity;
- adaptive compute cannot beat fixed schedules;
- gains vanish on procedural held-out tasks;
- small-scale mechanisms fail at the next scale;
- matched dense or recurrent baselines remain consistently stronger.

A negative result is a successful research outcome when it removes uncertainty and prevents an unjustified expensive scale-up.

## Change control

Material changes to the primary objective, approximately 300M target, gate ordering, success definition, or edge-runtime boundary require a new version of [`project_charter.md`](project_charter.md) through a dedicated proposal and explicit approval.

Existing preregistered protocols remain immutable historical records.
