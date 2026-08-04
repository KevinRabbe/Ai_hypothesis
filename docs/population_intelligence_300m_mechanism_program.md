# Population Intelligence 300M Mechanism Evidence Program

Status: **PREPARATION ONLY — NO EXPERIMENT, ARCHITECTURE FREEZE, OR PROTECTED RESULT ACCESS**

This document refines Gate 3 of the [Population Intelligence 300M roadmap](roadmap.md). It prevents the six-day reference-training wait, later context loss, or attractive implementation ideas from turning the 50M, 100M, or 300M architecture into a preference-driven design.

The governing objective remains:

> Build and rigorously evaluate the smartest population-based language model possible within approximately 300 million learned parameters.

The model architecture is not frozen by this program. The program freezes **how mechanisms earn the right to enter that architecture**.

## Evidence ladder

The scale sequence remains:

1. approximately 19M diagnostic models;
2. approximately 50M mechanism integration;
3. approximately 100M realistic language-and-code evidence;
4. strongest justified approximately 300M model;
5. deterministic interactive worlds after the 300M cognitive system is understood;
6. larger scaling only after the 300M result;
7. cheap edge deployment as a separate later project.

A result at one size does not automatically transfer to the next size.

## Capability accounting

Every major evaluation must report three modes separately:

1. **ONE_PASS_CLOSED_BOOK** — no retrieval, tools, persistent adaptation, or extra recursive computation;
2. **RECURSIVE_POPULATION** — the same learned parameters with declared workers, rounds, search, and verification;
3. **FULL_SYSTEM** — the same model with explicitly declared memory, retrieval, tools, or persistent adapters.

At minimum, report:

- total learned parameters;
- active and trainable parameters;
- worker count and topology;
- recurrent rounds;
- inference FLOPs;
- generated reasoning tokens;
- routed messages and bytes;
- retrieved and persisted bytes;
- verifier and tool calls;
- latency, RAM, and VRAM;
- synchronization and coordination cost.

External memory or unlimited test-time compute must never be described as if it were contained in the 300M learned-parameter budget.

## Universal controls

Every decisive mechanism experiment requires:

- matched learned-parameter budgets;
- transparent active-compute accounting;
- matched training data and checkpoint selection;
- at least three predeclared initialization seeds;
- protected evaluation unavailable during mechanism design;
- preservation of negative, null, and failure results.

Protocol-specific thresholds must be preregistered before protected result access. This program deliberately does not invent one universal effect-size threshold for different tasks.

## Mechanism lanes

### M01 — Recurrent latent depth

Question: can repeated application of the same learned core improve difficult-task capability without adding learned parameters?

Required comparisons:

- one fixed checkpoint and worker count;
- rounds 1, 2, 4, 8, 16, and 32;
- equal-FLOP resampling or a wider/shallow control;
- no hidden retrieval or tools.

Primary evidence:

- capability by round;
- gain per additional inference FLOP;
- calibration by round;
- saturation or instability point.

Allowed claim: recurrence helps on the tested tasks at the tested scale.

Forbidden claim: recurrence creates unbounded intelligence or must scale.

### M02 — Diversity and private deliberation

Question: can workers develop causally useful independent error modes without becoming incoherent or collapsing into copies?

Required comparisons:

- unrestricted shared workers;
- equal-cost independent resampling;
- private-before-public deliberation;
- differentiated information access;
- adversarial counterexample workers.

Primary evidence:

- hidden-state and prediction similarity;
- error overlap;
- unique correction rate;
- minority-rescue rate;
- causal worker-ablation loss.

Different worker states are not sufficient evidence. At least some workers must uniquely correct errors or preserve decisive evidence.

### M03 — Adaptive test-time computation

Dependencies: M01 and M02.

Question: can a learned controller allocate workers, rounds, candidates, and verification only where more computation is useful?

Required comparisons:

- fixed schedule matched to mean compute;
- fixed schedule matched to maximum compute;
- random allocation;
- controller-signal ablations.

Primary evidence:

- capability/compute Pareto frontier;
- halting regret;
- wasted-compute rate;
- allocation correlation with observed difficulty;
- failure-to-escalate rate.

### M04 — Verifier-guided generation and repair

Question: does generate-test-diagnose-revise beat one-pass generation and equal-cost independent resampling when exact evidence exists?

Required comparisons:

- one-pass generation;
- equal-cost independent resampling;
- learned verifier only;
- exact execution verifier;
- injected verifier failures.

Primary evidence:

- exact task success;
- false-accept rate;
- repair gain per attempt;
- diagnosis localization;
- verification compute.

Exact compilers, tests, formal checks, or deterministic environment transitions remain authoritative where available.

### M05 — Memory versus learning separation

Question: which persistent mechanism causes factual recall, procedure acquisition, transfer, restart survival, and forgetting?

Required comparisons:

- complete context replay;
- raw retrieval only;
- compressed memory only;
- persistent adapter only;
- memory plus adapter;
- no-persistence control.

Primary evidence:

- factual recall;
- procedural transfer;
- unseen compositional generalization;
- fresh-process persistence;
- retention drop;
- persisted bytes.

Retrieval alone must not be called continual learning.

### M06 — Sequential continual learning

Dependency: M05.

Question: can the frozen-base system acquire several changing skills or procedures while preserving old capability and rejecting contradictions?

Required comparisons:

- single-skill baseline;
- multiple predeclared skill orders;
- no replay;
- bounded replay;
- isolated adapters;
- consolidation.

Primary evidence:

- acquisition by skill;
- forward and backward transfer;
- catastrophic forgetting;
- cross-skill interference;
- contradiction rejection;
- restart persistence.

One successful post-training adaptation is not general continual learning.

### M07 — Hierarchical communication

Dependency: M02.

Question: can local groups, summaries, or blackboards preserve decisive evidence while reducing communication and consensus collapse?

Required comparisons:

- no communication;
- flat all-to-all communication;
- flat sparse top-k communication;
- local groups with summaries;
- shared blackboard.

Primary evidence:

- capability;
- routed bytes and messages;
- effective worker utilization;
- decisive-evidence retention;
- group-level causal ablation.

Lower bandwidth alone is not success.

### M08 — Conditional memory allocation

Dependency: M05.

Question: does reallocating learned parameters from dense computation to sparse addressable memory improve reasoning as well as recall?

Required comparisons:

- dense model at matched total parameters;
- matched active FLOPs;
- matched training tokens;
- retrieval-only baseline;
- memory-capacity sweep.

Primary evidence:

- factual recall;
- compositional reasoning;
- code and algorithm success;
- active FLOPs;
- lookup bandwidth;
- effective recurrent depth.

More stored parameters are not automatically more reasoning capacity.

### M09 — Scale-stable parameterization

Dependencies: M01, M03, and M07.

Question: can optimizer, activation, routing, recurrence, and message scales tuned on proxy models transfer predictably to larger population models?

Required comparisons:

- independent retuning at the target size;
- proxy-to-target transfer;
- width, worker, and round sweeps;
- router and message-scale ablations.

Primary evidence:

- hyperparameter-transfer error;
- activation and gradient scale drift;
- router-entropy drift;
- training stability;
- target-performance regret.

A proxy result cannot automatically authorize the 300M settings.

### M10 — Verified search distillation

Dependency: M04.

Question: can expensive verified population search be distilled into cheaper direct behavior without teaching unverifiable artifacts?

Required comparisons:

- final-answer-only training;
- unfiltered trajectory distillation;
- verified trajectory distillation;
- student-generated trajectories;
- equal training-token control.

Primary evidence:

- direct student success;
- search-compute reduction;
- verifier pass rate;
- out-of-distribution transfer;
- teacher dependence.

Distillation is a later optimization and does not replace proving the reasoning and verification mechanism first.

## Promotion to the 50M integration model

A mechanism may enter the approximately 50M integration candidate only after all of the following are present:

1. prospective preregistration;
2. protocol-specific success threshold passed;
3. matched dense or recurrent control;
4. matched-compute control;
5. causal ablation;
6. end-to-end cost accounting;
7. replication across predeclared seeds;
8. known failure region recorded;
9. claim restricted to observed scales and tasks.

A failed mechanism may return only through a new prospective protocol addressing the identified failure. It must not be quietly modified after protected results and presented as the same test.

## 50M integration rules

- Only promoted mechanisms may enter the candidate architecture.
- Mechanism interactions require fresh ablations at 50M.
- The model must be compared with a matched dense transformer, recurrent dense baseline, equal-compute resampling, and relevant memory/adaptation controls.
- Parameter allocation across lexical encoding, recurrent core, routing, verification, memory, and plasticity must be justified by evidence.
- The 50M stage selects an architecture family; it does not prove that family at 300M.

## 100M and 300M decision boundary

The approximately 100M model must establish realistic language, code, verification, repeated adaptation, worker specialization, and training stability. Its evidence selects the 300M design.

Before the major 300M run, freeze:

- parameter allocation;
- worker, recurrence, and routing design;
- adaptive-compute controller;
- verifier boundary;
- memory and persistent-adaptation boundary;
- tokenizer, context, curriculum, and data mixture;
- optimizer and scale-transfer rules;
- matched baselines;
- protected final suites.

The architecture must not be redesigned from protected final metrics.

## Explicitly deferred work

The following do not belong in the small-scale mechanism laboratory:

- Pokémon-like emulator or procedural RPG integration;
- pixels-and-controller evaluation;
- physical robotics;
- microcontroller clusters;
- quantized edge deployment;
- scaling beyond 300M.

Deterministic interactive worlds remain a later integration benchmark. Population Edge Runtime remains a separate downstream project and must not constrain the intelligence architecture.

## Machine-readable authority

The exact lane order, dependencies, controls, metrics, allowed claims, forbidden claims, promotion requirements, and scope boundaries are mirrored in:

`ai_hypothesis/population_language/intelligence_300m_mechanism_program.py`

Material changes require a dedicated versioned proposal under the project-charter change-control process.
