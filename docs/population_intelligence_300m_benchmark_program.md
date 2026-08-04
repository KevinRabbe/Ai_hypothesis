# Population Intelligence 300M Benchmark Evidence Program

Status: **PREPARATION ONLY — NO INSTANCE GENERATION, MODEL EXECUTION, OR PROTECTED RESULT ACCESS**

This document turns the [mechanism evidence program](population_intelligence_300m_mechanism_program.md) into a controlled capability-evaluation plan. It defines what the project must measure before mechanisms can be promoted into the approximately 50M integration model and before a 100M result can justify the final 300M architecture.

The benchmark program is designed to prevent five common errors:

1. treating memorized public benchmarks as evidence of online learning;
2. calling retrieval continual learning;
3. hiding extra inference compute inside the 300M parameter claim;
4. using learned judges when exact execution evidence exists;
5. redesigning tasks, thresholds, or splits after protected results are visible.

## Split roles

Every decisive benchmark protocol must use four disjoint split roles:

| Split | Role |
|---|---|
| development | visible for implementation, debugging, and generator checks |
| calibration | visible only for predeclared hyperparameter or candidate selection |
| validation | protected until architecture and selection are frozen |
| test | protected until explicit final-execution authorization |

Development, calibration, validation, and test generator seeds must be disjoint. Validation and test labels, exact traces, and hidden tests must be unavailable during design.

Public benchmarks may provide supportive context, but they must not be the sole evidence used to select population mechanisms. The decisive evidence should be procedural or otherwise held out under a versioned generator and scorer.

## Capability modes

Every benchmark must report:

1. **ONE_PASS_CLOSED_BOOK**;
2. **RECURSIVE_POPULATION**;
3. **FULL_SYSTEM**.

The result must identify learned parameters, active parameters, workers, rounds, inference FLOPs, routing, retrieved bytes, persistent writes, verifier calls, tool calls, latency, RAM, and VRAM. Failures, malformed outputs, timeouts, and verifier rejections count against the result.

At least three predeclared initialization seeds are required for decisive claims.

## Global baselines

Every major mechanism benchmark should include, where applicable:

- a matched dense transformer;
- a matched recurrent dense model;
- the population model with the target mechanism ablated;
- equal-compute independent resampling.

Task-specific baselines are added below. Parameter matching alone is insufficient; active computation and end-to-end coordination cost must also be reported.

## B01 — Procedural compositional rule induction

Mechanism lanes: M01, M02, M03, M05, M06.

Question: can the model infer new latent rules from limited examples and apply them to deeper unseen compositions while preserving old capability?

Exact oracle: a deterministic rule executor available only to the scorer.

Difficulty axes:

- number of new rules;
- composition depth;
- example coverage;
- distractor rules;
- number of sequentially learned rule sets.

Required metrics:

- direct-holdout accuracy;
- composition accuracy by depth;
- paired gain confidence bound;
- original-task retention drop;
- fresh-process restart drift;
- adaptation parameter and byte budgets.

Required baselines:

- no adaptation;
- full context replay;
- retrieval only;
- a dense adapter with matched trainable parameters.

Required failure slices:

- unseen operator;
- unseen composition;
- conflicting examples;
- adaptation beyond the declared budget.

Allowed claim: the model acquires and composes the tested rule family.

Forbidden claim: the result proves general continual learning.

## B02 — Algorithmic state tracking

Mechanism lanes: M01, M03, M07, M09.

Question: can recurrent population computation maintain and update exact latent state over long sequences more reliably than matched controls?

Exact oracle: a reference state machine.

Difficulty axes:

- sequence length;
- state dimension;
- dependency distance;
- branching factor;
- irrelevant-event rate.

Required metrics:

- final-state exact accuracy;
- intermediate-state accuracy;
- first error position;
- gain per inference FLOP;
- routed bytes;
- recovery after distractors.

Required failure slices:

- long dependency;
- high distractor rate;
- state collisions;
- recurrent-round saturation.

Success on this family is evidence for state tracking, not by itself for planning.

## B03 — Graph planning and counterfactual search

Mechanism lanes: M01, M02, M03, M07.

Question: can diverse workers search competing plans, preserve minority evidence, and revise after counterexamples under controlled compute?

Exact oracle: deterministic graph search and a transition simulator.

Difficulty axes:

- graph size;
- path depth;
- deceptive branch count;
- constraint count;
- dynamic edge changes.

Required metrics:

- valid-plan rate;
- optimality gap;
- counterexample recovery;
- minority-rescue rate;
- compute to solution;
- halting regret.

Required baselines:

- greedy one-pass planning;
- equal-compute beam search or resampling;
- flat population communication;
- hierarchical population communication.

Required failure slices:

- deceptive local optimum;
- conflicting constraints;
- late invalidation;
- no valid plan.

Success on synthetic graphs does not prove a general world model.

## B04 — Verified code synthesis

Mechanism lanes: M04, M10.

Question: can the model generate programs satisfying hidden executable specifications, and can verified search later be distilled into cheaper direct behavior?

Exact oracle:

- parser;
- compiler or interpreter;
- static checks where relevant;
- hidden tests;
- time and memory limits.

Difficulty axes:

- specification length;
- algorithmic complexity;
- API surface;
- hidden edge-case count;
- required revision steps.

Required metrics:

- compile rate;
- hidden-test pass rate;
- exact success rate;
- false verifier acceptance;
- attempts to success;
- verification compute.

Required baselines:

- one-pass code generation;
- equal-compute resampling;
- learned verifier only;
- exact-verifier repair loop.

Required failure slices:

- compiles but is wrong;
- overfits visible tests;
- API misuse;
- exceeds resource limits.

Compilation or visible tests alone are never sufficient correctness evidence.

## B05 — Code diagnosis and repair

Mechanism lanes: M04, M10.

Question: can the model localize failures and make minimal verified repairs more efficiently than complete regeneration?

Exact oracle: a fault-injection manifest plus hidden regression tests.

Difficulty axes:

- fault count;
- distance from symptom to root cause;
- codebase size;
- test ambiguity;
- interacting faults.

Required metrics:

- fault-localization accuracy;
- repair success;
- regression-free rate;
- patch size;
- attempts to repair;
- false-diagnosis rate.

Required failure slices:

- misleading stack trace;
- multiple root causes;
- plausible but incorrect patch;
- nonlocal regression.

Small patches do not automatically demonstrate causal understanding.

## B06 — Changing organization memory and procedure learning

Mechanism lanes: M05, M06, M08.

Question: can the system learn people, roles, procedures, exceptions, and changes over time without replaying the complete history?

Exact oracle: a versioned synthetic organization database and policy engine.

Example world state may include:

- employees and teams;
- responsibilities and permissions;
- projects and dependencies;
- approval procedures;
- organization-specific terms;
- exceptions;
- chronological changes.

Difficulty axes:

- organization size;
- change rate;
- procedure depth;
- exception count;
- time since observation.

Required metrics:

- current-fact accuracy;
- stale-fact error rate;
- procedure success;
- transfer to novel cases;
- fresh-process persistence;
- retention on unrelated organizations.

Required baselines:

- full-history context;
- retrieval only;
- persistent adapter only;
- memory plus adapter.

Required failure slices:

- employee or role change;
- conflicting update;
- exception to policy;
- stale retrieval.

Fact recall alone is not procedural learning. This benchmark explicitly separates the two.

## B07 — Conditional memory capacity allocation

Mechanism lanes: M05, M08.

Question: at matched total parameters and active compute, what allocation between dense reasoning and addressable memory maximizes useful capability?

Exact oracle: task-specific exact scorers plus a versioned memory-query manifest.

Difficulty axes:

- memory parameter fraction;
- lookup sparsity;
- knowledge novelty;
- reasoning depth;
- lookup collision rate.

Required metrics:

- recall accuracy;
- reasoning accuracy;
- code success;
- active FLOPs;
- lookup bandwidth;
- capability per total learned parameter.

Required baselines:

- all-dense learned parameters;
- external retrieval only;
- conditional memory at matched total parameters;
- dense model at matched active compute.

Required failure slices:

- memory collision;
- missing entry;
- misrouted lookup;
- reasoning-capacity starvation.

The best ratio at small scale must not be frozen for 300M without scale-transfer evidence.

## B08 — Adaptive compute challenge set

Mechanism lanes: M01, M02, M03, M09.

Question: can the model identify which instances benefit from additional workers, rounds, search, or verification without spending maximum compute everywhere?

Exact oracle: a mixture of exact task oracles with predeclared latent difficulty factors.

Difficulty axes:

- minimum useful rounds;
- minimum useful worker count;
- ambiguity;
- verification need;
- unsolvable-instance rate.

Required metrics:

- quality/compute frontier;
- regret relative to the best available compute schedule;
- over-compute rate;
- under-compute rate;
- correct stopping on unsolvable instances;
- difficulty calibration.

Required baselines:

- fixed low compute;
- fixed mean compute;
- fixed maximum compute;
- random compute allocation.

Required failure slices:

- easy but uncertain-looking;
- hard but confidently wrong;
- unsolvable;
- misleading verifier signal.

Controller confidence is not automatically a general measure of task difficulty.

## Result-record contract

Every result row must identify:

- benchmark and version;
- split and generator/scorer provenance;
- model and checkpoint hash;
- initialization seed;
- capability mode;
- total, active, and trainable parameters;
- workers and recurrent rounds;
- inference FLOPs;
- routed messages and bytes;
- retrieved and persisted bytes;
- verifier and tool calls;
- latency, RAM, and VRAM;
- primary and secondary metrics;
- artifact provenance.

Aggregate reports must retain per-seed and per-difficulty-slice evidence rather than publishing only one mean.

## Decision rules

- A benchmark protocol must freeze generator version, split seeds, metrics, thresholds, and baselines before protected access.
- Exact oracles override learned judges wherever an exact oracle exists.
- Validation may inform only the selection declared by the protocol.
- Test remains unavailable until candidate and evaluation procedure are frozen.
- Negative and null results are valid evidence.
- A mechanism is promoted only under the separate promotion requirements in the mechanism program.
- Benchmark success at 19M or 50M does not prove 100M or 300M transfer.

## Deferred integration benchmarks

The following remain later, after the cognitive architecture is qualified:

1. structured deterministic RPG state;
2. screenshots plus extracted text;
3. pixels and controller actions;
4. randomized or procedural RPG worlds.

They are valuable long-horizon integration tests, but they must not be used now to mix perception, control, planning, memory, and learning into one undiagnosable result.

Physical robotics and Population Edge Runtime remain outside this benchmark program.

## Machine-readable authority

The exact benchmark order, mechanism bindings, oracles, difficulty axes, metrics, baselines, failure slices, split roles, result schema, and claim boundaries are mirrored in:

`ai_hypothesis/population_language/intelligence_300m_benchmark_program.py`

Material changes require a dedicated versioned proposal under the project-charter change-control process.
