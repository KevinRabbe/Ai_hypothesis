# Research Questions

## Primary question

> Within an approximately 300 million learned-parameter budget, can a recurrent population model produce stronger reasoning, coding, continual-learning, and adaptive-computation capability than matched dense alternatives, and which mechanisms are responsible for any advantage?

The project must answer this through isolated mechanism tests, matched baselines, transparent compute accounting, and staged scaling rather than one final demonstration.

## RQ1 — Fixed-checkpoint worker scaling

At a fixed checkpoint, how does capability change as worker count increases while learned parameters remain unchanged?

Measure:

- task accuracy and calibration;
- compositional generalization;
- unique correct evidence contributed by additional workers;
- worker-state similarity and functional diversity;
- routed messages and communication cost;
- active memory, latency, and FLOPs;
- where gains saturate or reverse.

The question is not whether more workers generate more votes. It is whether they add useful computation or information.

## RQ2 — Recurrent latent depth

Can repeated application of the same learned population core improve difficult-task performance without increasing parameter count?

Test controlled reasoning-round schedules and measure:

- accuracy gain per additional round;
- accuracy gain per additional inference FLOP;
- convergence, oscillation, and error amplification;
- whether training depth transfers to greater inference depth;
- whether intermediate worker states become more useful or merely more confident.

## RQ3 — Worker diversity and anti-collapse

Which mechanisms produce useful independent error modes without making the population incoherent?

Candidate mechanisms include:

- private deliberation before communication;
- different context or memory access;
- temporary learned roles;
- diversity regularization;
- adversarial or counterexample objectives;
- controlled stochasticity.

Measure hidden-state similarity, proposal diversity, causal contribution, correction of another worker's error, and robustness after worker ablation.

## RQ4 — Adaptive test-time computation

Can the model learn when to allocate more workers, rounds, candidate generation, retrieval, or verification?

Possible signals include:

- predictive entropy;
- worker disagreement;
- unresolved subgoals;
- verifier rejection;
- recent improvement or stagnation;
- failure of predicted consequences.

Compare against fixed-compute baselines under matched end-to-end budgets. The controller must also learn when additional computation is unlikely to help.

## RQ5 — Persistent post-training learning

Can a frozen base checkpoint acquire a bounded new skill from limited verified examples, retain it across context reset and process restart, generalize to unseen compositions, and preserve old-task performance?

Separate measurements are required for:

- immediate acquisition;
- direct holdout improvement;
- compositional generalization;
- fresh-process persistence;
- immutable base-state verification;
- adaptation bytes and trainable parameters;
- retention and interference;
- rejection of invalid or poisoned updates.

## RQ6 — Memory versus learning

Which outcomes require retrieval, episodic memory, semantic memory, or neural adaptation?

Compare at least:

- complete context replay;
- raw retrieval only;
- compressed learned memory;
- persistent adapter only;
- memory plus adapter;
- population reasoning over each memory form.

A system has not learned a reusable skill merely because the answer is present in a retrievable record.

## RQ7 — Conditional memory allocation

How much of the fixed 300M budget should be dense recurrent computation versus sparsely addressed learned memory?

Compare allocations at matched total parameters, training data, active FLOPs, and inference budget.

Measure:

- factual recall;
- reasoning and coding performance;
- compositional generalization;
- memory lookup bandwidth;
- active parameter count;
- whether conditional memory frees the recurrent core for deeper computation.

## RQ8 — Verification-guided generation and repair

How much capability can a compact model gain from generate-test-diagnose-revise loops?

For code and deterministic tasks, compare:

- one-pass generation;
- independent resampling;
- learned candidate ranking;
- compiler or execution feedback;
- process-level diagnosis;
- population-based repair;
- exact verifier acceptance.

Measure verified success per unit of inference compute and the rate at which learned verifiers accept plausible but incorrect work.

## RQ9 — Search and distillation

Can expensive population search produce verified trajectories that are later distilled into cheaper behavior?

Study whether:

- successful search traces improve the base model;
- decisive intermediate states can be identified;
- distillation reduces required workers or rounds;
- teacher dependence decreases over time;
- distilled gains remain on held-out task families.

## RQ10 — Hierarchical communication

Can large worker populations communicate without all-to-all cost or rapid consensus collapse?

Compare:

- flat sparse routing;
- local worker groups;
- group summaries;
- hierarchical global exchange;
- blackboard or shared-memory communication.

Measure message bandwidth, latency, information loss, unique evidence retention, routing entropy, and causal contribution of each level.

## RQ11 — Parameter allocation inside 300M

What fraction of the budget should be assigned to:

- lexical representation and decoding;
- recurrent worker core;
- routing and communication;
- conditional memory;
- sparse specialists;
- verifier or value systems;
- persistent adaptation capacity?

The answer must come from ablations and matched alternatives rather than scaling the current small model proportionally.

## RQ12 — Data quality and curriculum

Which staged data curriculum produces the greatest general capability per parameter and training token?

Candidate progression:

1. language representation;
2. deterministic transformations;
3. compositional reasoning;
4. algorithms and state tracking;
5. code prediction and execution;
6. program repair and verification;
7. long-horizon decomposition;
8. tool use;
9. continual skill acquisition;
10. interactive environment learning.

Measure transfer, forgetting, contamination resistance, and whether gains survive procedurally generated held-out tasks.

## RQ13 — Fixed-budget dense and sparse baselines

Under matched conditions, how does the population model compare with:

- a conventional dense transformer;
- a recurrent dense model;
- a model with equivalent test-time sampling;
- sparse expert models;
- retrieval-augmented models;
- memory-only and adapter-only ablations?

Normalize or explicitly report:

- learned parameters;
- active parameters;
- training tokens and FLOPs;
- inference FLOPs;
- hardware;
- latency and memory;
- tools, retrieval, and verifier usage.

## RQ14 — Scale-stable parameterization

Can optimization and communication rules be designed so that hyperparameters transfer across approximately 19M, 50M, 100M, and 300M scales?

Study scaling of:

- worker width and count;
- router dimension and top-k;
- recurrent depth;
- residual and message magnitude;
- learning rates and initialization;
- adapter rank;
- conditional-memory capacity;
- gradient and activation scales.

Scaling is successful only when transferred settings remain close to optimal and preserve stable training dynamics.

## RQ15 — Longitudinal continual learning

Can the model learn from a changing stream over days or simulated organizational time without replaying the complete history?

A synthetic organization benchmark should separate:

- factual updates;
- episodic recall;
- procedural learning;
- transfer to new cases;
- conflicting information;
- personnel or rule changes;
- selective forgetting;
- consolidation of repeatedly verified knowledge.

## RQ16 — Deterministic interactive worlds

Can the qualified model observe, remember, plan, act, predict consequences, recover from mistakes, and learn over thousands of actions in a deterministic virtual environment?

Evaluation should progress from:

1. structured emulator state and constrained actions;
2. screenshots plus extracted text;
3. pixels and controller inputs;
4. randomized or procedurally generated RPG-like worlds.

A fixed game completion is not sufficient because a walkthrough may be memorized. World randomization must test genuine online learning and planning.

## RQ17 — AI-assisted model research

Can the model help improve its successor through bounded, verifiable experiments?

The loop should be:

- read repository state and prior evidence;
- propose one falsifiable hypothesis;
- implement one bounded change;
- execute tests or a controlled training probe;
- classify the result as supporting, negative, null, or invalid;
- propose the next experiment.

Measure verified uncertainty removed per GPU-hour and per unit of human review time, not lines of code produced.

## RQ18 — Scaling beyond 300M

After the 300M architecture is qualified, which mechanisms transfer to approximately 1B and larger scales?

This is a later hypothesis. It must be tested through controlled pilot runs before committing to a large training program.

Potentially scale-sensitive mechanisms include routing entropy, worker diversity, communication bandwidth, optimal recurrence, data mixture, memory allocation, and expert balance.

## Explicitly separate systems question

Ultra-cheap execution on microcontrollers, edge accelerators, or many inexpensive chips is not part of the primary research question. It belongs to a later Population Edge Runtime project unless a deployment mechanism independently improves capability within the 300M intelligence objective.
