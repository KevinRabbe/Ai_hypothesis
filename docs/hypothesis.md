# Population Intelligence Hypothesis

## Problem statement

A conventional dense language model generally applies a largely fixed learned computation graph to every token and request. Easy and difficult tasks therefore tend to activate similar parameter sets even when they require very different amounts or kinds of reasoning.

This project studies whether an approximately 300M learned-parameter budget can produce more useful intelligence when it is organized around recurrent population computation rather than one fixed dense forward pass.

The population is not a collection of autonomous chat agents. Workers are neural states inside one model. They may share the same learned parameters while receiving different information, developing different hypotheses, retrieving different memories, or performing different temporary roles.

## Primary hypothesis

> An approximately 300M population model can achieve stronger reasoning, coding, continual-learning, and adaptive-computation capability than a conventional dense model of comparable learned-parameter budget by repeatedly reusing a shared neural core across diverse worker states, allocating additional computation only where useful, preserving causally distinct evidence, and grounding revision in memory and verification.

This hypothesis is about **intelligence per learned parameter under transparent compute accounting**. It does not assume that the population model will always be faster, cheaper, or better on every workload.

## Sub-hypotheses

### H1 — Population scaling

At a fixed checkpoint, increasing the number of worker states can improve capability on tasks that benefit from additional independent or complementary computation.

The useful effect must exceed coordination cost and must not be explained solely by repeated identical sampling.

### H2 — Recurrent latent depth

Reusing the same learned population core for additional reasoning rounds can improve difficult-task performance without increasing learned parameter count.

A valid result must show improvement as a function of controlled additional computation and must report diminishing returns or instability.

### H3 — Diversity without incoherence

Workers become more useful when they maintain functionally different hypotheses, information access, or temporary roles before communication.

Too little diversity produces clone collapse. Too much unconstrained diversity produces noise. The project seeks a measurable operating region between those failures.

### H4 — Adaptive test-time computation

A learned controller can allocate workers, rounds, candidate generation, and verification depth according to task difficulty, disagreement, uncertainty, unresolved subgoals, or recent progress.

The claim succeeds only when adaptive allocation beats fixed-compute baselines under matched end-to-end budgets.

### H5 — Verification-guided revision

A compact model can achieve stronger reliable performance through a generate-test-diagnose-revise loop than through one-pass generation alone.

For code and deterministic environments, execution evidence remains authoritative. A learned verifier may prioritize candidates or diagnose failures but must not override exact execution results.

### H6 — Memory and learning are complementary

Facts, episodes, procedures, and generalized skills should not all be stored in the same way.

The model should separate:

- temporary worker state for the current computation;
- episodic memory for particular experiences;
- semantic or conditional memory for stable retrievable knowledge;
- persistent bounded neural adaptation for reusable learned procedures or transformations;
- immutable base weights for broadly trained capability.

A retrieval success is not automatically a learning success. Persistent learning requires useful behavior after context reset and restart, generalization beyond stored examples, and bounded interference with prior capability.

### H7 — Conditional memory can free reasoning capacity

Some learned capacity may be more useful as sparsely addressed knowledge or pattern memory than as dense computation used on every token.

The relevant question is whether reallocating part of the fixed 300M budget to conditional memory improves reasoning, code, or continual learning at matched total parameters and active compute—not merely factual recall.

### H8 — Data and curriculum determine effective capacity

A 300M model cannot afford to spend most of its capacity modeling duplication, noise, or unverified reasoning.

A staged curriculum built from high-quality language, code, procedural environments, exact execution feedback, verified search trajectories, and controlled continual-learning tasks may produce substantially more capability than an undifferentiated data mixture.

### H9 — Scale-stable mechanisms

Mechanisms discovered at approximately 19M, 50M, and 100M can inform the 300M architecture when their optimization and communication behavior are explicitly retested across scale.

No mechanism is assumed to scale automatically. The hypothesis includes the possibility that worker count, width, routing, recurrence, memory allocation, and training dynamics require scale-dependent changes.

### H10 — Deterministic interactive worlds provide an embodied-learning test

A deterministic virtual environment can test observation, memory, planning, action, consequence prediction, recovery, and continual learning before physical robotics is justified.

Structured emulator state should be tested before pixels and controller-only input so perception failures can be separated from reasoning and learning failures. Fixed games must later be randomized or replaced by procedural worlds to prevent walkthrough memorization from being mistaken for intelligence.

## Fixed-budget framing

The primary integration budget is approximately 300M learned parameters.

Candidate internal allocations are experimental rather than predetermined. Capacity may be distributed among:

- lexical representation and decoding;
- shared recurrent reasoning core;
- routing and communication;
- sparse specialists;
- conditional memory;
- verifier or value systems;
- persistent adaptation capacity.

Every allocation must be compared against alternatives under normalized data, training, parameter, and inference conditions.

## Evidence, not voting

Worker count is not truth.

A minority worker may hold decisive evidence because it received a different context fragment, memory retrieval, subgoal, or counterexample. Aggregation must therefore preserve:

- provenance;
- unique findings;
- contradictions;
- prediction confidence;
- causal contribution;
- access to original evidence.

Majority agreement is an uncertainty signal, not a sufficient decision rule.

## Deterministic computation boundary

Learned computation should be used where learning adds value. Deterministic implementations should remain authoritative for exact operations such as:

- parsing and schema validation;
- arithmetic and exact comparison;
- sorting and deduplication;
- permissions and resource limits;
- state transitions;
- compilers, tests, and formal checks;
- reproducible environment control;
- artifact hashing and provenance verification.

The model may decide when to invoke these operations and may reason over their results, but it should not replace reliable exact computation with unnecessary prediction.

## Null and failure outcomes

The primary hypothesis is unsupported for a tested configuration when one or more of the following dominate:

- additional workers become correlated copies with no causal benefit;
- recurrent rounds amplify errors or converge to unproductive loops;
- routing, synchronization, memory movement, or aggregation cost exceeds the capability gained;
- adaptive compute cannot outperform well-chosen fixed budgets;
- memory retrieval creates the appearance of learning without generalization;
- persistent adaptation causes unacceptable forgetting or cannot survive restart;
- learned verification rewards plausible but incorrect reasoning;
- sparse memory or experts reduce usable reasoning capacity;
- curriculum gains disappear on held-out compositions or environments;
- dense matched baselines remain consistently stronger;
- mechanisms that work at small scale fail to transfer toward 300M.

Negative results remain valuable because they identify which organizational mechanisms do not justify their complexity.

## Deployment boundary

Running a qualified model on microcontrollers, edge accelerators, or many inexpensive chips is a separate downstream systems problem. It may reuse sparse activation, hierarchical communication, quantization, or conditional memory, but edge deployment is not part of the primary intelligence hypothesis and must not constrain the 300M architecture prematurely.
