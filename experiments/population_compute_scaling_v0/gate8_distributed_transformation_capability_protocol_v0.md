# Gate-8 distributed-transformation capability scaling v0 — protocol

## Status

**DATA-FROZEN CAPABILITY-SCALING PROTOCOL — GENERATOR, TRAINING, BASELINE AND EXECUTION CLOSED.**

Base: exact qualified Gate-7 precision-result head:

`7cd29aa02d8ad0d4819978cd04f4a39b94a9bb0c`

Gate-7 established:

`G7_PRECISION_INFORMATION_CEILING_DOMINANT`.

The old benchmark is therefore closed for further routing, communication, specialization, recycling, topology or recurrent-scheduling intervention. Gate-8 changes the task family so additional runtime workers can create additional solvable capability.

## Research question

Can one fixed 19,649-parameter shared neural organism solve increasingly deep distributed composition problems as runtime population, active workers and recurrent coordination increase?

A separate secondary question is evaluated on the same exact worlds:

How does that organism compare with a conventional frozen approximately 1B-parameter pretrained model in capability and efficiency?

The 1B model is a reference, not a matched-training baseline. No parity claim may erase the difference between task-trained 19,649 shared parameters and a broadly pretrained 1B model.

## Benchmark: distributed transformation graph

Each world contains a rooted directed graph with opaque randomized node identifiers.

- One unique root-to-target path is relevant.
- Every directed edge carries one transform identifier.
- The transform library contains eight deterministic non-commuting bijections over a sixteen-symbol alphabet.
- The root symbol and target node are public.
- The correct answer is the symbol produced by applying the relevant path's transforms in exact order.
- Distractor branches use the same transform library and surface format.
- Relevant edges are exactly one eighth of all graph edges.
- There is no candidate enumeration or brute-force terminal search.
- An exact symbolic oracle computes the answer and complete relevant path.

One population worker receives exactly one edge shard plus the public query. No worker receives the complete graph or more than one edge. For depth greater than one, no individual worker can derive the final answer alone.

The population organism may use sparse recurrent communication, but each active worker may emit at most one eight-bit code per round from a fixed 256-code vocabulary. All active workers, messages, communicated bits, recurrent updates, wall time, memory and normalized compute must be recorded.

## Frozen condition matrix

Population ladder:

`32, 64, 128, 256, 512, 1024` workers.

Depth ladder:

`4, 8, 16, 32, 64, 128` ordered transformations.

A condition is valid only when:

`8 × depth <= population`.

This creates 21 population/depth conditions. Each condition receives 512 fresh test worlds.

Training uses only:

- populations 32, 64 and 128;
- depths 4, 8 and 16;
- three deterministic training seeds;
- 262,144 generated training worlds per seed.

Checkpoint selection must use held-out worlds inside the training population/depth region. The populations 256, 512 and 1024 and depths 32, 64 and 128 remain extrapolation conditions. No population-specific retraining is permitted.

## Population capability frontier

For every population, define the maximum solved depth as the deepest valid condition satisfying both:

- point accuracy at least 0.90;
- 95% confidence lower bound at least 0.85.

Positive capability scaling requires all of the following:

1. Every reported population frontier is solved under the frozen rule.
2. Maximum solved depth is non-decreasing with population.
3. At least three adjacent population steps strictly increase solved depth.
4. The final solved depth is at least four times the first solved depth.
5. Both frozen causal-ablation conditions pass.

Frozen scaling outcomes:

- `G8_POSITIVE_CAPABILITY_SCALING`
- `G8_CAPABILITY_PRESENT_NO_SCALING`
- `G8_NEGATIVE_CAPABILITY_SCALING`
- `G8_CAPABILITY_SCALING_INCONCLUSIVE`

## Causal guards

At `(population=512, depth=64)` and `(population=1024, depth=128)`, run:

- full population organism;
- no-communication ablation;
- shuffled-worker/edge assignment ablation.

For each ablation, the lower confidence bound of full-minus-ablation accuracy must exceed 0.20. This prevents a positive scaling label from being explained by a monolithic shortcut, static output bias or workers that make no causal contribution.

## Conventional 1B reference

Frozen model family:

`google/gemma-3-1b-it`.

The execution-admission branch must bind an exact immutable model revision and every required weight/tokenizer SHA256 before any benchmark world is generated.

Primary reference mode:

- instruction-tuned frozen weights;
- no task-specific gradient update, adapter or retrieval system;
- bfloat16 primary inference;
- deterministic greedy decoding, temperature zero;
- eight fixed public demonstrations from a separate namespace;
- maximum 24,576 input tokens;
- maximum 64 generated tokens;
- canonical graph serialization and exact answer marker;
- the same complete graph, root symbol and target query used by the population organism.

Any world exceeding the frozen reference input budget invalidates the benchmark encoder; it may not be silently truncated or omitted.

The reference is evaluated on all 21 shared conditions. Accuracy, prompt tokens, generated tokens, FLOPs or normalized compute, wall time, peak memory and energy proxy where available must be reported.

## Frozen 1B comparison classifier

Population-minus-reference accuracy uses 20,000 deterministic paired bootstrap samples over identical worlds.

- `G8_POPULATION_EXCEEDS_1B_REFERENCE`: pooled CI low is above zero and no condition clearly favors the reference beyond the five-point margin.
- `G8_POPULATION_NONINFERIOR_TO_1B_REFERENCE`: pooled CI low is above -0.05 and no condition clearly favors the reference beyond the margin.
- `G8_1B_REFERENCE_SUPERIOR`: pooled CI high is below -0.05.
- `G8_1B_REFERENCE_MIXED`: at least one condition clearly favors each system.
- `G8_1B_REFERENCE_COMPARISON_INCONCLUSIVE`: all other admissible evidence.

This comparison is secondary to the population-scaling question. A 1B result cannot rescue absent population capability scaling, and positive population scaling does not by itself establish 1B parity.

## Required controls and reporting

Every condition must also report:

- exact symbolic oracle accuracy;
- random-answer accuracy;
- single-worker accuracy;
- no-communication population accuracy;
- shuffled-edge and shuffled-message controls;
- accuracy by training/interpolation/extrapolation region;
- maximum solved depth by population;
- active-worker fraction;
- communicated bits;
- recurrent rounds and updates;
- capability per learned parameter;
- capability per active worker;
- capability per communicated bit;
- capability per recurrent update;
- capability per normalized compute;
- wall time and peak memory.

## Leakage and shortcut prohibitions

- Opaque node labels must be independently permuted per world.
- Transform identifiers may reveal only the primitive identity, never path position or answer.
- Relevant and distractor edges must share the same marginal formats.
- World ordering and serialization may not place relevant edges contiguously.
- Train, validation, test and 1B-demonstration namespaces must be disjoint.
- Test worlds may not influence architecture choice, checkpoint choice, prompt choice or reference revision.
- No result-dependent prompt retries, self-consistency samples or adaptive compute are allowed in v0.

## Staged admission

The next implementation must remain staged:

1. generator + exact oracle + leakage tests;
2. canonical encoder + 1B token-budget proof;
3. fixed-parameter organism architecture and training admission;
4. baseline revision/weight binding;
5. joint execution and independent audit.

No stage may expose fresh test answers before all preceding contracts are qualified.