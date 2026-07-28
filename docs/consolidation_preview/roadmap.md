# Research Roadmap — Consolidation Preview

> Intended to replace `docs/roadmap.md` only after the first Gate-2 development artifact/provenance is safe and the consolidated tree is qualified.

## Primary objective

> **Determine whether a fixed learned-parameter system can gain useful capability and resource efficiency by reusing the same learned machinery across a dynamic population of weak runtime neural states.**

The project does not reward population count, architectural complexity or compute usage by themselves. Each gate must remove one decision-relevant uncertainty and earn the next experiment.

# Gate 0 — Fixed-parameter population scaling — COMPLETED POSITIVE

## Question

> With learned parameters held fixed, can additional reusable runtime neural computation plus additional available distributed source scope reproducibly produce additional capability?

## Canonical result

Relay-v1 frozen confirmation used:

- one 26,669-parameter shared model per independently trained seed;
- confirmation seeds `1 / 2 / 3`;
- populations `1 / 4 / 16 / 64 / 256`;
- relay depths `2 / 4 / 8`;
- untouched confirmation worlds;
- normalized bounded recurrent communication;
- matched no-communication control.

All 3/3 confirmation seeds passed the frozen Gate rule.

Mean exact solve at population 256:

- relay-2: 99.63%;
- relay-4: 99.33%;
- relay-8: 98.27%.

No-communication exact solve remained 0%; incomplete-information exact solve remained 0% wherever incomplete worlds existed.

## Supported conclusion

Fixed learned machinery can reproducibly convert additional runtime neural computation/source scope into additional capability on the controlled relay task.

## Hard boundary

The relay function is serializable at matched learned recurrent-update count.

Gate 0 therefore does not prove that simultaneous width creates function-level capability unavailable to equivalent serial compute.

That boundary activated Gate 1.

# Gate 1 — Work/span resource frontier — COMPLETED POSITIVE

## Question

> Does simultaneous/batched population execution provide a useful practical resource frontier over mathematically equivalent serial schedules on the real target GPU?

## Frozen comparison

Schedules:

- parallel normalized;
- low-memory serial normalized;
- compute-matched cached serial normalized.

Matrix:

- population `1 / 4 / 16 / 64 / 256`;
- relay `2 / 4 / 8`;
- batch 1 and batch 64;
- eager execution;
- complete precision-aware correctness preflight before timing.

## Result

On the RTX 4060 Ti 16 GB target, parallel execution was faster than both serial controls in every admitted cell.

Descriptive complete-matrix CUDA-event geomeans were approximately:

- 16.01× versus cached serial;
- 19.89× versus low-memory serial.

Parallel latency grew strongly sublinearly over the tested population ladder while serial schedules scaled roughly with serialized worker work.

## Supported conclusion

The target GPU can convert simultaneous population width into a strong practical eager-CUDA latency/throughput advantage for this frozen relay computation.

## Hard boundaries

Gate 1 does not establish:

- organization-specific capability advantage;
- per-FLOP superiority;
- real-workload superiority;
- scaling beyond 256 states;
- compiled/graph-mode frontier;
- multi-machine behavior.

## Gate-1 v0 history

The original frozen FP32 tensor-allclose preregistration failed before timing and remains recorded as failed.

A separate precision-aware v1 protocol was frozen after correctness diagnostics but before admitted timing. The v0 tolerance was not silently enlarged.

# Gate 2 — Organization-specific persistent-state capability — ACTIVE DEVELOPMENT

## Question

> **With learned parameters, inspected information and total learned update count held fixed, can a larger population of persistent runtime neural states improve delayed associative capability by reducing state interference/locality collisions?**

This gate deliberately removes Gate 0's source-scope confound: every compared width receives the same entity observations.

## Workload — delayed keyed traces

Each world contains entities with keys and 4-bit payloads.

The model receives:

1. four payload/evidence rounds;
2. four interference/retention rounds;
3. a delayed query for one entity's payload.

Entity counts:

- 16;
- 64;
- 256.

Population widths:

- 1;
- 4;
- 16;
- 64;
- 256;

Widths are truncated at the entity count.

At every width/control, each world uses exactly:

`8 × entity_count`

learned recurrent updates.

The information presented is also held fixed.

## Stable persistent organization

A parameter-free world permutation determines entity rank.

Stable mapping uses:

`slot = rank mod width`

Therefore larger width reduces how many entities must share one persistent state without adding learned parameters.

At entity count 256, collision load is:

```text
width 1   → 256 entities/state
width 4   → 64 entities/state
width 16  → 16 entities/state
width 64  → 4 entities/state
width 256 → 1 entity/state
```

## Learned substrate

Current development substrate:

- one shared GRU-style learned update rule;
- state width 64;
- query width 24;
- four payload-bit logits;
- one checkpoint reused across every width/control in an evaluation;
- query identity kept outside the observation stream;
- parallel and serial persistent schedules mechanically qualified for output equivalence.

These are development settings, not a claim of optimality.

## Causal controls

### 1. Serial persistent

Same state bank, same mapping, same function, same observations and same recurrent updates, time-multiplexed rather than simultaneously batched.

Output difference is a correctness failure, not a capability result.

Purpose: separate neural organization/capability from execution scheduling.

### 2. Reshuffled locality

Same width, state count, information and learned work, but entity-to-slot mapping changes each round.

Purpose: test whether stable local state identity matters rather than width alone.

Width 1 stable vs reshuffled is an exact identity control.

### 3. Reset state

Same width, routing and learned work, but runtime neural states reset every round.

Purpose: test whether persistent state itself carries the capability rather than repeated stateless processing.

## Development-only phase

The first target-GPU run is a learnability/development probe.

It uses only development-world seed domains and explicitly reports:

`scientific_decision = DEVELOPMENT_ONLY_NOT_ASSIGNED`

Development evidence may justify changing the training recipe. It may not become a Gate verdict.

The development result must be interpreted under the pre-result map rather than inventing a success criterion after inspection.

Possible branches include:

1. substrate does not learn;
2. substrate learns but width does not help;
3. width helps but causal controls fail;
4. clean directional persistent/locality pattern;
5. mixed difficulty/frontier boundary.

Each branch defines another development action or a stop/redirect. None automatically opens confirmation.

## Confirmation lock

Before confirmation can open, freeze:

- final model architecture;
- training recipe;
- optimizer/hyperparameters;
- world construction;
- stable/reshuffled/reset control matrix;
- width/entity ladder;
- numerical schedule-equivalence rule;
- confirmation decision rule;
- confirmation training-seed set;
- confirmation world-count/split policy.

Confirmation then requires untouched worlds and multiple new training seeds.

No development checkpoint or development threshold may be promoted post hoc as confirmation evidence.

## Gate-2 continuation criterion

Continue if evidence supports an organization-specific effect under matched information/work, for example:

- larger stable persistent width improves delayed capability;
- reshuffling materially damages the gain;
- resetting state materially damages the gain;
- serial-persistent execution preserves the same neural result;
- resource cost remains practically meaningful.

The final confirmation rule must be frozen before confirmation exposure; the above is the causal direction, not a post-result numeric threshold.

## Negative/redirect outcome

Treat Gate 2 as negative for the tested substrate if qualified confirmation shows that:

- persistent width does not improve capability;
- reshuffled/reset controls perform equivalently, removing the proposed causal mechanism;
- a simpler recurrent/serial state organization matches the capability under the same practical budgets.

A negative Gate 2 does not invalidate Gate 0 or Gate 1.

It would mean the tested population organization does not earn further scaling.

# Gate 3 — Larger population frontier — LOCKED

## Activation requirement

Do not test 1K+ runtime states merely because Gate 1 showed the GPU can batch 256 states.

Activate only after confirmed evidence that an organization-specific workload benefits from population organization under matched practical budgets.

## Question

> How far can the useful population frontier extend before memory, communication, batching, selectivity or state-interference benefits saturate?

Candidate counts after activation:

- 1,024;
- 4,096;
- 16,384;
- larger only while marginal value remains measurable.

Measure marginal capability/value per:

- worker update;
- wall-clock time;
- peak state memory;
- communicated byte/scalar;
- useful information/state retained.

Stop at the first meaningful saturation region rather than targeting an impressive worker number.

# Gate 4 — Dynamic activation — LOCKED

## Activation requirement

Requires a useful fixed-width organization frontier first.

## Question

> Can the system allocate population compute according to task difficulty/uncertainty instead of always using maximum width/depth?

Potential variables:

- active state count;
- recurrent depth;
- inspected source scope;
- communication budget;
- early stopping/continuation.

Start with deterministic allocation.

Do not add a learned scheduler until deterministic traces reveal a concrete limitation.

Compare against matched fixed-width execution.

# Gate 5 — Information transport and integration — LOCKED

## Activation requirement

Measurements must show that information movement/preservation/integration is becoming a limiting resource.

## Question

> How much useful information can a population move, retain and integrate before coordination dominates neural computation?

Potential measurements:

- message/evidence production rate;
- useful-message fraction;
- communication bytes/scalars;
- synchronization cost;
- local-to-shared promotion;
- integration backlog when persistent external work is used;
- verification throughput;
- compression/summary loss;
- rare decisive information retention;
- provenance recoverability.

Preferred progression:

1. small bounded shared signal;
2. local groups/neighborhoods;
3. hierarchical summaries;
4. more complex routing only after measured need.

Never solve the communication problem by broadcasting complete global state everywhere.

# Independent systems variable — compiler / graph execution

Compiler optimization is not a neural-research gate.

Gate 1 established an eager-CUDA reference.

If future profiling makes it decision-relevant, compare the same frozen function/schedule under clearly separated execution modes such as:

- eager;
- `torch.compile` default;
- reduced-overhead/graph modes;
- custom kernels only after profiling identifies a concrete launch/fusion bottleneck.

Never report compiler speedup as neural capability gain.

# Deferred future direction — evolutionary organism optimization

Preserved at:

`docs/future/evolutionary_organism_direction.md`

Activate only after the substrate exhibits a meaningful phenotype worth optimizing.

First clean future question:

> Under matched total training compute and fixed deployed architecture/parameter count, can an evolutionary archive of gradient-trained shared-weight lineages discover a better held-out population-compute capability/resource frontier faster or more reliably than ordinary independent training runs?

Evolution searches organism space; runtime population computation searches problem space.

Keep the first evolutionary test architecture-fixed and multi-objective. Preserve lineage diversity and strict confirmation/test isolation.

# Repository / scientific hygiene rules

- Current active code should reflect current experiments, not every historical implementation program.
- Historical negative/failed preregistrations remain directly preserved.
- Deferred infrastructure can leave the active checkout while staying in Git history.
- A result is not valid merely because its code/tests execute.
- Mechanics qualification and neural evidence must remain explicitly separated.
- Development data may change a development recipe; confirmation data may not.
- Larger experiments must earn their cost through earlier evidence.
