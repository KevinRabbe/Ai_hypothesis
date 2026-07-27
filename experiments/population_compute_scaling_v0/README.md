# Population Compute Scaling Gate v0

## Purpose

This experiment replaced the earlier immediate objective of finding the smallest independently useful worker.

The primary question is:

> **With learned parameters held fixed, does increasing runtime population computation produce increasing system-level capability?**

The experiment is intentionally falsifiable. Population size is not a goal by itself. A positive result must survive fixed-parameter identity, matched source scope, no-communication controls, exact accounting, serial-compute interpretation, and untouched confirmation.

Ant colonies remain an inspiration for the weak-local-unit / strong-population possibility. They are not an implementation blueprint.

## Frozen scientific boundary

The learned model is one shared parameter set. Runtime worker count does **not** create additional learned weights.

A runtime worker is temporary state plus a local observation/context. Many runtime workers may reuse the same learned update machinery.

Across one population-size curve, all of the following remain identical:

- learned parameter count;
- exact parameter fingerprint/checkpoint;
- worker/update architecture;
- training data and training procedure;
- benchmark examples;
- output decoder;
- hardware;
- compiler/execution mode.

The measured runtime variables are:

- available/active worker count;
- recurrent population-update depth;
- communication budget/topology;
- total learned worker updates.

Compiler optimization remains a separate systems variable and is never mixed into the neural result.

## First benchmark family — collective relay

Each world contains 256 local key/value relationships. A query begins at one key. The answer is reached only after following a multi-hop chain whose required relationships are distributed across local worker contexts among distractors.

A single local worker never receives the complete chain.

The task separates:

1. **scope availability** — whether every required chain relationship is present inside the active worker prefix;
2. **communication/utilization** — whether the shared learned computation can turn those distributed relationships into the answer;
3. **recurrent depth** — how many population updates are available to continue the chain.

The benchmark is synthetic. It tests an architectural property, not language intelligence or general intelligence.

### Relay v1 answer-frontier repair

The original relay construction was scientifically contaminated because the final answer-producing edge could become visible before the declared first-complete population threshold. A model could therefore sometimes obtain the answer value without actually executing the full relay.

`collective-relay-v1-answer-frontier` fixes only that benchmark defect:

- relay-2 and relay-4 cycle through first-complete thresholds `4 / 16 / 64 / 256`;
- relay-8 cycles through `16 / 64 / 256`;
- all required chain edges lie inside the declared threshold;
- at least one required chain edge lies beyond the previous frozen population point;
- the **final answer-producing edge itself** is forced into that frontier region.

Therefore every population point below the threshold lacks both a complete chain and direct visibility of the final answer value.

The threshold is benchmark metadata only and is never supplied as neural input.

## Population counts

The frozen initial ladder is:

```text
1
4
16
64
256
```

Larger populations such as `1,024 / 4,096 / 16,384` are considered only after Gate v0 is resolved. The project does not jump to 10K-100K workers merely because those counts are representable.

## Required controls

### A — no communication

Workers receive their local contexts and run the same shared neural update, but population-produced state cannot move between workers during recurrent updates.

A final pooled readout still produces one system answer. This measures what additional scope plus one-shot set aggregation can achieve without recurrent inter-worker information transfer.

### B — sparse shared communication

Workers exchange only one bounded shared signal field.

This is the primary topology tested by Gate v0.

#### `sparse_shared_v0` — legacy transport ablation

The first implementation used independent sigmoid gates followed by an unnormalized sum. Development diagnostics established two correctness/training bottlenecks:

- end-to-end relay loss did not provide enough credit assignment for selector/gate learning at larger widths;
- many small nonmatch messages accumulated under the unnormalized sum and corrupted the shared query at width 256.

The legacy implementation remains runnable as an ablation. Its failure is preserved as evidence rather than overwritten.

#### `sparse_shared_v1` — repaired canonical sparse transport

The repaired implementation keeps the same shared-field topology and learned parameter count while applying two demonstrated fixes:

1. training-only supervision teaches the existing gate which local record matches each clean relay query;
2. inference uses parameter-free softmax normalization across gate logits before combining candidate values.

No oracle information enters inference and no learned parameters are added.

`SPARSE_SHARED_V1` is therefore a corrected implementation of the preregistered sparse-shared design, not a third communication topology invented after the result.

### C — serial compute control

The same repaired relay computation is also executed with one live learned state per sample while time-multiplexing all N record-local updates and using an online softmax accumulator.

The serial and parallel schedules use:

- the same learned weights;
- the same source records;
- the same `N × relay_hops` learned worker-update count;
- the same candidate reduction;
- the same output head.

The full `1 / 4 / 16 / 64 / 256` regression establishes numerical schedule equivalence for arbitrary fixed weights.

This means the current repaired relay benchmark does **not** show additional function-level capability caused specifically by simultaneous state residency. Width currently provides parallel execution / lower sequential depth, while the same mathematical function can be serialized at equal learned-update count.

That result narrows the scientific claim but does not invalidate fixed-parameter runtime-compute scaling.

## Communication-topology budget

Gate v0 originally allowed only:

1. one bounded shared field;
2. one hierarchical-summary rescue topology if the shared-field topology genuinely failed after ordinary debugging.

The normalized v1 repair stays inside topology 1. It fixes weighting/credit assignment; it does not introduce a new routing graph.

`hierarchical_summary_v0` therefore remains the only unused topology-level rescue allowed by the original experiment design. It is not activated merely because the legacy v0 implementation failed.

## Mandatory scope/capability decomposition

Raw solve rate is insufficient because increasing population also exposes more source records.

Every condition records:

- `task_count`;
- `information_complete_count`;
- `solved_information_complete_count`;
- `solved_count`.

Derived measurements are:

- **information-complete rate**;
- **solve rate given complete information**;
- **solve rate given incomplete information**;
- **raw solve rate**.

Communication and no-communication conditions at the same population point must use identical benchmark worlds and therefore identical information-complete counts.

The primary decomposition is:

```text
more active runtime computation/scope
    -> more required information becomes available
    -> the shared learned computation either uses or fails to use it
    -> raw system capability changes
```

A positive raw curve with weak solve-given-complete indicates scope exposure without competent distributed utilization. A communication advantage on the same complete-information worlds is stronger evidence that recurrent shared computation matters.

## Measurements

For every condition record:

- task count and solved count;
- information-complete count/rate;
- solve given complete information;
- solve given incomplete information;
- solve rate by relay difficulty;
- learned parameter count and exact checkpoint fingerprint;
- active/available workers;
- recurrent rounds;
- total learned worker updates;
- messages/signals emitted;
- communicated scalar values/bytes;
- peak worker-state bytes;
- wall time;
- device execution time when measurable;
- compiler/execution mode as provenance only.

Primary graph:

> **capability vs runtime population compute at fixed learned parameters**

It is always shown beside the information-complete curve.

Secondary views include capability versus worker updates, conditional capability versus width, communication volume, and parallel-versus-serial execution cost.

## Frozen Gate-v0 interpretation

The following thresholds were frozen before the first trained scaling curve was inspected.

For one independent training seed, the seed-level gate requires:

1. at least **two nontrivial relay difficulty tiers** where:
   - the 256-worker endpoint improves raw solve rate over the 1-worker condition by at least **5 percentage points**; and
   - at least **three of four** adjacent population steps are non-decreasing within a **1 percentage point** tolerance;
2. at least **one multi-hop tier** where the communicating 256-worker endpoint beats its matched no-communication endpoint by at least **5 percentage points**;
3. identical learned parameter count and checkpoint fingerprint across every population/control point for that seed;
4. identical benchmark worlds and information-complete counts between communicating and no-communication points;
5. valid worker-update, communication, state-memory and scope accounting.

Conditional solve rate is reported and interpreted but receives no post-hoc success threshold.

### Frozen cross-seed aggregation

Before confirmation was opened, the cross-seed rule was made explicit:

- use exactly new training seeds **1, 2 and 3**;
- each seed is assessed independently under the seed-level rule above;
- the final relay-v1 confirmation gate passes only if **all 3 / 3 seeds pass**;
- there is no 2-of-3 majority rule;
- extra seeds cannot be added after results are visible to rescue a failed seed.

The complete execution contract is in [`confirmation_protocol_v1.md`](confirmation_protocol_v1.md).

## Development evidence already obtained

### Clean relay-v1 seed-0 curve

A development-only one-checkpoint run on relay-v1 produced:

| Workers | relay-2 | relay-4 | relay-8 |
| ---: | ---: | ---: | ---: |
| 1 | 0.0% | 0.0% | 0.0% |
| 4 | 24.9% | 25.0% | 0.0% |
| 16 | 49.9% | 49.9% | 33.1% |
| 64 | 74.9% | 74.8% | 66.4% |
| 256 | 99.7% | 99.4% | 99.0% |

For every condition with incomplete information, exact solve given incomplete information was 0%.

Solve-given-complete stayed approximately 99-100% across the usable ladder, while no-communication exact solve was 0% at every width/difficulty.

This is **development evidence only**. See [`relay_v1_clean_development_result_v0.md`](relay_v1_clean_development_result_v0.md).

### Serial schedule equivalence

The repaired normalized computation is exactly serializable at matched learned worker-update count within floating-point tolerance, with serial peak live neural-state residency of 1 instead of N.

See [`serial_schedule_equivalence_result_v0.md`](serial_schedule_equivalence_result_v0.md).

## What the development evidence supports

The clean evidence supports the narrow statement:

> **With learned parameters fixed, additional reusable runtime neural computation plus additional available distributed source scope can produce additional capability.**

It does not yet establish:

- frozen-confirmation reproducibility;
- more capability per learned update than equal-work serial execution;
- superiority over a dense baseline;
- real-workload advantage;
- language/coding/general intelligence scaling.

## Kill / redirect criteria

Stop or redirect this benchmark family when the frozen confirmation criteria fail after correctness is established, or when later resource-frontier tests show that the population organization provides no useful capability/latency/throughput frontier.

A positive population curve does not establish superiority over a 1B dense model. It only earns the next experiment.

Likewise, a positive confirmation result with serial equivalence must be described as **runtime-compute scaling**, not as intrinsic capability arising from simultaneous colony state.

## Sequential plan — current state

### Gate 0A — executable benchmark contract ✅

Deterministic relay worlds, controlled scope thresholds, answer-frontier integrity, population conditions and exact accounting are implemented and qualified.

### Gate 0B — minimal shared-weight population ✅

One shared learned update system can be instantiated over 1→256 temporary runtime states with fixed learned parameters.

### Gate 0C — sparse communication debugging / development ✅

Development work localized and repaired:

- compositional shared node representation;
- hop-local state reset;
- selector credit assignment through training-only supervision;
- width-dependent transport dilution through normalized gate competition;
- relay-v1 answer-frontier benchmark shortcut.

The clean development curve is preserved. The repaired function is also known to be serializable at matched worker-update count.

### Gate 0C.1 — canonical protocol freeze — ACTIVE

The successful diagnostic mechanism is being frozen into permanent `relay_protocol_v1` / `relay_experiment_v1` code with versioned checkpoints, locked confirmation configuration and executable qualification.

A final seed-0 development rerun through the canonical implementation must match the preserved diagnostic curve before confirmation is opened.

### Gate 0D — frozen confirmation — NOT OPENED

After canonical seed-0 reproduction succeeds:

- run training seeds 1 / 2 / 3 sequentially on one runner;
- evaluate the untouched confirmation split;
- apply the already-frozen 3-of-3 seed rule;
- record both a positive or negative result without changing the protocol.

### Gate 0E — hierarchy rescue

Only if the canonical sparse-shared protocol fails the frozen confirmation gate for substantive reasons may `hierarchical_summary_v0` be considered under the original topology budget.

### Decision after Gate 0D

- positive fixed-parameter compute scaling -> move to larger counts / richer tasks / explicit resource-frontier comparisons;
- negative result -> stop or redirect this benchmark family rather than adding arbitrary mechanisms;
- positive capability but poor runtime/communication economics -> investigate information-transport and scheduling efficiency separately.
