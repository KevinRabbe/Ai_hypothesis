# AI Hypothesis — Neural Population Compute Research

> Consolidation preview. Intended to replace the top-level README only after the current Gate-2 development artifact/provenance is safely captured and the final consolidated tree is qualified.

## Research objective

This repository investigates whether a **fixed learned-parameter budget** can be used more effectively by reusing the same learned machinery across a population of weak temporary neural states instead of binding capability growth mainly to more stored weights.

The target is one computational organism:

- one fixed/shared learned parameter set;
- many weak runtime neural states;
- recurrent reuse of the same learned machinery;
- bounded communication;
- state/locality that may persist when the workload requires it;
- dynamic activation as a later gate;
- explicit accounting of work, latency, memory and communication.

The workers are not miniature autonomous agents. An individual state may be nearly useless alone.

The core question is whether useful capability can emerge from how many states are available, what local information they hold, how computation is reused over time, and how information is integrated.

## Current scientific status

```text
Gate 0 — fixed-parameter capability scaling       COMPLETED POSITIVE
Gate 1 — practical work/span resource frontier   COMPLETED POSITIVE
Gate 2 — organization-specific persistent state  ACTIVE DEVELOPMENT
Gate 3 — larger population frontier              LOCKED
```

No Gate-2 confirmation result is claimed yet.

## Gate 0 — fixed learned parameters can use more runtime computation/source scope

The confirmed relay-v1 experiment used:

- one shared learned relay model per training seed;
- **26,669 learned parameters** independent of runtime population size;
- independently trained confirmation seeds `1 / 2 / 3`;
- runtime populations `1 / 4 / 16 / 64 / 256`;
- relay depths `2 / 4 / 8`;
- 1,000 untouched confirmation worlds per relay depth;
- bounded recurrent normalized communication;
- matched no-communication control.

All **3 / 3** confirmation seeds passed the frozen rule.

Across those seeds, mean exact solve at population 256 was:

- relay-2: **99.63%**;
- relay-4: **99.33%**;
- relay-8: **98.27%**.

Critical controls remained clean:

- exact solve on incomplete-information worlds was 0% wherever incomplete worlds existed;
- no-communication exact solve was 0% across all confirmation seed × difficulty × population conditions;
- parameter count/fingerprint remained fixed across every population/control point inside one seed.

Supported claim:

> **With learned parameters fixed, additional reusable runtime neural computation plus additional available distributed source scope can reproducibly produce additional capability on the tested synthetic relay task, with bounded recurrent communication required.**

This does not prove general intelligence, real-workload superiority, or per-FLOP superiority.

Canonical evidence:

- `experiments/population_compute_scaling_v0/confirmation_protocol_v1.md`
- `experiments/population_compute_scaling_v0/relay_v1_confirmation_result_v0.md`
- `experiments/population_compute_scaling_v0/confirmation_gate_v1.json`

## Serial-equivalence boundary

The repaired relay computation is mathematically serializable at matched learned update count.

Parallel normalized execution and a one-live-state serial execution can implement the same relay function while using the same `N × relay_hops` learned recurrent update budget.

Therefore Gate 0 does **not** establish extra function-level capability merely from simultaneous wide state.

This forced the next question to become a systems/resource question rather than treating population width itself as magic.

Evidence:

- `experiments/population_compute_scaling_v0/serial_schedule_equivalence_result_v0.md`

## Gate 1 — target GPU can exploit simultaneous population execution

Gate-1 v1 measured the frozen relay computation on the real local target:

- NVIDIA GeForce RTX 4060 Ti 16 GB;
- PyTorch `2.9.1+cu130`;
- CUDA runtime `13.0`;
- frozen seed-1 relay checkpoint;
- 26,669 learned parameters;
- populations `1 / 4 / 16 / 64 / 256`;
- relay depths `2 / 4 / 8`;
- batch 1 latency and batch 64 throughput;
- eager CUDA execution.

Compared schedules:

1. simultaneous/batched parallel normalized;
2. low-memory serial normalized;
3. compute-matched cached serial normalized.

Before timing, Gate-1 v1 required complete FP32/FP64 correctness corroboration across the full 30-cell matrix.

### Result

Parallel execution was faster than both serial controls in **30 / 30** preregistered cells.

Across the complete matrix, descriptive CUDA-event geomean speedup was approximately:

- **16.01×** versus compute-matched cached serial;
- **19.89×** versus low-memory serial.

At relay-8 / population 256 / batch 64:

- parallel: about `3.36 ms`;
- cached serial: about `493.64 ms`;
- low-memory serial: about `650.16 ms`.

Parallel execution paid higher O(N) activation/state residency, but the absolute measured memory cost remained modest for this small relay model on the 16 GB target.

Supported claim:

> **For the frozen relay-v1 computation on the RTX 4060 Ti, eager simultaneous/batched population execution provides a robust practical latency/throughput advantage over mathematically equivalent serial schedules across the preregistered matrix.**

This does not prove the same frontier for larger models, populations above 256, compiled execution, real workloads or multi-machine execution.

Canonical evidence:

- `experiments/population_compute_scaling_v0/resource_frontier_protocol_v1.md`
- `experiments/population_compute_scaling_v0/gate1_v1_target_gpu_result.md`

### Gate-1 v0 remains preserved

The original Gate-1 v0 target-CUDA run failed its frozen FP32 tensor-allclose criterion before timing.

The project did not silently loosen that preregistration.

Precision diagnostics later showed exact decoded equivalence and near-double-precision schedule agreement, motivating a separate precision-aware Gate-1 v1 protocol frozen before admitted timing.

The failed v0 rule remains part of the scientific record:

- `experiments/population_compute_scaling_v0/gate1_v0_cuda_equivalence_result.md`

## Gate 2 — active organization-specific capability test

Gate 0's relay is deliberately too reducible to determine whether persistent distributed neural state has intrinsic capability value beyond serializable computation.

Gate 2 therefore uses a workload where state interference/locality can matter while holding inspected information and learned work fixed.

Current development workload: **delayed keyed traces**.

Frozen structural ladder:

- entity counts `16 / 64 / 256`;
- widths `1 / 4 / 16 / 64 / 256`, truncated at entity count;
- four payload/evidence rounds;
- four interference/retention rounds;
- delayed final query requests one entity's 4-bit payload;
- exact `8 × entity_count` learned recurrent updates per world at every width/control.

Stable routing uses a parameter-free world permutation and deterministic entity-to-slot assignment.

Controls:

1. **serial persistent** — same state bank/function time-multiplexed; output difference is a correctness failure;
2. **reshuffled locality** — same width/state count/information/work, but entity-to-slot mapping changes across rounds;
3. **reset state** — same width/routing/work, but persistent neural state is erased each round.

Width-1 stable versus reshuffled is an exact identity control.

The first local CUDA run is **development only**. It may establish learnability or expose training/protocol weaknesses, but it cannot produce a Gate-2 verdict.

Confirmation remains locked until architecture, optimizer/training recipe, evaluation matrix and confirmation decision rule are frozen and untouched confirmation worlds are evaluated across new training seeds.

Key files:

- `experiments/population_compute_scaling_v0/gate2_persistent_state_capacity_protocol_v0.md`
- `experiments/population_compute_scaling_v0/gate2_execution_semantics_v0.md`
- `experiments/population_compute_scaling_v0/gate2_development_execution_note.md`
- `ai_hypothesis/population_compute/gate2_persistent_state_capacity.py`
- `ai_hypothesis/population_compute/gate2_persistent_model.py`
- `ai_hypothesis/population_compute/gate2_development.py`

## Why Gate 2 matters

Gate 1 established that the GPU can execute simultaneous population width efficiently for the current relay.

Gate 2 asks the stronger architecture question:

> **Can population organization itself improve capability under a matched practical information/work budget when persistent local state or locality matters?**

A negative result remains useful. It would mean that, for this tested workload/substrate, simpler recurrent/serial organization is sufficient and the project should not scale population count merely because the hardware can execute it.

## Research principles

1. **Fixed learned parameters and runtime population are separate variables.**
2. **Workers are weak processing states, not full agents.**
3. **Runtime state/communication are resources, not free capacity.**
4. **Matched serial/recurrent controls are mandatory when they can implement the same function.**
5. **Same-information controls are mandatory when population width changes source scope.**
6. **Use deterministic algorithms when exact logic is sufficient.**
7. **Keep communication bounded; do not broadcast complete global state to every worker.**
8. **Compiler optimization is a separate systems variable.**
9. **Consumer hardware is a first-class target.**
10. **Negative results remove uncertainty and can terminate a direction.**
11. **Do not increase worker count merely because it is technically possible.**
12. **Confirmation/test boundaries remain untouched until a protocol is frozen.**

## Repository layout after consolidation

The current canonical implementation surface is intentionally small:

```text
ai_hypothesis/population_compute/        active neural population substrate
experiments/population_compute_scaling_v0/ protocols + permanent evidence
scripts/                                 local Gate runners/finalizers
tests/                                   focused population-compute contracts
docs/future/                             deferred future research directions
docs/history/                            compact summaries of earlier programs
```

Earlier Research Ledger/runtime, large-scope and independently weighted Step-2 implementations remain available in Git/PR history but are not required by the active population-compute core.

## Deferred directions

High-potential ideas are preserved without making them active prematurely.

### Evolutionary organism optimization

`docs/future/evolutionary_organism_direction.md`

Potential future question: under matched total training compute and fixed deployed parameter count, can an evolutionary archive of gradient-trained shared-weight organism lineages discover better population-compute frontiers faster or more reliably than ordinary independent training?

This becomes relevant only after the substrate demonstrates a phenotype worth optimizing.

### Dynamic activation

Only after fixed-width organization is useful: can harder/uncertain tasks spend more active states or recurrent depth while easy tasks terminate cheaply?

### Information transport/integration

Only when measurements show communication/integration becoming limiting: how much useful information can be moved, preserved and integrated before coordination dominates neural compute?

### Larger populations

1K+ runtime states are not a milestone by themselves. Gate 3 remains locked until organization-specific evidence justifies locating a larger useful frontier.

## Current next step

Complete and analyze the first Gate-2 **development-only** CUDA run without opening confirmation.

The result should decide the next development action under the pre-written interpretation map; it should not be forced into a positive narrative.
