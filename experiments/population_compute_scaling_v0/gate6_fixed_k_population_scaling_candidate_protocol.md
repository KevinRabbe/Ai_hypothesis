# Gate-6 candidate — fixed-K routing under larger live populations

**STATUS: PREPARATION ONLY — NOT ADMITTED.**

This document may be prepared while Gate-5 confirmation is running, but Gate-6 must not generate scientific worlds, run an admitted evaluator, or assign any outcome unless Gate-5 confirmation independently returns `GATE5_CONFIRMED_BOUNDED_SCORE_ACTIVATION` and a later commit explicitly promotes this candidate protocol to an admitted frozen protocol.

Base preparation head: Gate-5 confirmation pre-result head `d6541dc1e2f9c15c4c30408b2616ec04f10affe9`.

## Scientific question

Gate-5 asks whether bounded score visibility can preserve near-global learned routing at the population sizes created by its depth-8 search. Gate-6 asks the next scaling question:

> With the learned scorer, learned parameters, candidate depth, public information, active neural width, per-candidate refinement, frontier-construction work, and Stage-B work held fixed, does a constant bounded routing visibility K remain useful as the number of simultaneously live hypotheses increases?

The causal variable is **live-population size available to the router**. The experiment must not obtain larger populations by giving them deeper candidate states or additional frontier-construction work.

## Core anti-confound design

Use depth-10 worlds and first construct the **same complete depth-8 frontier of 256 scored persistent hypotheses for every condition**.

- Complete binary frontier size at depth 8: `256`.
- Frontier construction is identical before the population-size treatment is applied.
- Every candidate therefore has the same prefix depth and has received the same learned recurrence schedule.
- The same 256-candidate parent frontier exists before deterministic thinning.

Then derive smaller live populations by an answer-blind deterministic nested thinning rule from that same 256-candidate frontier.

Frozen candidate population ladder for the first development version:

- `N64`
- `N128`
- `N256`

The thinning must be nested (`N64 ⊂ N128 ⊂ N256`) and depend only on public runtime seed + candidate path. It must not inspect hidden answers or neural scores.

This design keeps **frontier-construction learned work identical across N**. No condition receives extra learned work merely because its live population is larger.

## Shared learned machinery

Reuse the exact same three frozen Gate-3 v1 checkpoints used by Gates 3–5:

- learned parameter count: `19,649` per checkpoint;
- no training or fine-tuning;
- same recurrent scorer and candidate encoding;
- no population-size embedding;
- no slot ID;
- no N-specific learned parameters.

Compiler, CUDA graphs, fusion and mixed precision remain separate experimental variables and are not enabled by this protocol.

## Search topology

World depth: `10`.

### Stage A — identical for every N and scheduler

Build the complete depth-8 frontier generation-synchronously.

Parent-expansion count:

`1 + 2 + 4 + ... + 128 = 255` parent expansions.

Each parent expansion uses the existing two child lanes and eight recurrent updates per child.

Stage-A learned recurrent updates/world:

`255 × 2 × 8 = 4,080`.

After Stage A, deterministically thin the common 256-candidate frontier to the requested N.

### Stage B — matched routing/search work

Candidate preparation target for the first version: `128` scheduled parent activations after thinning.

Each Stage-B activation:

- selects exactly one live parent;
- expands two children;
- gives each child exactly eight recurrent updates;
- therefore consumes exactly `16` learned recurrent updates.

Stage-B learned recurrent updates/world:

`128 × 16 = 2,048`.

If a treatment exhausts its live reserve before all 128 Stage-B slots, remaining slots must execute the same preregistered learned sink update primitive so total learned work remains fixed.

Total learned recurrent work/world:

`4,080 + 2,048 = 6,128`.

This total is identical across every N and scheduler condition.

## Routing conditions

Primary fixed-K treatment is inherited from Gate-5's preregistered primary treatment, not selected from Gate-5 descriptive results:

1. `global_score` — full live-reserve score visibility.
2. `bounded_score_k16` — inspect at most 16 sampled candidate scores before parent selection.
3. `bounded_hash_k16` — identical K16 sampling mechanics but zero neural-score comparisons before parent selection.

Optional descriptive frontier condition, allowed only if frozen before Gate-6 exposure:

4. `bounded_score_k8`.

K8 must remain descriptive in v0 and cannot rescue a failed K16 primary result.

## Strict information boundary

For bounded conditions, the parent must be irrevocably selected before any non-sampled candidate score is read.

Post-selection full-population rankings are allowed only as evaluation telemetry and must never feed back into runtime state, routing, pruning, or future sampling.

The K16 score/hash conditions must use the same deterministic sampling function whenever their incoming reserves are identical. Later reserve divergence caused by different parent choices is an intended causal consequence.

## Fixed information

Every paired condition for a world receives exactly the same:

- hidden path distribution;
- public noisy hints;
- hint reliability;
- candidate encoding;
- complete depth-8 Stage-A frontier before thinning;
- deterministic thinning rule for a given N;
- Stage-B slot count;
- active child lanes;
- recurrent updates per child;
- total learned work.

Suggested inherited hint reliability for candidate v0: `0.70`.

## Development world namespace

**Do not define or instantiate the scientific development namespace while this file remains preparation-only.**

A later admission commit must freeze a fresh namespace disjoint from Gates 0–5 before any Gate-6 development world is generated.

## Primary scaling comparisons

For each checkpoint and each population size `N ∈ {64, 128, 256}`:

- learned-routing effect:
  `bounded_score_k16 - bounded_hash_k16`;
- global-gap effect:
  `bounded_score_k16 - global_score`.

The experiment should report raw per-world coverage vectors and deterministic paired bootstrap intervals.

## Candidate non-inferiority rule

Candidate v0 retains Gate-5's already-preregistered five-percentage-point margin rather than inventing a new one after seeing Gate-6 data:

`delta_NI = 0.05`.

At a population tier N, K16 is considered non-inferior to global only when:

`CI_low(K16 - global) > -0.05`.

Learned K16 routing is established at N only when:

`CI_low(K16 - hashK16) > 0`.

## Candidate directional outcomes

These labels are preparation-only and must be rechecked for mutual exclusivity before admission.

- `G6_S0_FIXED_K_NOT_ESTABLISHED` — learned K16 routing is not established at the base N64 tier.
- `G6_S1_FIXED_K_DEGRADES_WITH_POPULATION` — K16 is valid at smaller N but fails the learned-routing or non-inferiority criterion at a larger N.
- `G6_S2_ROBUST_FIXED_K_POPULATION_SCALING` — K16 learned routing and five-point non-inferiority both hold at N64, N128 and N256 on all three checkpoints.
- `G6_S3_CHECKPOINT_SENSITIVE_SCALING` — the population-scaling status differs across checkpoints.
- `G6_S4_BOUNDED_ROUTING_HARMFUL_AT_SCALE` — K16 is significantly harmful relative to the hash control at any preregistered primary tier.

No Gate-6 confirmation protocol may be opened from preparation alone.

## Required telemetry

Per world / condition:

- coverage / exact hidden-path generation;
- productive Stage-B slots;
- learned sink slots;
- total learned recurrent updates;
- initial thinned live-population size;
- live population by Stage-B slot;
- score observations by Stage-B slot;
- selected visible score rank;
- selected global score rank computed only after selection;
- generated terminal count;
- unique generated terminal count.

Aggregate reporting should include score-observation cost relative to global as N increases.

## Claims boundary

Even a future positive Gate-6 result would support only the narrow statement that a fixed bounded-score visibility rule retained useful routing as the controlled live population increased over the tested range.

It would **not** establish:

- physical decentralization;
- communication-network scalability;
- 1K/10K/100K-worker capability;
- asynchronous distributed execution;
- per-FLOP, per-joule or wall-clock superiority;
- general intelligence.

Those require later gates.
