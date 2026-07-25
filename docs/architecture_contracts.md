# Architecture Contracts

## Purpose

This document freezes the population runtime's long-lived component boundaries while allowing each component's implementation to remain minimal at first and scale internally later.

The rule is:

> Final architecture from day 1; minimal implementation from day 1.

Scale should change implementation strategy, storage layout, indexing, batching, partitioning, and policy quality without changing the meaning of the core interfaces.

## Stable architecture boundary

The runtime has five durable component roles:

1. **Worker Bank** - provides architecturally identical learned workers with independently learned weights.
2. **Worker Runtime** - executes bounded attempts against Work Items.
3. **Research Ledger** - persists immutable events and provenance.
4. **State Projector** - derives current views from ledger history.
5. **Scheduler** - allocates bounded work from projected state and resource constraints.

Large-scale integration is not a sixth autonomous service. It is expressed through the same Work Items, attempts, ledger events, projections, and scheduling mechanisms, potentially arranged hierarchically when scale requires it.

## Contract 1 - Work Item

A Work Item is the bounded unit assigned by the scheduler.

It should contain stable identifiers and references rather than copied global state.

Minimum logical fields:

- `work_item_id`;
- `thread_id`;
- `objective`;
- `purpose` (`EXPLORE`, `PROGRESS`, `CHALLENGE`, `VERIFY`, or `SYNTHESIZE`);
- relevant source/evidence/hypothesis references;
- bounded context view;
- constraints and resource budget;
- parent/dependency references when applicable;
- projection revision used to build the item.

A Work Item is immutable once execution begins. If the world changes materially, a later attempt receives a new Work Item.

## Contract 2 - Attempt Result

A worker does not return a vote or an unstructured full history. It returns structured changes produced by one bounded attempt.

Minimum logical fields:

- `attempt_id`;
- `work_item_id`;
- `thread_id`;
- `worker_id` / checkpoint provenance;
- observations/evidence produced;
- hypotheses proposed, strengthened, weakened, or rejected;
- contradictions found;
- possibilities eliminated;
- open questions;
- requested follow-up work;
- progress/stagnation indicators;
- side effects or tool actions performed;
- resource usage;
- completion status.

The exact hidden neural trajectory is not part of the durable contract.

## Contract 3 - Ledger Event

All durable state changes are represented as append-only ledger events.

A ledger event must contain:

- immutable event ID;
- event type;
- timestamp / logical sequence;
- thread and attempt provenance where applicable;
- stable references to sources/evidence/hypotheses;
- payload schema/version;
- causal/parent references where needed.

Published events are never silently rewritten. Corrections are new events that invalidate, supersede, or qualify previous events.

This lets the storage implementation evolve from a simple local database to partitioned/distributed storage without changing the semantic model.

## Contract 4 - Projected State

The State Projector converts ledger history into bounded current views.

Important projections include:

- Work Thread state;
- evidence state;
- hypothesis state;
- contradiction state;
- coverage state;
- verification state;
- integration backlog;
- scheduler metadata;
- compact knowledge deltas.

Projections are derived and rebuildable. The ledger remains canonical.

A projection may be aggressively summarized as long as decisive underlying evidence remains recoverable by stable reference.

## Contract 5 - Scheduler Decision

The scheduler consumes projected metadata and resource availability, not the entire research history.

A decision allocates one of a small set of actions:

- continue depth;
- rotate worker;
- add width;
- fork;
- challenge;
- verify;
- synthesize;
- pause;
- complete.

The scheduler controls four stable allocation dimensions:

- width;
- depth;
- scope;
- purpose.

Scheduler v0 is deterministic scoring plus structured-random exploration. Future heuristics or learned value estimates may replace the policy internally without changing this contract.

## Contract 6 - Knowledge Delta

At scale, downstream components should normally exchange changes to knowledge rather than complete states.

Examples:

- evidence `E91` was invalidated;
- hypothesis `H17` gained an independent contradiction;
- source region `R42` became covered;
- verification for claim `C12` passed;
- thread `W8` became obsolete because its objective was resolved elsewhere.

A delta must preserve enough references to retrieve the underlying evidence and causal history.

This contract is the main defense against information volume growing linearly through every layer of the system.

## Contract 7 - Integration hierarchy

Knowledge integration must be able to scale recursively without introducing a special global brain.

The same contracts apply at every level:

```text
worker attempts
    -> local/thread integration
    -> branch/topic integration
    -> cross-topic integration
    -> global knowledge changes
```

An integration operation is still a Work Item. It is still executed as a bounded attempt. Its result still becomes ledger events. Its output still propagates as evidence or knowledge deltas.

Therefore hierarchical integration can be introduced later without redesigning worker semantics, persistence, or scheduling.

## Contract 8 - Backpressure

Population width must not grow without regard to information absorption.

The scheduler must be able to reduce or redirect discovery work when:

- raw evidence production exceeds integration throughput;
- integration backlog age/depth grows;
- verification backlog grows;
- duplicate evidence rate rises;
- rare-evidence retention degrades;
- scheduler/integration overhead dominates useful work.

Available neural compute is not by itself a reason to activate more workers.

The useful population ceiling is the population whose useful evidence production can still be preserved, organized, verified, and exploited at acceptable end-to-end cost.

## Day-1 implementation rule

The first implementation should use the simplest mechanism satisfying these contracts.

Examples:

- one local durable ledger rather than distributed storage;
- one in-process projector rather than a projection service;
- one deterministic scheduler rather than a learned controller;
- one integration path rather than a hierarchy;
- compact structured records rather than a generalized message bus.

When scale forces a change, split or optimize the implementation behind the existing contract before changing the architecture.

## Architecture-change rule

A new top-level component, communication mode, worker role, or scheduler dimension is added only when measured evidence shows that the existing contracts cannot express or efficiently solve a real bottleneck.

Do not redesign for hypothetical scale when an internal implementation upgrade is sufficient.
