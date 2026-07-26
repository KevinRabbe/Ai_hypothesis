# Thread-level knowledge consolidation v0

## Purpose

Raw evidence partitions are a throughput primitive, not a final semantic boundary.

Hash sharding can place related evidence in different partitions. After each partition produces compact provisional knowledge, the system therefore needs a higher-level pass that can reconnect those findings without rereading all raw evidence.

v0 adds that first hierarchy step:

```text
raw evidence
      ↓
partitioned integration
      ↓
partition-local knowledge deltas
      ↓
thread-level consolidation
      ↓
thread-level provisional knowledge
```

The same homogeneous workers and Worker Runtime execute both levels.

## No hierarchy database

The hierarchy is represented by ordinary knowledge references.

A thread-level consolidation delta has:

```text
kind = THREAD_CONSOLIDATION
source_reference_ids = [lower knowledge delta IDs ...]
```

That reference edge is the durable consolidation relationship.

No separate tree/graph database or “integration result store” is introduced.

The Research Ledger remains canonical; the hierarchy is a projection of knowledge-delta events.

## Source eligibility

`ThreadConsolidationPlanner` starts from exact partition lineage from PR #40.

A lower knowledge delta is eligible when:

1. it was actually produced by a partition attempt for the selected Work Thread;
2. its current knowledge status is not `RETRACTED`;
3. no active thread-consolidation delta currently references it.

Eligible statuses therefore include:

- `PROVISIONAL`;
- `VERIFIED`;
- `DISPUTED`.

This is intentional.

A disputed minority finding must not disappear merely because it conflicts with dominant knowledge. Consolidation should see unresolved contradictions and preserve them in the higher-level representation.

## Retraction semantics

A lower source delta that is retracted disappears from future consolidation input.

A **higher consolidation delta** that is retracted has the opposite effect:

```text
thread consolidation references K1, K2
        ↓
K1 and K2 considered consumed
        ↓
thread consolidation RETRACTED
        ↓
K1 and K2 become pending again
```

No mutable “consumed” flag is required.

Current consolidation state is derived entirely from append-only knowledge plus assessment history.

## Thread-local provenance strictness

Exact partition provenance is required only for the Work Thread being consolidated.

Missing legacy provenance on Thread B does not block consolidation of fully proven Thread A.

For the selected Work Thread, however, the planner refuses to infer missing historical partition identity.

This preserves independent progress while keeping each consolidation scientifically reproducible.

## Cross-partition selection

Default provisional configuration:

```text
selection_limit = 32 knowledge deltas
minimum_source_deltas = 2
```

The selection policy is deterministic and mechanical:

1. group pending lower deltas by historical partition ID;
2. order each partition's deltas by creation sequence;
3. order partitions by the creation sequence of their oldest pending delta;
4. select round-robin across partitions until the bounded limit is reached.

Example:

```text
Partition A: A1 A2 A3
Partition B: B1
Partition C: C1 C2

selection:
A1 B1 C1 A2 C2 A3 ...
```

This gives the consolidation worker cross-partition visibility before spending its entire context on a single hot partition.

It is not semantic relevance ranking and can be replaced later behind the same boundary.

## Minimum useful consolidation

A consolidation Work Item is considered ready only with at least two source deltas.

One remaining delta does not need another neural compression pass merely to reproduce itself.

It remains pending until more partition knowledge arrives or another policy explicitly handles it.

## Bounded worker context

`prepare_thread_consolidation_work(...)` reuses the existing bounded knowledge preparation path.

The worker receives only selected compact knowledge records, including:

- delta IDs;
- summaries;
- source references;
- causal event IDs;
- current knowledge status;
- prior assessment state.

The full lower-level evidence remains in the ledger and is reachable through provenance when necessary.

The context is marked:

```text
context_view   = SYNTHESIZE
synthesis_mode = THREAD_CONSOLIDATION
```

and includes selected source partition IDs plus total pending source/partition counts.

## Worker contract

The Work Item references are exactly the lower knowledge delta IDs the worker may consume.

Constraints request:

- output kind `THREAD_CONSOLIDATION`;
- explicit references to consumed lower knowledge;
- preservation of unresolved contradictions;
- provisional output rather than automatic truth promotion.

Worker Runtime already enforces that a generated knowledge delta may only reference information authorized by the Work Item.

Therefore the same generic trust boundary works at both hierarchy levels:

```text
raw integration:
Work Item references evidence IDs
→ KnowledgeDelta references evidence IDs

thread consolidation:
Work Item references knowledge delta IDs
→ KnowledgeDelta references knowledge delta IDs
```

## Same Worker Runtime

No new executor is needed.

A normal consolidation attempt is:

```text
WorkPreparation
      ↓
WorkItem(purpose=SYNTHESIZE)
      ↓
ordinary WorkerAssignment
      ↓
WorkerRuntime
      ↓
homogeneous Worker Bank
      ↓
AttemptResult(
    knowledge_delta.kind = THREAD_CONSOLIDATION,
    knowledge_delta.reference_ids = lower delta IDs
)
```

The resulting higher delta enters the same Knowledge State projector as every other claim.

It starts `PROVISIONAL` and can later be:

- verified;
- disputed;
- retracted.

## Causal integrity

An active thread-consolidation delta cannot consume lower knowledge that was only created later in ledger history.

The planner explicitly requires:

```text
lower delta created_sequence
    <
consolidation created_sequence
```

Normal Worker Runtime already prevents unauthorized future references during live execution; replay validation keeps imported/corrupt history from silently changing consolidation state.

## Why not verify everything before consolidation

Requiring every lower partition delta to be verified first would create unnecessary verification load and could delay the discovery of cross-partition contradictions.

v0 instead allows provisional/disputed information into consolidation while preserving status metadata.

The higher output remains provisional.

This follows:

> diversity for discovery; redundancy for verification.

Consolidation is an information-organization step, not truth promotion.

## Information scaling effect

Suppose 1,000 raw evidence items are compressed by partition integration into 80 partition-local knowledge deltas.

Thread consolidation might then operate on bounded batches of those 80 deltas rather than rereading all 1,000 evidence records.

The exact compression ratio is empirical and not assumed.

The architecture goal is:

```text
large lower-level information volume
      ↓
compact but provenance-preserving deltas
      ↓
smaller higher-level integration surface
```

while retaining the ability to drill back to rare raw evidence when a higher-level conclusion is challenged.

## Future levels

The same pattern can recurse:

```text
partition knowledge
      ↓
THREAD_CONSOLIDATION
      ↓
branch / topic consolidation
      ↓
cross-topic consolidation
      ↓
global knowledge
```

Each level should consume compact outputs from the level below and preserve references downward.

A future general hierarchy planner may parameterize the consolidation kind/ownership level, but v0 keeps the first real level explicit and inspectable.

## Non-goals

v0 does not add:

- semantic clustering;
- learned relevance retrieval;
- automatic scheduler routing into thread consolidation;
- automatic verification;
- truth promotion;
- a hierarchy database;
- specialized integration workers;
- a claim that 32 inputs is optimal.

It only makes the first higher integration level executable using existing final architecture contracts.
