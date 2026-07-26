# Integration hierarchy information-volume telemetry v0

## Purpose

The project now has a real two-level integration hierarchy:

```text
raw evidence
      ↓
partition-local knowledge
      ↓
thread-level consolidated knowledge
```

This makes the information-scaling question measurable for the first time:

> Is higher-level integration actually reducing the number of active information objects the rest of the system must handle?

`IntegrationHierarchyTelemetryProjector` answers that as a **count / fan-in / frontier** question.

It deliberately does **not** claim that fewer objects mean better or lossless knowledge.

## The central distinction

There are two independent questions:

### Information volume

How many objects must the next level process?

Examples:

- raw evidence count;
- partition-local knowledge count;
- pending lower knowledge count;
- active thread-consolidation count;
- active hierarchy frontier count.

### Information quality

Did the higher-level representation preserve the rare evidence, contradictions and facts that matter?

That requires separate evidence/verification tests.

Therefore:

> Count reduction is an information-bandwidth metric, not a semantic compression score.

## Raw evidence layer

The snapshot records:

- total durable `EVIDENCE_ADDED` identities;
- duplicate evidence identity remains invalid history.

It also measures how much of that raw evidence is referenced by observed partition-produced knowledge:

```text
unique raw evidence referenced by partition knowledge
----------------------------------------------------
raw evidence count
```

This is `partition_evidence_reference_fraction`.

It is a provenance-coverage metric, not proof that every evidence record was semantically preserved.

## Partition integration layer

Using exact historical partition lineage, telemetry records:

- partition assignment count;
- unique historical partition IDs;
- started partition-attempt count;
- unstarted allocated partition count;
- partition-produced knowledge delta count;
- active partition-produced knowledge count;
- current status counts (`PROVISIONAL`, `VERIFIED`, `DISPUTED`, `RETRACTED`);
- total raw-evidence references made by those partition knowledge deltas;
- unique raw evidence referenced.

### Assignment count versus unique partition IDs

A logical shard can be allocated repeatedly over time as new evidence enters or earlier work remains unresolved.

Therefore:

- `partition_assignment_count` counts historical allocation instances;
- `unique_partition_id_count` counts unique logical partition IDs.

Those are intentionally separate.

## Thread-consolidation layer

For `THREAD_CONSOLIDATION` knowledge deltas, telemetry records:

- total consolidation deltas;
- currently active consolidation deltas;
- status counts;
- total references to known partition-produced knowledge;
- references from active consolidations;
- unique active partition deltas consumed;
- active partition deltas still pending above the partition layer;
- mean and maximum active consolidation fan-in.

It also exposes integrity/shape diagnostics:

- active higher deltas that still reference a retracted lower partition delta;
- cross-thread partition references inside `THREAD_CONSOLIDATION`;
- source references that are not recognized partition-produced knowledge.

These are not silently normalized away.

## Active hierarchy frontier

The key current information-surface metric is:

```text
active_hierarchy_frontier_count =
    pending active partition-local deltas
  + active thread-level consolidation deltas
```

A lower partition delta that is referenced by an active thread consolidation is no longer counted separately in the active frontier.

Its full history and provenance remain durable; the frontier metric only describes what the next integration level needs to treat as current top-level information.

### Example

Suppose the partition layer currently has three active deltas:

```text
K1  K2  K3
```

and one active higher delta consumes K1 + K2:

```text
K1 ─┐
    ├─> T1
K2 ─┘

K3 remains unconsolidated
```

Then:

```text
active partition deltas = 3
active frontier         = 2   (T1 + K3)
```

The count-only reduction factor is:

```text
3 / 2 = 1.5
```

This says only that the next layer sees two active objects instead of three.

It does not prove that T1 is correct, complete or better than K1 + K2.

## Count-reduction factor

`partition_to_frontier_count_reduction_factor` is:

```text
active partition knowledge count
--------------------------------
active hierarchy frontier count
```

It is deliberately named a **count-reduction factor**, not compression quality.

Interpretation:

- `1.0`: no reduction in current object count;
- `>1.0`: higher integration currently reduces the number of active information objects;
- `<1.0`: the higher level currently introduces more active objects than it removes.

A value greater than one is not automatically good if semantic quality is poor.

## Fan-in

For active thread-level consolidation deltas, the telemetry measures how many distinct known partition-produced deltas they reference.

This gives:

- mean active fan-in;
- maximum active fan-in;
- unique active lower deltas consumed per active higher delta.

Fan-in helps distinguish:

```text
10 higher deltas each summarize 1 lower delta
```

from:

```text
10 higher deltas each integrate 20 lower deltas
```

without assigning a quality score to either.

## Retraction behavior

### Higher consolidation retracted

When a `THREAD_CONSOLIDATION` delta is retracted:

- it no longer counts as active higher knowledge;
- its lower partition sources return to the pending frontier;
- the count-reduction factor moves back toward the lower-level surface.

This follows the append-only hierarchy semantics already used by the consolidation planner.

### Lower partition delta retracted

If an active higher consolidation still references a partition delta that is now retracted:

- the retracted lower delta is excluded from active partition knowledge;
- it is not counted as successfully consumed active knowledge;
- the reference remains visible as `active_retracted_partition_source_reference_count`.

This is hierarchy integrity debt that later verification/recomputation can address.

The telemetry does not silently rewrite the higher delta.

## Non-partition and cross-thread references

A thread consolidation may contain references that are not recognized partition-produced knowledge.

Those are counted separately.

Likewise, a `THREAD_CONSOLIDATION` referencing a partition delta from another Work Thread is counted as a cross-thread reference.

The current v0 thread-consolidation contract expects that count to remain zero. The telemetry keeps it observable rather than assuming the invariant held historically.

A later graph/branch consolidation level may intentionally consume cross-thread knowledge under a different knowledge kind.

## Incomplete historical provenance

Partition allocations created before durable partition provenance cannot be assigned exact historical partitions.

The hierarchy snapshot exposes:

- `partition_lineage_complete`;
- `missing_partition_provenance_decision_ids`.

Metrics that require exact partitions are derived only from complete lineage records.

`require_complete_partition_lineage()` provides a strict boundary for scientific comparisons that must not undercount legacy partition work.

The projector never guesses old shard membership from current configuration.

## Relationship to integration bandwidth telemetry

PR #33 answers:

> How quickly is evidence entering and leaving the unresolved integration queue?

This hierarchy telemetry answers:

> After integration, how many durable information objects remain active at each current hierarchy level?

Combined, future runs can measure:

```text
raw evidence / second
      ↓
unique evidence absorption / second
      ↓
partition knowledge produced / second
      ↓
thread-level active frontier size
      ↓
verification / dispute / retraction behavior
```

That is much closer to the project's real systems question than raw worker throughput alone.

## What would constitute a useful hierarchy

A promising integration hierarchy would eventually show all of the following:

- incoming raw evidence can be absorbed fast enough to avoid runaway backlog;
- partition knowledge preserves enough source provenance;
- higher levels reduce the active information surface;
- duplicate/redundant integration remains bounded;
- rare decisive evidence remains recoverable;
- verification catches harmful synthesis;
- compute/storage/control overhead stays acceptable.

No single metric in this snapshot proves all of those.

## Why this comes before another hierarchy level

The architecture can mechanically recurse to branch/topic/global consolidation.

But after two levels exist, adding more levels blindly would only create more machinery.

This telemetry gives us the exact observables needed later to tell whether another level is reducing an actual information-volume bottleneck or merely adding another transformation.

Construction can still proceed independently, but the architecture no longer needs to guess what its current hierarchy is doing.

## Non-goals

v0 does not add:

- another integration level;
- semantic quality scoring;
- a lossless-compression claim;
- learned hierarchy routing;
- scheduler tuning;
- automatic repair of stale/retracted hierarchy edges;
- wall-clock rate measurement;
- another persistence store.

It only makes the current evidence → partition → thread hierarchy quantitatively visible as information volume and provenance structure.
