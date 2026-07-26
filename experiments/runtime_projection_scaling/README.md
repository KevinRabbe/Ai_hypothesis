# Runtime Projection Scaling Benchmark

## Question

The indexed runtime slices #45–#52 are intended to change the systems cost of reconstructing current integration work from:

```text
replay all immutable history every control cycle
```

to:

```text
advance new tail once
+
query compact current materialization
+
fetch only bounded worker payload
```

This benchmark measures that engineering boundary directly.

It does **not** measure neural-worker quality, useful uncertainty reduction, population scaling, or end-to-end research throughput.

## Compared paths

### Replay reference

```text
IntegrationPartitionAllocator.plan(ledger)
```

This remains the semantic reference implementation and rebuilds deterministic current partitions from canonical ledger history.

### Indexed path

```text
IndexedIntegrationPartitionPlanner.plan(sequence=N, thread_id=...)
```

This uses the incremental integration index plus one-time deterministic shard assignments from #52.

## Workload

One Work Thread receives a fixed number of pending evidence records.

Then increasing numbers of **irrelevant immutable history events** are appended:

```text
0
1,000
10,000
50,000
```

by default.

Those extra events deliberately do not change the correct partition plan.

Therefore any planning-time growth caused only by those events is bookkeeping/history-reconstruction cost rather than additional useful current work.

## Measurements

For each history size the harness records:

### `indexed_catchup_ms`

One indexed plan immediately after the new history tail is appended.

This includes advancing the materialized integration view through only the newly appended tail since the previous measurement point.

Expected systems shape:

```text
roughly related to new_tail_event_count
```

not total historical event count.

### `indexed_warm_median_ms`

Median repeated indexed planning time after the materialized view is already at the target revision and all pending evidence has a deterministic shard assignment.

This is the steady repeated control-cycle measurement.

Expected shape:

```text
indexed shard aggregate queries
+
bounded selected evidence payload lookup
```

### `replay_median_ms`

Median repeated full replay partition-planning time at the same canonical revision.

Expected shape:

```text
increases with total immutable history inspected
```

### `replay_to_indexed_warm_ratio`

A descriptive ratio:

```text
replay_median_ms / indexed_warm_median_ms
```

This is environment-specific and is not used as a test gate.

### `plan_equivalent`

The critical correctness check.

Both paths must produce the same ordered partition fingerprint:

- stable partition ID;
- shard index/count;
- backlog count;
- oldest pending sequence;
- bounded evidence IDs.

A fast indexed result that differs from the replay reference is a failure, not an optimization.

## Why irrelevant history is intentional

The architecture uses an append-only Research Ledger, so historical state is expected to grow indefinitely.

A mature system may have millions of events that remain scientifically/provenance-important but no longer affect a particular current scheduling decision.

The benchmark isolates whether that immutable history is repeatedly transported through the hot control path.

It therefore tests one of the project's central systems rules:

> Preserve all durable cognitive work without requiring every future compute decision to reread all of it.

## Running

```bash
PYTHONPATH=. python experiments/runtime_projection_scaling/run_projection_scaling.py
```

Example custom run:

```bash
PYTHONPATH=. python experiments/runtime_projection_scaling/run_projection_scaling.py \
  --history-events 0,1000,10000,50000 \
  --pending-evidence 128 \
  --repeats 7 \
  --output results/projection_scaling.json
```

## Interpretation rules

### A result can support

- replay planning cost grows with immutable history in this workload;
- indexed catch-up depends on newly appended tail rather than replaying the full ledger;
- warmed indexed planning reduces repeated bookkeeping cost;
- indexed and replay paths remain semantically equivalent for the partition plan.

### A result cannot support

- tiny workers are better than dense models;
- population scaling is superlinear/linear/sublinear;
- knowledge integration preserves semantic truth;
- a particular worker width is optimal;
- SQLite will remain optimal at arbitrary production scale;
- the observed ratio generalizes to other hardware/workloads.

## Why no speedup assertion in unit tests

Wall-clock microbenchmarks are noisy under CI, shared runners and virtualized environments.

The unit regression therefore asserts only:

- valid benchmark configuration;
- monotonically increasing workload points;
- plan equivalence;
- sane non-negative measurement fields.

Performance is reported as evidence, not encoded as a brittle correctness threshold.

## Next research benchmark

This is a **systems benchmark**, not the architecture's main capability benchmark.

Once the current empirical reducer gate and runtime qualification are settled, the higher-value experiment remains the large-scope task where one worker cannot inspect the entire input and population width should increase useful possibility/scope coverage.

That benchmark should separately measure:

- unique scope coverage per wall-clock time;
- useful evidence generated;
- duplicate/correlated work;
- evidence retained/integrated;
- uncertainty removed;
- end-task accuracy;
- fixed-compute population-vs-dense comparison.
