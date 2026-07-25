# Integration bandwidth telemetry v0

## Purpose

At large population scale, learned-worker compute may stop being the dominant limit. The runtime can instead become bottlenecked by the rate at which generated evidence is preserved, dispositioned, connected, and converted into usable knowledge.

This document freezes the first measurement surface for that problem.

The telemetry is deliberately read-only. It does not change scheduler policy, integration policy, evidence semantics, knowledge status, or persistence.

## Architectural question

The relevant scaling condition is not merely:

`workers available`

It is:

`useful evidence production rate` versus `knowledge integration / absorption rate`.

If evidence arrives faster than the runtime can resolve it for long enough, the backlog grows even when neural compute remains available.

## Durable snapshot

`IntegrationTelemetryProjector` derives one snapshot from ordinary Research Ledger events.

It records:

- total evidence produced;
- unique evidence that has received a disposition;
- raw disposition-reference traffic;
- current unresolved evidence backlog;
- knowledge-delta count;
- how many evidence records are referenced by current knowledge-delta production;
- total evidence references consumed by knowledge deltas;
- first-disposition counts by disposition kind;
- repeated disposition count;
- disposition references to nonexistent evidence;
- mean and maximum evidence-to-first-disposition latency in **ledger sequences**;
- mean and oldest current backlog age in **ledger sequences**.

### Why sequence latency

`LedgerEvent` intentionally contains no wall-clock timestamp. The durable replay metric is therefore sequence distance:

`first disposition sequence - evidence creation sequence`

This is stable across replay and storage implementations, but it is not seconds.

The telemetry must never label sequence distance as wall-clock latency.

## Wall-clock bandwidth window

`IntegrationBandwidthWindow.between(previous, current, elapsed_seconds=...)` converts two durable snapshots plus an externally measured wall-clock interval into rates.

It reports:

- new evidence / second;
- unique evidence dispositioned / second;
- raw disposition references / second;
- knowledge deltas / second;
- backlog growth / second;
- absorption ratio.

The absorption ratio is:

`newly dispositioned unique evidence / newly generated evidence`

It is intentionally allowed to exceed `1.0` while the system drains older backlog.

When no new evidence arrives, the ratio is undefined (`None`) rather than infinite; the absolute disposition rate still shows backlog draining.

## Unique absorption versus processing traffic

Repeated integration work must not make the system appear more capable than it is.

Therefore two counters are separate:

1. `dispositioned_evidence_count` — unique evidence resolved at least once;
2. `disposition_reference_count` — all disposition references, including repeated work.

Example:

- 100 unique evidence items become dispositioned;
- 20 of them are unnecessarily dispositioned a second time.

Then:

- unique absorption = 100;
- disposition traffic = 120;
- redisposition overhead = 20.

Future optimization can reduce redundant integration traffic without changing knowledge semantics.

## Disposition categories

The first disposition is used for category accounting:

- `INTEGRATED`
- `DUPLICATE`
- `IRRELEVANT`
- `INVALID`
- `LOCAL_ONLY`

A second disposition does not silently rewrite the first classification. It increments `redisposition_count` so contradictory or repeated processing remains observable.

## Knowledge-reference metrics

Knowledge-delta provenance can contain evidence IDs, source documents, prior knowledge IDs, or other durable references.

Telemetry therefore intersects knowledge-delta source references with the known evidence registry. Non-evidence provenance is legitimate and is not treated as an error.

The snapshot exposes:

- `knowledge_referenced_evidence_count`: unique known evidence referenced by produced knowledge deltas;
- `knowledge_source_reference_count`: total references from those deltas to known evidence;
- `knowledge_reference_fraction`: fraction of generated evidence that has become direct knowledge provenance;
- `evidence_per_knowledge_delta`: average known-evidence references consumed per produced delta.

These are information-flow diagnostics, not a semantic compression score. A small delta count can be good or bad depending on whether important information was preserved.

## Thread-local telemetry

A snapshot may be scoped to a Work Thread.

Thread scoping means:

- evidence belongs to the thread where it was generated;
- dispositions are applied by evidence identity even when a synthesis worker runs on another thread;
- knowledge-delta production counts deltas targeted to that thread.

This preserves the distinction between where information originated and where integration compute happened.

## Integrity diagnostics

The projector rejects:

- non-increasing ledger sequence order;
- duplicate durable evidence IDs;
- malformed disposition kinds;
- a disposition that appears before the referenced evidence was created.

It also exposes, rather than hides:

- repeated dispositions;
- disposition references to nonexistent evidence.

The telemetry remains observational; it does not itself mutate or repair history.

## What this does not measure yet

v0 deliberately does not claim:

- semantic information gain;
- uncertainty removed;
- importance-weighted evidence value;
- rare-evidence retention quality;
- bytes/second through the ledger;
- neural integration compute cost;
- hierarchy efficiency;
- global knowledge correctness;
- optimal scheduler backpressure thresholds.

Those require either workload semantics or real measurements.

## Scaling interpretation

A future large-population run can now distinguish several failure modes.

### Healthy

- evidence rate rises;
- unique disposition rate keeps pace;
- backlog remains bounded;
- backlog age remains bounded.

### Integration saturation

- evidence rate remains high;
- unique disposition rate falls below evidence rate;
- backlog growth stays positive;
- oldest backlog age keeps increasing.

### Redundant integration

- raw disposition-reference rate is high;
- unique disposition rate is much lower;
- `redisposition_count` grows.

### Aggressive filtering

- disposition rate keeps pace;
- most first dispositions become duplicate/irrelevant/local-only;
- knowledge-reference fraction stays low.

This may be correct or may indicate destructive filtering; semantic benchmarks are needed to distinguish them.

### Backlog draining

- evidence generation slows;
- unique disposition rate remains high;
- backlog growth is negative;
- absorption ratio can exceed 1.0.

## Construction rule

This telemetry exists now so future information-volume problems are measurable without changing the final architecture.

It is not a new research gate and does not block construction.

Hierarchical integration, learned integration routing, storage partitioning, and deeper information-value metrics remain deferred until measured traces make them decision-relevant.
