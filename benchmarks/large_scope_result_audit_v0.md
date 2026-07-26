# Large-Scope Result Audit v0

## Purpose

`large-scope-relevance-v0` deliberately separates benchmark mechanics from the later scientific interpretation of frozen Worker-v1 results.

The benchmark can produce a valid result in which diverse workers help, hurt, or make no measurable difference. Before any of those outcomes are interpreted, the result artifact itself must satisfy the causal controls and accounting rules that make the comparison meaningful.

This document defines that integrity boundary.

> A structurally valid result is admissible research evidence. It is not automatically a successful research result.

The auditor therefore validates invariants and renders measurements, but does not create a scalar score, acceptance threshold, or automatic roadmap-gate decision.

## Input

The auditor consumes the JSON emitted by:

```text
python -m ai_hypothesis.large_scope.run_relevance ...
```

No checkpoint files, Torch runtime, GPU, Research Ledger, or worker execution are required to audit an already-created artifact.

The implementation is pure Python:

- `ai_hypothesis.large_scope.result_audit`
- `ai_hypothesis.large_scope.audit_relevance`

Public API:

```text
LargeScopeResultAudit
ResultAuditIssue
audit_large_scope_result(...)
render_large_scope_audit_markdown(...)
```

## CLI

Development result:

```text
python -m ai_hypothesis.large_scope.audit_relevance \
    --input results/large_scope_relevance_v0/development.json \
    --output results/large_scope_relevance_v0/development.audit.md
```

Exit codes:

- `0`: JSON parsed and all benchmark-integrity checks passed;
- `2`: JSON parsed but violates one or more benchmark-integrity invariants.

Unreadable or malformed JSON remains a command error rather than a research result.

The frozen `test` split is rejected unless `--allow-test-split` is explicit.

## Top-level integrity

The auditor requires:

- benchmark version exactly `large-scope-relevance-v0`;
- known split;
- positive world count;
- positive, unique, strictly increasing widths;
- only known worker modes;
- configured window count covering the largest width;
- loaded population width covering the largest diverse-worker width;
- exact local-window evaluation accounting;
- no unexpected world-level `acceptance_threshold`.

For both worker modes:

```text
local_window_evaluations
=
world_count × number_of_modes × sum(widths)
```

A mismatch means the result does not represent the workload described by its own metadata.

## Condition-summary arithmetic

Every `(mode, width)` condition must exist exactly once.

For each condition:

```text
positive_world_count + negative_world_count = world_count
```

and:

```text
target_inspected_count <= positive_world_count
target_retrieved_count <= target_inspected_count
```

Reported rates must be exactly consistent with their counts, within the configured numerical tolerance:

```text
target_coverage_rate
= target_inspected_count / positive_world_count

target_retrieval_rate
= target_retrieved_count / positive_world_count

retrieval_given_inspected
= target_retrieved_count / target_inspected_count
```

Zero denominators require `null`, not a manufactured rate.

Continuous diagnostic fields must be finite when present.

## Same-scope causal control

For a fixed width, `same_worker` and `diverse_workers` inspect the exact same deterministic world/window prefix.

Therefore they must agree on quantities determined before neural worker identity matters:

- positive/negative world count;
- target-inspected count;
- target-coverage rate.

If those differ, the artifact is not a valid worker-diversity comparison.

Because inspection prefixes are nested, target coverage may stay flat or increase with width, but may not decrease.

## Paired-summary integrity

When both worker modes are present, every requested width must have exactly one paired summary.

Paired deltas are always defined as:

```text
diverse_workers - same_worker
```

The retrieval contingency must close exactly:

```text
both
+ same_only
+ diverse_only
+ neither
= target_inspected_count
```

and:

```text
both + same_only    = same_target_retrieved_count
both + diverse_only = diverse_target_retrieved_count
same_only + diverse_only = retrieval_discordant_count
```

The auditor independently recomputes:

- same-worker retrieval-given-inspection;
- diverse-worker retrieval-given-inspection;
- their paired delta;
- the exact two-sided discordance probability from the discordant counts.

It does not trust those numbers merely because they exist in the JSON.

Paired standard errors must be finite and non-negative when present.

## Width-1 control

Width 1 is the benchmark's exact causal sanity check.

Both modes:

1. inspect the same single window;
2. use the exact same deterministic base checkpoint.

Therefore width 1 must contain no worker-diversity difference.

The auditor requires:

```text
same_only_retrieved_count = 0
diverse_only_retrieved_count = 0
retrieval_discordant_count = 0
```

same/diverse retrieval counts must match, and the exact discordance probability must be `null` because there are no discordant pairs.

All paired neural-output deltas and their standard errors must be `null` or numerically zero within `zero_tolerance`.

A nonzero width-1 difference is treated as a benchmark/runtime integrity failure, not as evidence for or against the population hypothesis.

## Invalid-result rendering

The Markdown renderer always prints the integrity errors first.

If integrity is invalid, it intentionally omits condition/paired result tables:

```text
Result tables are omitted because benchmark-integrity validation failed.
```

This has two purposes:

- malformed keys or values cannot crash the reporting path;
- invalid measurements are not visually presented as if they were admissible scientific evidence.

## Valid-result rendering

A valid artifact produces two threshold-free tables.

### Condition table

Shows, by width and worker mode:

- deterministic target coverage;
- target retrieval;
- retrieval conditional on inspection;
- mean target rank;
- mean target-minus-distractor evidence gap;
- mean/max negative-world candidate evidence.

### Paired table

Shows direct `diverse_workers - same_worker` diagnostics:

- retrieval-given-inspection delta;
- same-only and diverse-only retrieval counts;
- exact discordance probability;
- target-rank delta;
- target-evidence delta;
- strongest-distractor evidence delta;
- target-gap delta;
- negative-world candidate-evidence delta.

Sign conventions:

- retrieval delta > 0 favors diverse workers;
- target-rank delta < 0 favors diverse workers;
- target-evidence delta > 0 favors diverse workers;
- distractor-evidence delta < 0 favors diverse workers;
- target-gap delta > 0 favors diverse workers;
- negative-world candidate-evidence delta < 0 means lower false-positive pressure under diverse workers.

These are diagnostic directions, not success thresholds.

## Scientific boundary

The auditor answers:

> Is this result artifact internally consistent with the frozen benchmark contract and safe to interpret?

It does not answer:

> Did the population architecture win?

That later judgment must use the actual development measurements and preserve the benchmark's decomposition:

1. possibility/scope coverage;
2. retrieval conditional on inspection;
3. paired diversity effect;
4. target quality versus distractor pressure;
5. negative-world false-positive pressure;
6. compute/wall-time cost.

No single metric is promoted into a hidden scalar objective in v0.

## Executable qualification

The hardened result auditor is covered by the same Python 3.11 + CPU-Torch large-scope qualification lane as the benchmark itself.

Clean #55 run `30205165459` passed **37/37** focused large-scope tests on head `9c6b17696894699f983a24d516f0fd5582ba73b5`.

The same head's inherited indexed-runtime architecture lane also passed.

The 37-test large-scope suite includes:

- deterministic benchmark/world construction;
- worker-control and paired-metric invariants;
- runtime-bridge equivalence;
- CLI benchmark artifact generation;
- valid result audit;
- count/rate arithmetic corruption;
- paired contingency/probability corruption;
- malformed-summary rendering;
- test-split lock;
- width-1 control violations.

This is executable integrity qualification only. The frozen 16-checkpoint Worker-v1 development experiment remains a separate, still-pending empirical step.
