# Large-Scope Relevance Program v0 — Historical / Deferred Summary

## Status

**Mechanically qualified historical benchmark program. No admitted real population-advantage result. Deferred from the current shared-weight Gate-2 line.**

The complete implementation/history remains available through the historical `ai_hypothesis.large_scope` code and PRs, especially #24–#32, #54/#55 and #86.

Repository-hygiene CI later proved that the current focused `population_compute` package/test suite passes with `ai_hypothesis.large_scope` physically absent from the checkout.

## Original question

The large-scope program asked whether a population of independently weighted small Worker-v1 checkpoints could inspect a larger input/source scope usefully under controlled neural work.

A synthetic large world was composed from many local Step-1 relevance windows. The benchmark deliberately separated two effects:

1. **scope expansion** — more local regions are inspected;
2. **worker-weight diversity** — different independently trained checkpoints inspect those regions.

This distinction remains a useful experimental lesson even though independent worker weights are no longer the canonical architecture.

## Core causal control

For the same world and same width, both modes inspected the same deterministic window prefix:

### `same_worker`

One checkpoint was reused across all inspected windows.

This measured scope-only scaling.

### `diverse_workers`

Distinct independently weighted checkpoints inspected the same windows.

This added worker-weight diversity while holding inspected scope fixed.

The width-1 condition was later hardened so both modes use exactly the same checkpoint and same window, making width 1 an exact identity control.

This is a generally valuable pattern:

> When adding population diversity, compare it against a same-scope control rather than attributing ordinary coverage gains to diversity.

## Threshold-free evaluation

The benchmark intentionally avoided introducing an arbitrary global relevance threshold in its construction stage.

It instead recorded measurements such as:

- inspected-scope coverage;
- target retrieval;
- retrieval given target inspection;
- target rank;
- target evidence;
- strongest distractor evidence;
- target-minus-distractor evidence gap;
- positive/negative world candidate evidence.

The later paired metrics compared `diverse_workers - same_worker` on exactly matched worlds/scope.

## Paired statistics

Because both modes saw the same world and same inspection prefix, the strongest causal comparison was paired.

The program tracked:

- both/same-only/diverse-only/neither retrieval outcomes;
- exact discordance probability;
- paired target-rank delta;
- paired target-evidence delta;
- paired distractor-evidence delta;
- paired target-gap delta;
- paired candidate-evidence changes on positive and negative worlds.

This avoids interpreting differences between two unrelated aggregate samples as a population-diversity effect.

## Persistent-runtime bridge

The benchmark was also mapped onto the historical Research Ledger/Work Thread runtime.

One local window became one bounded scoped Work Item. The bridge preserved:

```text
scheduler decision
→ scoped Work Item
→ ATTEMPT_STARTED
→ selected Worker-v1 execution
→ evidence contribution
→ terminal attempt state
```

A deterministic fixed benchmark scheduler reproduced exact requested width and worker/window plans so persistent execution could be checked against the direct benchmark before introducing adaptive behavior.

## Persistent equal-budget baseline

The historical persistent-scope experiments required a neutral equality contract before claiming value from persistent execution.

For a total local budget `N`, a persistent multi-round run was expected to reproduce the direct fixed-prefix width-`N` run while no adaptive deviation or failure occurred.

The comparison froze:

- same checkpoints;
- same worlds;
- same worker mode;
- same local evidence configuration;
- same deterministic window order;
- same deterministic worker order;
- same total local neural evaluation count.

Structural mismatches were treated as experiment corruption; small numeric output drift was measured separately.

This remains a useful baseline principle for any future adaptive population controller:

> First prove the persistent/adaptive runtime reproduces the non-adaptive computation under the same work budget; only then attribute deviations to the adaptive policy.

## Cross-world batching

Persistent large-scope worlds were later batched so many independent Work Threads could share one neural execution call per round.

This removed an artificial one-launch-per-world penalty while preserving each world's scheduler/provenance state.

Again the reusable rule is:

> Fuse compatible neural execution, not logical experiment identity.

## Timing diagnostic

A later profiler wrapped the selected-worker neural boundary and separated:

- total direct wall time;
- selected-worker learned execution time;
- residual non-selected-worker time;
- persistent setup/run/end-to-end time;
- selected-worker time in persistent execution;
- ledger/storage volume.

The residual was intentionally not mislabeled as “scheduler time” because it could contain projection, persistence, tensor handling and other work.

No real target-hardware result from this persistent large-scope profiler became an admitted scientific architecture result before the project pivoted to the shared-weight population-compute program.

## Qualification history

Historical PR #54 qualified the hardened large-scope benchmark mechanics and reported 27 focused tests passing on its clean head.

Historical PR #55 added a structural result auditor and reported 37 focused tests passing on its clean head.

Historical PR #86 later added a strict large-scope result-contract validator and qualified that contract in CI.

These qualifications establish mechanics/integrity checks, not a positive Worker-v1 population result.

## What was never established

Do not reinterpret the existence of this benchmark stack as evidence that independently weighted worker populations won.

The real development experiment depended on a local frozen bank of Worker-v1 checkpoints that was not stored in GitHub.

The program explicitly recorded that the actual checkpoint-based large-scope development result remained pending/not supplied through the repository evidence path.

Therefore no canonical claim should be made that:

- weight diversity improved large-scope retrieval;
- the persistent runtime improved capability;
- the persistent runtime improved throughput;
- adaptive scope beat fixed-prefix scope;
- the large-scope program passed a research gate.

## Why the program is deferred

The project later adopted a cleaner primary hypothesis:

> Hold learned parameters fixed and vary reusable runtime population computation/state.

Independent Worker-v1 checkpoints make total learned capacity grow with population width, which is a confound for that new question.

The current `population_compute` line therefore uses shared/reused learned machinery and weak runtime states rather than a bank of independently weighted models.

The large-scope benchmark remains useful as a source of experimental-design patterns, but its old implementation should not remain active merely because it is extensive.

## Reusable lessons

Preserve these ideas for future richer workloads:

1. separate source-scope gain from population/diversity gain;
2. use exact same-scope paired controls;
3. make width 1 an identity control when possible;
4. freeze deterministic inspection order before comparing policies;
5. report threshold-free evidence separation before inventing an acceptance threshold;
6. distinguish structural experiment equivalence from floating-point output equivalence;
7. compare adaptive/persistent execution against a neutral equal-work baseline first;
8. measure neural execution and orchestration overhead separately without overnaming residual time;
9. keep development/confirmation/test seed domains isolated;
10. never claim a neural result from mechanics-only CI.

## Reactivation conditions

A future Gate may justify a new large-scope workload when the active shared-weight organism needs to process source scope that cannot fit into one local state/context.

A reactivated version should use the current shared-weight organism and current resource-accounting discipline rather than restoring independently weighted Worker-v1 checkpoints by default.

Potential future causal questions include:

- can dynamic activation choose useful source regions under a fixed latency/work budget?;
- can persistent distributed state avoid rereading large local contexts?;
- does locality improve capability at matched information and work?;
- can population exploration inspect more useful possibilities within the same wall-clock budget?

Those would be new experiments, not a continuation verdict from the old benchmark.
