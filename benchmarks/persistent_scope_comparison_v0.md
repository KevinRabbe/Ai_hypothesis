# Persistent Scope Comparison v0

## Question

What end-to-end cost does the persistent runtime add when it is forced to perform the **same learned work** as the direct large-scope benchmark?

This is a systems/control baseline, not an adaptive-allocation test.

The comparison freezes:

- the same Worker v1 checkpoints;
- the same benchmark worlds;
- the same worker mode;
- the same evidence configuration;
- the same total local neural-evaluation count;
- the same deterministic source-region order;
- the same deterministic worker order.

Only the execution organization changes.

## Normalized budget

The direct condition uses:

```text
direct_width = step_width × rounds
```

The persistent condition uses:

```text
step_width local attempts/world/round
× rounds
```

For v0, `direct_width` must not exceed the world window count. The neutral comparison intentionally stops before persistent execution enters redundant verification.

In diverse-worker mode, direct width must also fit within the loaded checkpoint population so the direct control and persistent checkpoint sequence describe the same distinct worker set.

## Conditions

### Direct

The existing batch-native direct benchmark evaluates all worlds in one chunk at `direct_width`.

### Persistent

The persistent multi-world harness creates one Work Thread per world and performs `rounds` cross-thread batches.

For a chunk of `W` worlds:

```text
direct:
    1 selected-worker call × (W × direct_width windows)

persistent:
    rounds selected-worker calls
    each call = W × step_width windows
```

The neural evaluation count is identical.

The persistent condition additionally performs:

- thread projection;
- scheduler decisions;
- Work Item construction;
- ATTEMPT lifecycle persistence;
- evidence persistence;
- coverage projection;
- final persistent evaluation projection.

Those costs are intentional and measurable.

## World chunking

`--world-batch-size` bounds the number of persistent Work Threads alive in one temporary experiment ledger and the number of worlds flattened into one direct call.

Default:

```text
64 worlds
```

The runner regenerates the next deterministic chunk after finishing the previous one. It does not retain the complete dataset in memory.

## Timing boundaries

Checkpoint loading and benchmark-world generation are outside the condition timing because they are common inputs.

An unmeasured direct forward pass warms each worker mode before timing begins.

### Direct time

Measured around `evaluate_scope_batch(...)` with device synchronization before and after.

### Persistent setup time

Measures:

- SQLite Research Ledger creation;
- persistent Work Thread creation/recovery;
- coverage planners;
- scheduler/control composition.

### Persistent run time

Measured around `PersistentScopeWorldBatchExperiment.run_rounds(...)` with device synchronization before and after.

It includes scheduler/control work, ledger writes, neural execution, and final persistent projection.

### Persistent end-to-end time

```text
persistent_setup_seconds + persistent_run_seconds
```

The report includes both run-only and end-to-end local-evaluation throughput.

No compiler mode is changed by this benchmark. Compiler/runtime optimization remains a separate systems variable.

## Storage/accounting

For every temporary persistent chunk the runner records:

- total Research Ledger event count;
- SQLite main database bytes;
- WAL bytes;
- shared-memory bytes while the ledger is open.

Chunk storage bytes are summed in the final result so persistence cost is visible instead of hidden.

The temporary chunk ledger is deleted after its metrics are incorporated.

## Equivalence qualification

Before interpreting timing, direct and persistent conditions must describe the same experiment.

### Structural identity — hard requirement

The comparison aborts on differences in:

- split;
- world seed;
- worker mode;
- width;
- inspected window order;
- worker-index order;
- target presence/index;
- whether the target was inspected.

These are experimental-identity errors, not model-output differences.

### Output equivalence

The comparison records:

- maximum absolute evidence/uncertainty/margin drift;
- local decoded-label mismatches;
- candidate mismatches;
- target-rank mismatches;
- target-evidence presence mismatches;
- strongest-distractor evidence-presence mismatches.

Default numeric tolerance:

```text
1e-5
```

A semantic mismatch or numeric drift above tolerance makes the comparison fail qualification but still produces a result JSON and a nonzero exit status.

This allows batching-shape numerical drift to be inspected rather than confusing it with source/worker identity corruption.

## Common metric surface

A non-redundant `PersistentScopeEvaluation` is projected back into the existing `ScopeEvaluation` contract.

Therefore direct and persistent quality summaries use the **same** `ScopeMetricsAccumulator` implementation.

The persistent-to-direct projection is rejected if:

- attempt count and evidence count differ;
- a persistent attempt lacks a local evidence record;
- the persistent budget already contains repeated source regions;
- no candidate evidence exists.

This prevents the neutral overhead baseline from silently turning into a verification/redundancy experiment.

## CLI

Example:

```bash
python -m ai_hypothesis.large_scope.run_persistent_comparison \
  --checkpoints checkpoint_1.pt checkpoint_2.pt checkpoint_3.pt checkpoint_4.pt \
  --device cuda \
  --backend vmap \
  --split development \
  --world-count 1000 \
  --world-batch-size 64 \
  --window-count 16 \
  --step-width 2 \
  --rounds 4
```

This compares direct width 8 against four persistent width-2 rounds.

The frozen test split is rejected unless `--allow-test-split` is supplied explicitly.

## Output

The JSON records:

- exact comparison version;
- split/seeds/world count;
- chunk size;
- step width/rounds/direct width;
- evidence configuration;
- Worker v1 architecture;
- exact checkpoint metadata and stable IDs;
- ordered worker-bank identity;
- direct quality summaries;
- persistent quality summaries;
- equivalence summaries by mode;
- direct elapsed time;
- persistent setup/run/end-to-end time;
- local-evaluations/second;
- persistent/direct time ratio;
- persistent ledger events and storage bytes.

Exit code:

```text
0 = all direct/persistent worlds equivalent within tolerance
2 = one or more output-equivalence failures
```

Structural experiment mismatches raise immediately.

## Interpretation

### Equivalent + modest overhead

The persistent runtime preserves the learned experiment and is a viable base for adaptive-scope tests.

### Equivalent + large overhead

The neural architecture may still be valid, but runtime organization must be profiled before adaptive capability claims are meaningful.

Likely next systems targets would be chosen from measured cost, not guessed in advance: ledger writes, projection scans, Python orchestration, tensor preparation, batching, or compiler/launch overhead.

### Not equivalent

Do not interpret performance timing or quality differences as adaptive value.

First identify whether the divergence comes from:

- worker order;
- source order;
- numerical batching drift;
- persistence/reconstruction logic;
- evidence conversion.

## What this does not establish

This benchmark does not establish:

- an adaptive routing advantage;
- useful dynamic width;
- optimal persistent step size;
- optimal world chunk size;
- compiler benefit;
- knowledge-integration saturation;
- a Gate 5 result.

It establishes the controlled execution baseline those later experiments require.
