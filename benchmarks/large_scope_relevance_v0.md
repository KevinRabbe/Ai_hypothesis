# Large-Scope Relevance Benchmark v0

## Status

**Construction artifact only. Gate 5 (adaptive allocation) is not being declared active by this document. No result is claimed until the frozen Worker v1 checkpoints are executed against the benchmark.**

This benchmark exists to make the next architecture-specific scope question executable without retraining Worker v1 or inventing a second neural task.

## Question

Step 1 established that one tiny worker can perform a useful local transformation. Step 2A established that independently weighted workers contain non-identical useful information when they inspect the same local input.

The next scope-specific question is:

> When the total input is larger than one worker's local context, can a population convert additional inspected scope into useful target evidence, and do independently weighted workers add value beyond the scope increase alone?

The benchmark deliberately separates those two effects.

## Reused frozen primitive

The benchmark reuses Step 1 task family E, `E_RELEVANCE`, unchanged.

One local worker still receives exactly:

```text
32 rows × 16 floating-point features
```

and the ordinary Step 1 task-control row. No new token, embedding, head, label, or worker architecture is introduced.

Local labels remain:

```text
RELEVANT
NOT_RELEVANT
UNCERTAIN
```

The frozen Step 1 generator already makes `RELEVANT` depend on two noisy signatures appearing in the correct local relationship while negatives contain wrong order, excessive separation, missing signatures, noise, and near misses. Hard examples can include occlusion. This is why task E was originally described as a proxy for later large-input allocation.

## Large world

A large-scope world is a tuple of ordinary Step 1 relevance windows.

Default v0:

```text
window_count = 16
```

The count is configurable so later runs can use 64, 256, or more windows without changing the local worker contract.

Each world is either:

```text
positive: exactly one RELEVANT window
negative: zero RELEVANT windows
```

All non-target windows are generated as either:

- `NOT_RELEVANT` answerable distractors; or
- controlled `UNCERTAIN` distractors.

Default difficulties:

```text
target_difficulty = hard
distractor_difficulty = hard
ambiguous_distractor_fraction = 0.125
```

Target-present and target-absent worlds alternate deterministically by world-seed parity unless explicitly overridden for a focused diagnostic.

## No local-data retraining

Every window is produced by the frozen Step 1 generator.

Large-scope construction changes only **which already-valid local examples coexist in one larger world**. It does not alter the learned transformation the worker must perform.

This matters scientifically: an initial large-scope result can be attributed to organization/scope behavior rather than new worker training.

## Split isolation

Large-scope worlds use split-local deterministic generation and disjoint reserved Step 1 window-seed ranges:

```text
development:   [3_000_000_000, 3_300_000_000)
confirmation:  [3_300_000_000, 3_600_000_000)
test:          [3_600_000_000, 3_900_000_000)
```

Therefore a local relevance window generated for one large-scope split cannot reappear in another split through seed reuse.

Process policy:

1. Develop benchmark mechanics and any future decision threshold on `development` only.
2. Freeze those choices before `confirmation` interpretation.
3. Keep `test` unopened until the relevant research gate explicitly calls for it.

The CLI refuses `--split test` unless `--allow-test-split` is supplied explicitly.

## Nested inspection order

World layout and inspection order use separate deterministic hash domains.

For every world, one no-duplicate inspection permutation is generated. Widths are prefixes of that same permutation:

```text
width 1  ⊂ width 4 ⊂ width 16 ⊂ ...
```

Consequences:

- increasing width adds new inspected scope;
- it never replaces a region already seen by a smaller width;
- duplicate discovery work is absent from this initial scope benchmark;
- comparisons can attribute changes cleanly to added scope and worker assignment.

## Two primary conditions

Every condition at a given width inspects **the same window indices**.

### A. Scope-only control — `same_worker`

Choose one checkpoint deterministically for that world and use the same worker weights for every inspected window.

```text
W inspected windows
×
one repeated worker checkpoint
```

This spends W local neural transformations and measures the effect of inspecting more scope without adding weight diversity.

### B. Scope + population diversity — `diverse_workers`

Assign a distinct independently weighted checkpoint to each inspected window when `W <= population_width`.

```text
W inspected windows
×
W distinct worker checkpoints
```

The assignment is deterministic and nested across widths.

This condition has the same inspected scope and number of local neural transformations as the scope-only control. The intended difference is independently learned weights.

## Why this comparison matters

If both conditions improve identically with width, the gain may be explained by ordinary tiling/coverage rather than population diversity.

If `diverse_workers` improves target evidence quality, target ranking, robustness, or false-positive behavior relative to `same_worker` on the same windows and equal local-transform count, that is evidence for additional architecture-specific value from the independently weighted population.

No result should be described as a population advantage merely because width inspected more regions.

## Evidence path

The selected-worker runtime is reused directly:

```text
inspection prefix
    ↓
worker-index plan
    ↓
HomogeneousWorkerBank.forward_selected(...)
    ↓
Step01Output
    ↓
Step 2 build_evidence_matrix(...)
    ↓
continuous local evidence per inspected window
```

For every inspected window the evaluator keeps:

- worker index;
- local label after the existing 0.5 uncertainty decoding boundary;
- continuous `RELEVANT` evidence;
- continuous `NOT_RELEVANT` evidence;
- uncertainty probability;
- invalid-label mass;
- top valid-label margin.

The world candidate is the inspected window with maximum continuous `RELEVANT` evidence, with deterministic first-inspected tie breaking.

## No world-level threshold in v0 construction

The initial evaluator does **not** decide that the strongest candidate is globally accepted as relevant.

A threshold would affect the positive/negative error trade-off and must be calibrated empirically. Hard-coding one during benchmark construction would mix a new reducer decision with the scope question.

Instead v0 reports the evidence distributions needed to decide later whether a useful threshold exists.

## Threshold-free metrics

Per `(split, worker mode, width)`:

### Scope

- positive-world target coverage count/rate;
- target-inspected count.

`target_coverage_rate` is primarily a deterministic sanity check on inspection scope.

### Retrieval

- target retrieval count/rate, where the highest-RELEVANT-evidence candidate is the true target;
- retrieval conditional on target having been inspected;
- mean target evidence rank when inspected.

The conditional metric is particularly important because it removes the trivial effect of whether width happened to reach the target.

### Evidence separation

- mean target `RELEVANT` evidence when inspected;
- mean strongest distractor `RELEVANT` evidence;
- mean target-minus-strongest-distractor evidence gap;
- mean positive-world candidate evidence;
- mean negative-world candidate evidence;
- maximum observed negative-world candidate evidence.

These expose the false-positive competition created by inspecting more distractors without prematurely choosing an acceptance threshold.

## Key decomposition

For positive worlds:

```text
target retrieval
≈
target was inspected
×
target won local evidence competition once inspected
```

The benchmark therefore distinguishes:

```text
coverage failure
from
local evidence/ranking failure
```

and then compares the second component between repeated weights and diverse weights.

## Expected width behavior is not assumed monotonic

More inspected scope creates two opposing effects:

1. higher probability of reaching the true target;
2. more opportunities for a distractor to produce a spurious high `RELEVANT` score.

Therefore a larger width can legitimately improve coverage while worsening evidence competition. That is useful evidence about the architecture rather than a benchmark failure.

## Initial widths

With the current 16 frozen Worker v1 checkpoints and default 16-window world:

```text
1
4
16
```

Use larger world/width values only after a correspondingly larger frozen checkpoint bank exists or when explicitly testing repeated-worker scope without the diverse condition.

## Interpretation matrix

### Strongest positive signal

`diverse_workers` materially outperforms `same_worker` on retrieval-given-inspection and/or target-vs-distractor separation under the same width and inspected windows.

Interpretation: independent weights add useful local-search/evidence diversity beyond tiling alone.

### Scope-only success

Both modes improve similarly and conditional retrieval is approximately equal.

Interpretation: larger inspected scope helps, but this benchmark has not demonstrated extra value from independent worker weights.

### False-positive pressure dominates

Coverage rises with width while target rank/evidence gap deteriorates sharply and negative-world candidate evidence rises.

Interpretation: local evidence quality or integration must improve before wider scope is useful.

### No useful scope scaling

Increasing width does not improve target retrieval enough to offset added local-transform cost.

Interpretation: the tested worker/task organization does not support the scope hypothesis at this regime.

All are valid research outcomes.

## Resource accounting

The CLI records:

- world count;
- window count;
- widths;
- modes;
- loaded checkpoint metadata;
- Worker v1 architecture configuration;
- evidence configuration;
- total elapsed wall time with CUDA synchronization at run boundaries;
- exact number of local neural window evaluations.

It deliberately does not sum heterogeneous per-attempt resource maps or manufacture a scalar efficiency score.

## Bounded execution

Condition summaries use a streaming accumulator. Memory therefore scales with the number of `(mode, width)` conditions, not with world count.

Worlds are deterministic from split + seed and do not need to be stored in memory or result JSON to be reproducible.

## Example development command

```powershell
$checkpoints = 1..16 | ForEach-Object {
    "results\step01\checkpoint_50k_extended_15k\seed_$($_)\best.pt"
}

python -m ai_hypothesis.large_scope.run_relevance `
    --checkpoints $checkpoints `
    --device cuda `
    --backend vmap `
    --split development `
    --world-count 1000 `
    --window-count 16 `
    --widths 1 4 16 `
    --output results\large_scope_relevance_v0\development.json
```

Do not add `--allow-test-split` during ordinary development.

## Non-goals of v0

This benchmark does not yet prove or test:

- adaptive scheduler quality;
- learned scope routing;
- zoom-in policies;
- recursive/hierarchical integration;
- population versus dense-model superiority;
- same-total-parameter organization;
- real-document/image/code workloads;
- a calibrated global relevance decision threshold;
- Gate 5 success.

It is the smallest controlled bridge from the existing **local worker + same-input population** evidence to the architecture's intended **different workers inspecting different parts of a larger input** behavior.
