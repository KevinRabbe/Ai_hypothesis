# Gate-8 distributed-transformation capability protocol correction v0

## Status

**DATA-FROZEN PRE-EXPOSURE CORRECTION — EXECUTION REMAINS CLOSED.**

Base qualified protocol head:

`e73541115e8ddd122f336463dc1a9ffdbf82df46`

No benchmark world, training example, checkpoint, prompt trial, 1B inference, or test answer existed when this correction was frozen.

## Inconsistency

The qualified protocol contained both:

- “Relevant edges are exactly one eighth of all graph edges.”
- A condition is valid when `8 × depth <= population`.

The relevant edges are the edges on the unique root-to-target path, so their count is exactly `depth`. The two statements are simultaneously true only when `population = 8 × depth`, which would not produce the frozen 21-condition matrix.

## Superseding rule

The single inconsistent sentence is replaced by:

> The unique relevant path contains exactly `depth` edges. Relevant path edges are at most one eighth of all graph edges; equality holds when `population = 8 × depth`.

Formally:

```text
relevant_path_edges = depth
total_edges = population
distractor_edges = population - depth
relevant_fraction = depth / population <= 1/8
```

This retains the complete frozen condition rule:

`8 × depth <= population`.

It also preserves all 21 population/depth conditions, the one-worker-per-edge rule, the unique path, all classifier thresholds, training/test namespaces, causal controls, and the frozen 1B reference contract.

## Scientific reason

Changing the generator to manufacture additional “relevant” edges outside the unique answer path would create undefined task semantics and could introduce shortcut or leakage structure. Treating the path edges as the only relevant edges is the minimal interpretation consistent with the research question.

## Boundary

This correction opens no:

- world generator;
- symbolic oracle execution;
- canonical encoder;
- organism architecture or training;
- baseline revision or weights;
- benchmark execution;
- result or audit path.

The next implementation stage must bind both the original protocol head and this correction head before generating any world.
