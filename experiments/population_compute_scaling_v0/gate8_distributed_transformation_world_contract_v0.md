# Gate-8 distributed-transformation world contract v0

## Status

**GENERATOR + EXACT SYMBOLIC ORACLE ADMITTED — TRAINING, ENCODER, 1B BASELINE AND SCIENTIFIC EXECUTION CLOSED.**

Exact qualified protocol head:

`e73541115e8ddd122f336463dc1a9ffdbf82df46`

Exact qualified protocol-correction head:

`124065691d257d483a37be4200452f1f7ca50063`

This stage creates benchmark mechanics only. It does not select an organism architecture, train weights, serialize prompts, load a tokenizer, bind 1B weights, or evaluate any scientific test condition.

## Fixed transform library

The symbol alphabet is `0..15`.

The world contract freezes eight explicit permutations over that alphabet. Every primitive is:

- a bijection;
- unique;
- pairwise non-commuting with every other primitive.

Transform IDs expose only primitive identity. They contain no path position, relevance bit, answer bit, population, depth, or world identity.

Every generated graph uses an exactly balanced transform multiset: each primitive appears `population / 8` times. Relevant and distractor edges therefore share the same edge schema and global transform marginal.

## Graph construction

For each valid `(population, depth)` condition:

- the graph contains exactly `population` directed edges;
- it contains exactly `population + 1` nodes;
- every non-root node has exactly one parent;
- all nodes belong to one rooted directed tree;
- the public target has one unique root-to-target path;
- that path contains exactly `depth` edges;
- all remaining `population - depth` edges are distractors;
- relevant path fraction is `depth / population <= 1/8`;
- the public edge list is shuffled until relevant workers do not occupy one contiguous block.

The target node is never used as the parent of a distractor edge, preserving an unambiguous leaf query without changing edge format.

## Worker surface

Worker `i` receives exactly one shard:

```text
worker_index
source_node
target_node
transform_id
```

The public query contains only:

```text
root_node
target_node
root_symbol
```

The answer and relevant-worker path live in a separate truth object. They are absent from the public dataclass and public dictionary surface.

## Opaque labels and namespaces

Every node receives a fixed-width opaque hexadecimal label derived from the complete split/world identity. Labels are independently remapped for every world and do not expose internal topology indices.

The following namespaces are disjoint:

- contract qualification;
- training;
- validation;
- scientific test;
- fixed 1B demonstrations.

Frozen bounds are enforced before construction:

- training world indices: `0..262143` per seed;
- test world indices: `0..511` per condition;
- demonstration indices: `0..7`.

CI generates accepted worlds only from the contract namespace. Its train/test/demonstration calls are rejection tests with out-of-range indices and terminate before labels, topology, transforms, answers, or truth objects are created.

## Exact oracle

The oracle receives only the public graph and query. It:

1. verifies one rooted tree with `population + 1` nodes;
2. reconstructs the unique target-to-root parent chain;
3. reverses it into root-to-target order;
4. verifies exact path length `depth`;
5. applies primitive transformations in that order;
6. returns the answer symbol and path evidence for internal audit.

The generated truth must exactly equal a fresh oracle reconstruction from public data.

## Leakage controls

Qualification proves:

- all 21 frozen conditions generate valid contract worlds;
- one worker maps to one edge;
- transform marginals are exactly balanced;
- relevant workers are interleaved with distractors;
- public objects contain no answer, truth, relevance, or path fields;
- node-label sets differ across worlds;
- root-to-target path length is exact;
- split bounds reject before generation;
- no Torch, NumPy, tokenizer, prompt, neural model, optimizer, CUDA, or training surface exists.

## Next boundary

The next stage may add only:

- one canonical public graph encoder;
- one canonical worker-shard encoder;
- exact tokenizer-independent size accounting;
- exact Gemma tokenizer binding and the 24,576-token budget proof.

It may not train the population organism, load 1B weights, choose prompts from test results, or execute the benchmark.
