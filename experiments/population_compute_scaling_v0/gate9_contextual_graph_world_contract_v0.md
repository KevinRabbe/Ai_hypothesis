# Gate-9 contextual graph-world contract v0

## Status

**PUBLIC/TRUTH GRAPH-WORLD CONTRACT QUALIFIED FOR CONTRACT WORLDS ONLY — THE
SCIENTIFIC TEST ASSIGNMENT KEY, TEST-WORLD GENERATION, ARCHITECTURE, TRAINING,
CHECKPOINTS, EXECUTION AND RESULTS REMAIN CLOSED.**

Bound heads:

```text
Gate-9 protocol                    e5e20e8de6707d35f1a7a9315a5a9a67deacc9a1
operator/support contract          be6451e1af82b18749bd0313a9c02ca62c4eee5c
graph-query support correction     0dd7e5417bab5d3af074772a60725b95d22be76f
```

This slice establishes the exact graph semantics and leakage boundaries without
opening the frozen scientific test namespace.

## Rooted tree contract

Every world contains:

```text
population edges
population + 1 nodes
one root
one public target
one unique root-to-target path of exact length depth
population - depth distractor edges
one contextual affine operator per edge
```

Canonical construction first creates the relevant path, then attaches each new
distractor node to one deterministic earlier node. The resulting graph is a
rooted directed tree: every non-root node has exactly one incoming edge, the
root has none, and the public root-to-target path is unique.

## Opaque public representation

Internal node indices are replaced by 24-hex SHA-256 labels. Canonical edges are
then deterministically shuffled into contiguous public worker indices.

A public worker contains only:

```text
worker_index
source_node
target_node
nine support pairs
```

The public query contains only:

```text
root_node
target_node
root_symbol
```

It exposes no operator counter, operator key, canonical edge index, path flag,
answer, private truth, condition role or relevance label.

The public world ID is SHA-256 of the complete canonical JSON public payload. It
contains no private truth field.

## Exact public-support graph oracle

The oracle uses public topology to trace the unique incoming chain from target
back to root, reverses it, and composes each worker's operator reconstructed
from that worker's public support pairs.

For every relevant path position it also records whether the incoming byte is
one of the nine support inputs. Graph queries are accepted in either case under
the qualified correction; no operator or world is rejected.

Private truth must exactly match:

- the path worker-index sequence inferred from public topology;
- the answer reconstructed from public supports only;
- the support-hit Boolean vector;
- one unique contract operator counter per public worker.

## Exact operator allocation

### Contract worlds

Contract tests use a separate disjoint interval beginning at `2^60` and exactly
8 worlds per each of the 21 frozen conditions. Its size is:

```text
sum(population over 21 conditions) * 8 = 82,176 operators
```

One fixed contract-only SHA-256 key defines an affine permutation over this
complete interval. Exhaustive qualification proves every counter is used
exactly once.

### Future scientific test worlds

The frozen graph-test interval remains:

```text
start 2^48
count 2,629,632
```

Condition/world/canonical-edge ordinals are assigned through an exact keyed
affine permutation modulo `2,629,632`:

```text
permuted = (a * ordinal + b) mod count
gcd(a, count) = 1
```

The inverse exists exactly. No counter may be skipped or reused.

The scientific assignment key is deliberately unbound in this contract. It
must be generated and frozen only after all three model checkpoints have been
independently admitted and before any test world is generated. It is never a
model input.

This prevents the operator identity recoverable from public support from acting
as a stable encoding of condition, world, or edge position during training.
The key may be published in the later immutable execution evidence after model
weights are fixed.

## Contract-only generation

The only world generator admitted here is:

```text
generate_gate9_contract_world(...)
```

It accepts valid population/depth conditions and world indices `0..7`. A test
split is rejected. There is no default or embedded scientific assignment key
and no `generate_gate9_test_world` path.

## Qualified regressions

The contract proves:

- deterministic public/truth generation across representative low and maximum
  conditions;
- exact public-support oracle equality and support-hit accounting;
- unique world identities and non-canonical public worker order;
- exhaustive one-to-one use of all 82,176 contract counters;
- exact invertibility of the future keyed test allocation;
- no public counter/key/canonical/truth/answer fields;
- rejection of topology, support, world-ID and truth tampering;
- scientific test generation, architecture, training and execution remain
  closed.

## Closed boundary

This branch contains only:

- deterministic contract-tree construction;
- public and private contract schemas;
- contract-only operator allocation;
- future test-allocation permutation mechanics with no bound key;
- exact public-support graph oracle;
- support-hit accounting;
- structural tests, documentation and CI.

It contains no neural architecture, Tensor library, optimizer, checkpoint,
training example builder, scientific test assignment key, test-world generator,
runtime, result artifact or classifier execution.

## Next admission boundary

The next slice may freeze the 19,649-parameter contextual worker architecture
and its model-input serialization using contract worlds only. Training remains
closed until architecture, parameter count, support/query encoding, local
computation budget and leakage regressions qualify independently.
