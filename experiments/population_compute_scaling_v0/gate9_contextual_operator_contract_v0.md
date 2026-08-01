# Gate-9 contextual affine-operator contract v0

## Status

**DETERMINISTIC OPERATOR / PUBLIC-SUPPORT / ORACLE CONTRACT ONLY — GRAPH
WORLDS, ARCHITECTURE, TRAINING, CHECKPOINTS, SCIENTIFIC TEST EXECUTION AND
RESULT CLASSIFICATION REMAIN CLOSED.**

Frozen protocol head:

```text
e5e20e8de6707d35f1a7a9315a5a9a67deacc9a1
```

This slice implements the minimum next boundary admitted by the Gate-9
preregistration. It establishes the exact local task family and independently
recoverable public evidence before any neural architecture or world generator
exists.

## Exact uint64 counter permutation

One unsigned 64-bit counter is mapped to one unsigned 64-bit operator key by the
frozen SplitMix64 output permutation:

```text
z = counter + 0x9E3779B97F4A7C15
z = (z xor (z >> 30)) * 0xBF58476D1CE4E5B9
z = (z xor (z >> 27)) * 0x94D049BB133111EB
key = z xor (z >> 31)
```

Every operation is modulo `2^64`. The contract includes the exact inverse using
odd modular multiplicative inverses and inverse right-xor shifts. Therefore:

```text
inverse(key(counter)) = counter
```

for the complete uint64 domain, not only the frozen ranges.

## Exact affine family

The 64 key bits are interpreted as:

```text
bits  0..27  strict-lower entries of unit-lower L
bits 28..55  strict-upper entries of unit-upper U
bits 56..63  eight-bit bias b
```

Rows and columns use bit indices `0..7`. The local function is:

```text
A = L U over GF(2)
f(x) = A x XOR b
```

Both triangular factors have unit diagonal. Their product is invertible, so
every operator is a bijection over all 256 symbols.

The mapping from `(L,U,b)` to an affine operator is injective. The implementation
recovers the unique unit-lower/unit-upper factorization of `A` without pivoting
and rejects matrices outside this family.

Consequently all `2^64` keys identify distinct affine operators.

## Public support contract

Every operator exposes exactly the protocol's nine inputs:

```text
0, 1, 2, 4, 8, 16, 32, 64, 128
```

The order is one globally fixed SHA-256-derived permutation. It is identical for
every operator. This closes an otherwise possible side channel in which an
operator-specific support order could encode part of the hidden key.

The model-facing public evidence contains only the nine `(input, output)` byte
pairs. It contains no key, counter, range name, split, worker ID or operator ID.

## Exact public-support oracle

The oracle reconstructs the affine map from public pairs only:

```text
b             = f(0)
column_i(A)   = f(2^i) XOR f(0)
```

It then:

1. reconstructs all eight matrix rows;
2. requires the unique unit `L U` decomposition;
3. repacks the exact 64-bit key for audit equality;
4. regenerates all nine support outputs;
5. applies the reconstructed map to the query.

Scientific queries must be outside the nine support inputs. A missing pair,
duplicate input, changed basis, malformed byte, non-family matrix or support
reconstruction mismatch fails closed.

## Frozen split proof

The contract binds the exact disjoint counter intervals:

```text
train       [0,             0 + 262144)
validation  [2^32,          2^32 + 32768)
local test  [2^40,          2^40 + 4096)
graph test  [2^48,          2^48 + 2629632)
```

Because the intervals are disjoint and the counter permutation and operator
mapping are injective, every pairwise operator intersection is exactly zero.
The audit reports all six pairwise intersections explicitly.

Boundary counters from every range are round-tripped through:

```text
counter -> key -> operator -> public support -> reconstructed operator -> key
```

## Qualified regressions

The contract tests:

- exact SplitMix64 inversion on uint64 boundaries plus 20,000 deterministic
  random values;
- key/factor/support/oracle round-trip on 4,099 operators;
- complete 256-symbol bijectivity for boundary operators;
- one operator-independent support order;
- malformed support and non-family matrix rejection;
- exact frozen split counts and zero intersections;
- absence of graph, model, training, checkpoint, scientific-test and artifact
  surfaces.

## Closed boundary

This branch admits only:

- deterministic counter-to-key mapping;
- deterministic key-to-affine-operator construction;
- public nine-example support generation;
- exact public-support reconstruction and oracle evaluation;
- mathematical split-disjointness audit;
- contract-only tests and CI.

It contains no graph topology, world namespace, root/target query, worker
architecture, Torch, NumPy, optimizer, checkpoint, training loop, test-world
execution, result reader or classifier execution.

## Next admission boundary

The next slice may implement only the Gate-9 public/truth graph-world contract
using these qualified local operators and supports:

- rooted directed tree construction;
- opaque node identities and shuffled worker assignment;
- unique operator allocation per edge from the frozen graph-test interval;
- exact public-support path oracle;
- namespace and leakage regressions.

Architecture, training and scientific execution remain closed.
