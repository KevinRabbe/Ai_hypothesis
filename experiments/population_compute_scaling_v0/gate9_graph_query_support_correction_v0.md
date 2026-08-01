# Gate-9 graph-query support policy correction v0

## Status

**DATA-BLIND PREREGISTRATION CORRECTION — NO OPERATOR OR WORLD HAS BEEN
GENERATED, NO ARCHITECTURE EXISTS, AND TRAINING / SCIENTIFIC EXECUTION REMAIN
CLOSED.**

Bound heads:

```text
original Gate-9 protocol  e5e20e8de6707d35f1a7a9315a5a9a67deacc9a1
operator/support contract be6451e1af82b18749bd0313a9c02ca62c4eee5c
```

## Corrected ambiguity

The original protocol used one global statement that the query excludes the
nine support inputs. That is correct for the isolated novel-operator test, where
one fresh query can be selected outside the support basis.

It is not an admissible requirement for the distributed graph benchmark. A
worker's query is the preceding worker's emitted byte. The admitted affine
family can map an initially novel query onto any byte, including a support
input. For example, the identity linear map with bias `3` maps query `3` to
`0`, one of the support inputs.

Across paths as deep as 128, preventing every support hit would require one or
more of:

- rejecting otherwise-valid frozen operators;
- skipping counters in the frozen graph-test interval;
- rejecting worlds after examining path values;
- changing the support basis or operator distribution.

Each would condition the science on hidden path behaviour and violate the exact
operator-allocation contract.

## Frozen corrected policy

```text
isolated local query outside support basis  REQUIRED
graph query outside support basis           NOT REQUIRED
reject operator after graph support hit      FORBIDDEN
skip frozen operator counter                 FORBIDDEN
reject world after graph support hit         FORBIDDEN
```

The operator, support basis, graph matrix, thresholds, controls, classifier and
split ranges are unchanged.

The qualified public-support oracle already supports both policies explicitly:

- `require_novel_query=True` for isolated local induction;
- `require_novel_query=False` for graph execution.

## Mandatory descriptive evidence

Every graph condition must report support hits at the world/path-position unit:

```text
total path queries = worlds * depth
support-hit count
support-hit rate
support-hit count at each path position 0..depth-1
```

These values are descriptive and do not alter accuracy weights or thresholds.
They expose how often graph workers could answer by direct support lookup rather
than affine reconstruction.

The exact evidence vector has length `depth`, each position count lies within
`0..256`, and the sum must equal the condition total.

## Scientific interpretation

A positive Gate-9 result still requires:

- near-perfect isolated induction on queries outside the support basis;
- large causal losses under shuffled context and query-only controls;
- causal communication at the frozen high-scale conditions;
- the preregistered rising population/depth frontier.

Therefore occasional graph support hits cannot establish novel induction by
themselves. The isolated test and controls remain the causal protection.

## Closed boundary

This correction contains only:

- immutable local-versus-graph query-policy constants;
- a support-hit reporting schema;
- synthetic policy regressions;
- correction-only CI.

It contains no operator generator, graph generator, model, optimizer,
checkpoint, training path, scientific world, execution wrapper, result reader
or classifier invocation.

## Next boundary

After qualification, the graph-world contract may use the corrected policy and
must preserve every frozen graph-test operator counter without rejection or
skipping.
