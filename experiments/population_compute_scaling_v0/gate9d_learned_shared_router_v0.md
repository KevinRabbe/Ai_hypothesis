# Gate-9D learned shared router v0

## Status

**DEVELOPMENT-ONLY SUPERVISED ROUTING DIAGNOSTIC. NOT AUTOMATIC COORDINATE
DISCOVERY, NOT END-TO-END ANSWER-LOSS LEARNING, AND NOT POPULATION
CONFIRMATION.**

The corrected sparse population result proves the affine bridge can be executed
exactly with bounded communication. This slice replaces only the hard-coded
worker activation predicates with one shared neural router.

## Router

Every worker receives only:

```text
local support input bits  8
query bits                8
```

It predicts two binary gates:

```text
bias broadcast
basis contribution
```

The router never sees the support output, operator counter, operator key, global
worker index, target answer, or another worker's state.

Architecture:

```text
16 -> Linear(16,64) -> ReLU -> Linear(64,2)
1,218 learned parameters
```

## Supervision boundary

Local routing labels are supplied directly:

```text
bias = support_input == 0
contribution = support_input is one-hot
               and the corresponding query bit is active
```

Therefore a pass establishes learned shared routing, not automatic discovery.
Message payloads and XOR aggregation remain fixed.

## Evaluation

Three independent initializations are trained on the exhaustive local routing
domain. The frozen routers are then evaluated on 64 fresh affine operators and
population sizes 9, 16, 64, and 256 with:

- full support;
- worker permutation;
- support-output shuffling;
- deterministic distractor workers.

A pass requires exact routing, exact full and permuted execution, and shuffled
support below two percent exact accuracy for every seed and population size.

## Closed boundaries

No checkpoint is published. No later Gate-9D stage, Gate-9 v0 science,
population confirmation, end-to-end answer-loss routing, or frozen-result
mutation is admitted.
