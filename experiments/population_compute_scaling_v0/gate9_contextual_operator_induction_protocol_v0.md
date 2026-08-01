# Gate-9 contextual affine-operator induction v0 — protocol

## Status

**DATA-FROZEN PRE-EXPOSURE GATE-9 PROTOCOL — OPERATOR GENERATION,
WORLD GENERATION, ARCHITECTURE, TRAINING, CHECKPOINT LOADING, SCIENTIFIC
TEST EXECUTION AND RESULT CLASSIFICATION REMAIN CLOSED.**

Exact immutable Gate-8 source result:

```text
result head       a063be4bf04979b026370f87cdd0aa05712cdd05
execution head    474b2590e5e138134bcb993e1d8114c473f0455b
summary SHA-256   a63f1c6c7cb7facdc71a48e5df05297cc823017ea342dc052310d36c97394462
condition ledger  276969b304beee1edbeb3979c44a12db4b256b436e6d82c86ff92da7ce64f44d
manifest SHA-256  bc7ff2ca604c914a2bb610d0089450454c47fbb36eb76954db28ea898c3ced59
```

Gate-8 closed with:

```text
G8_POSITIVE_CAPABILITY_SCALING
G8_POPULATION_EXCEEDS_1B_REFERENCE
```

Its three distinct checkpoints nevertheless compiled to the identical exhaustive
2,048-entry local transition function. Gate-9 therefore does not ask whether
that already-learned finite algebra composes again. It asks whether the same
fixed learned parameter budget can infer a new local operator from context at
every worker and still obtain population capability scaling.

## Frozen scientific question

> Does fixed learned machinery retain population capability scaling when every
> worker must infer a previously unseen local operator from public support
> examples, with no operator identity, per-operator parameter, population
> feature or test-time weight update?

A positive result requires all three properties simultaneously:

1. accurate isolated induction of unseen local operators;
2. causal dependence on each worker's support context;
3. causal population communication with a rising solved-depth frontier.

Failure of any one property receives a distinct frozen outcome.

## Operator family

Symbols and messages are exact eight-bit values:

```text
symbol domain      GF(2)^8
symbol count       256
message width      8 bits
support examples   9
learned parameters 19,649
checkpoint seeds   0, 1, 2
```

Each local operator is an affine bijection:

```text
f(x) = A x XOR b
A    = L U
```

where:

- `L` is an 8x8 unit lower-triangular binary matrix;
- `U` is an 8x8 unit upper-triangular binary matrix;
- `b` is an eight-bit bias.

The 28 strict-lower bits, 28 strict-upper bits and eight bias bits form one
64-bit key. Every key produces one unique affine bijection. Uniqueness follows
because unit-lower/unit-upper factorization is unique: if `L1 U1 = L2 U2`,
then `L2^-1 L1 = U2 U1^-1`, and the only matrix both unit lower- and
unit upper-triangular is the identity.

Frozen family size:

```text
2^64 = 18,446,744,073,709,551,616 operators
```

The 64-bit key exists only inside the future generator and audit. It is never a
model input.

## Public support context

Every worker receives exactly nine public input-output examples for its local
operator:

```text
support inputs = 0, 1, 2, 4, 8, 16, 32, 64, 128
```

These are zero plus the eight canonical basis vectors. Their order is
deterministically permuted through an answer-independent namespace. The worker
also receives one eight-bit query value, equal to its incoming message.

The query is never one of the nine support inputs. The support pairs uniquely
determine the affine map:

- `f(0)` gives the bias;
- `f(e_i) XOR f(0)` gives column `i` of `A`.

The exact symbolic oracle must reconstruct the operator only from the same nine
public pairs and then apply it to the query. A world is invalid if that
reconstruction is not unique or disagrees with the generator's private
operator.

This deliberately tests contextual operator execution rather than retrieval of
a previously trained operator identity.

## Injective operator split

Sequential 64-bit counters are mapped through the frozen
`splitmix64_bijection_v0` permutation before their bits are interpreted as
`L`, `U` and `b`. The counter permutation must be proven bijective.

Frozen disjoint counter ranges:

```text
training        start 0        count   262,144
validation      start 2^32     count    32,768
local science   start 2^40     count     4,096
graph science   start 2^48     count 2,629,632
```

All pairwise split intersections must be exactly zero. The scientific artifact
must preserve exact operator-identity hashes and an independent proof of:

- injective counter-to-operator mapping;
- exact split counts;
- no train/validation/scientific operator overlap;
- no local-test/graph-test overlap;
- no operator key, counter, split, world ID or worker ID in model input.

A collision or exposed key invalidates the artifact before accuracy is
interpreted.

## Fixed training boundary

The future architecture must contain exactly 19,649 learned parameters with no
padding tensor or per-operator storage. One parameter set is reused for every
operator, worker, population and recurrent round.

The top-level data allocation is frozen to:

```text
training operator episodes   262,144 per checkpoint seed
validation operator episodes  32,768 per checkpoint seed
checkpoint seeds              0, 1, 2
```

The exact architecture, optimizer, batching, checkpoint schedule and admission
criteria remain closed. They require separate pre-exposure contracts.

Training and checkpoint selection may use only the frozen training and
validation operator ranges. No graph-science world, local-science operator or
scientific outcome may be generated before all three checkpoints are
independently admitted.

## Isolated novel-operator test

All three frozen checkpoints evaluate the same 4,096 unseen local-test
operators. For each operator the model receives:

- the nine shuffled support pairs;
- one query outside the support set;
- no graph, population, depth, round, operator key or persistent cache.

Primary local induction for every checkpoint requires:

```text
accuracy                  >= 0.995
95% CI low                >= 0.990
accuracy - shuffled       >= 0.50
accuracy - query-only     >= 0.50
paired CI low, both       >  0.50
```

The local controls are:

### `shuffled_context`

Assign the complete valid support set of another local-test operator to the
current query. Support-set validity and marginal byte distributions remain
unchanged, but operator identity is wrong.

### `query_only`

Remove every support pair while preserving the query and execution budget. No
operator key or fallback descriptor is introduced.

If any checkpoint fails the isolated induction rule, the primary outcome is:

```text
G9_NOVEL_OPERATOR_INDUCTION_FAILED
```

No graph scaling result can rescue failed local induction.

## Distributed graph benchmark

The graph family preserves the Gate-8 structural scaling question while
replacing every finite primitive transform with a unique contextual operator:

- one rooted directed tree;
- `population` edges and `population + 1` nodes;
- one unique root-to-target path of exactly `depth` edges;
- `population - depth` distractor edges;
- one worker per edge;
- opaque node labels and shuffled worker assignment;
- one unique affine operator and support set per edge;
- one uniform eight-bit root symbol;
- target answer equal to exact path-wise operator composition.

The shared worker receives only:

```text
nine local support pairs
incoming eight-bit message
```

It emits exactly one eight-bit message. It receives no operator key, topology
identity, population, depth, round, node ID, worker ID, relevance label or
target flag.

Population/depth matrix:

```text
population = 32, 64, 128, 256, 512, 1024
depth      = 4, 8, 16, 32, 64, 128
valid iff  = 8 * depth <= population
```

This yields exactly 21 conditions. Every condition uses:

```text
split                 test
seed                  0
world indices         0..255
worlds per condition  256
checkpoint seeds      0, 1, 2
```

The complete scientific graph matrix contains 5,376 unique worlds and
2,629,632 unique local operators.

## Runtime and communication semantics

The future runtime must be separately frozen, but the protocol-level invariants
are already fixed:

- root symbol starts as one eight-bit mailbox message;
- a worker executes only when its source node has a mailbox;
- every active full-mode worker reads all nine local examples and the mailbox;
- every active worker emits at most one eight-bit result;
- delivery is synchronous to the next recurrent round;
- maximum recurrent rounds equal public query depth;
- terminal answer is the message emitted by the unique target-entering worker;
- learned parameters remain exactly 19,649 independent of active workers;
- communicated bits equal delivered messages times eight;
- support-context bytes are reported separately from inter-worker
  communication and are not counted as messages.

## Mandatory graph controls

Every checkpoint and condition reports:

### `full`

Exact contextual worker plus synchronous message delivery.

### `no_communication`

Workers may evaluate local context and emit locally, but every inter-worker
delivery is suppressed.

### `shuffled_context`

Deterministically permute complete support sets across worker slots inside the
same world. Topology, incoming messages, operator marginals, support validity,
worker count and communication budget remain fixed.

### `query_only`

Remove support pairs from every worker while preserving query messages,
topology, rounds and parameter count.

### `random_answer`

Answer-blind deterministic random eight-bit symbol.

### `oracle`

Exact public-support reconstruction and path composition. Accuracy must be
exactly 1.0 or the artifact is invalid.

## Statistics

One condition's primary accuracy is:

1. accuracy over 256 worlds for checkpoint seed 0;
2. accuracy over the same worlds for checkpoint seed 1;
3. accuracy over the same worlds for checkpoint seed 2;
4. arithmetic mean of those three seed accuracies.

The 95% interval uses 20,000 deterministic paired bootstrap samples. The
resampling unit is world index. The same sampled indices are applied across all
three checkpoints before equal-seed averaging.

Frozen namespace:

```text
gate9-contextual-operator-induction-bootstrap-v0
```

Raw per-seed outcomes remain mandatory. Training seeds may not be treated as
768 independent worlds.

## Population frontier

For each population, choose the deepest valid condition satisfying both:

```text
accuracy   >= 0.90
95% CI low >= 0.85
```

If no condition is solved, carry the shallowest valid condition with
`solved=false`.

The positive frontier requires:

- all six population frontiers solved;
- non-decreasing solved depth;
- at least three strict adjacent depth increases;
- final solved depth at least four times the first;
- both communication causal guards pass.

## Causal guards

Frozen high-scale causal conditions:

```text
(512, 64)
(1024, 128)
```

At both conditions, calculate paired world-level differences after equal-seed
averaging:

```text
full - no_communication
full - shuffled_context
full - query_only
```

Every lower 95% confidence bound must be strictly greater than `0.20`.

The support-context guards are interpreted before population scaling. If either
support guard fails at either condition, the outcome is:

```text
G9_CONTEXT_NOT_CAUSAL
```

A communication guard failure does not prove absence of contextual induction;
it blocks a positive population-scaling label and leaves the result
inconclusive unless another frozen branch applies.

## Frozen outcomes

```text
G9_CONTEXTUAL_POPULATION_SCALING
G9_CONTEXTUAL_CAPABILITY_PRESENT_NO_SCALING
G9_CONTEXTUAL_CAPABILITY_NEGATIVE_SCALING
G9_CONTEXT_NOT_CAUSAL
G9_NOVEL_OPERATOR_INDUCTION_FAILED
G9_CONTEXTUAL_CAPABILITY_INCONCLUSIVE
```

Classifier order:

1. validate exact operator splits and evidence matrices;
2. require isolated novel-operator induction on all three checkpoints;
3. require causal support-context use at both high-scale conditions;
4. apply frontier and communication rules;
5. distinguish positive, flat, negative and inconclusive scaling.

No result-dependent threshold change, operator-range extension, extra training,
alternate support basis, additional query, rescue checkpoint or second
scientific namespace is admitted.

## Required evidence

The future artifact must preserve:

- every operator split identity and intersection audit;
- all local-test predictions and both local controls;
- every graph world/operator/worker identity;
- every per-seed prediction for every mode;
- oracle answers reconstructed from public support only;
- active workers, support examples read, worker updates, messages, bits and
  wall time;
- per-condition correctness vectors and hashes;
- all paired bootstrap intervals;
- six frontier rows;
- both causal rows;
- exact model/checkpoint/parameter identities;
- explicit no-test-training and no-test-weight-update proofs;
- independent result classification.

The independent auditor may not import the scientific executor, model runtime or
statistics implementation.

## External design context

Gate-9 deliberately separates retrieval of trained tasks from inference of a
new task. Synthetic in-context-learning research reports that task diversity
can produce transitions between specialized retrieval-like solutions and
solutions that generalize to broader task spaces, while other studies show that
out-of-distribution task induction can fail even when ordinary in-context
performance appears strong. Neural algorithmic-reasoning benchmarks likewise
treat larger-size and distribution-shift generalization as separate scientific
problems. These papers motivate the split and controls; they do not determine
the frozen Gate-9 classifier.

Non-binding references:

- Goddard et al., *When can in-context learning generalize out of task
  distribution?*, ICML 2025.
- Wang et al., *Can In-context Learning Really Generalize to
  Out-of-distribution Tasks?*, ICLR 2025.
- Veličković et al., *The CLRS Algorithmic Reasoning Benchmark*, 2022.
- Mahdavi et al., *Towards Better Out-of-Distribution Generalization of Neural
  Algorithmic Reasoning Tasks*, 2022.

## Closed boundary

This protocol branch contains only:

- standard-library immutable constants and evidence dataclasses;
- preregistered classifiers;
- synthetic classifier and structural regressions;
- this scientific protocol;
- branch-scoped protocol-only CI.

It contains no NumPy, Torch, operator generator, world generator, affine solver,
model, optimizer, checkpoint, artifact reader, runner, wrapper, scientific
world, test answer or result.

## Next admission boundary

The next slice must implement and exhaustively qualify only the deterministic
64-bit-counter-to-affine-operator mapping, public nine-example support contract,
exact public-support oracle, split-disjointness proof and contract-only
generator tests.

Architecture, training and scientific execution remain closed after that slice.
