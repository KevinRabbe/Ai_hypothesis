# Gate-9 contextual failure decomposition protocol v0

## Status

**DEVELOPMENT-ONLY PRE-EXECUTION DIAGNOSTIC PROTOCOL. GATE-9 V0 REMAINS
IMMUTABLY CLASSIFIED `G9_NOVEL_OPERATOR_INDUCTION_FAILED`. THIS SLICE ADMITS
NO MODEL EXECUTION, CHECKPOINT LOAD, TRAINING, SCIENTIFIC WORLD GENERATION, OR
POPULATION EVALUATION.**

Qualified final Gate-9 v0 result head:

```text
33f2860795a1b70e5fbe20998f4fe8a2a6fc8452
```

The purpose is not to rescue Gate-9 v0. The purpose is to identify the earliest
missing prerequisite before designing a new architecture or Gate-9 v1 protocol.

## Why decomposition is required

Across the three frozen Gate-9 seeds, full-context accuracy was exactly equal
to query-only accuracy in aggregate:

```text
full-context correct  380 / 98,304
query-only correct    380 / 98,304
```

That proves the v0 worker did not demonstrate useful support-example induction,
but it does not distinguish among four materially different failure modes:

1. inability to fit even one byte mapping;
2. inability to let support context causally control answers;
3. inability to condition on several held-in operators;
4. inability to generalize to unseen operators after the first three abilities
   are established.

The diagnostic ladder tests those prerequisites in that order and stops at the
first nonpassing stage.

## Exact frozen machinery

The diagnostic retains the exact v0 worker architecture and learned budget:

```text
architecture head     c689cc3f38f6f642916ee1a702d7de7bd0e43b
learned parameters    19,649
support examples      9
non-support queries   247
```

Every stage starts from scratch under three fixed diagnostic initializations:

```text
910900
910901
910902
```

Optimizer mechanics remain aligned with Gate-9 v0:

```text
AdamW
maximum LR             1e-3
minimum LR             1e-4
warmup                  16 steps
cosine decay            fixed final step
betas                   0.9, 0.95
epsilon                 1e-8
weight decay            1e-4
gradient clipping       1.0
float32                 yes
AMP / TF32 / compile    off / off / off
```

There is no early stopping, checkpoint selection, seed replacement, or retry.

## Diagnostic operator namespace

All counter-based diagnostic operators begin at `2^56`. This is disjoint from
the frozen Gate-9 v0 training, validation, local-science, and graph-science
ranges.

The paired-context collision stage uses two exact affine keys:

```text
0000000000000000
ff00000000000000
```

They have the same identity linear map and opposite byte bias. Their exact
inverse SplitMix64 counters are:

```text
61c8864680b583eb
e8cf068191d03bbc
```

For every non-support query byte, the two required answers differ by `0xff`.
The same query therefore cannot be answered correctly for both operators unless
the support context causally changes the output.

## Ordered stages

### 1. Single-operator query fit

```text
operators              1
queries/operator       247
training examples      247
steps                  1,024
batch size             247
```

This stage asks only whether the architecture and optimizer can fit one complete
non-support byte mapping. Support is constant, so passing does not establish
context use.

Failure outcome:

```text
G9D_BASIC_QUERY_MAPPING_FAILED
```

### 2. Paired-operator context collision

```text
operators              2
queries/operator       247
training examples      494
steps                  2,048
batch size             494
```

Every query appears under both operators with contradictory target bytes. Full
support must outperform both swapped support and query-only controls by more
than `0.50` exact accuracy.

Failure outcome:

```text
G9D_CONTEXTUAL_CONTROL_FAILED
```

### 3. Held-in multi-operator fit

```text
operators              16
queries/operator       247
training examples      3,952
steps                  4,096
batch size             512
```

Training and evaluation use the same sixteen diagnostic operators. This stage
tests contextual capacity and optimization across multiple mappings before any
claim about unseen-operator induction.

Failure outcome:

```text
G9D_HELD_IN_OPERATOR_FIT_FAILED
```

### 4. Unseen-operator generalization

```text
training operators     256
evaluation operators   64
operator overlap       0
queries/operator       247
training examples      63,232
evaluation examples    15,808
steps                  8,192
batch size             512
```

This stage may execute only after all three initialization seeds pass every
prior stage. It tests operator induction on a disjoint development-only split;
it does not run Gate-9 v0 local or graph scientific worlds.

Failure outcome:

```text
G9D_UNSEEN_OPERATOR_GENERALIZATION_FAILED
```

If every stage passes all three seeds:

```text
G9D_V0_FAILURE_NOT_LOCALIZED
```

That outcome would show the small ladder did not reproduce the original failure;
it would not retroactively alter Gate-9 v0.

## Thresholds and seed rule

Every seed must satisfy:

```text
exact byte accuracy    >= 0.995
bit accuracy           >= 0.999
oracle accuracy        == 1.0
```

Contextual stages additionally require:

```text
full - shuffled        > 0.50
full - query-only      > 0.50
```

A stage advances only when all three diagnostic initialization seeds pass.

```text
3 pass                 advance
3 fail                 return stage-specific failure
mixed pass/fail        G9D_DIAGNOSTIC_INCONCLUSIVE
missing result         G9D_DIAGNOSTIC_INCOMPLETE
```

The ladder stops after the first nonpassing stage. Later stages remain unrun,
preventing post-hoc search across multiple explanations.

## Interpretation boundaries

The stage outcomes localize the earliest demonstrated deficiency:

- stage 1 failure: basic mapping capacity or optimization;
- stage 2 failure: causal support pathway or query-context fusion;
- stage 3 failure: multi-operator contextual capacity or optimization;
- stage 4 failure: unseen-operator generalization;
- all stages pass: the compact diagnostic did not reproduce the v0 failure.

No outcome alone identifies the exact parameter tensor, optimizer mechanism, or
architectural repair. Any repair requires a separately frozen execution or v1
protocol.

## Closed boundaries

This protocol imports no Torch and performs no execution. It contains no
checkpoint, trainer, optimizer step, operator materialization, scientific
assignment key, test-world generation, population runtime, or result mutation.

Gate-9 v0 local and graph scientific tests remain permanently closed. The
positive Gate-8 result remains unchanged.
