# Population Language L0 architecture v0

## Status

Architecture contract only. This file does not report a trained model.

## Baseline: matched causal transformer

The reference baseline is a conventional decoder-only transformer:

```text
input token IDs
  -> learned token + position embeddings
  -> 6 pre-norm causal transformer blocks
  -> final layer normalization
  -> tied embedding projection + output bias
```

Each block contains eight-head causal self-attention and a two-layer GELU feed-forward network. Its exact parameter count is 18,964,544.

KV caching is deliberately outside L0. Training is teacher-forced and sequences are only 28 tokens. A KV cache would affect generation latency, not the learning question tested here.

## Population organism

The organism is a causal recurrent system with a variable number of persistent workers and one shared set of weights.

### State

For runtime worker count `W`, the organism carries:

```text
worker_state: [batch, W, 992]
```

The state begins at zero at `<bos>` and persists token by token until `<eos>`. Its size changes with runtime workers, but learned parameters do not.

### Deterministic worker coordinates

Every worker receives a fixed sinusoidal/hash coordinate vector generated from its normalized rank and the current runtime worker count. Coordinates are not learned and contain no lookup table indexed by worker identity.

Their purpose is symmetry breaking: identical workers may specialize dynamically without acquiring separate weights.

### Token step

At each causal token step:

1. Look up the token and absolute-position embeddings.
2. Add the deterministic coordinate vector for each worker.
3. Apply one shared `992 -> 992` token initializer.
4. Execute six recurrent communication rounds.
5. Mean-pool the worker states.
6. Apply final normalization and the tied vocabulary projection to predict the next token.

No future token is present when the state update is executed, so causality does not depend on a mask over a full sequence.

### Communication round

Every round reuses the same learned modules:

```text
state
  -> shared low-rank query/key router (router width 96)
  -> top-4 incoming messages per worker
  -> shared 992 -> 992 message encoder
  -> shared GRU-like recurrent update
  -> shared 992 -> 5,440 -> 992 GELU update
  -> residual state
```

Routing ties are resolved deterministically by sender index. The top-k value, six-round count, temperature, and normalization rule remain fixed at every evaluated worker count.

All-pairs scores are acceptable for L0 because the largest population is 256. Later gates must replace quadratic routing if organization cost dominates.

### Readout

Worker states are mean-pooled, then passed through a shared final layer normalization and the tied token-embedding matrix. There is no learned attention pooler whose capacity could scale with worker count.

## Exact population parameter accounting

```text
token + position embeddings                    (64 + 32) * 992
shared token initializer                        992 * 992 + 992
shared GRU-like update                     6 * 992 * 992 + 3 * 992
shared message encoder                          992 * 992 + 992
shared low-rank router                      2 * 992 * 96 + 2 * 96
shared feed-forward update             2 * 992 * 5,440 + 5,440 + 992
final layer normalization                       2 * 992
tied language-model bias                              64
----------------------------------------------------------
total learned parameters                         18,964,800
```

The worker count does not occur in this equation.

## Fixed-weight scaling mechanism

Increasing workers changes only runtime state and communication opportunities:

```text
16 workers   -> 15,872 recurrent state values
32 workers   -> 31,744
64 workers   -> 63,488
128 workers  -> 126,976
256 workers  -> 253,952
```

A positive result requires that the same learned local rule uses this additional runtime substrate to improve held-out language behavior. Merely increasing identical repeated computation without improved capability is not evidence for the hypothesis.

## Organism-state cache versus KV cache

The persistent worker state is the organism analogue of an autoregressive cache, but it is not a transformer KV cache:

- transformer KV memory grows with layers and sequence length;
- L0 organism state grows with workers and remains fixed across sequence length;
- the organism recurrently compresses its prefix into worker state;
- information loss from that compression is part of the experiment.

A later caching gate will compare decode latency, memory, and quality between conventional transformer KV state and organism persistent state. L0 records state bytes but makes no cache-efficiency claim.

## Implementation order

1. Implement and qualify the deterministic dataset loader and metric surface.
2. Implement the matched transformer and verify exact parameter accounting.
3. Implement the shared population cell and verify worker-count-independent parameters.
4. Run CPU overfit and causal-leakage tests.
5. Run one-seed GPU development training.
6. Freeze the three-seed reference protocol before final execution.
