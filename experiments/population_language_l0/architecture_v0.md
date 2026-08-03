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

The approximately 19M parameter budget is divided into:

- a lexical encoder applied once per token;
- a 271,680-parameter worker core reused across workers and rounds;
- a lexical decoder applied once per predicted token.

This prevents runtime workers from multiplying the complete parameter budget.

### State

For runtime worker count `W`, the organism carries:

```text
worker_state: [batch, W, 128]
```

The state begins at zero at `<bos>` and persists token by token until `<eos>`. Its size changes with runtime workers, but learned parameters do not.

### Deterministic worker coordinates

Every worker receives a fixed sinusoidal/hash coordinate vector generated from its normalized rank and the current runtime worker count. Coordinates are not learned and contain no lookup table indexed by worker identity.

Their purpose is symmetry breaking: identical workers may specialize dynamically without acquiring separate weights.

### Token step

At each causal token step:

1. Look up 512-wide token and absolute-position embeddings.
2. Run the shared lexical encoder `512 -> 14,544 -> 128` once for the token.
3. Add deterministic worker coordinates and apply one shared `128 -> 128` initializer to each worker.
4. Execute six recurrent communication rounds using only the 271,680-parameter worker core.
5. Mean-pool the worker states.
6. Run the shared lexical decoder `128 -> 14,544 -> 512` once.
7. Apply final normalization and the tied vocabulary projection to predict the next token.

No future token is present when the state update is executed, so causality does not depend on a mask over a full sequence.

### Communication round

Every round reuses the same small modules:

```text
state
  -> shared low-rank query/key router (router width 32)
  -> top-4 incoming messages per worker
  -> shared 128 -> 128 message encoder
  -> shared GRU-like recurrent update
  -> shared 128 -> 512 -> 128 GELU update
  -> residual state
```

Routing ties are resolved deterministically by sender index. The top-k value, six-round count, temperature, and normalization rule remain fixed at every evaluated worker count.

All-pairs scores are acceptable for L0 because the largest population is 256. Later gates must replace quadratic routing if organization cost dominates.

### Readout

Worker states are mean-pooled before the lexical decoder. There is no learned attention pooler whose capacity could scale with worker count.

## Exact population parameter accounting

```text
token + position embeddings                       (64 + 32) * 512
lexical encoder                         512 * 14,544 + 14,544
                                      + 14,544 * 128 + 128
shared token initializer                           128 * 128 + 128
shared GRU-like update                        6 * 128 * 128 + 3 * 128
shared message encoder                             128 * 128 + 128
shared low-rank router                          2 * 128 * 32 + 2 * 32
shared worker FF update                    2 * 128 * 512 + 512 + 128
lexical decoder                         128 * 14,544 + 14,544
                                      + 14,544 * 512 + 512
final layer normalization                          2 * 512
tied language-model bias                                  64
-------------------------------------------------------------
total learned parameters                            18,967,968
shared repeated worker core                            271,680
```

The worker count does not occur in either parameter equation. The worker core is approximately 1.43% of total learned parameters.

## Fixed-weight scaling mechanism

Increasing workers changes only recurrent state and communication opportunities:

```text
16 workers   -> 2,048 recurrent state values
32 workers   -> 4,096
64 workers   -> 8,192
128 workers  -> 16,384
256 workers  -> 32,768
```

A positive result requires that the same learned local rule uses this additional runtime substrate to improve held-out language behavior. Merely increasing identical repeated computation without improved capability is not evidence for the hypothesis.

## Compute fairness

The lexical encoder and decoder run once per token regardless of worker count. Only the small worker core scales with runtime population.

The implementation must report:

- lexical FLOPs per token;
- worker-core FLOPs per worker and round;
- router score/reduction FLOPs;
- total active FLOPs at each worker count;
- memory traffic and peak recurrent-state bytes.

A configuration fails the engineering comparison when organization overhead grows faster than capability benefit.

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
3. Implement the lexical encoder/decoder and small shared population cell.
4. Verify worker-count-independent parameters and causal state behavior.
5. Run CPU overfit and causal-leakage tests.
6. Run one-seed GPU development training.
7. Freeze the three-seed reference protocol before final execution.
