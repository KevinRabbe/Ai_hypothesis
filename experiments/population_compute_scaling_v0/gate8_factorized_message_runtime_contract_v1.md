# Gate-8 factorized-message runtime contract v1

## Status

**PRE-EXPOSURE CONTRACT-WORLD RUNTIME. TRAINING, CHECKPOINT LOADING, SEEDS 1/2, SCIENTIFIC-TEST WORLDS, AND THE 1B REFERENCE REMAIN CLOSED.**

Base: exact qualified v1 factorized-message architecture head:

`c3ab64008c816fa1eb6f9d6f8f0a1a99ed195ec8`

Fixed learned-parameter budget:

`19,649`

## Purpose

This stage binds only deterministic execution semantics for the repaired Gate-8 v1 worker core. It does not train the core and does not claim transformation correctness.

The runtime consumes a qualified public `contract` world and an already constructed v1 core in evaluation mode. It never reads stored truth, invokes the symbolic oracle, or applies a primitive transform itself.

## Root-message seeding

The public root symbol enters the organism only through the initial 8-bit mailbox message:

```text
carrier = 0
symbol  = public root symbol
code    = (carrier << 4) | symbol
```

For the 16 possible root symbols, the initial codes are exactly `0..15`.

There is no root-symbol neural input after this message is constructed.

## Synchronous execution

The public graph is a rooted tree with one incoming edge per non-root node. Therefore each node mailbox contains at most one message and requires no aggregation rule.

For round `r`:

1. Freeze the mailboxes produced by round `r - 1`.
2. Schedule every worker whose source node has a mailbox.
3. Give each scheduled worker its source mailbox code, local transform ID, and its own hidden state.
4. Run one update of the shared 19,649-parameter core.
5. Recompose one message from the carrier-head and symbol-head argmax values.
6. Deliver every emitted message to its target node for round `r + 1`.
7. Stop after the unique worker entering the query target has executed, or after the public query-depth cap.

No message is available in the same round in which it is emitted.

## Deterministic delivery

V1 has no activity head. In the full and shuffled-worker modes:

```text
one scheduled worker
= one recurrent update
= one emitted message
= one delivered 8-bit message
```

The runtime rejects any state in which one scheduled emission is suppressed.

Sparse activation remains a separate later efficiency experiment. It cannot re-enter this repaired capability runtime without a new protocol.

## Terminal answer

When the unique edge entering the public query target executes:

```text
predicted answer = argmax(symbol logits)
target message   = (argmax(carrier logits) << 4) | predicted answer
```

The runtime verifies that the target message’s low four bits equal the reported answer. There is no independent answer decoder.

## Runtime modes

### `full`

Execute the public topology and deliver every scheduled message.

### `no_communication`

Workers reachable from the initial root mailbox execute and emit, but no message is delivered. The run therefore terminates when no next-round mailbox exists.

This preserves the distinction between local computation and inter-worker communication.

### `shuffled_worker`

Apply a deterministic SHA-256-derived permutation to transform shards while keeping topology slots fixed. The permutation:

- is independent of model outputs and stored truth;
- is reproducible from public world ID and population;
- preserves the exact transform multiset;
- is forced away from identity.

This remains a causal control for correct worker-to-transform assignment.

## Exact accounting

For each round the runtime records:

- sorted mailbox nodes before execution;
- scheduled worker indices;
- aligned inbox codes;
- aligned emitted message codes;
- delivered worker/target/code triples;
- recurrent updates;
- communicated bits.

Global invariants:

```text
emitted messages = recurrent updates
communicated bits = delivered messages × 8
```

For `full` and `shuffled_worker`:

```text
delivered messages = emitted messages
```

For `no_communication`:

```text
delivered messages = 0
communicated bits  = 0
```

## Qualification surface

Contract-only regressions must prove:

1. root symbol appears as the first-round inbox code with carrier zero;
2. every scheduled worker emits and delivers in full mode;
3. target answer equals the emitted target-message symbol;
4. no-communication emits locally but delivers zero messages;
5. shuffled-worker assignment is deterministic, non-identity, and marginal-preserving;
6. the maximum `(population=1024, depth=128)` contract condition reaches the target in exactly 128 rounds;
7. all recurrent-update, emission, delivery, and bit counts are internally exact;
8. non-contract worlds are rejected before public-world validation;
9. model training mode and any parameter count other than 19,649 are rejected;
10. runtime source contains no truth read, symbolic oracle, transform application, optimizer, or checkpoint load.

## Boundary

This stage does not admit:

- `train`, `validation`, `test`, or `demonstration` worlds;
- optimizer construction or parameter updates;
- checkpoint loading or saving;
- seed-1 or seed-2 execution;
- scientific-test evaluation;
- tokenizer loading;
- reference-model weights;
- reference inference.

## Next stage

After this runtime contract qualifies, the next stage is a separate v1 training protocol. It must preregister factorized carrier/symbol supervision, optimizer schedule, world allocation, validation conditions, checkpoint selection, and admission thresholds before any v1 training world is generated.
