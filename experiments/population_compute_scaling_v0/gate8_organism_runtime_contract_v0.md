# Gate-8 deterministic organism runtime v0

## Status

**CONTRACT-ONLY SYNCHRONOUS ORGANISM RUNTIME ADMITTED — TRAINING, CHECKPOINTING, SCIENTIFIC TEST WORLDS AND THE 1B REFERENCE REMAIN CLOSED.**

Exact qualified architecture head:

`2afdcc9f13f138e97c7b3821cc2a5a77bd87cf0c`

The runtime accepts only a qualified public world whose split is exactly `contract`, and it rejects any other split before invoking world validation.

## Runtime ownership

The qualified world contract owns:

- opaque node labels;
- ordered worker slots;
- each worker's source node, target node and local transform ID;
- the public root node, query target and root symbol;
- the population and target-path depth.

The qualified 19,649-parameter architecture owns:

- worker hidden-state updates;
- message-code logits;
- activity logits;
- answer-symbol logits.

The runtime owns only deterministic scheduling, delivery, terminal readout and resource accounting. It does not apply primitive transforms, call the symbolic oracle, inspect path truth, or derive an answer algorithmically.

## Mailbox topology

Every non-root node in a valid Gate-8 world has exactly one incoming edge. Consequently:

- each node mailbox contains at most one 8-bit code;
- no mailbox aggregation or collision rule is needed;
- multiple outgoing sibling workers may read the same source-node mailbox;
- every delivery becomes visible only in the next synchronous round.

The runtime independently revalidates that:

- worker slots are ordered and contiguous;
- the root has no incoming worker;
- the query target has exactly one incoming worker;
- all `population` edges are reachable from the root;
- the graph contains `population + 1` nodes.

## Frozen inference semantics

```text
admitted split            = contract only
model mode                 = eval
execution context          = torch.no_grad
learned parameters         = exactly 19,649
root seed mailbox code     = 0
root seed communication    = 0 bits
round cap                  = public world depth
message choice             = argmax over 256 codes
activity gate              = activity_logit >= 0
answer choice              = argmax over 16 symbols
terminal reader            = unique worker entering query target
```

The root symbol is supplied separately as public query data to every scheduled worker. The root mailbox code merely starts the synchronous propagation and has no communication charge.

## One round

For round `r`:

1. A worker is scheduled exactly when its source node had a mailbox at the start of the round.
2. All scheduled workers are evaluated as one shared-core batch.
3. Their hidden states are updated at their fixed worker slots.
4. An activity-positive worker selects one message code by deterministic argmax.
5. Selected messages are placed into fresh target-node mailboxes.
6. Those mailboxes become visible only at round `r + 1`.
7. If the unique worker entering the query target was scheduled, its answer head supplies the terminal prediction and execution stops after accounting for that round.

Each scheduled worker performs one recurrent update. Each delivered message costs exactly 8 communicated bits. A worker can deliver at most one message in a round.

## Sparse activation

Sparsity has two independent sources:

- only workers whose source node has received a mailbox are evaluated;
- an evaluated worker may suppress outbound propagation through a negative activity logit.

The runtime reports separately:

- recurrent updates;
- activity-positive workers;
- delivered messages;
- communicated bits;
- per-round scheduled and emitting worker indices.

No hidden compute or message is inferred from wall time.

## Frozen causal ablations

### No communication

The neural core still evaluates root-mailbox workers, but every worker-to-node delivery is suppressed. Therefore:

```text
delivered_messages = 0
communicated_bits   = 0
```

The root seed is public query initialization and is not counted as inter-worker communication.

### Shuffled worker

A deterministic SHA-256-derived bijection permutes transform shards across the fixed topology slots:

```text
routing source/target nodes = unchanged
worker-slot order           = unchanged
transform multiset          = unchanged
transform-to-slot pairing   = permuted
```

The permutation depends only on public `world_id` and worker index. It uses neither truth, model outputs nor test performance. This destroys the correct evidence-to-route correspondence without changing topology, population, communication opportunities or transform marginals.

## Terminal and failure behavior

A target is reached only if its unique incoming worker is actually scheduled. If propagation dies before then:

```text
target_reached  = false
predicted_symbol= null
answer_logits   = null
```

Later evaluation must count an unreached target as incorrect; this runtime stage does not compute accuracy.

## Closed boundaries

This stage contains no:

- train, validation, test or demonstration world generation;
- optimizer, loss or backward pass;
- differentiable discrete-message surrogate;
- checkpoint serialization or selection;
- symbolic transform application or oracle call;
- 1B tokenizer, weights or inference;
- scientific result or classifier.

The next stage may admit the differentiable training runtime and frozen development protocol. Scientific-test execution remains closed until trained checkpoints, exact 1B weights and joint execution/audit contracts are independently qualified.
