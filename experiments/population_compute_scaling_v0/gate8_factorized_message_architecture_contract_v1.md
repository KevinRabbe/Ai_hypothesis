# Gate-8 factorized-message architecture contract v1

## Status

**PRE-EXPOSURE ARCHITECTURE CONTRACT. RUNTIME, TRAINING, CHECKPOINTS, SCIENTIFIC-TEST WORLDS, SEEDS 1/2, AND THE 1B REFERENCE REMAIN CLOSED.**

Base: exact qualified seed-0 causal-diagnostic result head:

`ad54c8daa7617d54e15932da76da08212d0d1444`

Fixed learned-parameter budget:

`19,649`

## Evidence-driven changes

The seed-0 causal diagnostic established:

```text
activity_gate_material           = false
answer_head_material             = true
continued_optimization_effective = false
core_interference_persists       = true
frozen_core_linearly_sufficient  = false
```

The v1 architecture therefore makes only changes tied to those findings.

### Root symbol becomes the initial message

The v0 worker received a separate root-symbol embedding on every edge. Non-root predictions remained dependent on this irrelevant feature.

V1 encodes the root symbol directly into the low four bits of the initial message:

```text
initial carrier = 0
initial symbol  = public root symbol
initial code    = (carrier << 4) | symbol
```

No root-symbol feature enters the neural worker.

### Message prediction is factorized

The v0 core predicted one of 256 joint codes through one monolithic head. The carrier component was nearly solved while symbol transformation remained materially weaker.

V1 predicts two explicit categorical values:

```text
carrier head = 16 classes
symbol head  = 16 classes
message code = (argmax(carrier) << 4) | argmax(symbol)
```

The complete communication channel remains exactly eight bits and 256 possible codes.

### No duplicate answer decoder

The causal terminal-decode intervention raised mean target accuracy from `0.4036458333333333` to `0.5755208333333334` by reading the message symbol instead of the independent answer head.

V1 has no answer head. The terminal answer is exactly:

```text
argmax(symbol logits)
```

The communicated symbol and reported answer cannot disagree.

### Deterministic delivery for the capability gate

Forcing activity raised mean target accuracy only to `0.4215494791666667` and did not satisfy the frozen materiality criterion. Rare false-negative activity decisions can still destroy long chains despite adding no demonstrated capability benefit.

V1 has no activity head. A future v1 runtime must deliver one message for every topologically scheduled worker. Sparse activation remains a separate later efficiency experiment and cannot be mixed into this capability repair.

### Remove all irrelevant local-role inputs

Every edge performs the same operation:

```text
next symbol = transform(current symbol)
next carrier = current carrier + 1 mod 16
```

The worker does not require root identity, query-target identity, population size, depth, round flags, worker identity, or node identity. V1 therefore admits only:

```text
carrier nibble
symbol nibble
transform ID
```

Topology and target selection remain external deterministic runtime responsibilities.

## Exact architecture

```text
carrier embedding width = 7
symbol embedding width  = 11
transform embedding     = 3
worker input width       = 21
shared GRU hidden width  = 65
carrier output classes   = 16
symbol output classes    = 16
```

The symbol embedding receives more width than the carrier embedding because the symbol undergoes one of eight noncommuting transformations, while the carrier follows one fixed cyclic update.

The transform embedding requires only eight distinguishable learned vectors; three continuous dimensions are sufficient to represent eight unique categories while the 65-dimensional shared core performs the nonlinear transition.

## Exact no-padding parameter ledger

| Component | Parameters |
|---|---:|
| carrier embedding `16 × 7` | 112 |
| symbol embedding `16 × 11` | 176 |
| transform embedding `8 × 3` | 24 |
| one shared learned initial hidden state | 65 |
| GRU input weights `3 × 65 × 21` | 4,095 |
| GRU recurrent weights `3 × 65 × 65` | 12,675 |
| GRU biases `2 × 3 × 65` | 390 |
| carrier head `65 × 16 + 16` | 1,056 |
| symbol head `65 × 16 + 16` | 1,056 |
| **Total** | **19,649** |

There is no padding parameter, unused reserve, per-worker parameter, per-population parameter, per-depth parameter, node embedding, or world-specific learned state.

## Contract invariants

1. All 256 messages round-trip exactly through high/low-nibble factorization.
2. The 16 root symbols produce initial codes `0..15` with carrier zero.
3. One shared core is reused across workers, populations, and rounds.
4. The forward signature contains only inbox code, transform ID, and hidden state.
5. Terminal answer equals the symbol component of the emitted message.
6. The architecture has exactly 12 parameter tensors and 19,649 scalar parameters.
7. No v0 root, role, runtime-flag, monolithic-message, activity, or answer interface exists.
8. Runtime, training, checkpoints, scientific-test worlds, and reference inference remain unadmitted.

## What this stage does not claim

This architecture contract does not show that v1 learns the transition table or composes transformations. It only admits a repaired fixed-budget substrate whose interfaces directly reflect the completed causal diagnosis.

The next stage, after qualification, is a separate deterministic v1 runtime contract. It must bind root-message seeding, synchronous message propagation, factorized code recomposition, terminal symbol readout, and exact communication accounting before any v1 training protocol is opened.
