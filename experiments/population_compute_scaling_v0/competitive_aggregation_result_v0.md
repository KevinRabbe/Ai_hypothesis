# Competitive relay aggregation diagnostic v0

Development-only result from temporary PR #66.

## Question

Does replacing independent sigmoid-gated message summation with a parameter-free softmax competition over active worker gate logits rescue fixed-width relay-2 scaling?

## Result

The structural suite passed 13/13 tests. Learned parameter count remained 26,669.

| Width | Sparse exact | Sparse bit accuracy | No-comm exact | Mean last-100 loss |
| ---: | ---: | ---: | ---: | ---: |
| 4 | 100% | 100% | 0% | 0.00605 |
| 16 | 0% | 58.79% | 0% | 0.6642 |
| 64 | 0% | 53.66% | 0% | 0.6916 |
| 256 | 0% | 49.38% | 0% | 0.6995 |

## Interpretation

Simple population-size normalization is not sufficient. Competitive softmax preserves and slightly strengthens the already-learnable width-4 regime, but width 16 still fails completely and larger widths remain near chance.

Therefore the primary failure is not merely growth in total message magnitude from unnormalized distractor emission.

## Follow-up architecture audit

The audit exposed a separate flaw in the intended compositional message contract. Relay currently constructs candidate message content as:

`tanh(query_projection(value_bits))`

and the shared cell then computes:

`tanh(weighted_message_sum)`.

With perfect one-hot selection this yields:

`tanh(tanh(query_projection(value_bits)))`

while a fresh query is represented as:

`tanh(query_projection(value_bits))`.

Thus the implementation does **not** make an emitted node exactly equal to the same node as the next query representation, despite the intended #64 design.

The next repair should remove the inner candidate-message `tanh`, leaving the shared field's existing final `tanh` as the single bounding transform. Then perfect selection gives exact query/value representation identity by construction.

No Gate-v0 conclusion is claimed.
