# Gate-9D answer-loss router feasibility v8

## Status

Development-only. The deterministic zero-input bias broadcast remains fixed.
Only contribution routing is learned from final answer error.

## Question

Can the four-parameter local-summary contribution gate be learned without any
worker-level routing labels?

## Variants

- `answer_only`: differentiable XOR-sign answer loss only.
- `answer_plus_global_message_budget`: answer loss plus a global penalty forcing
  the number of messages toward `popcount(query)`.

The budget term contains no worker identity or local routing target.

## Train/evaluation split

Training uses 64 operators from a fresh counter range and nominal population
16. Evaluation uses 64 disjoint operators at populations 9, 16, 64, and 256.

## Boundaries

This does not learn bias routing, discover affine coordinates, use operator
identity, open frozen Gate-9 science, or claim population confirmation.
