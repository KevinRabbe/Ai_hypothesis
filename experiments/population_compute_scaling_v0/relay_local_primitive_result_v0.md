# Relay local primitive diagnostic v0

Development-only result from temporary PR #63 / workflow run `30217870711`.

## Result

The current shared-weight population can learn the local and one-hop primitives, but fails when a retrieved node must become the query for the next relay hop.

Measured on CPU Torch:

- key/query equality: 99.83% held-out accuracy; 100% match recall; 99.66% non-match specificity;
- one-hop sparse shared lookup:
  - 4 active workers: 99.44% exact solve;
  - 16 active workers: 95.85% exact solve;
- one-hop no-communication control: ~0.05% exact at both widths;
- relay-2 sparse shared execution: effectively 0% exact even on information-complete worlds.

The relay-2 training loss briefly reached ~0.45 but ended near ~0.70, while the one-hop loss converged near ~0.05.

## Localization

This rules out two broad failure explanations for the current configuration:

1. the local worker is unable to learn random key/query equality;
2. the population shared field is unable to perform one matching key/value retrieval.

The first failed primitive is recurrent composition:

`query -> matching worker value -> next shared query -> next matching worker value`.

## Architecture implication

The current relay path seeds `shared` with `query_projection(start_bits)`, but later shared fields come from the generic cell's unrelated `message_projection(state)`. The training objective must therefore discover an arbitrary latent protocol in which emitted values become valid future query representations.

The next development variant should make that protocol compositional by construction while keeping the learned parameter set fixed across runtime population sizes: use one shared node projection for both the incoming query and candidate emitted value content, with the neural worker still learning the gate/match/update behavior.

No Gate-v0 success or failure is claimed from this diagnostic.
