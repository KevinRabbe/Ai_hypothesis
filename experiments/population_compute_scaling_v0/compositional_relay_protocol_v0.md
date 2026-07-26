# Compositional relay protocol v0

## Motivation

The development-only local primitive diagnostic in closed PR #63 showed:

- key/query equality is learnable;
- one-hop population lookup is learnable;
- two-hop relay composition fails.

The previous relay path seeded the shared field with `query_projection(start_bits)` but produced later shared fields from an unrelated learned `message_projection(state)`. Multi-hop training therefore had to invent an arbitrary latent protocol in which an emitted value accidentally became a valid future query representation.

## Variant

This variant keeps one fixed learned parameter set across all runtime population sizes and changes only the relay message protocol:

- one learned node projection maps fixed node bits into the shared message space;
- the start query uses that projection;
- each worker's candidate value uses the exact same projection;
- the worker still learns its recurrent state update and scalar emission gate;
- gated candidate values are summed into the next bounded shared field;
- no oracle chain edge, lookup table, answer key, specialized worker type, or learned parameter is added as population size grows.

The purpose is to make the protocol compositional by construction:

`node X as emitted value == node X as next-query representation`.

## Scientific boundary

This is a development architecture repair after a localized failed primitive, not a positive Gate-v0 result.

The immediate test is relay-2 learnability under the same development-only diagnostic. If relay-2 remains effectively unsolved, do not proceed to expensive population curves; localize the next recurrent failure instead.
