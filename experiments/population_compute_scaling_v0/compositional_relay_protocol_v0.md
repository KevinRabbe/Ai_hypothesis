# Relay hop protocol v0

## Motivation

The development-only local primitive diagnostic in closed PR #63 showed:

- key/query equality is learnable: 99.83% held-out accuracy;
- one-hop population lookup is learnable: 99.44% exact at 4 workers and 95.85% at 16;
- relay-2 is effectively unsolved with persistent local worker state.

The failure therefore begins when a retrieved value must become the query for the next hop.

## Architecture repair

Two mechanisms are required by the development ablations:

1. **Compositional node messages.** One learned node projection maps fixed node bits into the shared message space. The start query and every worker's candidate value use that same projection, so `node X as emitted value == node X as next-query representation`.
2. **Hop-local worker state.** The shared field is the recurrent state across relay hops. Each worker's hidden state is reinitialized from its immutable local record before processing the next shared query, preventing a previous match from contaminating later re-selection.

The generic `SharedPopulationCell` keeps persistent local state as its default. Relay explicitly opts into hop-local reset semantics.

Workers still learn the recurrent update and scalar emission gate. No oracle chain edge, lookup table, answer key, specialized worker type, or population-dependent learned parameter is added.

## Development ablations

All numbers below are development-only CPU diagnostics; they are not Gate-v0 results.

At fixed population 4 with information-complete relay-2 worlds:

- compositional messages + **persistent** local state: 3.91% exact;
- original learned state→message projection + **reset** local state: 1.95% exact;
- compositional messages + **reset** local state: **99.22% exact**, 99.87% bit accuracy;
- the corresponding no-communication control with both repairs: 0% exact.

Therefore neither repair alone is sufficient in this experiment; together they make the two-hop primitive reliably learnable at width 4.

## Remaining failure

Training one repaired checkpoint across mixed populations 4/16/64/256 still fails:

- width 4, solve given information-complete: 2.34%;
- width 16: 0.39%;
- width 64: 0%;
- width 256: 0%.

This changes the active uncertainty. The local recurrent primitive is no longer the primary blocker. The next diagnostic must determine whether the repaired architecture can learn relay-2 **independently** at fixed widths 16, 64, and 256.

Interpretation:

- if those fixed-width models learn, the remaining problem is mixed-width curriculum / scale normalization;
- if capability collapses as fixed width grows, the remaining problem is population aggregation/selectivity.

Do not run the full Gate-v0 population curve until that distinction is resolved.
