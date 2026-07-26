# Relay hop protocol v0

## Motivation

The development-only local primitive diagnostic in closed PR #63 showed:

- key/query equality is learnable: 99.83% held-out accuracy;
- one-hop population lookup is learnable: 99.44% exact at 4 workers and 95.85% at 16;
- relay-2 is effectively unsolved with persistent local worker state.

The failure therefore begins when a retrieved value must become the query for the next hop.

## Architecture repair

Two mechanisms are required by the development ablations:

1. **Exactly compositional node messages.** One learned node projection maps fixed node bits into the shared message space. A fresh query is `tanh(node_projection(bits))`. Candidate values now contribute the **raw** `node_projection(value_bits)` to the shared reducer, whose existing final `tanh` produces that exact same bounded representation when the candidate is selected. This avoids the earlier accidental `tanh(tanh(...))` distortion.
2. **Hop-local worker state.** The shared field is the recurrent state across relay hops. Each worker's hidden state is reinitialized from its immutable local record before processing the next shared query, preventing a previous match from contaminating later re-selection.

The generic `SharedPopulationCell` keeps persistent local state as its default. Relay explicitly opts into hop-local reset semantics.

Workers still learn the recurrent update and scalar emission gate. No oracle chain edge, lookup table, answer key, specialized worker type, or population-dependent learned parameter is added.

## Historical development ablations

The following numbers were measured before the exact compositional correction, when candidate content was prematurely bounded and the shared reducer applied a second `tanh`. They localized the state-reset requirement but do **not** qualify the corrected exact-identity implementation.

At fixed population 4 with information-complete relay-2 worlds:

- prematurely bounded compositional messages + **persistent** local state: 3.91% exact;
- original learned state→message projection + **reset** local state: 1.95% exact;
- prematurely bounded compositional messages + **reset** local state: **99.22% exact**, 99.87% bit accuracy;
- the corresponding no-communication control: 0% exact.

These results established that hop-local state reset matters strongly. The message contract itself has since been corrected so perfect candidate selection maps a node value to the exact same representation used for that node as a new query.

## Historical width-scaling diagnostics

Before the exact-identity correction:

- one mixed-population checkpoint over 4/16/64/256 failed beyond the small-width regime;
- separate fixed-width checkpoints failed already at width 16;
- parameter-free softmax competition preserved width 4 but did not rescue width 16+.

Those negative results remain useful localization evidence, but the corrected exact-identity implementation must be re-qualified before deciding the next bottleneck.

## Current immediate gate

Run the existing #64 structural suite and mixed-population relay-2 diagnostic on the corrected exact-identity implementation.

If the corrected protocol still shows a width-related collapse, rerun the fixed-width capacity diagnostic from the corrected head before changing aggregation or model capacity again.

No Gate-v0 population-scaling result is claimed.
