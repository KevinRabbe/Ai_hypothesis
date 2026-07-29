# Gate-5 v0 — pre-result K16 control clarification

## Status

**FROZEN BEFORE ANY GATE-5 DEVELOPMENT WORLD IS GENERATED OR INSPECTED.**

This note resolves one wording ambiguity in `gate5_bounded_score_activation_protocol.md` without changing any scientific treatment, K value, work budget, world namespace, non-inferiority margin or outcome rule.

## Clarification

The protocol requires `bounded_score_k16` and `bounded_hash_k16` to use the same K16 bounded-sampling mechanics.

The exact invariant is:

- both use the same deterministic `k16` sampling-group identifier;
- both use the same canonical path ordering, start/stride derivation and `min(16, N)` visibility bound;
- **if the incoming live reserves are identical at a slot, the visible candidate subsets are exactly identical**;
- after the two policies make different parent selections, their later live reserves may differ, so later visible candidate identities are not required to be identical across conditions;
- that later divergence is an intended downstream consequence of the treatment (learned-score selection versus answer-blind hash selection), not a different sampler.

Therefore the phrase “exact same K=16 visible candidate subset” in the primary protocol is to be read as **same K16 sampling function on the same incoming reserve**, not as an impossible requirement that two causally diverged search trajectories contain the same candidates forever.

No Gate-5 result had been generated or inspected when this clarification was frozen.
