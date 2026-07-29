# Gate-3 v0 — Development stop decision

## Status

**GATE-3 v0 DEVELOPMENT CLOSED AFTER SEED 0**

This decision is made after the first admitted development checkpoint and before running any additional Gate-3 development training seed or opening any confirmation world.

It is therefore the robustness-seed policy required by the frozen confirmation boundary.

## Observed seed-0 outcome

Independent audit classification:

`A_NO_OR_INCOMPLETE_BREADTH_EFFECT`

The five preregistered primary directions were:

```text
H6 W64 vs W1            -0.07421875
H8 W256 vs W1           -0.05078125
H8 W256 vs W64          -0.0078125
H8 stable vs collapsed  -0.02734375
H8 stable vs reshuffled  0.0
```

The H6 W64-vs-W1 and H8 W256-vs-W1 95% paired bootstrap intervals lie entirely below zero.

The stable exact-solve curves also decline strongly as width grows:

```text
H4: 0.2578, 0.1172, 0.0430
H6: 0.0898, 0.0469, 0.0273, 0.0156
H8: 0.0508, 0.0469, 0.0352, 0.0078, 0.0000
```

## Decision

Do **not** run Gate-3 v0 development seeds 1 or 2.

Do **not** open Gate-3 v0 confirmation.

Rationale:

1. The development outcome is not borderline or merely underpowered. The central breadth comparisons point in the opposite direction with paired intervals fully below zero.
2. Running additional seeds after such a result would spend compute primarily to search for a favorable checkpoint rather than to resolve a genuinely ambiguous development signal.
3. The preregistered interpretation map explicitly defines this pattern as Outcome A: no useful breadth effect under the measured workload/scorer.
4. A negative development result is scientifically useful and should redirect architecture rather than be tuned away under the same protocol.

## What is closed

Closed under v0:

- the current 19,873-parameter shared recurrent scorer;
- the `B(D)=2^D` fixed-work allocation rule;
- the delayed binary hypothesis-tree task as implemented;
- the stable/collapsed/reshuffled runtime semantics;
- the seed-0 development recipe;
- any attempt to obtain a positive Gate-3 verdict by rerunning or extending this exact v0 recipe.

## What remains open

This does **not** close the overall population-computation research program.

Gate 0, Gate 1 and Gate 2 remain valid results.

The negative result narrows the next research question:

> Why does broader fixed-work population hurt here, and what mechanism would allow a computational organism to create more useful possibilities without sacrificing the per-hypothesis processing required to evaluate and preserve them?

A next experiment must be versioned separately and justified by diagnosis of this failure. Candidate directions may include a different work-allocation rule, cheap non-learned branching/filtering, selective/dynamic activation, population communication, shared intermediate computation, or separating hypothesis generation from expensive learned refinement.

None of those alternatives may rewrite or reinterpret the Gate-3 v0 result.

## Confirmation boundary

`confirmation_opened = false`

Gate-3 v0 confirmation remains permanently unopened unless a future explicit scientific amendment decides to test reproducibility of the negative effect for a separate purpose. Such an amendment would not be a positive-path confirmation protocol.
