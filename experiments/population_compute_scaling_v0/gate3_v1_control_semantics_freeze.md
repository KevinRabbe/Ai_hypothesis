# Gate-3 v1 — Collapsed-control semantics freeze

## Status

**FROZEN BEFORE ANY GATE-3 v1 DEVELOPMENT RESULT OR TRAINING RUN**

This note resolves one mechanical ambiguity in the initial Gate-3 v1 protocol before any v1 scorer training or development evidence exists.

## Qualification finding

The initial implementation represented the collapsed-diversity control by filling the nominal reserve with duplicate copies of the current top hypothesis.

A pre-result regression then exposed a semantic defect: if those duplicate backing slots are individually schedulable, the search can repeatedly activate duplicate copies of the same logical prefix. That is not a pure diversity control; it changes the search algorithm and can prevent the committed hypothesis from reaching a terminal leaf.

No Gate-3 v1 training or development result had been run when this defect was found.

## Frozen interpretation

For Gate-3 v1, reserve capacity `L` has two meanings that must remain distinct:

1. **nominal physical/state-bank capacity** — the resource allocation associated with the condition;
2. **logical schedulable hypothesis population** — the distinct candidate identities available to best-first search.

For the primary stable treatment, up to `L` distinct hypotheses may be logically schedulable.

For the collapsed-diversity control:

- nominal capacity remains `L`;
- only the current top hypothesis is logically schedulable;
- any duplicate backing/state-bank slots are resource placeholders, not additional search candidates;
- expanding the collapsed hypothesis advances that one logical path once;
- the control therefore behaves logically like a committed `L=1` search while retaining the larger condition's nominal state-bank allocation and exact learned-work budget.

This is the intended causal control:

> Does the benefit come from retaining **distinct alternative hypotheses**, rather than from merely allocating a larger reserve/state bank?

## Required identities

Before any Gate-3 v1 development evidence is admitted, qualification must prove:

- `L1 stable == L1 collapsed` exactly;
- a large collapsed reserve follows the same logical candidate path and frontier-exhaustion schedule as `L1 stable` for the same scorer/world;
- collapsed logical reserve population never exceeds one;
- total learned work remains the frozen tier total through matched sink work;
- nominal `L` remains recorded in telemetry even though logical collapsed population is one.

## What does not change

This clarification changes none of the preregistered v1 scientific variables:

- two active neural child lanes;
- eight recurrent updates per evaluated child;
- fixed search-round budget;
- exact total learned work;
- shared learned parameters;
- noisy world evidence;
- best-first score ordering;
- primary exact search-coverage outcome;
- confirmation boundary.

It only prevents duplicate physical slots from being misinterpreted as distinct logical hypotheses in the diversity control.

Changing this interpretation after the first admitted Gate-3 v1 development result would require a new protocol version.
