# Gate-7 high-scale routing-bandwidth continuation result protocol v0

## Status

**DATA-BLIND POST-RUN INTERPRETATION CONTRACT — RESULT RECORD CLOSED.**

This contract was frozen while the admitted continuation execution was running and before any continuation result artifact was inspected.

Exact qualified continuation-execution head:

`19ee6b4e228c56b32a11b11b1c61b35bf640e2c8`

The contract adds no artifact reader, result values, world generator, checkpoint loader, Torch dependency, scientific runner, PowerShell wrapper or new evidence namespace.

## Purpose

The continuation can produce several scientifically distinct outcomes. This contract prevents the observed pattern from determining:

- which values are reported;
- whether negative tiers are hidden;
- whether an asymptotic law is fitted;
- whether a resource stop is misclassified as scientific evidence;
- which family of question is considered next.

It does not choose the exact next experiment. It only freezes the interpretation and family-level decision boundary.

## Required provenance

A permanent continuation result may be recorded only from an independently audited artifact that reports:

- exact execution head;
- `artifact_valid = true`;
- empty audit error list;
- result SHA-256;
- independent-audit SHA-256;
- recursive-manifest SHA-256;
- `training_performed = false`;
- `checkpoint_selection_performed = false`;
- `second_continuation_opened = false`.

A failed wrapper or rejected audit is preserved for diagnosis and is not converted into a scientific result record.

## Required tier reporting

Every completed population must retain the complete frozen evidence surface:

- population identity;
- global-reference viability;
- global score/hash coverage, paired delta and 95% interval for T0, T1 and T2;
- checkpoint-stratified global-reference delta and interval;
- score/hash coverage for every fixed K16, K32, K64, K128, K256 and K512 condition;
- score-vs-hash and score-vs-global paired intervals for every K and checkpoint;
- complete ordered passing-K set;
- smallest passing K when one exists;
- descriptive `K/N` ratio;
- exact tier outcome.

No K may be omitted because it failed, passed unexpectedly, or appears inconsistent with neighboring populations.

## Tier outcomes

```text
G7_CONTINUATION_K_REQUIRED
G7_CONTINUATION_NO_K_LE_512
G7_CONTINUATION_REFERENCE_NOT_VIABLE
```

Interpretation:

- `K_REQUIRED`: the global learned reference is viable and at least one tested K passes all six checkpoint criteria. `K_required(N)` is the smallest tested passing K, while the full passing set remains primary evidence.
- `NO_K_LE_512`: the global learned reference is viable but no tested K up to 512 passes. This is a tested-budget result, not proof that bounded routing fails in general.
- `REFERENCE_NOT_VIABLE`: the global learned reference itself fails its frozen viability rule. This is a global-signal or representation result, not a routing-budget result.

## Campaign outcomes

```text
G7_POST_CONFIRMATION_LADDER_COMPLETE
G7_POST_CONFIRMATION_RESOURCE_FRONTIER_REACHED
```

A complete campaign must contain all four ordered populations:

```text
N16384
N32768
N65536
N131072
```

A resource-frontier campaign may contain only an exact contiguous completed prefix, and the frontier must be the next uncompleted population.

## Cross-population interpretation boundary

Allowed:

- report the observed `K_required(N)` staircase for completed viable-reference tiers;
- preserve every passing-K set;
- report `K/N` as a descriptive ratio;
- compare checkpoint consistency and interval width;
- state exactly where the global reference, tested routing budget or hardware became limiting.

Forbidden:

- fit an asymptotic scaling exponent from this four-population ladder;
- interpolate or predict outcomes for an unobserved population;
- combine the earlier 64-world screen with 512-world confirmation/continuation points as equal-precision evidence;
- infer monotonicity from only the smallest passing K;
- treat `NO_K_LE_512` as proof that all bounded routing fails;
- treat a resource frontier as scientific non-viability;
- open a rescue K, second continuation or post-result condition.

## Frozen next-question hierarchy

The next research family is selected in this order:

### 1. Resource engineering

Condition:

`G7_POST_CONFIRMATION_RESOURCE_FRONTIER_REACHED`

Next family:

`G7_NEXT_QUESTION_RESOURCE_ENGINEERING`

The scientific result remains restricted to completed populations. The resource-frontier tier and all larger populations remain unobserved.

### 2. Global signal or representation

Condition:

The full campaign completes, but at least one tier is `G7_CONTINUATION_REFERENCE_NOT_VIABLE`.

Next family:

`G7_NEXT_QUESTION_GLOBAL_SIGNAL_OR_REPRESENTATION`

Routing cannot be diagnosed against a failed global learned reference. The next study must first isolate representation, task signal or global search viability.

### 3. Routing mechanism or tested budget

Condition:

The full campaign completes, every global reference is viable, and at least one tier is `G7_CONTINUATION_NO_K_LE_512`.

Next family:

`G7_NEXT_QUESTION_ROUTING_MECHANISM_OR_BUDGET`

The next study may compare routing mechanisms or extend the pre-frozen visibility budget, but its exact design must be separately preregistered.

### 4. Coordination efficiency

Condition:

The full campaign completes, every global reference is viable, and every tier has at least one passing K.

Next family:

`G7_NEXT_QUESTION_COORDINATION_EFFICIENCY`

Only in this case does the evidence support moving from routing-bandwidth sufficiency to the next population-computation question: whether worker activation, recycling, specialization, communication topology or recurrent scheduling can improve capability per active worker, communicated bit and runtime update under the same learned-parameter budget.

## Boundary

This protocol does not open the continuation result record or any subsequent experiment. After the running artifact completes, the result must first pass the existing independent auditor unchanged. Only then may an immutable result branch copy the observed values, hashes and the family-level classification defined here.
