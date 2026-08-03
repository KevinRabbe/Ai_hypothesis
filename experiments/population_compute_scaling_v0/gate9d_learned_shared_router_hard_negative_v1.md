# Gate-9D learned shared router hard-negative v1

## Status

**DEVELOPMENT-ONLY CORRECTION OF THE SUPERVISED ROUTING EXPERIMENT. THIS DOES
NOT CLAIM AUTOMATIC COORDINATE DISCOVERY, DOES NOT USE END-TO-END ANSWER LOSS,
AND DOES NOT OPEN POPULATION CONFIRMATION OR LATER GATE-9D STAGES.**

## V0 failure localization

The v0 router learned the bias gate exactly, but its contribution gate collapsed
to nearly unconditional basis routing:

```text
expected contribution messages per episode  approximately 4.11
observed contribution messages per episode  approximately 9.00
```

The reported contribution accuracy near `0.9805` was dominated by the negative
class. Contribution-positive states occupy only 1,024 of the 65,536 exhaustive
local `(worker_input, query)` states. Aggregate accuracy therefore concealed
failure on the selected-versus-unselected basis distinction.

## Corrective curriculum

V1 keeps the same shared 1,218-parameter router and the same fixed XOR
population execution. It changes only local supervised routing training and
measurement.

Each training batch contains equal-sized strata:

1. `worker_input == 0` — bias positives;
2. one-hot worker selected by its query bit — contribution positives;
3. one-hot worker not selected by its query bit — hard contribution negatives;
4. nonzero non-one-hot workers — distractor negatives.

No class-weighted uniform sampling is used.

## Required routing metrics

The aggregate result must report, per seed:

- bias true-positive and true-negative counts;
- selected-basis recall;
- unselected-basis specificity;
- distractor specificity;
- contribution precision and recall;
- joint exact routing accuracy;
- predicted versus expected contribution messages per episode.

A router is not admitted merely because aggregate local-state accuracy is high.

## Population evaluation

The frozen message payload and XOR aggregation are evaluated on fresh affine
operators at populations `9`, `16`, `64`, and `256`, including worker
permutation and shuffled-support controls.

A pass requires every seed and population size to satisfy:

```text
bias routing exact                         1.0
selected-basis recall                      1.0
unselected-basis specificity               1.0
distractor specificity                     1.0
full execution exact                       1.0
permuted execution exact                   1.0
shuffled-support exact                     <= 0.02
observed contribution message count        exact expected count
```

## Interpretation boundary

A pass would establish that the local communication predicate can be learned by
one shared neural router when its rare hard-negative structure is represented
explicitly in the training distribution.

It would still not establish automatic coordinate discovery or learning from
answer loss alone.

The qualification slice is intentionally limited to six source, runner,
documentation, test, and workflow files; no binary artifact is admitted.
