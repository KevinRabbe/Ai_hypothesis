# Gate-9D learned shared router v0 — audited development result

The uploaded development bundle is structurally valid and bound to execution
head `d974277db6e270433876228b7534d44456ecee3e`.

## Result

The bias gate reached exact accuracy. The contribution gate did not learn the
query-conditioned basis predicate. It converged to routing almost every basis
worker for every query:

```text
expected contribution messages per episode  approximately 4.11
observed contribution messages per episode  approximately 9.00
```

The exhaustive-domain contribution accuracy of approximately `0.9805` is
misleading because contribution positives occupy only 1,024 of 65,536 local
states. Predicting the wrong rule on the rare selected/unselected basis states
can retain high aggregate accuracy while destroying XOR execution.

Observed complete execution remained at chance:

```text
full exact accuracy  0.004048582995951417
full bit accuracy    approximately 0.5003
```

## Interpretation

This is a supervised-routing training failure, not evidence of an XOR
aggregation defect. The next diagnostic must report positive recall, hard
negative specificity, confusion counts, and message-count calibration rather
than aggregate routing accuracy alone.

The corrective training distribution must explicitly balance:

- zero-input bias workers;
- selected basis workers;
- unselected basis workers;
- non-basis distractor workers.

This note is development-only. It does not modify any frozen Gate-9 or Gate-9D
result and does not claim automatic coordinate discovery.
