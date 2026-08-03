# Gate-9D learned shared router zero-false-positive v2

Development-only follow-up. The hard-negative v1 router learned every positive
and hard-negative basis state exactly, but a distractor false-positive rate near
`0.001` amplified with nominal population size and corrupted XOR aggregation.

The v2 hypothesis keeps the same network and balanced training. It replaces the
arbitrary logit threshold `0` with an exhaustive, per-gate calibrated threshold.
A threshold is admissible only when the complete 65,536-state routing truth table
is strictly separable:

```text
max negative logit < min positive logit
```

The selected threshold is the midpoint. If either gate is not strictly
separable, execution fails closed and no population pass is claimed.

This is still supervised routing over a finite Boolean domain. It is not
end-to-end answer-loss learning and not automatic coordinate discovery.
