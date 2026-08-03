# Gate-9D learned shared router exhaustive-margin v3

Development-only supervised routing follow-up. The zero-FP v2 run showed that threshold calibration alone is insufficient: two seeds were strictly separable, while seed 0 retained an overlapping contribution-gate representation.

v3 keeps the same 1,218-parameter router and fixed XOR population executor. It changes the training objective and coverage only:

- every optimizer step evaluates all 65,536 local routing states;
- each gate balances positive and negative losses independently;
- positives are pushed above `+2` and negatives below `-2`;
- the exact worst positive and worst negative worker/query states are recorded;
- population execution remains fail-closed unless both gates are strictly separable.

This tests whether explicit exhaustive margin training can make the existing router reliably population-safe. It remains supervised routing, not answer-loss learning, automatic coordinate discovery, later-stage execution, or population confirmation.
