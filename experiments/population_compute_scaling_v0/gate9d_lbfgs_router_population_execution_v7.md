# Gate-9D LBFGS router population execution v7

Development-only end-to-end validation of the reliable eight-parameter local-summary router.

The router is retrained with full-batch LBFGS for all three seeds, calibrated on all 65,536 local routing states, and executed on 128 fresh affine operators at populations 9, 16, 64, and 256.

Pass requires exact full and permuted execution, exact bias/contribution message counts, and shuffled-support accuracy at or below 0.02 for every seed and population size.

This still uses supervised routing labels. It does not use answer-loss training, claim automatic coordinate discovery, modify frozen Gate-9 evidence, or claim population confirmation.
