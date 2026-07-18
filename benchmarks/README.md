# Benchmarks

Benchmarks in this repository are designed to answer research questions, not to maximize headline scores.

## Rules

1. Define the task and success metric before training the evaluated model configuration.
2. Keep train, validation, and test data separated.
3. Record deterministic baselines where applicable.
4. Compare configurations using the same evaluation data.
5. Preserve difficult, contradictory, ambiguous, and negative examples.
6. Do not remove failed cases because they reduce the score.
7. Measure end-to-end resource cost, not only neural FLOPs.
8. Preserve source evidence when testing distributed information processing.
9. Do not treat worker agreement as ground truth.
10. Report both positive and negative results.

## Step 1 benchmark goal

The first benchmark suite should locate the transition between:

- useful learned local transformations;
- weak but still useful signal;
- mostly noisy output;
- transformations better replaced by deterministic logic.

The benchmark specification will be added before Step 1 training begins.
