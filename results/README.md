# Results

This directory will contain reproducible experiment outputs and compact summaries of what each experiment established.

## Result policy

For every experiment, record:

- experiment identifier;
- code/configuration version;
- architecture and parameter count;
- training-data version;
- random seed or seed policy;
- hardware used;
- training duration;
- evaluation metrics;
- inference/resource measurements;
- failures and anomalies;
- interpretation;
- whether the result supports, weakens, or leaves the current hypothesis unresolved.

Negative and null results must be retained.

Raw large artifacts such as model checkpoints should not automatically be committed to Git. Store references and checksums when appropriate.

## Comparison principle

Population configurations and dense baselines must be compared using end-to-end costs. Coordination, batching, memory movement, aggregation, and scheduling are part of the cost of the population architecture and must not be excluded from performance comparisons.
