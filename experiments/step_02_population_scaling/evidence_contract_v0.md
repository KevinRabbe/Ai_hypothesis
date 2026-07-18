# Step 2 Evidence Contract v0

## Principle

Workers contribute evidence, not votes.

The evidence contract must preserve continuous support, uncertainty, contradiction, and provenance long enough for the population reducer to inspect them. A rare strong signal must not disappear merely because most workers prefer another label.

This v0 contract is deliberately derived from the existing Step 1 outputs so that the first population experiment does not require changing the worker architecture.

## Raw worker output

For every sample, worker `i` emits:

```text
WorkerRawOutput
- worker_index: integer
- label_logits: float[11]
- uncertainty_logit: float
```

The 11 label logits correspond to the existing Step 1 non-`UNCERTAIN` label vocabulary.

The task identity and valid task label set are known deterministically from the benchmark schema.

## Derived worker evidence

The reducer derives a structured evidence packet without adding trainable parameters.

```text
WorkerEvidencePacketV0
- worker_index
- task
- label_probabilities_all
- valid_label_probabilities
- invalid_label_mass
- uncertainty_probability
- reliability
- evidence_score_per_valid_label
- top_valid_label
- top_margin
```

### Label probabilities

Compute a softmax across all 11 label logits to retain visibility into task-invalid output mass.

`label_probabilities_all = softmax(label_logits)`

For the current task, identify the valid non-`UNCERTAIN` labels and renormalize their probability mass for task-local comparison.

### Invalid-label mass

`invalid_label_mass` is the total probability assigned to labels that are invalid for the current task.

This value is never silently discarded. A worker that strongly allocates probability to task-invalid labels is less reliable for population aggregation even if its final decoded output happens to be valid.

### Uncertainty probability

`uncertainty_probability = sigmoid(uncertainty_logit)`

### Reliability

Initial deterministic reliability:

`reliability = (1 - uncertainty_probability) * (1 - invalid_label_mass)`

This is a v0 operational definition, not a claim that the formula is optimal. Any later replacement must be versioned and compared against this baseline.

### Per-label evidence score

For valid task labels, define a centered log-support score.

For worker `i` and valid label `l`:

```text
raw_support_i(l)
= log(p_i(l) + eps)
  - mean(log(p_i(k) + eps) for valid k != l)
```

Then:

`evidence_i(l) = reliability_i * clip(raw_support_i(l), -E_max, E_max)`

`eps` and `E_max` are fixed constants recorded in the aggregation configuration.

For two-label tasks this behaves similarly to a reliability-weighted log-odds signal. For multi-label tasks it measures support for one label relative to the other valid alternatives.

## Population evidence summary

The reducer combines worker packets into:

```text
PopulationEvidenceSummaryV0
- population_width
- sum_evidence_per_label
- mean_evidence_per_label
- max_evidence_per_label
- top_k_evidence_per_label
- top_k_worker_ids_per_label
- support_count_per_label
- mean_uncertainty
- max_uncertainty
- uncertainty_quantiles
- mean_invalid_label_mass
- max_invalid_label_mass
- disagreement_entropy
- protected_minority_labels
- protected_minority_worker_ids
```

## Evidence-preserving fields

### Cumulative evidence

`sum_evidence_per_label` and `mean_evidence_per_label` represent broad population support.

### Strongest evidence

`max_evidence_per_label` ensures that one unusually strong worker signal remains visible even when the population average favors another label.

### Top-k evidence

The strongest `k` pieces of evidence for each label are retained with worker provenance. This avoids reducing the entire population to one mean and one maximum.

The initial `k` should remain small and fixed, for example 3 or 5, and must be recorded in the aggregation configuration.

### Support count

A count of workers whose evidence exceeds a validation-calibrated support threshold may be recorded as a diagnostic.

It must not become the primary decision rule. A label is not selected simply because it has the most supporters.

### Disagreement entropy

Worker argmax labels may be used to calculate disagreement entropy as a diagnostic and uncertainty signal.

This is not majority voting. The argmax distribution is not itself the final prediction.

## Protected minority evidence

A label becomes a protected minority candidate when one or more workers provide evidence above a validation-calibrated strong-evidence threshold.

Protected evidence must retain:

- label;
- evidence strength;
- worker identity;
- rank among that label's strongest supporting workers.

Protection means the evidence survives aggregation. It does not mean the minority label automatically wins.

The final decision layer may:

- accept the broad population result if the opposing minority signal is weak;
- return `UNCERTAIN` when broad support and protected minority evidence remain materially contradictory;
- select the minority-supported label only when the deterministic evidence rule justifies it.

## Decision rule v0

The exact numeric thresholds are calibrated on validation data and frozen before test evaluation.

The initial deterministic rule should use:

1. the label with strongest mean population evidence as the primary candidate;
2. the margin between the strongest and second-strongest mean evidence;
3. protected minority evidence for competing labels;
4. population uncertainty statistics;
5. invalid-label mass.

A sample should resolve to `UNCERTAIN` when the configured rule finds insufficient support or unresolved contradiction.

No test-set result may be used to tune these thresholds.

## Required aggregation baselines

Every population-width experiment should report at least:

1. single-worker performance distribution;
2. majority vote — naive control only;
3. mean-logit ensemble;
4. mean-probability ensemble;
5. evidence-preserving reducer v0.

This allows us to determine whether the proposed evidence mechanism contributes anything beyond ordinary ensembling.

## Population-specific evidence metrics

### Oracle-any-correct coverage

Fraction of samples for which at least one worker produced the correct task-valid answer.

This is not an achievable production score by itself. It estimates how much useful information exists somewhere in the population before aggregation.

### All-wrong rate

Fraction of samples where every worker is wrong.

### Minority-rescue opportunity

Fraction of samples where:

- majority vote is wrong;
- at least one minority worker is correct.

This measures how often a non-voting aggregator has a chance to recover useful minority evidence.

### Minority-rescue rate

Among minority-rescue opportunities, fraction correctly resolved by the evidence-preserving reducer.

### Minority-suppression rate

Fraction of samples where strong correct minority evidence existed but the final reducer discarded or failed to preserve it and returned a wrong answer.

### Majority-harm rate

Fraction of samples where the majority-vote answer was correct but the evidence-preserving reducer changed the result to an incorrect answer.

### Evidence-utilization gap

Difference between oracle-any-correct coverage and final population accuracy.

A large gap indicates that useful evidence exists in the population but the reducer is failing to use it.

## Provenance scope in Step 2

Because every worker receives the same compact benchmark sample in Step 2, provenance initially identifies the worker/checkpoint that produced evidence.

Source-region or document-position provenance is deferred until information partitioning experiments, where workers may receive different portions of a larger input.

## Versioning rule

Any change to:

- reliability calculation;
- evidence transformation;
- clipping;
- top-k retention;
- strong-evidence thresholds;
- contradiction rules;
- final decision logic;

must increment the evidence/aggregation contract version and be evaluated separately.
