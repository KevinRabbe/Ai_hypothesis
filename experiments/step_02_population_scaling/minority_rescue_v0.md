# Step 2A — Minority Rescue Diagnostic v0

## Why this experiment exists

The 16-worker 50K population already demonstrates that width contains additional useful information:

- one-worker oracle coverage: about 94.83%;
- 16-worker oracle-any-correct coverage: about 97.77%;
- current reducer accuracy: about 95.26%;
- 463 validation cases contain a true minority opportunity at width 16;
- the v0 reducer rescued only 19 of those cases.

The immediate bottleneck is therefore not worker shrinking and not proof that additional workers can contain additional signal. The bottleneck is **evidence utilization**.

The v0 evidence contract already preserves maximum and top-k evidence for every label. However, the final v0 primary label is selected from mean evidence. A strong minority label can therefore be recorded or cause abstention without usually being able to become the selected class.

## Research question

> Can inference-visible population evidence distinguish a genuinely useful strong minority candidate from a noisy minority outlier well enough to improve class selection without unacceptable harm to correct primary decisions?

This is narrower than "build a better reducer". It is intended to locate the failure boundary.

## Frozen inputs

For this experiment:

- keep the 50K Step 1 worker architecture frozen;
- use the already trained independent 50K workers;
- do not train smaller workers;
- do not expand to 64 or 256 workers yet;
- do not change the v0 production reducer;
- do not access the frozen test set;
- do not mix compiler or adaptive-allocation changes into this experiment.

## Two-stage design

### Stage A — candidate proposal

For each answerable validation sample:

1. keep the v0 mean-evidence primary label;
2. consider only valid, protected, non-primary labels;
3. propose the alternative with the strongest single-worker evidence;
4. record inference-visible evidence features for that proposal.

Candidate proposal is intentionally high-recall. A rare decisive signal should be allowed to become a candidate even when population mean evidence suppresses it.

### Stage B — rescue gate

Fit a tiny logistic gate on the development half of validation data. The gate receives only inference-visible features, including:

- candidate mean, maximum, and top-k evidence;
- candidate support fraction;
- reliability and margin of the strongest supporting worker;
- candidate-versus-primary evidence gaps;
- primary margin;
- population disagreement;
- population uncertainty and invalid-label mass.

The diagnostic target is whether switching from the current primary label to the proposed minority candidate would correct the sample.

The target is used only during validation calibration. It is not part of inference.

## Data separation

Only answerable validation samples are used for the label-selection diagnostic.

- even answerable row indices: development/calibration;
- odd answerable row indices: untouched confirmation.

The threshold is selected on development data under an explicit total-harm budget and then applied once to confirmation data.

The frozen test set remains unopened.

## Measurements

Report separately:

### Candidate proposal quality

- primary accuracy;
- primary error count;
- candidate proposal rate;
- number of primary errors for which the proposed candidate is correct;
- fraction of primary errors recoverable by the proposal rule.

### Gate quality

- switch count;
- rescued error count;
- harmed correct-primary count;
- net gain count;
- rescue recall;
- switch precision;
- total harm rate;
- accuracy before and after the gate.

## Interpretation

### Outcome 1 — candidate recall is low

The useful minority signal exists somewhere in the population, but maximum protected evidence does not reliably propose it. Improve the evidence representation or candidate proposal before designing a more complex gate.

### Outcome 2 — candidate recall is good, gate cannot separate signal from noise

The population contains useful alternatives, but current inference-visible evidence is insufficient to know when to trust them. Worker output/evidence quality is the next bottleneck.

### Outcome 3 — confirmation improves under a small harm budget

Promote the calibrated rescue mechanism to a reducer-v1 candidate and compare it against mean-logit, mean-probability, majority, and reducer-v0 controls.

### Outcome 4 — development improves but confirmation does not

Treat the gate as overfit and reject it. Do not tune against confirmation or test.

## Run

Use the same 16 frozen 50K worker checkpoints already used by the Step 2A scaling gate:

```powershell
python -m ai_hypothesis.step02.run_minority_rescue `
  --checkpoints <seed_1_best.pt> <seed_2_best.pt> ... <seed_16_best.pt> `
  --device cuda `
  --backend vmap `
  --count 20000 `
  --batch-size 256 `
  --max-harm-rate 0.001 `
  --output results/step02/minority_rescue/validation_result.json
```

No `--split test` option exists intentionally.
