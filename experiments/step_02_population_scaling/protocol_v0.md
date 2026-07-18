# Step 2 Population Scaling Protocol v0

## Research gate

Do not freeze the primary Step 2 worker size until the Step 1 multi-seed confirmation sweep is complete.

The selected primary worker architecture should be chosen from the statistically confirmed viable region, using both capability and cost evidence rather than one best single-seed score.

Candidate worker sizes that remain scientifically interesting after Step 1 may also be retained for Step 2B fixed-budget organization experiments.

## Step 2A objective

Measure how useful population behavior changes as the number of architecturally identical workers increases.

Primary question:

> Does increasing homogeneous population width add useful learned evidence beyond a single worker, and where do gains saturate relative to end-to-end cost?

## Population-bank design

Instead of training a separate model for every tested width, train a maximum-width homogeneous worker bank and evaluate nested subsets.

Initial maximum width:

`W_max = 256`

Primary evaluation widths:

- 1;
- 4;
- 16;
- 64;
- 256.

A deterministic permutation of worker indices defines the nested subsets:

```text
W=1   -> first 1 worker
W=4   -> first 4 workers
W=16  -> first 16 workers
W=64  -> first 64 workers
W=256 -> all workers
```

This ensures that increasing width adds workers instead of replacing the entire population with a different independently trained system.

To reduce subset-ordering bias, evaluate multiple deterministic worker permutations from the same bank. For confirmatory results, train multiple independent population banks with different population seeds.

Initial recommendation:

- development: 1 population bank;
- confirmation: at least 3 independent population banks;
- 5 banks if variance is high near the decision boundary.

## Worker independence policy for Step 2

Each worker has:

- identical architecture;
- independent parameter initialization;
- independent stochastic model state;
- the same overall data distribution;
- the same optimizer and training hyperparameters.

For Step 2 v0, workers may receive the same training batches in the same order. Diversity then comes primarily from initialization and stochastic model behavior.

Systematic experiments on stronger or weaker diversity mechanisms belong to Step 3.

Exact weight clones do not count as additional population width.

## Training implementation

The target implementation trains the population as a vectorized homogeneous worker bank where practical.

A reference loop implementation may be used to verify mathematical correctness on tiny widths, but performance results must use grouped/vectorized execution.

All workers in one bank should receive equal optimization opportunity.

Record:

- training steps per worker;
- samples seen per worker;
- total wall-clock training time;
- GPU time where measurable;
- peak VRAM;
- peak RAM;
- total trainable worker parameters;
- effective total parameter-updates or an equivalent normalized compute estimate.

## Aggregation variants

Every evaluated width must be tested with the same set of aggregation variants:

1. majority vote — naive control only;
2. mean-logit ensemble;
3. mean-probability ensemble;
4. evidence-preserving reducer v0.

Aggregation thresholds for the evidence-preserving reducer are calibrated using training/validation data only and frozen before test evaluation.

The test set must remain untouched by threshold selection.

## Width-specific calibration

Two reporting modes should be retained:

### Shared calibration

Use one aggregation configuration across all widths.

Purpose: determine how width behaves without retuning the reducer.

### Width-calibrated

Allow validation-only calibration for each width separately.

Purpose: estimate the best capability available at each width when the reducer is appropriately configured.

Both should be reported when feasible because they answer different questions.

## Evaluation data

Use the same frozen Step 1 benchmark version unless a benchmark limitation is discovered that invalidates population measurement.

All widths and aggregation variants must use identical test samples.

No population width may receive a privileged test split.

## Core quality metrics

For every width report:

- final accuracy;
- macro task accuracy;
- by-task accuracy;
- by-difficulty accuracy;
- invalid output rate;
- uncertainty precision;
- uncertainty recall;
- stability across population banks and subset permutations.

## Population-information metrics

Report:

- oracle-any-correct coverage;
- all-wrong rate;
- worker disagreement entropy;
- minority-rescue opportunity;
- minority-rescue rate;
- minority-suppression rate;
- majority-harm rate;
- evidence-utilization gap.

These metrics are necessary because final accuracy alone cannot tell whether additional workers discover useful information that the reducer fails to exploit.

## Scaling interpretation

Possible outcomes:

### No population gain

Single-worker and larger-width results remain statistically indistinguishable.

Interpretation: additional independent workers are not adding useful information under the current setup.

### Information gain but aggregation failure

Oracle-any-correct coverage rises with width, but final population accuracy does not.

Interpretation: the population contains additional useful evidence, but the reducer is failing to use it.

### Useful population gain

Final quality, uncertainty, robustness, or evidence recovery improves reproducibly with width.

Interpretation: population scaling is adding usable value.

### Saturation

Quality improves up to a width and then stops improving materially.

Interpretation: this identifies a useful fixed-width region for later adaptive-allocation experiments.

### Negative scaling

Larger widths reduce quality or uncertainty reliability.

Interpretation: correlation, aggregation, or contradiction handling may be causing harmful population effects.

## Performance measurements

For each width measure end to end:

- neural worker execution time;
- evidence-packet construction time;
- GPU-to-CPU transfer time if applicable;
- aggregation time;
- final decision time;
- total latency;
- samples per second;
- worker evaluations per second;
- peak VRAM;
- peak RAM.

The performance report must distinguish useful neural work from coordination overhead.

## Step 2A exit criterion

Step 2A passes if increasing width produces a reproducible useful improvement before end-to-end overhead dominates.

A useful improvement may be:

- higher final task quality;
- better uncertainty behavior;
- higher minority-rescue capability;
- lower catastrophic-error rate;
- higher robustness under perturbation.

If oracle coverage increases but final quality does not, Step 2A should not be declared a full success; the correct conclusion is that aggregation requires improvement.

## Step 2B — Fixed-budget homogeneous population competition

After Step 2A demonstrates that population width adds useful information, compare candidate worker sizes under an approximately equal total worker-parameter budget.

Initial target budget:

`B_target ≈ 5,000,000 worker parameters`

For each confirmed worker architecture with actual parameter count `P_worker`:

`W = round(B_target / P_worker)`

Record the resulting exact budget:

`B_actual = W * P_worker`

Use the nearest practical worker count and report the percentage mismatch from `B_target`.

Each configuration remains homogeneous.

Example form:

```text
Population A: many smaller workers
Population B: medium number of medium workers
Population C: fewer larger workers
```

Do not mix worker shapes inside one population.

## Step 2B controls

Keep constant or carefully normalized:

- benchmark version;
- training data distribution;
- samples seen per worker;
- optimizer policy;
- training-step ceiling;
- hardware;
- precision;
- evidence contract;
- aggregation rule;
- evaluation set.

Because total worker parameters are approximately fixed, using the same training steps per worker also keeps the total scale of parameter-updates approximately comparable. Actual wall-clock and measured compute must still be reported because different architectures can have different hardware efficiency.

Step 2B initially uses 100% worker activation.

Adaptive activation is excluded until the population-size effect is understood.

## Step 2B primary question

> Under approximately equal total learned worker capacity, is capability better organized as many smaller workers, fewer larger workers, or an intermediate population?

This experiment may reveal differences in:

- final quality;
- diversity;
- oracle evidence coverage;
- minority rescue;
- robustness;
- batch efficiency;
- memory footprint;
- latency.

## Relationship to later fixed-budget competition

Step 2B compares different homogeneous population organizations against each other.

It does not replace the later broader fixed-budget competition against dense or conventional model organizations.

## Stop conditions

Pause or redirect Step 2 if:

- independently initialized workers remain functionally identical enough that width adds no new information;
- population training cannot be vectorized efficiently enough to be practical;
- evidence aggregation destroys most of the additional oracle coverage;
- coordination overhead grows faster than useful population value;
- results are dominated by benchmark ceiling effects.
