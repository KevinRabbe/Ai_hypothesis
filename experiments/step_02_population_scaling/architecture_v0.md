# Step 2 Population Architecture v0

## Purpose

Define the smallest population architecture needed to test whether multiple architecturally identical learned workers provide useful information beyond a single worker.

This architecture is intentionally simpler than the eventual runtime. It excludes adaptive routing, heterogeneous worker shapes, hierarchical recombination, and storage-tier management.

## Population invariant

Within one population experiment, every worker has exactly the same architecture and tensor shapes.

Workers may have different learned weights, but they must not differ in:

- model width;
- model depth;
- attention-head count;
- feed-forward width;
- input shape;
- output shape.

This invariant exists to preserve the central implementation advantage of uniform batching.

Different worker sizes such as approximately 25K, 50K, and 75K are treated as different homogeneous population configurations, never mixed inside one population.

## Worker architecture

Step 2 reuses a Step 1-confirmed `Step01Unit` architecture without adding a new trainable evidence head for the first population experiment.

Each worker receives the same benchmark input representation:

- maximum sequence length: 32 elements;
- feature width: 16 floats;
- mask;
- task identity encoded using the existing benchmark representation.

Each worker produces:

- 11 label logits;
- 1 uncertainty logit.

The population layer converts these outputs into the structured evidence contract defined in `evidence_contract_v0.md`.

This keeps the first population test focused on population behavior rather than simultaneously changing the worker architecture.

## Worker weights

Workers are independently initialized and learned under the same architecture and overall training distribution.

Step 2 uses one default diversity policy:

- independent parameter initialization;
- independent stochastic training trajectory;
- same benchmark generator and data distribution;
- same optimizer family and hyperparameter policy.

The exact effect of weight correlation and diversity is deferred to Step 3. Step 2 only requires enough non-identity that adding workers can, in principle, add information.

Exact checkpoint cloning does not count as population growth because identical weights receiving identical inputs cannot add independent learned evidence.

## Population execution

For a configured width `W`:

1. receive one benchmark sample or batch of samples;
2. execute all `W` workers;
3. collect one evidence packet per worker and sample;
4. reduce the evidence packets into one population evidence summary;
5. produce one final task prediction and uncertainty state.

Step 2A uses 100% activation of the configured population width.

There is no dynamic worker selection in v0.

## GPU execution requirement

The benchmarked implementation must target vectorized or grouped worker execution rather than a Python loop that launches one tiny GPU job per worker.

Conceptually, worker parameters should be organized as an additional population dimension where practical:

`[worker, ...parameter dimensions...]`

and activations should retain a compatible worker dimension during population execution.

The exact PyTorch mechanism may use vectorized functional execution or another implementation that preserves mathematical independence while avoiding thousands of isolated kernel launches.

A simple loop may exist only as a correctness reference implementation. It must not be used as the performance result for the population architecture.

## Initial population widths

Primary Step 2A widths:

- 1;
- 4;
- 16;
- 64;
- 256.

Larger widths are added only if useful signal is still increasing and hardware permits meaningful measurement.

## Population reducer

The primary reducer is evidence-preserving rather than vote-based.

It receives continuous worker evidence and retains at least:

- cumulative support per valid label;
- strongest individual support per valid label;
- top-k support per valid label;
- worker uncertainty statistics;
- disagreement and contradiction statistics;
- provenance identifying which worker produced protected strong evidence.

Majority vote is evaluated only as a control baseline.

Mean-logit or mean-probability ensembling is also evaluated as a conventional ensemble baseline.

The evidence-preserving reducer must be able to retain a strong minority signal even when most workers support another answer. A minority signal does not automatically override the population; it must remain visible to the decision rule rather than being destroyed by averaging.

## Decision layer v0

The first decision layer should remain deterministic and compact.

It consumes the population evidence summary and uses validation-calibrated thresholds to decide among:

- a task-valid predicted label;
- `UNCERTAIN` when evidence is insufficient;
- `UNCERTAIN` when protected contradictory evidence remains unresolved.

Thresholds are calibrated on training/validation data and frozen before the test set is evaluated.

The test set must never be used to tune aggregation thresholds.

A learned aggregator may be studied later as a separate variant, but it is not required to establish the first population-scaling result.

## CPU/GPU boundary for Step 2

The neural worker execution belongs on the GPU.

The first reducer may run on the CPU because the evidence packet is compact, but transfer and reduction time must be measured separately. If CPU reduction becomes measurable overhead, a GPU reducer should be benchmarked as an alternative.

Step 2 does not yet attempt to build the final heterogeneous scheduler. It only records enough timing to reveal whether aggregation is already becoming a bottleneck.

## Population configuration identity

Every result must record:

- worker architecture configuration;
- actual trainable parameters per worker;
- population width;
- total worker-parameter count;
- worker seeds or population initialization identity;
- training protocol version;
- evidence-contract version;
- aggregation version;
- code revision;
- device;
- precision;
- batch size.

This is required so that later fixed-budget comparisons cannot accidentally compare different hidden configurations.
