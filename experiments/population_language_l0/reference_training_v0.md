# Population Language L0 reference training v0

## Status

**Executable reference-training implementation only. No reference result exists until the exact CUDA runner completes and its evidence bundle verifies.**

This slice follows the locally qualified reference preflight. It does not modify the qualified model, dataset, or objective implementations.

## Fixed scientific contract

Each matched model is trained independently at the qualified approximately 19 million parameter scale for every preregistered initialization seed:

```text
seeds                  120100, 120101, 120102
optimizer              AdamW
betas                   (0.9, 0.95)
weight decay            0.1
optimizer steps         4,096
global batch            256 episodes
peak learning rate      3e-4
warmup                  205 updates
post-warmup schedule    cosine decay to zero
precision               CUDA BF16 autocast
parameters/state        FP32
clip norm               1.0
population train workers 32
population rounds       6
population top-k        4
```

The full next-token cross-entropy over every non-padding target is the only training objective. Answer tokens receive no additional loss weight. The tiny diagnostic's first-answer auxiliary loss is not imported.

The fixed final update-4096 checkpoint is evaluated. There is no early stopping and no validation-selected checkpoint. Test semantics therefore cannot influence training or checkpoint selection.

## Exact data order

The reference train set contains 131,072 deterministic episodes. The 4,096 × 256 schedule exposes exactly 1,048,576 episodes, or eight complete epochs.

Within each epoch, the implementation uses the locked bijection:

```text
ordinal = (position × 65,537 + epoch × 8,191) mod 131,072
```

Because 65,537 is odd and the dataset size is a power of two, every train ordinal appears exactly once per epoch. Both systems and all seeds receive the same ordered global batches.

The complete schedule hash is:

```text
2813a743f3bbb32a182ee7bf83efde654fa95359977a0ad050236141575e4f64
```

The implementation also records the locked first-256 fingerprints for train, validation, and test.

## Microbatch boundary

The qualified engineering preflight established a common microbatch of 8, producing 32 accumulation passes per optimizer update. That remains the default:

```text
microbatch                 8
gradient accumulation     32
effective global batch   256
```

The CLI makes microbatch explicit and requires it to divide 256 exactly. A different value changes only accumulation mechanics, not ordered episodes, effective batch, optimizer updates, seeds, or objective. It must be separately CUDA-qualified before being used for the scientific execution.

Evaluation microbatch is also explicit and affects only memory/runtime.

## Hot-path data handling

All three deterministic splits are materialized once before optimization and kept as a compact GPU-resident tensor cache. Training then uses index selection from that cache.

This avoids regenerating and hashing more than six million repeated episode presentations across the six model/seed runs. The cache content remains exactly the qualified `l0_data` materialization.

The cache build duration and resident bytes are recorded in evidence.

## Training evidence

For each model and seed, the runner records every 64 updates plus the first and final update:

- optimizer update;
- learning rate;
- full next-token NLL for the exact global batch;
- global gradient norm before clipping;
- synchronized elapsed time;
- peak allocated CUDA bytes;
- peak reserved CUDA bytes.

A progress file is atomically rewritten at every recorded point. If a multi-hour run is interrupted, the partial output directory remains diagnostic evidence and must not be reused.

The final model state is saved once per model/seed. Evidence contains:

- the checkpoint file SHA-256;
- a serialization-independent canonical SHA-256 over sorted tensor names, dtypes, shapes, and bytes;
- exact parameter count;
- training schedule hash;
- training tokens;
- estimated active training FLOPs;
- wall time and peak VRAM.

Optimizer state is not preserved because the protocol uses the fixed final checkpoint and does not resume a partial scientific run.

## Evaluation

Transformer validation and test evaluation use the fixed final checkpoint.

The population organism is trained only with 32 workers. Its one fixed checkpoint is evaluated without parameter mutation at:

```text
16, 32, 64, 128, 256 workers
```

Every evaluation records:

- next-token NLL and perplexity;
- answer-span NLL;
- greedy five-token answer exact accuracy;
- color-token accuracy;
- shape-token accuracy;
- relation-token accuracy;
- exact accuracy after swapping the two contextual definition blocks;
- answer-token agreement between original and swapped definition order;
- estimated forward FLOPs per episode;
- answer exact accuracy per estimated active GFLOP;
- synchronized wall time.

Population rows additionally record:

- active worker count;
- routed messages in total, per processed token, and per episode;
- BF16 persistent-state bytes per episode;
- mean top-k router entropy;
- normalized router entropy;
- selected-sender coverage;
- effective worker utilization;
- sender-selection coefficient of variation.

Router statistics are collected with forward hooks on the qualified query/key projections. The probe reconstructs the exact top-k scores and tie break without changing model outputs or learned parameters.

## FLOP estimates

Multiply-add is counted as two FLOPs. Estimates include the dominant dense projections, attention score/value products, population routing score products, selected-message aggregation, recurrent projections, and worker feed-forward projections.

Layer normalization, activation functions, softmax elementwise work, embedding lookup, and optimizer elementwise operations are not included. These values are analytical active-compute estimates, not hardware throughput measurements.

## Validity classification

The run is classified `POPULATION_LANGUAGE_L0_REFERENCE_RUN_VALID` only when:

1. all three seeds are present in preregistered order;
2. exact live parameter counts match the protocol formulas;
3. models use the same schedule hash, global batch, optimizer-step count, microbatch, and accumulation count;
4. all checkpoints and evaluation rows contain valid finite evidence;
5. every population worker-count row uses the same canonical checkpoint hash;
6. the transformer reaches at least 95% validation answer exact accuracy in every seed.

Failure of any requirement produces `POPULATION_LANGUAGE_L0_REFERENCE_RUN_INVALID`. The complete result bundle is still preserved and packaged when execution reaches classification.

## Fixed-parameter population criterion

Only a valid run receives a population-scaling conclusion.

Across the three-seed test aggregate, support requires both:

1. 256 workers improve answer exact accuracy over 16 workers by at least five percentage points, or reduce answer-span NLL by at least 10%;
2. at least three of the four consecutive worker-count transitions are non-degrading in answer exact accuracy within a 0.5-point tolerance.

The result is reported as either:

```text
SUPPORTS_FIXED_PARAMETER_POPULATION_SCALING
DOES_NOT_SUPPORT_FIXED_PARAMETER_POPULATION_SCALING
```

An invalid reference run receives no scaling conclusion.

The mean population result at 32 workers is also compared descriptively with the matched transformer. Beating the transformer is not required for the first scaling criterion.

## Immutable execution boundary

The PowerShell wrapper sets:

```text
CUBLAS_WORKSPACE_CONFIG=:4096:8
```

before Python imports Torch.

The runner then requires:

- the exact `agent/population-language-l0-reference-training-v0` branch;
- a clean working tree before and after execution;
- a fresh output directory and ZIP path;
- evidence paths outside the repository;
- CUDA BF16 support;
- deterministic algorithms;
- TF32 disabled;
- the exact preregistered three seeds and 4,096 updates.

There is deliberately no `--steps` override.

## Evidence bundle

A complete run contains:

```text
run-start.json
run-config.json
git-head.txt
git-status.txt
seed-120100.json
seed-120101.json
seed-120102.json
progress/*.json
checkpoints/*.pt
summary.json
manifest.sha256
one ZIP archive
```

Checkpoint files are stored without redundant ZIP compression; JSON and text evidence are deflated. All files are covered by the manifest and final archive SHA-256.

## Runtime boundary

The qualified local preflight observed approximately 0.70 seconds for one population microbatch-8 BF16 AdamW step after initialization. At 32 accumulation passes, the three population seeds alone project to multiple days of GPU execution before full validation/test evaluation.

Therefore this PR qualifies the executable contract; it does not silently launch the scientific run. The exact local command is released only after CPU qualification and merge-tree verification.

## Explicit non-claims

This implementation does not itself establish:

- held-out contextual-word learning;
- fixed-parameter population scaling;
- competitive language modeling;
- natural-language competence;
- router efficiency;
- wall-clock or energy superiority;
- KV-cache superiority;
- a valid scientific result.

Those claims depend exclusively on the completed, hash-verified reference evidence.
