# Gate-8 v1 three-seed population scientific execution v0

## Status

**FIRST FROZEN SCIENTIFIC-TEST EXPOSURE FOR THE POPULATION ORGANISM ONLY. GEMMA MODEL LOADING, TOKENIZER LOADING, REFERENCE INFERENCE, TRAINING AND FINAL POPULATION-VERSUS-REFERENCE CLASSIFICATION REMAIN CLOSED.**

Base: exact qualified Gemma binding-result head:

`8237732aecbec083c66668de9fae132e0cc4c1f9`

This stage executes Phase A of the preregistered Gate-8 v1 scientific evaluation. It produces the complete three-checkpoint population evidence and all required population controls. It does not run the 1B reference.

## Exact source bindings

```text
scientific protocol  6bb89111a47713bea0a23bb1cae662ed5ec56b42
architecture         c3ab64008c816fa1eb6f9d6f8f0a1a99ed195ec8
qualified runtime    333d88ac4fc52f1651741fba224e0b4605feedd3
Gemma binding result 8237732aecbec083c66668de9fae132e0cc4c1f9
```

Exact checkpoints:

```text
seed 0  3005369a4830c12baee8ffa7fedc1bed0f1888784e1043bd88f4afd2b7cddde9
seed 1  cbcae487dd7f4c695e1d6a83a61926cd43f5ccf6add1a7469c16a15697d22d07
seed 2  e1e35b3864354e8f3398497a897b6a759dfa3454a33d866de63784a323f461e4
```

Each checkpoint must load with `weights_only=True`, reproduce its frozen metadata, contain the exact 12-tensor state dictionary, contain exactly 19,649 finite float32 learned parameters, and bind step 1,024.

## Frozen scientific matrix

```text
split                 test
test seed             0
world indices         0..511
worlds per condition  512
conditions            21
unique worlds         10,752
checkpoint seeds      0, 1, 2
```

The condition matrix remains population-major:

```text
(32,4)
(64,4) (64,8)
(128,4) (128,8) (128,16)
(256,4) (256,8) (256,16) (256,32)
(512,4) (512,8) (512,16) (512,32) (512,64)
(1024,4) (1024,8) (1024,16) (1024,32) (1024,64) (1024,128)
```

Every generated world is identity-checked, validated, required to have a unique world ID, and evaluated against the exact symbolic oracle. No result may change architecture, checkpoint choice, training, thresholds or execution after exposure.

## Exact neural-transition compilation

A Gate-8 v1 worker observes only an 8-bit inbox code and one of eight primitive transform IDs. Every edge worker starts from the same learned initial hidden state and executes at most once. Therefore one checkpoint has a finite complete local domain of:

`256 × 8 = 2,048 neural transitions`.

After every checkpoint hash and state-dict contract is verified—but before any scientific-test world is generated—the runner evaluates the actual neural worker on all 2,048 inputs and preserves its exact argmax message table.

This is compilation, not replacement by the symbolic transform library:

- the table contains only the checkpoint's neural predictions;
- no oracle target is used to modify or validate those predictions;
- a wrong neural transition remains wrong in every scientific world that uses it;
- qualification compares compiled execution against the already-qualified neural runtime on contract worlds only;
- no scientific-test world is generated in CI.

The neural model objects are discarded before test-world generation. Population wall time measures compiled deterministic execution and is labelled accordingly. It must not be represented as raw recurrent-network wall time or compared directly with Gemma inference wall time. Normalized neural compute remains:

`recurrent updates × 19,649 learned parameters`.

One-time table compilation time and each transition-table SHA-256 are reported separately.

## Population modes and controls

Every checkpoint and condition executes:

1. `full` — exact synchronous qualified communication.
2. `no_communication` — workers may emit, but no message is delivered.
3. `shuffled_worker` — the existing deterministic worker/transform permutation, preserving transform marginals.
4. `shuffled_message` — emitted messages are deterministically reassigned within each round, preserving message marginals.
5. `target_worker_only` — only the target edge worker receives the root message and performs one local update; no communication occurs.
6. `random_answer` — deterministic world-ID-derived answer in `0..15`.
7. `oracle` — exact symbolic answer.

The original positive-scaling causal guard remains based only on full versus no communication and full versus shuffled worker at `(512,64)` and `(1024,128)`.

## Statistical estimand

For each condition and mode:

- retain all three raw checkpoint-seed correctness vectors;
- average equally across checkpoint seeds for each shared world index;
- bootstrap the 512 shared world indices with replacement;
- use 20,000 deterministic PCG64 replicates;
- use the same sampled index multiplicities across all three checkpoints;
- use linear 2.5% and 97.5% quantiles.

The compact empirical-multinomial implementation is distributionally identical to explicit world-index resampling for this mean statistic. Its namespace, algorithm label and correctness-matrix SHA-256 are preserved. Causal differences use the corresponding paired per-world full-minus-ablation values before bootstrap.

## Outputs

The external result preserves:

- one raw JSONL row per checkpoint, world and mode;
- 225,792 raw rows exactly;
- all condition accuracies and confidence intervals;
- raw per-seed accuracies;
- resource accounting and normalized efficiency metrics;
- six population frontiers;
- both frozen causal-ablation rows;
- the frozen population-scaling classification;
- transition-table artifacts and hashes;
- Git, environment, checkpoint and execution provenance;
- a recursive outer SHA-256 manifest.

The raw ordering is:

`condition → world index → checkpoint seed → mode`.

Oracle and random controls contain no learned parameters; their per-learned-parameter fields are null rather than being attributed the organism's 19,649 parameters.

## Closed boundaries

```text
population scientific test generation  admitted
three checkpoint loading               admitted
population controls                    admitted
population scaling classification      admitted
training                               false
Gemma model loading                     false
reference tokenizer loading             false
reference inference                     false
joint reference comparison              false
```

## Qualification boundary

CI may compile the runner and execute contract-world equivalence probes only. CI must not invoke the production runner, generate a `test` world, load an admitted checkpoint artifact, load Gemma files or a tokenizer, perform training, or expose a scientific answer.

## Next stage

After the Phase-A result is independently audited and permanently recorded, Phase B may run the exact frozen Gemma reference on the identical 10,752 worlds. Only after both result sets are qualified may a standard-library finalizer execute the paired population-minus-reference bootstraps and frozen reference classifier.
