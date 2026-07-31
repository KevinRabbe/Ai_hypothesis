# Gate-8 v1 three-seed scientific evaluation v0 — protocol

## Status

**DATA-FROZEN PRE-EXPOSURE SCIENTIFIC-EVALUATION PROTOCOL — CHECKPOINT
LOADING, SCIENTIFIC-TEST WORLD GENERATION, REFERENCE-WEIGHT LOADING,
INFERENCE AND RESULT CLASSIFICATION REMAIN CLOSED.**

Base: exact qualified seed-2 result head:

```text
4f3159e5ba7abde6045543fd85e691f8f75ef7c4
```

The three training runs are complete and independently qualified. This stage
does not reopen training. It binds their immutable identities and freezes how
the original Gate-8 capability-scaling and 1B-reference questions will be
evaluated before any fresh test answer is exposed.

## Exact checkpoint bindings

| Seed | Qualified result head | Raw result SHA-256 | Selected checkpoint SHA-256 | Source manifest SHA-256 |
|---:|---|---|---|---|
| 0 | `f259620f7d3beab2f886c76271c753e9ebf96dc9` | `1e42eb53f6446e4eeb66bbb2090c8dad7551e2098b76f289b43cf0c05975e829` | `3005369a4830c12baee8ffa7fedc1bed0f1888784e1043bd88f4afd2b7cddde9` | `3db3284b37d4ddd7dfec03ab9fd6c0aa6193d59c0cb887fcb773927eaa13e3ac` |
| 1 | `66532cb72c2bb0703e7af395ef51bbbef31d9b3b` | `873cacdb5965b29c59a14d74fc0df7a32c036f35aeeda2cdd4cb5ac3640a7e8e` | `cbcae487dd7f4c695e1d6a83a61926cd43f5ccf6add1a7469c16a15697d22d07` | `22a22993ebe3aff46997fd83605aed25170db6abca631e5c109d8bcc33446133` |
| 2 | `4f3159e5ba7abde6045543fd85e691f8f75ef7c4` | `cc9dad3bd05982ff5390a8f23bff3bfe8227c5a4c4c457e6578426b186bb6df2` | `e1e35b3864354e8f3398497a897b6a759dfa3454a33d866de63784a323f461e4` | `2df3483f63e6c31e06a51fde57e57eb773bb183d3cf71a405407748645c89ef0` |

Every selected checkpoint is step 1,024, contains exactly 12 finite float32
tensors and exactly 19,649 learned parameters. The checkpoint binaries remain
external; the execution wrapper must reject any byte mismatch before loading.

Frozen implementation bindings:

```text
original capability protocol  e73541115e8ddd122f336463dc1a9ffdbf82df46
protocol correction           124065691d257d483a37be4200452f1f7ca50063
world contract                722c646eacfd05c51fb9d1e8887fe1620d53672c
encoder contract              9882256ae0152bc266dc4d96cab3bbeb0c4ef95b
tokenizer result              c7f5260189ef9ac1a1beb73596446316631090c7
v1 architecture              c3ab64008c816fa1eb6f9d6f8f0a1a99ed195ec8
v1 runtime                   333d88ac4fc52f1651741fba224e0b4605feedd3
v1 training protocol          a33dc123d090268a531d112251ea3ab53cb50062
seed-0 execution              1b449f0ed4998e9246c86803d4473d0ac9ebdac3
replication execution         31a8d115eb14d876997fb361b02258fbe3a30506
```

## One shared scientific-test set

All systems use the exact same worlds:

```text
split                 test
test seed             0
world indices         0..511
worlds per condition  512
conditions            21
```

The population-major condition matrix is unchanged:

```text
population = 32, 64, 128, 256, 512, 1024
depth      = 4, 8, 16, 32, 64, 128
valid iff  = 8 × depth <= population
```

Each of the three checkpoints evaluates every one of the 21 conditions. The
Gemma reference, controls and symbolic oracle use those same public worlds and
queries. No result-dependent omission, retry, prompt change, extra sample,
adaptive compute or alternate test seed is permitted.

The test namespace must not be generated until the exact reference model-weight
binding has independently qualified. Merely loading or hashing checkpoint
files does not open the test namespace.

## Primary three-seed aggregation

The primary organism accuracy for one population/depth condition is:

1. accuracy over 512 worlds for each checkpoint seed;
2. the arithmetic mean of the three seed accuracies.

Every per-seed result remains mandatory and visible. Pooling may not concatenate
the 1,536 predictions and silently treat training seeds as independent worlds.

The 95% condition interval uses 20,000 deterministic paired bootstrap samples.
The resampling unit is the world index. A bootstrap replicate samples 512 world
indices with replacement and applies the same sampled indices to all three
checkpoint seeds before equal-seed averaging.

Frozen bootstrap namespace:

```text
gate8-v1-three-seed-scientific-evaluation-bootstrap-v0
```

This interval quantifies test-world uncertainty conditional on the three frozen
training seeds. It is not represented as a large-sample estimate of arbitrary
future training-seed variance.

## Population capability frontier

For each population, choose the deepest valid condition satisfying both original
thresholds:

```text
point accuracy >= 0.90
95% CI low    >= 0.85
```

If no depth at a population is solved, its shallowest valid condition is carried
to the original classifier with `solved=false`; no synthetic depth-zero value or
post-hoc rescue is introduced.

The original classifier is applied unchanged:

- all population frontiers solved;
- non-decreasing maximum solved depth;
- at least three strict adjacent increases;
- final solved depth at least four times the first;
- both causal guards pass.

Frozen outcomes remain:

```text
G8_POSITIVE_CAPABILITY_SCALING
G8_CAPABILITY_PRESENT_NO_SCALING
G8_NEGATIVE_CAPABILITY_SCALING
G8_CAPABILITY_SCALING_INCONCLUSIVE
```

## Organism evaluation modes

Every checkpoint and every condition reports:

### `full`

The exact qualified v1 runtime, unchanged.

### `no_communication`

Workers may compute local emissions, but every delivery is suppressed. Delivered
messages and communicated bits are exactly zero.

### `shuffled_worker`

The already-qualified deterministic worker/transform reassignment control.
Topology, worker slots, transform marginals, population and round budget remain
fixed.

### `shuffled_message`

After a round's emissions are fixed, deterministically permute the emitted
eight-bit codes across the same ordered destination slots. Destinations, number
of deliveries, hidden states, topology and code multiset remain unchanged. The
permutation is keyed only by the frozen shuffled-message namespace, world ID,
checkpoint seed and round. It must be non-identity whenever at least two
deliveries exist; a zero/one-delivery round is unchanged.

### `target_worker_only`

Identify the unique public edge entering the query target. Initialize only that
worker with the qualified initial hidden state and the ordinary carrier-zero
root-symbol code, execute exactly one local update, and score its symbol head.
No other worker, recurrent round or message delivery is available. This is the
frozen single-worker shortcut control; it is not an alternative organism.

Two non-neural controls are also mandatory:

- deterministic random answer: SHA-256 of the frozen random-control namespace
  and world ID, reduced modulo 16;
- exact symbolic oracle: must equal 1.0 accuracy or the artifact is invalid.

## Causal guards

The original causal conditions remain:

```text
(512, 64)
(1024, 128)
```

For each condition, calculate paired world-level differences after equal-seed
averaging:

```text
full - no_communication
full - shuffled_worker
```

The same 20,000 world-index bootstrap replicates are shared between each full
and ablated prediction. Each lower 95% confidence bound must be strictly greater
than 0.20. `shuffled_message` and `target_worker_only` are required descriptive
controls but do not replace or modify the original causal classifier.

## Conventional 1B reference

Frozen reference identity:

```text
repository  google/gemma-3-1b-it
revision    dcc83ea841ab6100d6b47a070329e1ba4cf78752
mode        BF16, greedy temperature 0
demos       8 fixed separate-namespace demonstrations
input       <= 24,576 tokens
output      <= 64 tokens
updates     none
```

The tokenizer-only binding is already qualified:

```text
tokenizer result SHA-256
c8d6adb733cadbbd251d91d35f9d224e255705dac49ba144655717f9f4ab7b8d

tokenizer manifest SHA-256
21de192eb57c0759fbf2236fae2252e5319696b71689ada1471b74a9f1315a88
```

Model weights are not yet bound. A separate pre-exposure stage must freeze every
required model/config file SHA-256 at the exact revision before any test world
is generated or any reference inference occurs.

For one condition, the paired world-level difference is:

```text
mean(correct_seed0, correct_seed1, correct_seed2) - correct_reference
```

The 20,000 bootstrap samples resample world indices identically across all three
checkpoints and the reference. Condition rows report the maximum actual prompt
token count observed over their 512 worlds.

The pooled replicate gives each of the 21 conditions equal weight. Population
size, depth, prompt length and runtime do not change the primary accuracy
weight. The original five-outcome reference classifier is applied unchanged.

## Mandatory raw evidence

Every organism world row includes:

```text
checkpoint seed
population and depth
world index and world ID
evaluation mode
predicted symbol
oracle answer symbol
correctness
round count
active workers
recurrent updates
delivered messages
communicated bits
wall time
peak device memory
```

The execution must commit predictions before invoking the oracle for scoring.
The artifact must preserve raw per-world rows, per-seed condition summaries,
equal-seed pooled summaries, frontier rows, causal rows and reference rows.

Every condition reports accuracy and interval plus:

- mean active workers;
- mean communicated bits;
- mean recurrent updates;
- mean wall time and peak memory;
- capability per learned parameter;
- capability per active worker;
- capability per communicated bit;
- capability per recurrent update;
- capability per normalized compute.

Generated tokens, prompt tokens, reference wall time, memory and normalized
compute are reported separately. Accuracy classification and efficiency
reporting may not be merged into one post-hoc score.

## Independent audit

The future independent auditor must:

1. verify all three checkpoint/result/manifest identities;
2. verify exact reference revision and all bound file hashes;
3. regenerate all 10,752 test worlds from the frozen namespace;
4. reconstruct every oracle answer and world ID;
5. validate exact row coverage for every checkpoint, mode and condition;
6. independently rebuild equal-seed accuracies and all paired intervals;
7. independently rebuild frontiers, causal guards and reference pooling;
8. reapply the unchanged original classifiers;
9. prove no training, checkpoint selection or prompt adaptation occurred.

The auditor may not import the scientific executor or trust its summaries.

## Closed boundary

This protocol stage contains only:

- standard-library immutable constants and evidence dataclasses;
- wrappers around the already-qualified Gate-8 classifiers;
- synthetic protocol regressions;
- this scientific record;
- branch-scoped standard-library qualification CI.

It loads no checkpoint, imports no Torch, generates no world, loads no tokenizer
or model, performs no inference, reads no result artifact and exposes no test
answer.

## Next admission boundary

After this exact protocol qualifies, the next stage is the immutable Gemma
model-weight binding. Only after that binding qualifies may a separate guarded
joint scientific-execution and independent-audit slice be implemented.
