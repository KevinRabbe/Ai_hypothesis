# Post-Training Learning L0 — Execution Primitives v0

Status: **PRIMITIVES ONLY — NO CHECKPOINT, CALIBRATION, OR FINAL RESULT**

This slice implements the deterministic, checkpoint-independent building
blocks required by a later calibration runner. It is stacked on the qualified
calibration contract and does not inspect the active reference-training output.

## 1. Exact provenance

- source calibration branch:
  `agent/population-language-post-training-learning-l0-calibration-v0`
- source calibration head:
  `19aa701c475b19fc5b31409528948f21ad9fbdf4`
- execution-primitives branch:
  `agent/population-language-post-training-learning-l0-execution-primitives-v0`
- preregistration issue: `#202`

The slice contains no checkpoint path, no output-directory discovery, no GPU
launch entry point, and no calibration or final evaluator.

## 2. Learning-example encoding

A world example has the form:

```text
<bos> <query> operator... input-value <answer> target-value <eos>
```

The model input ends exactly at `<answer>`:

```text
<bos> <query> operator... input-value <answer>
```

The next-token target is the single `target-value` token. The target position
and `<eos>` are not included in the model input. This prevents accidental
answer-position leakage while preserving the legitimate input value, even when
the input and output value happen to be the same token.

The encoder validates the full example shape, operator chain, input value,
output value, vocabulary membership, input dtype, and allowed depth.

## 3. Deterministic adaptation schedule

The locked microbatch size is 8 and the adaptation split contains 64 examples.

For update index `u`, the ordinals are:

```text
(u * 8 + 0..7) mod 64
```

No shuffle, random sampler, curriculum, or early stopping is allowed.

Locked schedule SHA-256 values:

| Updates | Presentations | Schedule SHA-256 |
|---:|---:|---|
| 32 | 256 | `391e3cedb1290c5956cd0d8b72fea240054f20914b64f81317966c70173ac81d` |
| 64 | 512 | `0df432ac0bbfde71a84a199118d041467371b9c47f21c547d3aa06ebeced42ca` |
| 128 | 1,024 | `77d166ee7e7fcb579acd16b3295ab56e9f42aed37ed4fa884fbca388461a7bed` |
| 256 | 2,048 | `f8dbafd553ab4bca6d3d6b977a3cb8bc939adb18a86a46e9837d3c5bb9dd8958` |

The later runner must record the applicable schedule hash.

## 4. Locked optimizer constructor

The optimizer constructor accepts only the six declared trainable adapter
parameters and verifies that the candidate rank matches their exact parameter
count.

It creates AdamW with:

```text
learning rate    selected candidate value
betas            (0.9, 0.999)
epsilon          1e-8
weight decay     0
```

The later training loop remains responsible for:

- CUDA BF16 autocast with FP32 adapter weights;
- gradient-norm clipping at 1.0;
- one optimizer update per locked microbatch;
- no early stopping;
- verifying the immutable base hash before and after adaptation.

The existing reference-training canonical state hash is reused for the base
model rather than introducing another hashing format.

## 5. Tensor-only adaptation artifact

The artifact contains only the six declared FP32 tensors, in exact order:

1. `operator_embedding_delta`
2. `encoder_down`
3. `encoder_up`
4. `decoder_down`
5. `decoder_up`
6. `value_logit_bias`

The binary format records only:

- a fixed format magic;
- tensor count;
- each declared tensor name;
- shape;
- raw little-endian FP32 bytes.

It contains no:

- raw examples;
- world seeds;
- rule parameters;
- optimizer state;
- model state;
- retrieval state;
- executable code;
- arbitrary pickled objects.

The decoder rejects:

- missing, extra, or reordered tensors;
- wrong names, shapes, dtypes, layouts, or byte counts;
- non-finite values;
- truncated payloads;
- trailing bytes;
- artifacts exceeding 1 MiB;
- SHA-256 mismatch when an expected hash is supplied.

Artifact publication uses filesystem mode `xb`. An existing path therefore
fails instead of being overwritten. The write is flushed and fsynced before
the record is returned.

## 6. Paired bootstrap primitive

The final composition test uses exact paired correctness vectors:

```text
per episode gain = adapted_correct - baseline_correct
```

The lower confidence bound is locked to:

```text
procedure         DETERMINISTIC_PAIRED_BOOTSTRAP_PERCENTILE_V0
resamples         20,000
RNG               NumPy PCG64
seed              final_world_seed + 900000
lower percentile  0.025
quantile method   linear
```

Each resample draws the same number of episodes as the original paired vector,
with replacement, and records the mean paired gain.

The implementation deliberately makes each resample through one fixed RNG call.
Changing the processing chunk size therefore cannot change the RNG sequence or
the resulting confidence bound.

The primitive accepts caller-provided correctness vectors only. It does not
load final worlds or labels.

## 7. Deferred work

This slice deliberately does not implement:

- checkpoint discovery or manifest interpretation;
- loading the still-running reference checkpoint;
- the adaptation training loop;
- calibration-world execution;
- final-world execution;
- retention evaluation;
- subprocess restart orchestration;
- result publication;
- operational authorization.

The fresh-process runner should be stacked later, after the completed reference
checkpoint and its exact manifest can be inspected without touching an active
run.

## 8. Qualification target

CI must prove:

- exact prefix/target encoding;
- locked schedule hashes;
- exact optimizer settings and rank binding;
- tensor-only artifact round trip;
- create-once behavior;
- strict corruption rejection;
- deterministic, chunk-invariant paired bootstrap;
- existing protocol, adapter, and calibration contracts remain green;
- exact four-file scope.

This is implementation preparation only. It produces no scientific evidence.
