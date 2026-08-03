# Population Language L0 reference-scale engineering preflight v0

## Status

**Development-only CUDA engineering preflight. This is not a language-training result.**

## Purpose

Before releasing the approximately 19 million parameter development run, this preflight determines whether both matched systems can execute the preregistered full next-token objective safely on the local GPU.

It measures a common microbatch rather than assigning each system a different effective batch regime.

## Exact systems

The preflight instantiates the qualified models without reducing width, depth, worker count, or learned parameters:

```text
Transformer parameters   18,964,544
Population parameters    18,967,968
Population train workers         32
Population rounds                 6
Population top-k                  4
```

Any live PyTorch parameter-count mismatch aborts the run.

## Locked candidates

Both systems execute one real AdamW optimization step at each microbatch:

```text
1, 2, 4, 8 episodes
```

The objective is ordinary full next-token cross-entropy over all non-padding targets. It does not use the tiny diagnostic's answer-span or first-answer auxiliary losses.

The preflight uses:

```text
CUDA BF16 autocast
FP32 parameters and AdamW state
learning rate 3e-4
betas (0.9, 0.95)
weight decay 0.1
gradient clipping 1.0
initialization seed 120100
```

Within each model, the model and optimizer persist across candidate measurements. The first candidate includes Adam-state materialization; later candidates reuse that state. `allocated_before_bytes` and `reserved_before_bytes` are recorded so the incremental activation cost is interpretable.

This preflight does not compare optimization quality across candidate steps. It only establishes finite execution and engineering capacity.

## Recorded fields

For every successful model/microbatch row:

- full next-token loss;
- global gradient norm before clipping;
- synchronized wall time;
- CUDA allocated/reserved memory before the step;
- peak CUDA allocated/reserved memory during the step.

A CUDA out-of-memory event becomes a terminal failure row for that model. Earlier successful rows remain evidence.

## Classification

The recommended common microbatch is the largest candidate that succeeds for both systems.

The preflight passes only when:

1. both systems produce finite loss, gradients, timing, and memory fields;
2. the largest common successful microbatch is at least 4;
3. the common microbatch divides the locked global batch of 256 exactly.

The resulting gradient-accumulation count is:

```text
256 / recommended common microbatch
```

A recommendation below 4 blocks the reference run pending memory engineering. It does not count as a scientific language failure.

## Cache-state estimates

The bundle also records BF16 state-size formulas per sample:

```text
Transformer KV at 32 tokens:
2 × 6 layers × 32 tokens × 512 width × 2 bytes
= 393,216 bytes

Population persistent state:
workers × 128 width × 2 bytes
```

Therefore:

```text
16 workers      4,096 bytes
32 workers      8,192 bytes
64 workers     16,384 bytes
128 workers    32,768 bytes
256 workers    65,536 bytes
```

These are analytical state estimates only. L0 does not yet benchmark decode latency, memory traffic, information retention, or cache quality. The transformer does not use a KV cache during teacher-forced training, and the organism's persistent state is not claimed to be equivalent in capability.

## Evidence bundle

A valid local run produces:

```text
summary.json
transformer-rows.json
population-rows.json
git-head.txt
git-status.txt
run-config.json
manifest.sha256
one ZIP archive
```

The runner requires the exact branch, a clean tree, a fresh output path, CUDA BF16 support, and deterministic cuBLAS configuration before Torch initializes.

## Explicit non-claims

A pass does not demonstrate:

- contextual-word generalization;
- population scaling;
- competitive language modeling;
- stable 4,096-step training;
- superior cache efficiency;
- natural-language capability.

It only authorizes the next bounded reference-scale development slice.