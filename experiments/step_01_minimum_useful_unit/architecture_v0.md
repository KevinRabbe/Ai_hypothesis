# Step 1 Architecture v0 — Minimal Learned Processing Unit

## Purpose

This architecture is not intended to be a standalone language model or autonomous agent. It is a deliberately small learned processing unit used to measure how far neural computation can be reduced before useful local transformations collapse into noise.

The unit should receive a compact standardized feature sequence and return a compact structured signal.

Raw text, images, audio, and other modalities are out of scope for the unit itself in Step 1. Later systems may use shared or deterministic front-ends to convert raw modalities into the standardized representation consumed by the population.

## Design goals

The v0 architecture family must:

- scale from millions of parameters down into the low-thousands range without changing the fundamental computation pattern;
- avoid large vocabulary embeddings or language-generation heads that would dominate tiny parameter budgets;
- support efficient batching on GPU hardware;
- be simple enough to profile and reason about;
- support local sequence relationships rather than only independent scalar classification;
- use one compact output contract across the benchmark suite;
- allow uncertainty or abstention instead of forcing a guess.

## Standard input contract

Initial proposal:

- fixed maximum sequence length: 32 elements;
- per-element feature width: 16 floating-point values;
- optional binary validity mask;
- one task identifier encoded as part of the input rather than a separate large embedding table.

Conceptual shape:

```text
batch × 32 × 16
```

The benchmark generator is responsible for mapping each synthetic task into this common representation.

The representation should preserve the information needed for the task while remaining small enough that the worker, rather than the modality encoder, is what the experiment measures.

## Proposed backbone family

Use a small residual sequence encoder with the same overall block structure at every size.

Initial block:

```text
input projection
    ↓
N residual sequence-mixing blocks
    ↓
masked pooling / summary token
    ↓
small output head
```

Each residual block should contain:

1. normalization;
2. lightweight self-attention or another simple learned sequence-mixing operation;
3. residual connection;
4. normalization;
5. compact feed-forward transformation;
6. residual connection.

The first implementation should prefer a conventional tiny Transformer-style encoder because:

- the computation is dominated by matrix operations that batch well on GPU;
- it can model relationships across the local input sequence;
- width, feed-forward size, head count, and block count can be scaled systematically;
- the architecture is well understood and easy to profile.

This is an experimental baseline, not a commitment that the final population model must use Transformers.

## Output contract

The unit should produce compact logits or fixed-size vectors rather than free-form text.

Minimum output fields:

```text
class / signal logits
abstention or uncertainty logit
optional small auxiliary value vector
```

The exact number of classes depends on the benchmark task, but the shared head should remain small and fixed across size variants where practical.

No natural-language decoder is used in Step 1.

## Size scaling

The architecture family is scaled by changing only documented structural dimensions such as:

- model width;
- feed-forward width;
- attention head count;
- block count.

Candidate coarse parameter checkpoints:

```text
10M
3M
1M
300K
100K
30K
10K
3K
```

These are search points, not required final sizes.

A configuration generator should compute the actual trainable parameter count before training and record it in the experiment result.

Once a collapse region is found, perform a denser sweep around that range.

## Fairness rule

For a given benchmark version, size variants must use:

- the same input representation;
- the same output semantics;
- the same training-data distribution;
- the same train/validation/test split;
- the same loss definitions;
- the same evaluation pipeline.

Training duration may differ only when required to reach convergence. Any such difference must be recorded, because a tiny model that fails only because it was undertrained is not evidence of a capability boundary.

## Training objective

Initial objective:

```text
classification / structured-output loss
+
uncertainty or abstention objective where applicable
```

Avoid auxiliary complexity in the first sweep. JEPA-style latent prediction, population diversity objectives, routing losses, and multi-worker training belong in later experiments after the minimum useful-unit boundary is understood.

## What Step 1 is not measuring

Step 1 does not measure:

- language fluency;
- world knowledge;
- autonomous planning;
- tool use;
- long-context memory;
- population coordination;
- majority voting;
- final 1B-scale behavior.

It measures only whether a small learned unit can still turn a compact local input into a useful learned signal.

## Architecture success criterion

This architecture family is acceptable for Step 1 if:

1. large variants can solve the benchmark clearly above trivial baselines;
2. the same family can be reduced far enough to expose a measurable collapse region;
3. parameter count can be controlled predictably;
4. batching behavior can be measured cleanly;
5. the smallest variants remain mechanically valid even when their learned performance becomes poor.

If the architecture cannot scale down smoothly, that is an architecture failure, not evidence that tiny learned units are impossible.
