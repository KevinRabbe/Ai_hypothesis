# Population Language Post-Training Learning L0 bounded adapter v0

## Status

**Implementation boundary only. No calibration candidate has been executed or selected, and no final-world result is claimed.**

This slice implements the smallest neural adaptation surface admitted by the preregistered Post-Training Learning L0 protocol.

Exact qualified protocol base:

```text
48e8edb9ff39417bfb5cb44521318efa032a340a
```

## Frozen base

The adapter wraps the exact `18,967,968`-parameter `PopulationLanguageOrganism`. Every base parameter is set non-trainable. Construction fails if the base parameter count or any locked lexical/worker dimension differs.

The original Population Language L0 format bypasses the adapter and calls the frozen base directly. Mixed original-L0/adaptation batches fail closed.

## Six trainable tensors

```text
operator_embedding_delta   8 × 512
encoder_down               rank × 14,544
encoder_up                 128 × rank
decoder_down               rank × 128
decoder_up                 14,544 × rank
value_logit_bias            16
```

Supported ranks are `1, 2, 4, 6`. Rank 6 contains exactly `180,176` trainable parameters and `720,704` raw FP32 tensor bytes, remaining below the preregistered one-percent and one-MiB limits.

The two up-projections, operator deltas, and value bias initialize to zero. Consequently, the complete task path is an exact no-op in both FP32 and CPU BF16 autocast before learning.

## Task gate

The adapter activates only for prefixes of the exact form:

```text
<bos> <query> operator... value <answer>
```

with one to four preregistered operator tokens and one preregistered value token. The ground-truth answer token is not part of the input.

## Persistence

The persisted state is an ordered mapping containing exactly the six declared FP32 tensors. Loading rejects:

- missing, extra, or reordered tensors;
- shape or dtype drift;
- non-tensor values;
- NaN or infinite values.

No examples, world seeds, inferred rules, executable logic, optimizer state, or retrieval state can be represented by this API.

## Qualification boundary

The tests use the production-shape population model and verify:

- exact parameter and byte budgets;
- frozen base parameters;
- exact FP32 and BF16 zero-effect equivalence;
- exact original-L0 bypass;
- base-state hash stability after an adapter optimizer update;
- changed neural adaptation tensors after learning;
- exact output equivalence after loading the tensor-only state into a fresh deterministic base;
- strict artifact and mixed-batch rejection.

This slice does not choose rank, learning rate, update count, or any other calibration setting.
