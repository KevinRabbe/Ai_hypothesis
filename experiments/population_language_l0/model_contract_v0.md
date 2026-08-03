# Population Language L0 model contract v0

## Status

**Executable architecture contract only. No optimization or language result is reported.**

This slice implements the two preregistered model families and verifies their structural invariants before training.

## Implemented baseline

`MatchedCausalTransformer` implements:

- learned 64-token embeddings;
- learned 32-position embeddings;
- six pre-norm causal decoder blocks;
- eight-head self-attention;
- standard GELU feed-forward layers;
- final layer normalization;
- tied input/output embedding weights and one output bias.

The live PyTorch tensor count must equal the protocol formula exactly:

```text
18,964,544 learned parameters
```

## Implemented organism

`PopulationLanguageOrganism` implements:

- one token embedding and one position embedding per causal step;
- one lexical encoder invocation per token;
- persistent worker state of width 128;
- deterministic non-learned worker coordinates;
- six rounds of shared top-4 message routing;
- a 271,680-parameter shared recurrent worker core;
- deterministic mean pooling;
- one lexical decoder invocation per token;
- tied vocabulary projection and output bias.

The live PyTorch tensor count must equal:

```text
18,967,968 learned parameters
```

Changing runtime worker count changes state shape and routed-message count, but cannot change the module parameter set.

## Causal contract

Both models are tested with paired sequences that share a prefix and differ only in later tokens. Logits on the shared prefix must be bit-identical.

The transformer enforces this with a strict causal attention mask.

The organism processes one token at a time and carries only state produced by the current and preceding tokens. No full-sequence tensor is visible to its recurrent update.

## Data contract

`materialize_batch` creates teacher-forcing tensors padded to the locked sequence limit:

```text
input_ids   [batch, 31]
target_ids  [batch, 31]
loss_mask   [batch, 31]
answer_mask [batch, 31]
```

Each episode has exactly five semantic answer targets. Padding and `<eos>` are excluded from the answer mask; valid non-padding next-token targets remain in the ordinary loss mask.

## Explicit boundaries

This slice does not:

- train either model;
- select hyperparameters from validation results;
- claim contextual word learning;
- claim population scaling;
- benchmark KV cache or persistent-state latency;
- add learned worker identities;
- modify Gate-9 evidence.

## Next qualification

The next slice must run bounded optimization diagnostics:

1. overfit a tiny fixed episode set with both tiny model configurations;
2. verify gradients reach the lexical and worker-routing paths;
3. verify the organism can retain information across the definition/query delay;
4. measure per-step memory and runtime before releasing the approximately 19M reference development run.
