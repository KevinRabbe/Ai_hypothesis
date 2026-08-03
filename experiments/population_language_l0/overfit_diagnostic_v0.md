# Population Language L0 tiny overfit diagnostic v0

## Status

**Development-only optimization diagnostic. This is not the approximately 19M reference experiment.**

## Purpose

Before spending GPU time on the preregistered models, this diagnostic asks whether the executable architectures possess a complete learning path for contextual bindings.

It tests:

1. tiny transformer and population variants can fit a fixed four-example binding set;
2. gradients reach every required lexical, routing, recurrent, and decoder component;
3. the population organism retains definition values across the delay to the answer;
4. the result is not explained by a query-only shortcut.

## Binding set

All four episodes use the same query:

```text
<query> the dax is left_of the wug <sep>
```

Only the contextual color/shape definitions change. Definition order alternates. The five-token answers change consistently with those definitions.

Because query tokens and relation are identical, a model cannot distinguish the four required answers from the query alone.

## Tiny systems

Transformer:

```text
d_model 64
layers 2
heads 4
feed-forward 128
```

Population organism:

```text
token width 64
lexical encoder/decoder width 128
worker width 32
worker feed-forward 64
router width 8
workers 4
communication rounds 2
top-k 2
```

Both optimize answer-span cross-entropy for 256 full-batch AdamW steps at learning rate 0.003 from seed 120100.

Answer-span-only training is allowed here solely to test architecture and gradient viability. It is not the L0 reference objective, which remains full next-token cross-entropy.

## Required gradient paths

The transformer must expose finite nonzero gradients through token embeddings, attention projection, feed-forward input, and language-model bias.

The population organism must expose finite nonzero gradients through:

- token embeddings;
- both lexical encoder layers;
- worker initializer;
- router query and key projections;
- message encoder;
- recurrent input and hidden weights;
- worker feed-forward input and output;
- both lexical decoder layers;
- language-model bias.

## Pass criteria

Both systems must satisfy all conditions:

- all required gradient norms are finite and strictly positive on the initial backward pass;
- answer-token accuracy is 1.0;
- four-example answer exact accuracy is 1.0;
- first-answer-token accuracy is 1.0;
- replacing all four definition color/shape values with `<unk>` reduces first-answer accuracy to at most 0.5;
- ablation increases first-answer cross-entropy by at least 1.0.

The ablation criterion establishes causal use of definition values for the first answer token. It does not establish generalization.

## Explicit non-claims

A pass does not demonstrate:

- held-out contextual word learning;
- natural language competence;
- fixed-parameter population scaling;
- competitive performance against the matched transformer;
- efficient routing;
- KV-cache or persistent-state superiority.

A failure blocks the reference development run until the exact gradient, optimization, or memory-retention defect is localized.
