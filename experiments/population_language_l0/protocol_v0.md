# Population Language L0 protocol v0

## Status

**Protocol only. No training result exists on this branch.**

This is a separate scientific lineage from Gate-9 and Gate-9D. Results from this track must not modify, reinterpret, or overwrite contextual-operator evidence.

## Question

At an approximately 19 million learned-parameter budget, can a recurrent population organism learn contextual word binding and compositional sentence continuation, and does capability improve when runtime workers increase while weights remain fixed?

## Synthetic language world

The vocabulary contains exactly 64 tokens:

- eight special tokens;
- sixteen nonce words;
- eight colors;
- eight shapes;
- eight relations;
- sixteen grammar/function tokens.

Each 28-token episode provides two contextual definitions, a relational query, and a five-token answer. Example shape:

```text
<bos>
<def> dax means red triangle <sep>
<def> wug means blue circle <sep>
<query> the dax is left_of the wug <sep>
<answer> red triangle left_of blue circle <eos>
```

Nonce meanings are remapped independently per episode. A token such as `dax` therefore cannot acquire one permanent lexical meaning; the model must bind it from the current prefix.

Definition order is independently swapped. Query order remains semantically authoritative.

## Split contract

The split is determined only by the semantic tuple

```text
(lhs color, lhs shape, relation, rhs color, rhs shape)
```

using a locked SHA-256 bucket:

- train: buckets 0–79;
- validation: buckets 80–89;
- test: buckets 90–99.

Atomic colors, shapes, relations, and nonce words remain shared. Semantic combinations do not cross splits.

Reference sizes:

```text
train       131,072 episodes
validation    8,192 episodes
test         16,384 episodes
```

The first 256-episode fingerprints are locked in the contract tests.

## Compared systems

### Transformer baseline

```text
vocabulary             64
d_model                512
layers                   6
attention heads          8
feed-forward width    2,048
maximum sequence         32
learned parameters 18,964,544
```

The baseline is a pre-norm causal decoder with standard multi-head self-attention, GELU feed-forward layers, learned token/position embeddings, tied token embedding/output weights, and an untied output bias.

### Population organism

```text
vocabulary                    64
token width                  512
lexical encoder width     14,544
worker state width           128
worker FF width              512
router width                  32
lexical decoder width     14,544
training workers              32
evaluation workers 16, 32, 64, 128, 256
learned parameters     18,967,968
shared worker core        271,680
```

The organism has a lexical encoder and decoder that each run once per token, surrounding a much smaller worker-shared recurrent core. The core uses persistent worker state, deterministic non-learned worker coordinates, a shared low-rank message router, shared message encoding, a shared GRU-like update, and a shared feed-forward update.

Only the 271,680-parameter worker core is repeated over workers and communication rounds. The full approximately 19M parameter budget is not independently applied to every worker.

No learned parameter is indexed by worker identity or worker count.

The pairwise parameter mismatch is 3,424 parameters, or approximately 0.0181%, below the locked 0.5% tolerance.

## Training contract

Both systems receive identical ordered episodes and next-token targets.

Reference training:

```text
optimizer              AdamW
betas                   (0.9, 0.95)
weight decay            0.1
peak learning rate      3e-4
schedule                5% linear warmup, cosine decay
optimizer steps         4,096
global batch size       256 episodes
microbatch size         implementation-selected, recorded
sequence length         28, padded to 32
precision               BF16 autocast, FP32 optimizer state
gradient clipping       1.0
initialization seeds    120100, 120101, 120102
```

Gradient accumulation may differ only to satisfy memory limits. The ordered global episode batches, optimizer updates, and effective global batch size must remain identical.

The full next-token cross-entropy is optimized. Answer-span metrics are reported separately; the answer span is not given extra training weight in L0.

The population model trains only at 32 workers. Evaluation at every worker count uses the identical checkpoint, parameters, recurrent round count, and decoding rule.

## Metrics

For both systems and all three seeds:

- validation/test next-token negative log-likelihood and perplexity;
- answer-span negative log-likelihood;
- greedy five-token answer exact accuracy;
- color, shape, and relation token accuracy;
- definition-order swap invariance;
- training tokens, optimizer steps, wall time, peak VRAM, and estimated FLOPs.

For the organism at every worker count:

- all language metrics above;
- active worker count;
- routed messages per token and per episode;
- persistent-state bytes;
- router entropy and worker utilization;
- capability per active FLOP.

## Gate validity

The run is invalid rather than failed if:

- parameter mismatch exceeds 0.5%;
- datasets, token order, optimizer steps, or seeds differ between systems;
- a worker-count-specific learned table or checkpoint is used;
- test semantics enter training or checkpoint selection;
- the transformer baseline fails to reach 95% validation answer exact accuracy in every seed;
- the implementation applies the complete 19M parameter budget once per worker.

## Population-scaling criterion

L0 supports the fixed-parameter population hypothesis only when all conditions hold across the preregistered three-seed aggregate:

1. the 256-worker checkpoint improves test answer exact accuracy over 16 workers by at least 5 percentage points **or** reduces answer-span NLL by at least 10%;
2. at least three of the four consecutive worker-count transitions are non-degrading within a 0.5-point tolerance;
3. the 256-worker gain is not explained by a different checkpoint, learned parameter count, tokenizer, or training data;
4. routed-message and active-FLOP costs are reported rather than hidden.

Matching or beating the transformer at 32 workers is reported, but is not required for the first scaling gate. L0 tests the existence of fixed-weight population scaling before claiming competitive language modeling.

## Explicit non-claims

L0 does not claim natural-language competence, broad reasoning, automatic vocabulary discovery, useful conversational behavior, or superior inference efficiency. It is a controlled contextual-binding and compositional-language experiment.
