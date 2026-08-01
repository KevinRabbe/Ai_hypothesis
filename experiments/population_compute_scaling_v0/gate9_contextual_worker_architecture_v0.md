# Gate-9 contextual worker architecture v0

## Status

**EXACT 19,649-PARAMETER SHARED WORKER ARCHITECTURE — CONTRACT-ONLY
FORWARD/BACKWARD CHECKS ADMITTED; OPTIMIZER, TRAINING, CHECKPOINTS,
SCIENTIFIC TEST GENERATION, EXECUTION AND RESULTS REMAIN CLOSED.**

Qualified graph-world base:

```text
b6688aa8bf305f099ec320ea60dd5ccdffce4d51
```

The architecture consumes only the information available to one Gate-9 worker:
exactly nine public support pairs in the qualified global order and one incoming
eight-bit query. It has no topology, population, depth, operator identity or
persistent per-operator memory.

## Model-visible input

Every byte is converted deterministically to its eight least-significant-first
binary features. The model receives:

```text
support inputs   [batch, 9] bytes
support outputs  [batch, 9] bytes
query            [batch] bytes
```

The nine support inputs must equal the qualified global order:

```text
8, 0, 64, 4, 2, 1, 32, 128, 16
```

The serializer rejects a changed order, wrong row count, mismatched batch,
wrong dtype, malformed pair or byte outside `0..255`.

Explicitly absent from model input:

```text
operator counter or key
world or worker identity
source or target node
population or depth
recurrent round
relevance or target flag
train/validation/test split
```

## Exact architecture

```text
support pair byte bits         16 features
pair projection                Linear(16 -> 48)
qualified support-slot signal  learned [9,24], zero-padded to 48
support self-attention         48 dimensions, 4 heads
support feed-forward           48 -> 64 -> 48
normalization                  parameter-free layer normalization
support pooling                arithmetic mean over nine rows
query projection               Linear(8 -> 48)
query/support fusion           Linear(96 -> 24), tanh
output                          Linear(24 -> 8), no bias
output calibration             one learned scalar
byte prediction                zero-threshold each bit, repack 0..255
```

The support-slot modulation is tied to the one globally qualified support
order. It cannot encode operator identity. It permits the shared network to
distinguish the zero row and eight basis-vector rows without adding a hidden
operator descriptor.

## Exact parameter budget

```text
pair projection                 816
query projection                432
support-slot modulation         216
support multi-head attention  9,408
support feed-forward          6,256
query/support fusion          2,328
output bit head                 192
output scale                      1
-----------------------------------
total                         19,649
```

There is no padding tensor, unused reserve, per-operator table, learned byte
embedding, learned normalization affine, recurrent state table or
population-dependent parameter.

One parameter set is reused for every worker, population and graph round.

## Contract qualification

Synthetic contract checks establish:

- exact total and component parameter arithmetic;
- exactly 17 parameter tensors;
- strict support/query serialization and byte validation;
- finite `[batch,8]` bit logits and valid decoded bytes;
- every parameter tensor receives a finite, nonzero gradient in one synthetic
  binary-cross-entropy backward pass;
- changing support context changes logits at a fixed query;
- changing query changes logits at a fixed support context;
- no optimizer, checkpoint, graph-world generator, operator counter, training
  loop or scientific execution surface exists.

The backward pass is an architecture-connectivity test. It does not update
parameters and is not training evidence.

## Scientific interpretation boundary

This architecture is deliberately not a hard-coded affine solver. It is a small
shared attention network that must learn to infer and apply the operator from
support examples. Contract tests prove information flow and budget compliance,
not successful induction.

No claim is made about trainability, unseen-operator accuracy, graph scaling or
comparison with Gate-8.

## Closed boundary

This slice contains only:

- the exact PyTorch module;
- model-input serializer;
- deterministic bit decoder;
- architecture plan and parameter audit;
- synthetic forward/backward regressions;
- architecture-only CI and this record.

It contains no optimizer, loss schedule, training episode generator, checkpoint
format, model selection rule, local-science operator loader, scientific graph
assignment key, test-world generator, population runtime, result artifact or
classifier invocation.

## Next admission boundary

The next slice must freeze the training and checkpoint-selection protocol before
any checkpoint is produced. It must bind:

- exact training/validation operator ranges already preregistered;
- three seeds;
- fixed episode count and batching;
- exact optimizer, loss, schedule and precision;
- checkpoint serialization and independent validation criteria;
- strict prohibition on scientific local/graph operators and test assignment
  key.

Training execution remains closed until that protocol qualifies.
