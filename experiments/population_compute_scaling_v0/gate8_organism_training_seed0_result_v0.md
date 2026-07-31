# Gate-8 organism training seed-0 result v0

## Status

`G8_TRAINING_CHECKPOINT_NOT_ADMITTED`

This is the immutable result record for the first preregistered Gate-8 organism training execution.

The run completed normally. Non-admission is a scientific outcome, not an execution failure.

## Exact identities

```text
seed                 = 0
source head          = 6c68b51741a30229b1be23d522d0009507c806d5
architecture head    = 2afdcc9f13f138e97c7b3821cc2a5a77bd87cf0c
runtime head         = 1a2be148411bc71ba35fda12b035b724f06ec166
protocol head        = 869791e5b44089f9c79447b8ae212ce830f8496a
learned parameters   = 19,649
training worlds      = 262,144
optimizer steps      = 1,024
selected step        = 1,024
```

Uploaded source artifacts were verified before this record was created:

```text
result JSON SHA-256
5f477022ac45a80d8b05b112b3485e4112519ddef78e0bb23990a090e0cc92e2

selected checkpoint SHA-256
4aca6bfde7fa82cd2c1fec3613c4cc59303788616f352e8d419c90662d7b9a1b

source manifest SHA-256
bb814b9ebb5116f0a13ff2ce130c5ad8e32ed4bd80453ddc167143b6cbf0bb8d
```

The source manifest is reproduced in this slice for human inspection. The source-manifest SHA above binds the originally uploaded bytes; repository text normalization is not treated as byte identity.

## Frozen selector result

The selector correctly chose step 1,024 because it had the strongest frozen lexicographic selection tuple.

```text
message accuracy       = 0.9167085535386029
answer accuracy        = 0.6418421128216911
activity accuracy      = 0.9941442153033089
mean target accuracy   = 0.4036458333333333
minimum target accuracy= 0.19921875
validation loss        = 1.7918499257935976
inbox-code coverage    = 256 / 256
target-code coverage   = 256 / 256
```

Admission required every condition to reach at least 0.99 target accuracy and message accuracy to reach at least 0.995. The checkpoint therefore correctly failed admission.

## Complete-tree development results

| Population | Depth | Target accuracy | Target reach rate | Accuracy conditioned on target reached |
|---:|---:|---:|---:|---:|
| 32 | 4 | 0.521484375 | 0.95703125 | 0.5448979591836735 |
| 64 | 4 | 0.498046875 | 0.962890625 | 0.5172413793103449 |
| 64 | 8 | 0.353515625 | 0.953125 | 0.3709016393442623 |
| 128 | 4 | 0.537109375 | 0.978515625 | 0.5489021956087824 |
| 128 | 8 | 0.3125 | 0.939453125 | 0.33264033264033266 |
| 128 | 16 | 0.19921875 | 0.91015625 | 0.21888412017167383 |

Chance accuracy is 0.0625. The organism learned real local and composed capability, but errors accumulated strongly with depth.

## Checkpoint integrity

The uploaded checkpoint was loaded with `torch.load(..., weights_only=True)`.

It contains:

- the exact experiment version;
- protocol head `869791e5b44089f9c79447b8ae212ce830f8496a`;
- seed 0;
- step 1,024;
- declared parameter count 19,649;
- the expected 15-tensor state dictionary;
- an observed total of exactly 19,649 tensor parameters.

No unsafe generic pickle load was used.

## Post-hoc local transition diagnostic

This diagnostic is explicitly **not** an admission input. It generated no world, used no scientific-test data and did not modify the checkpoint. It exhaustively evaluated the finite local input table implied by the frozen architecture and transform library.

### Root-outgoing role

```text
cases                    = 128
exact message accuracy   = 0.96875
message-symbol accuracy  = 0.96875
message-carrier accuracy = 0.9921875
answer-head accuracy     = 0.7421875
activity accuracy        = 0.9765625
```

### Non-root ordinary role

```text
cases                    = 32,768
exact message accuracy   = 0.910003662109375
message-symbol accuracy  = 0.910369873046875
message-carrier accuracy = 0.9947509765625
answer-head accuracy     = 0.6312255859375
activity accuracy        = 0.994171142578125
```

### Non-root target-incoming role

```text
cases                    = 32,768
exact message accuracy   = 0.892822265625
message-symbol accuracy  = 0.893646240234375
message-carrier accuracy = 0.992950439453125
answer-head accuracy     = 0.607269287109375
activity accuracy        = 0.9969482421875
```

The carrier transition is nearly solved. Most remaining message errors are symbol-transform errors. The independent answer head is materially weaker than the message channel.

For non-root target-incoming transitions, predictions were invariant to the irrelevant public root-symbol input for only:

```text
message = 0.88427734375
answer  = 0.77587890625
```

This indicates that the frozen architecture did not fully suppress an irrelevant feature after the root transition.

## Interpretation boundary

The result supports the following diagnosis, but does not yet distinguish their relative causal weight:

1. 1,024 very-large-batch optimizer updates were insufficient to reach exact local transition precision before the cosine schedule decayed.
2. The monolithic 256-way message objective spent capacity on a carrier component that was already almost solved while symbol-transform precision remained below requirement.
3. The separate answer head duplicated the message symbol readout and remained substantially less accurate.
4. The always-positive activity target still allowed rare false-negative gates, which compounded into incomplete target reach at depth.
5. The non-root core remained partially sensitive to the irrelevant root-symbol embedding.

These are post-hoc hypotheses. No protocol parameter may be changed under the completed seed-0 result.

## Closed boundaries

The source result confirms:

```text
scientific test worlds generated = false
reference tokenizer loaded       = false
reference model weights loaded   = false
reference inference performed    = false
```

Seeds 1 and 2 remain unexecuted. Scientific-test worlds and the 1B reference remain closed.
