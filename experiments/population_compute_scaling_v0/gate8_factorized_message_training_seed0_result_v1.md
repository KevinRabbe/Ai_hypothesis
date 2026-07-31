# Gate-8 v1 factorized-message seed-0 result

## Frozen outcome

```text
G8_V1_TRAINING_CHECKPOINT_ADMITTED
```

Execution head:

```text
1b449f0ed4998e9246c86803d4473d0ac9ebdac3
```

Protocol, runtime and architecture bindings:

```text
protocol     a33dc123d090268a531d112251ea3ab53cb50062
runtime      333d88ac4fc52f1651741fba224e0b4605feedd3
architecture c3ab64008c816fa1eb6f9d6f8f0a1a99ed195ec8
```

## External artifact identities

```text
raw result JSON  1e42eb53f6446e4eeb66bbb2090c8dad7551e2098b76f289b43cf0c05975e829
selected checkpoint 3005369a4830c12baee8ffa7fedc1bed0f1888784e1043bd88f4afd2b7cddde9
source manifest  3db3284b37d4ddd7dfec03ab9fd6c0aa6193d59c0cb887fcb773927eaa13e3ac
```

The uploaded checkpoint loaded with `torch.load(..., weights_only=True)`. It contains exactly 12 tensors and 19,649 learned parameters. No checkpoint binary is committed to Git history; its immutable identity is the SHA-256 above.

## Training integrity

The result contains exactly 1,024 contiguous telemetry rows. Their world ranges cover `0..262143` without a gap or overlap, in batches of 256. The run processed 262,144 worlds and 23,767,648 supervised edge transitions.

All four frozen checkpoints were preserved and evaluated:

| Step | Admitted | Mean target | Minimum target | Exact message | Carrier | Symbol | Validation loss |
|---:|:---:|---:|---:|---:|---:|---:|---:|
| 256 | No | 0.6555989583 | 0.2402343750 | 0.9354642980 | 1.0 | 0.9354642980 | 0.5558421509 |
| 512 | Yes | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 0.0016954409 |
| 768 | Yes | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 0.0000590927 |
| 1024 | Yes | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 0.0000274653 |

The frozen selector chooses step 1,024. Steps 512, 768 and 1,024 tie on all discrete accuracies; step 1,024 has the lowest validation loss.

## Fresh validation boundary

Every candidate used exactly 512 worlds for each frozen training-regime condition, from validation indices `512..1023`:

```text
(32,4), (64,4), (64,8), (128,4), (128,8), (128,16)
```

These indices are disjoint from the v0 and causal-diagnostic validation evidence at `0..511`.

The selected checkpoint solved all 3,072 fresh validation worlds and achieved complete local coverage:

```text
inbox codes      256 / 256
target codes     256 / 256
target carriers   16 / 16
target symbols    16 / 16
```

## Checkpoint ledger

```text
initial_hidden_state          65
carrier_embedding            112
symbol_embedding             176
transform_embedding           24
GRU input weights           4095
GRU recurrent weights      12675
GRU input bias               195
GRU recurrent bias           195
carrier head                1056
symbol head                 1056
-------------------------------
total                      19649
```

## Closed boundaries

The source result records:

```text
seeds 1 and 2 executed        false
scientific-test worlds        false
reference tokenizer loaded    false
reference model weights       false
reference inference           false
```

This result admits seed 0 only. It establishes exact learned composition on the frozen development regime; it does not yet establish replication across seeds or scientific depth/population scaling.

## Next boundary

A separate qualified replication execution slice may unlock seeds 1 and 2 without changing architecture, objective, training schedule, validation indices, selection, admission thresholds or artifact format. Scientific-test worlds remain closed until all three seeds are independently audited.
