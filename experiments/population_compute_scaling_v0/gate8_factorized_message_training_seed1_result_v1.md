# Gate-8 v1 factorized-message seed-1 result

## Frozen outcome

```text
G8_V1_TRAINING_CHECKPOINT_ADMITTED
```

Replication execution head:

```text
31a8d115eb14d876997fb361b02258fbe3a30506
```

Qualified seed-0 result head:

```text
f259620f7d3beab2f886c76271c753e9ebf96dc9
```

Protocol, runtime and architecture bindings:

```text
protocol     a33dc123d090268a531d112251ea3ab53cb50062
runtime      333d88ac4fc52f1651741fba224e0b4605feedd3
architecture c3ab64008c816fa1eb6f9d6f8f0a1a99ed195ec8
```

## External artifact identities

```text
raw result JSON     873cacdb5965b29c59a14d74fc0df7a32c036f35aeeda2cdd4cb5ac3640a7e8e
selected checkpoint cbcae487dd7f4c695e1d6a83a61926cd43f5ccf6add1a7469c16a15697d22d07
source manifest     22a22993ebe3aff46997fd83605aed25170db6abca631e5c109d8bcc33446133
```

The uploaded checkpoint loaded with `torch.load(..., weights_only=True)`. It contains exactly 12 finite float32 tensors and 19,649 learned parameters. Its metadata binds seed 1, step 1,024, and the exact frozen architecture, runtime, and protocol heads. No checkpoint binary is committed to Git history.

## Training integrity

The result contains exactly 1,024 contiguous telemetry rows. Their world ranges cover `0..262143` without a gap or overlap, in batches of 256. The run processed 262,144 worlds and 23,767,648 supervised edge transitions.

All four frozen checkpoints were preserved and evaluated:

| Step | Admitted | Mean target | Minimum target | Exact message | Carrier | Symbol | Validation loss |
|---:|:---:|---:|---:|---:|---:|---:|---:|
| 256 | No | 0.8821614583 | 0.6250000000 | 0.9842134364 | 1.0 | 0.9842134364 | 0.3495128288 |
| 512 | Yes | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 0.0007687584 |
| 768 | Yes | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 0.0000222471 |
| 1024 | Yes | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 0.0000098894 |

The frozen selector chooses step 1,024. Steps 512, 768, and 1,024 tie on all discrete accuracies; step 1,024 has the lowest validation loss.

## Fresh validation boundary

Every candidate used exactly 512 worlds for each frozen training-regime condition, from validation indices `512..1023`:

```text
(32,4), (64,4), (64,8), (128,4), (128,8), (128,16)
```

The selected checkpoint solved all 3,072 fresh validation worlds and achieved complete local coverage:

```text
inbox codes      256 / 256
target codes     256 / 256
target carriers   16 / 16
target symbols    16 / 16
```

## Checkpoint ledger

```text
initial hidden state          65
carrier embedding            112
symbol embedding             176
transform embedding           24
GRU input weights           4095
GRU recurrent weights      12675
GRU input bias               195
GRU recurrent bias           195
carrier head                1056
symbol head                 1056
-------------------------------
total                      19649
```

## Replication-result flag clarification

The qualified replication runner writes the legacy field:

```text
seeds_1_and_2_executed = true
```

for either allowed replication seed. It denotes use of the seed-1/2 replication path, not simultaneous completion of both seeds. The immutable `seed = 1` field, checkpoint metadata, `PYTHONHASHSEED = 1`, and artifact provenance establish that this result is seed 1 only. Seed 2 remains unexecuted at this boundary.

## Closed boundaries

```text
seed 0 rerun                   false
seed 2 executed                false
scientific-test worlds         false
reference tokenizer loaded     false
reference model weights        false
reference inference            false
```

This result establishes one successful independent replication of exact learned composition on the frozen development regime. It does not yet establish the complete three-seed replication set or scientific depth/population scaling.

## Next boundary

After this exact result head qualifies, seed 2 may run from the unchanged qualified replication execution head. Scientific-test worlds remain closed until seed 2 is independently audited and permanently qualified.
