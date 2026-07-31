# Gate-8 v1 factorized-message seed-2 result

## Frozen outcome

```text
G8_V1_TRAINING_CHECKPOINT_ADMITTED
```

Replication execution head:

```text
31a8d115eb14d876997fb361b02258fbe3a30506
```

Qualified earlier result heads:

```text
seed 0  f259620f7d3beab2f886c76271c753e9ebf96dc9
seed 1  66532cb72c2bb0703e7af395ef51bbbef31d9b3b
```

Protocol, runtime and architecture bindings:

```text
protocol     a33dc123d090268a531d112251ea3ab53cb50062
runtime      333d88ac4fc52f1651741fba224e0b4605feedd3
architecture c3ab64008c816fa1eb6f9d6f8f0a1a99ed195ec8
```

## External artifact identities

```text
raw result JSON     cc9dad3bd05982ff5390a8f23bff3bfe8227c5a4c4c457e6578426b186bb6df2
selected checkpoint e1e35b3864354e8f3398497a897b6a759dfa3454a33d866de63784a323f461e4
source manifest     2df3483f63e6c31e06a51fde57e57eb773bb183d3cf71a405407748645c89ef0
```

The uploaded checkpoint loaded with `torch.load(..., weights_only=True)`. It contains exactly 12 finite float32 tensors and 19,649 learned parameters. Its metadata binds seed 2, step 1,024, and the exact frozen architecture, runtime, and protocol heads. No checkpoint binary is committed to Git history.

## Training integrity

The result contains exactly 1,024 contiguous telemetry rows. Their world ranges cover `0..262143` without a gap or overlap, in batches of 256. The run processed 262,144 worlds and 23,767,648 supervised edge transitions.

All four frozen checkpoints were preserved and evaluated:

| Step | Admitted | Mean target | Minimum target | Exact message | Carrier | Symbol | Validation loss |
|---:|:---:|---:|---:|---:|---:|---:|---:|
| 256 | No | 0.9882812500 | 0.9296875000 | 0.9988834157 | 1.0 | 0.9988834157 | 0.2109749328 |
| 512 | Yes | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 0.0004594156 |
| 768 | Yes | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 0.0000125042 |
| 1024 | Yes | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 0.0000055113 |

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

for either allowed replication seed. It denotes use of the seed-1/2 replication path, not simultaneous execution in one artifact. The immutable `seed = 2` field, checkpoint metadata, `PYTHONHASHSEED = 2`, and artifact provenance establish that this result is seed 2 only.

## Closed boundaries

```text
seed 0 rerun                   false
seed 1 rerun                   false
scientific-test worlds         false
reference tokenizer loaded     false
reference model weights        false
reference inference            false
```

## Three-seed replication outcome

Seeds 0, 1, and 2 independently produced admitted checkpoints under the same frozen architecture, objective, training schedule, validation boundary, selector, and admission thresholds. Each selected step-1,024 checkpoint achieved perfect discrete validation accuracy on all six development conditions.

This completes the training-replication prerequisite. It still does **not** establish scientific depth/population scaling: the unopened scientific-test matrix, causal ablations, and reference comparison remain separate boundaries.

## Next boundary

After this exact result head qualifies, the three-seed development result can be consolidated and the preregistered scientific evaluation path can be implemented without changing any trained checkpoint or reopening training.
