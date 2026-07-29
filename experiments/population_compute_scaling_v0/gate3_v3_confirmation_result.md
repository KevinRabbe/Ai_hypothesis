# Gate-3 v3 — generation-pressure confirmation result

## Status

**FINAL CONFIRMATION RESULT — `GATE3_V3_CONFIRMED_GENERATION_PRESSURE_BENEFIT`**

This record closes the Gate-3 v3 confirmation protocol.

Measured confirmation head:

`f58a2a4f626e39d674bacbe65effca2e7522736d`

Training performed: **none**.

Confirmation worlds: **512 untouched worlds** from the separately frozen confirmation namespace.

The exact three frozen Gate-3 v1 checkpoints were reused and SHA-256 verified.

## Frozen confirmation question

Under the unchanged generation-synchronous topology that mechanically creates a 128-hypothesis depth-7 frontier, does persistent latent capacity `L256` retain a positive exact-search advantage over `L64` on an untouched confirmation namespace while learned parameters, active neural width, scheduler and total learned recurrent work remain fixed?

## Unchanged scientific identities

Every confirmation condition used:

- depth: `8`;
- hint reliability: `0.70`;
- scheduled parent-expansion slots/world: `223`;
- active neural child lanes: `2`;
- recurrent updates/child: `8`;
- learned recurrent updates/world: `3,568`;
- learned parameters/checkpoint: `19,649`;
- no training or fine-tuning.

The preregistered structural pressure geometry held:

```text
stable L16:  depth7=16,  productive=79,  sink=144
stable L64:  depth7=64,  productive=191, sink=32
stable L256: depth7=128, productive=223, sink=0
```

Thus `L64` was forced to discard half of the 128 live depth-7 hypotheses while `L256` retained all 128, under the same total learned-work budget.

## Confirmation coverage

### Checkpoint 0

```text
stable L16      0.4375000
stable L64      0.857421875
stable L256     0.97265625
collapsed L256  0.080078125
reshuffled L256 0.765625
```

Primary `L256 - L64`:

- delta: `+0.115234375`
- paired-bootstrap 95% CI: `[0.087890625, 0.14453125]`

Controls:

- stable L256 - collapsed L256: `+0.892578125`, CI `[0.865234375, 0.91796875]`
- stable L256 - reshuffled L256: `+0.20703125`, CI `[0.16796875, 0.248046875]`

Secondary lower-population effect:

- stable L64 - stable L16: `+0.419921875`, CI `[0.37890625, 0.4609375]`

### Checkpoint 1

```text
stable L16      0.439453125
stable L64      0.857421875
stable L256     0.9687500
collapsed L256  0.080078125
reshuffled L256 0.748046875
```

Primary `L256 - L64`:

- delta: `+0.111328125`
- paired-bootstrap 95% CI: `[0.0859375, 0.138671875]`

Controls:

- stable L256 - collapsed L256: `+0.888671875`, CI `[0.861328125, 0.916015625]`
- stable L256 - reshuffled L256: `+0.220703125`, CI `[0.1796875, 0.259765625]`

Secondary lower-population effect:

- stable L64 - stable L16: `+0.41796875`, CI `[0.37890625, 0.4609375]`

### Checkpoint 2

```text
stable L16      0.41796875
stable L64      0.857421875
stable L256     0.9609375
collapsed L256  0.080078125
reshuffled L256 0.7421875
```

Primary `L256 - L64`:

- delta: `+0.103515625`
- paired-bootstrap 95% CI: `[0.078125, 0.130859375]`

Controls:

- stable L256 - collapsed L256: `+0.880859375`, CI `[0.8515625, 0.908203125]`
- stable L256 - reshuffled L256: `+0.21875`, CI `[0.1796875, 0.26171875]`

Secondary lower-population effect:

- stable L64 - stable L16: `+0.439453125`, CI `[0.396484375, 0.482421875]`

## Frozen acceptance rule

The confirmation protocol required the independent auditor to accept the artifact, all structural/work invariants to pass, and every checkpoint to have paired-bootstrap CI low `> 0` for:

1. stable `L256 - L64`;
2. stable `L256 - collapsed L256`;
3. stable `L256 - reshuffled L256`.

All preregistered acceptance conditions passed.

Independent audit:

```text
artifact_valid = true
errors = []
confirmation_outcome = GATE3_V3_CONFIRMED_GENERATION_PRESSURE_BENEFIT
scientific_status = FINAL_GATE3_V3_CONFIRMATION_EVIDENCE
```

## Confirmed scientific statement

Gate-3 v3 confirms, within this controlled no-replay generation-synchronous binary-search regime, that increasing persistent hypothesis-population capacity can improve exact search capability when the task/scheduler genuinely creates more simultaneously live alternatives than the smaller population can retain, while learned parameters, active neural width and total learned recurrent work remain fixed.

The controls further show that the observed gain depends on retaining **distinct alternatives** and preserving **candidate-specific neural continuity**.

The development/confirmation sequence is mechanistically coherent:

- Gate-3 v2: `L64` never became capacity-binding and `L256-L64 = 0`;
- Gate-3 v3 development: a forced 128-hypothesis frontier produced a positive `L256-L64` effect on all three checkpoints;
- Gate-3 v3 confirmation: the positive `L256-L64` effect replicated on 512 untouched worlds/checkpoint with all three confirmation CI lows above zero.

## Claims boundary

This result does **not** establish:

- AGI or general intelligence;
- arbitrary-task population scaling;
- unlimited useful population scaling;
- superiority to every serial/replay algorithm;
- per-FLOP or per-joule superiority;
- that matched sink work is equivalent in usefulness to productive candidate work;
- that the same mechanism transfers unchanged to open-ended real-world tasks.

## Provenance

- result SHA-256: `638f4965e015b2f6d0f709231cf5116fe17cec78006a1deec66ba66f29c07283`
- independent-audit SHA-256: `0a66bb4d2943a38d706ea7fafa581763ee4bdb81ae5ae88644fd23c51d90a1dc`
- recursive manifest SHA-256: `1b8170860fe709f12ac09db8cdc5f09c27deab2d3f2c7ec8c6edec1782495f43`
- output root: `F:\gate3_v3_generation_pressure_confirmation_v0`
