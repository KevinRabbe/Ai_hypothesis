# Gate-3 v3 — generation-pressure development result

## Status

**VALID DEVELOPMENT RESULT — `V3_G1_ROBUST_GENERATION_PRESSURE_BENEFIT`**

This is development evidence only. It is not confirmation.

Measured scientific head:

`f9c05833fe01583f9a32e61a9484cd5bd91e5670`

Training performed: **none**.

Confirmation opened during this run: **false**.

The exact three frozen Gate-3 v1 checkpoints were reused and SHA-256 verified.

## Frozen question

Under the preregistered generation-synchronous topology, a depth-7 frontier of 128 unique hypotheses is mechanically created before the final search phase. With learned parameters, active neural width, per-child recurrent refinement and total learned recurrent work fixed, does persistent latent capacity `L256` improve exact no-replay search coverage over `L64` when `L64` is genuinely capacity-binding?

## Structural pressure result

The preregistered geometry was satisfied:

```text
stable L16:  depth7=16,  productive=79,  sink=144
stable L64:  depth7=64,  productive=191, sink=32
stable L256: depth7=128, productive=223, sink=0
```

The shared scheduled-work budget remained exactly 223 parent-expansion slots/world with two active child lanes and eight recurrent updates/child:

`223 × 2 × 8 = 3,568 learned recurrent updates/world`.

Thus `L64` was forced to discard half of the mechanically generated 128-hypothesis frontier while `L256` could retain all 128. No learned parameters or active-neural lanes scaled with capacity.

## Exact development coverage

### Checkpoint 0

```text
stable L16      0.421875
stable L64      0.86328125
stable L256     0.96484375
collapsed L256  0.08203125
reshuffled L256 0.7265625
```

Primary `L256 - L64`:

- delta: `+0.1015625`
- paired-bootstrap 95% CI: `[0.06640625, 0.140625]`

Controls:

- stable L256 - collapsed L256: `+0.8828125`, CI `[0.84375, 0.921875]`
- stable L256 - reshuffled L256: `+0.23828125`, CI `[0.1796875, 0.30078125]`

Lower population effect:

- stable L64 - stable L16: `+0.44140625`, CI `[0.3828125, 0.50390625]`

### Checkpoint 1

```text
stable L16      0.4453125
stable L64      0.86328125
stable L256     0.95703125
collapsed L256  0.08203125
reshuffled L256 0.71484375
```

Primary `L256 - L64`:

- delta: `+0.09375`
- paired-bootstrap 95% CI: `[0.05859375, 0.12890625]`

Controls:

- stable L256 - collapsed L256: `+0.875`, CI `[0.83203125, 0.9140625]`
- stable L256 - reshuffled L256: `+0.2421875`, CI `[0.18359375, 0.30078125]`

Lower population effect:

- stable L64 - stable L16: `+0.41796875`, CI `[0.35546875, 0.4765625]`

### Checkpoint 2

```text
stable L16      0.453125
stable L64      0.86328125
stable L256     0.9609375
collapsed L256  0.08203125
reshuffled L256 0.7265625
```

Primary `L256 - L64`:

- delta: `+0.09765625`
- paired-bootstrap 95% CI: `[0.0625, 0.13671875]`

Controls:

- stable L256 - collapsed L256: `+0.87890625`, CI `[0.8359375, 0.91796875]`
- stable L256 - reshuffled L256: `+0.234375`, CI `[0.171875, 0.29296875]`

Lower population effect:

- stable L64 - stable L16: `+0.41015625`, CI `[0.34375, 0.47265625]`

## Frozen classification

The independent auditor returned:

```text
artifact_valid = true
errors = []
directional_outcome = V3_G1_ROBUST_GENERATION_PRESSURE_BENEFIT
scientific_status = DEVELOPMENT_ONLY_NO_GATE_VERDICT
```

The frozen G1 rule required all three checkpoints to have:

- `L256-L64` paired-bootstrap CI low `> 0`;
- stable-vs-collapsed CI low `> 0`;
- stable-vs-reshuffled CI low `> 0`;
- all structural pressure/work invariants passing.

All requirements passed.

## Interpretation

This development result supports the narrow mechanism that larger persistent hypothesis populations can improve exact search capability when the scheduler/task genuinely creates more simultaneous live alternatives than the smaller population can retain, while learned parameters, active neural width and total learned recurrent work remain fixed.

The contrast with Gate-3 v2 is mechanistically informative:

- v2: `L64` never became capacity-binding, so `L256-L64 = 0`;
- v3: the topology forced a 128-hypothesis frontier, `L64` discarded half of it, and `L256-L64` became positive across all three independent frozen checkpoints.

This does **not** establish AGI, arbitrary-task scaling, unlimited population scaling, superiority to every serial/replay algorithm, per-FLOP/per-joule superiority, or confirmation.

## Confirmation boundary

The preregistered development protocol permits only `V3_G1` to proceed to a **separately frozen confirmation protocol** using:

- an untouched confirmation namespace;
- the same three frozen checkpoints;
- unchanged generation-synchronous scheduler;
- unchanged depth/reliability;
- unchanged learned-work budget;
- a confirmation acceptance rule frozen before any confirmation world is generated or inspected.

No confirmation world was generated or inspected by the development run.

## Provenance

- result SHA-256: `0e0f87379cf9acf32ef799dc4bd5a6b99d6e684c7bb4e8be0c35388f079e2dfa`
- independent-audit SHA-256: `2d74260a224a5a6c107d9df29882456b5acca450dce217b28abe89e3022e11ec`
- recursive manifest SHA-256: `d749b58c9a9b010d4c10cf617034c7016cfab72386f596acc9a749bf5a742ba8`
- output root: `F:\gate3_v3_generation_pressure_development_v0`
