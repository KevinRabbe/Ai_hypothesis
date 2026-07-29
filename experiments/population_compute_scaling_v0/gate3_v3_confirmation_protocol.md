# Gate-3 v3 — generation-pressure confirmation protocol

## Status

**FROZEN BEFORE ANY GATE-3 v3 CONFIRMATION WORLD IS GENERATED OR INSPECTED**

This confirmation protocol is opened mechanically because Gate-3 v3 development returned the only development class permitted to proceed:

`V3_G1_ROBUST_GENERATION_PRESSURE_BENEFIT`.

Development evidence is preserved unchanged in:

`experiments/population_compute_scaling_v0/gate3_v3_generation_pressure_development_result.md`.

The fully qualified development/result lineage head used as this protocol base is:

`c9a2c82a966cbbb14697a56183213f07c6e004a3`.

No confirmation world may be generated or inspected until this protocol and its implementation qualify.

## Confirmation question

> On an untouched world namespace, does the preregistered generation-pressure advantage of persistent `L256` over capacity-binding `L64` reproduce across all three frozen learned checkpoints while learned parameters, active neural width, scheduler, per-child refinement and total learned recurrent work remain unchanged?

## Frozen checkpoints

Exactly the same three checkpoints used in development are reused. No training, fine-tuning, recalibration or replacement is allowed.

### checkpoint 0

- SHA-256: `e63748a1182472d03c540f9123c3442ce44e130519e2176681034568826db590`
- parameter fingerprint: `e4f9990f08e85393a155637cfa50b5831d779770447ec3716fc9e67946992afc`
- learned parameters: `19,649`

### checkpoint 1

- SHA-256: `8ce384627f5543fa4fe78498f9990f8214cf2a4afed0c9b734e86761ad13c989`
- parameter fingerprint: `2a57984f755ee1404fd828c2df36a5efff8f13a541b7d7c93a891014fbf4897c`
- learned parameters: `19,649`

### checkpoint 2

- SHA-256: `103826bbd9451b965eced5134d1674cb8e893f5d3b378cb828312ffdb6fc9a37`
- parameter fingerprint: `8afaf956f200f41ea914eafdd1b5f151dd303cae1552165c74e12bb8c945af02`
- learned parameters: `19,649`

## Frozen runtime / scheduler

All causal runtime mechanics remain exactly the Gate-3 v3 development mechanics:

```text
depth:                         8
hint reliability:              0.70
generation scheduler:          synchronous binary generations
scheduled parent slots/world:  223
active neural child lanes:     2
recurrent updates/child:       8
total learned updates/world:   3,568
score quantization:             unchanged Gate-3 v1 rule
compiler:                      disabled
mixed precision:               disabled
CUDA graphs:                    disabled
```

The depth-7 pressure invariant remains mandatory:

- unpruned pre-capacity frontier: exactly `128` unique hypotheses;
- stable `L64`: retain exactly `64`, expand exactly `64` final parents, execute exactly `32` matched sink slots;
- stable `L256`: retain all `128`, expand exactly `96` final parents, execute `0` sink slots.

Any violation invalidates the confirmation artifact.

## Frozen condition matrix

The complete development matrix is reproduced unchanged for each checkpoint:

```text
stable L16
stable L64
stable L256
collapsed L256
reshuffled L256
```

Three checkpoints × five conditions = **15 confirmation cells**.

## Untouched confirmation namespace

Confirmation uses a namespace that did not exist in the development generator:

```text
gate3-v3-generation-pressure-confirmation-hidden
gate3-v3-generation-pressure-confirmation-hints
gate3-v3-generation-pressure-confirmation-runtime
gate3-v3-generation-pressure-confirmation-bootstrap
```

These labels must not be used by development code.

Exactly **512 confirmation worlds** are generated once per condition/checkpoint from indices `0..511`.

The same 512 confirmation worlds are reused across capacities, controls and checkpoints for paired comparisons.

This increases statistical precision without changing per-world compute or causal mechanics.

## Runtime-visible information

Runtime receives only:

- depth `8`;
- eight public noisy hints;
- answer-independent deterministic runtime/tie-break seed.

Runtime must never receive:

- hidden solution path;
- success/failure status during execution;
- checkpoint index as a learned input;
- capacity as a learned input;
- confirmation acceptance state.

## Frozen primary and controls

For each checkpoint, reconstruct from raw paired 512-world coverage vectors:

1. **primary:** stable `L256 - L64`;
2. stable `L256 - collapsed L256`;
3. stable `L256 - reshuffled L256`;
4. secondary: stable `L64 - L16`.

Use deterministic paired bootstrap intervals with exactly **4,000 samples** from the confirmation bootstrap namespace.

## Frozen confirmation acceptance rule

Assign:

`GATE3_V3_CONFIRMED_GENERATION_PRESSURE_BENEFIT`

**only if all of the following hold:**

1. the independent confirmation auditor accepts the artifact with no errors;
2. all structural pressure/work invariants pass for every world;
3. for **each of all three frozen checkpoints**, stable `L256 - L64` has paired-bootstrap 95% CI low `> 0`;
4. for **each of all three checkpoints**, stable `L256 - collapsed L256` has CI low `> 0`;
5. for **each of all three checkpoints**, stable `L256 - reshuffled L256` has CI low `> 0`.

The lower `L64-L16` comparison is reported as secondary evidence but is **not** required for confirmation because the confirmation claim is specifically the beyond-L64 generation-pressure effect.

If any primary checkpoint fails criterion 3 while controls and artifact validity remain intact, assign:

`GATE3_V3_CONFIRMATION_NOT_ESTABLISHED`.

If artifact, structural work, or control criteria fail, assign:

`GATE3_V3_CONFIRMATION_INVALID_OR_MECHANISM_FAILED`.

No post-result threshold, world-count, capacity, scheduler or checkpoint change is permitted.

## Claims allowed if confirmed

A successful confirmation supports the narrow statement:

> Across three independently trained but frozen shared scorers and an untouched confirmation world set, a larger persistent latent hypothesis population (`L256`) improves exact no-replay search coverage over a capacity-binding smaller population (`L64`) under the preregistered generation-synchronous binary search topology, while learned parameters, active neural width and total learned recurrent work per world remain fixed.

It does **not** establish:

- AGI or general intelligence;
- arbitrary-task generalization;
- unlimited population scaling;
- superiority over every possible serial/replay/search algorithm;
- per-FLOP or per-joule superiority;
- that matched sink work is causally equivalent to productive candidate work;
- geographic/distributed-hardware benefit.

## Stop rule

After the 15 confirmation cells are inspected:

- no additional confirmation worlds;
- no replacement checkpoints;
- no added capacities;
- no scheduler changes;
- no changed bootstrap rule;
- no changed acceptance threshold.

Whatever outcome occurs is the final Gate-3 v3 confirmation evidence.
