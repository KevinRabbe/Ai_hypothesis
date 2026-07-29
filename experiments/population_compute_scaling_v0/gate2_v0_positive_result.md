# Gate-2 v0 — Positive capability + eager-CUDA resource result

## Status

**POSITIVE_V0**

Gate-2 v0 is complete and positive under the frozen capability-confirmation and eager-CUDA resource protocols.

This result is intentionally narrow. It establishes the frozen Gate-2 question on the measured architecture and RTX 4060 Ti execution path; it does not establish general intelligence, general workload superiority, per-FLOP optimality, compiler superiority, or scaling beyond the measured protocol.

## Frozen capability half

Measurement source head:

`c2a26a17a94746ca88f29950197131689405917b`

Confirmation training seeds:

`3, 4, 5`

Frozen confirmation recipe:

- 1,000 optimizer steps per training seed;
- training batch 32;
- 512 untouched confirmation worlds per entity tier;
- full 36-cell evaluation matrix;
- deterministic 2,000-sample paired bootstrap;
- all four primary comparisons required to have `95% CI low > 0` independently for every seed;
- no pooling and no seed replacement.

Independent raw-world audit result:

```text
artifact_valid = true
capability_confirmation_passed = true
errors = []
seed 3 = pass
seed 4 = pass
seed 5 = pass
```

All 12 / 12 frozen primary confidence-interval lower bounds were strictly above zero.

Primary CI lower bounds:

| Seed | C64 W64 > W1 | C256 W256 > W1 | C256 stable > reshuffled | C256 stable > reset |
|---:|---:|---:|---:|---:|
| 3 | 0.04296875 | 0.068359375 | 0.056640625 | 0.052734375 |
| 4 | 0.03125 | 0.0546875 | 0.05078125 | 0.041015625 |
| 5 | 0.029296875 | 0.044921875 | 0.0546875 | 0.048828125 |

Capability interpretation:

> At fixed learned parameter count, fixed inspected information and fixed learned recurrent update count, larger populations of persistent runtime neural states reproducibly improved held-out delayed-associative capability. Stable locality and persistence controls support reduced state interference as the mechanism targeted by Gate 2.

## Frozen eager-CUDA resource half

Resource implementation/result line:

`agent/gate2-post-confirmation-prep-v0`

Qualified pre-resource result-record head:

`490bdedda721de9c315474a521516b0448b1a58a`

Primary hardware:

`NVIDIA GeForce RTX 4060 Ti`

Observed runtime environment:

```text
torch = 2.9.1+cu130
CUDA runtime = 13.0
GPU = NVIDIA GeForce RTX 4060 Ti
```

Frozen resource protocol:

- checkpoint: confirmation training seed 3, selected before confirmation results;
- checkpoint SHA-256: `8c5c03df82b3c43e67a51f9169d5a7e8ca8348215aeb46d72bb580a27d6bf7c2`;
- C64 widths: 1 / 4 / 16 / 64;
- C256 widths: 1 / 4 / 16 / 64 / 256;
- batch sizes: 1 / 64;
- 18 correctness-preflight cells before timing admission;
- exact decoded parallel/serial identity required;
- equal learned-update and state-bank telemetry required;
- 10 warmups + 50 CUDA-event timing trials per schedule/cell;
- deterministic parallel/serial order interleaving;
- eager CUDA only;
- compiler disabled;
- no CUDA graphs, fusion, mixed precision, architecture change or timing tuning.

Observed frozen resource result:

```text
all_preflights_passed = true
resource_frontier_passed = true

c64_w64_b1   = true
c64_w64_b64  = true
c256_w256_b1 = true
c256_w256_b64 = true
```

The guarded resource runner then invoked the independent resource auditor, which reported:

```text
Capability confirmation passed: True
Resource frontier passed: True
Overall Gate-2 v0 positive: True
```

## Gate-2 v0 verdict

The frozen decision map required both halves:

```text
capability confirmation = PASS
AND
eager-CUDA resource frontier = PASS
```

Both conditions are satisfied.

**Gate-2 v0 verdict: POSITIVE_V0.**

## What this result supports

Gate-2 v0 supports the claim that, for this frozen delayed keyed-trace workload and shared learned model:

1. increasing the population of persistent runtime neural states can improve held-out capability while learned parameters, inspected information and learned recurrent update count remain fixed;
2. stable state locality and persistence materially contribute to that improvement;
3. the simultaneous population implementation has a useful practical eager-CUDA execution frontier relative to the frozen serial persistent baseline on the RTX 4060 Ti at the preregistered endpoints.

## What this result does not support

This result does **not** establish:

- general intelligence or AGI;
- arbitrary task generalization;
- superiority to all serial architectures;
- superiority per FLOP or per joule;
- optimality of the current eager-CUDA implementation;
- compiler / CUDA-graph / fusion benefit;
- scaling beyond the tested state-population and entity-count range;
- that every workload benefits from more runtime states.

Compiler/runtime optimization remains a separate experimental variable.

## Artifact provenance known at result recording

Finalized confirmation archive SHA-256:

`bd8f748be6127d66699a339992c71c4e4bea06e9893b5cd858a58bb0824d2415`

Pre-finalization confirmation archive SHA-256:

`02dc99218f39ef9fd003821da39e1d3ea4b9176056c911eda04971db34d739d6`

Independent confirmation-audit JSON SHA-256:

`51ac0374eca1621df3f047a2eb52a8f1abc30b0b45ac9650606e3c4447318c5b`

The local resource directory was produced at:

`F:\gate2_resource_frontier_v0`

Its final archive and recursive-manifest hashes should be appended to this record after preservation. The scientific Gate-2 verdict above is based on the completed frozen runner plus independent resource auditor output; archival hashing is provenance hardening, not an additional scientific decision criterion.

## Next gate

The pre-result decision map specified that Gate 3 remains locked unless both Gate-2 halves pass.

That condition is now met.

**Gate 3 is scientifically unlocked, but its exact protocol must be frozen before any Gate-3 development result is inspected.**
