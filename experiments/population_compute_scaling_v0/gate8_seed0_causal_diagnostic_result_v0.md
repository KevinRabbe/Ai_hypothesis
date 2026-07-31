# Gate-8 seed-0 causal diagnostic result v0

## Status

`G8_SEED0_CAUSAL_DIAGNOSTIC_COMPLETE`

This record binds the completed seed-0 causal diagnostic executed from exact head:

`01199a848c9daa296bc3f9009e6aebc3a6b0357b`

Protocol head:

`0fa9ec48c31b36c90d58da827139457fd812b98c`

The complete raw result JSON remains an external evidence artifact with SHA-256:

`386ee8252bcb9ff79a161f6753bac81e215f0b0c4f86324113cc9489cc4c1c0c`

The exact recursive source manifest is committed alongside this record and has SHA-256:

`0544d62ca5c4c53c56c2090a1ea8b59a0c3e2e5c54968429e4783b7d5b461f25`

## Artifact inspection

All uploaded checkpoints loaded with `torch.load(..., weights_only=True)`, contained 15 tensors, and totaled exactly 19,649 learned parameters.

| Probe | Step | Trainable parameters | SHA-256 |
|---|---:|---:|---|
| head-only | 256 | 9,009 | `c0e014b1f9daec46a73f1895f20d51258a4f5f2e284232680737c3aac4765e29` |
| full-resume | 256 | 19,649 | `7f602f7ff7867870e8ffedc9d5920eb3421e01ccedcfed71f290245115094269` |
| full-resume | 512 | 19,649 | `0db9f10adaa78bcb6718185b7ddfe5a251b84a71ad7fb89902002012920b975c` |

Each checkpoint binds the original non-admitted seed-0 checkpoint:

`4aca6bfde7fa82cd2c1fec3613c4cc59303788616f352e8d419c90662d7b9a1b`

## Frozen causal findings

```text
activity_gate_material           = false
answer_head_material             = true
continued_optimization_effective = false
core_interference_persists       = true
frozen_core_linearly_sufficient  = false
```

These are independent findings; they are not a replacement admission decision.

## Runtime interventions

| Probe | Mean target accuracy | Minimum target accuracy |
|---|---:|---:|
| baseline | 0.4036458333333333 | 0.19921875 |
| forced activity | 0.4215494791666667 | 0.21875 |
| terminal low-four-bit message decode | 0.5755208333333334 | 0.236328125 |
| forced activity plus message decode | 0.5992838541666666 | 0.265625 |

Forcing activity improved mean target accuracy by only `0.01790364583333337`, which did not satisfy the frozen materiality rule.

Using the message channel's low four bits as the terminal answer improved mean target accuracy by `0.171875`. Therefore the learned message channel contains substantially more task-relevant answer information than the independent answer head exposes.

## Optimization probes

### Head-only final checkpoint

```text
step                     = 256
message accuracy         = 0.9549273322610294
answer accuracy          = 0.6477804744944853
activity accuracy        = 0.9965784409466911
mean target accuracy     = 0.4876302083333333
minimum target accuracy  = 0.318359375
message root invariance  = 0.9033203125
answer root invariance   = 0.77294921875
```

### Full-resume final checkpoint

```text
step                     = 512
message accuracy         = 0.9360495174632353
answer accuracy          = 0.6638039981617647
activity accuracy        = 0.9949987074908089
mean target accuracy     = 0.4527994791666667
minimum target accuracy  = 0.25
message root invariance  = 0.89697265625
answer root invariance   = 0.78955078125
```

The head-only probe produced the stronger terminal-composition result despite updating fewer parameters for fewer steps. Full-model continuation improved selected local metrics but did not satisfy the preregistered continued-optimization criterion and did not remove root-feature interference.

## Scientific interpretation

The seed-0 failure is not primarily explained by the activity gate or by a simple shortage of additional optimization steps.

The evidence supports three dominant defects:

1. The separate answer head is a materially lossy duplicate decoder.
2. The monolithic 256-way message head entangles carrier prediction with symbol transformation.
3. The shared core retains dependence on the irrelevant root-symbol feature after the root transition.

The frozen core was not judged linearly sufficient under the preregistered head-only criterion.

## Boundary

The diagnostic did not:

- run seeds 1 or 2;
- generate scientific-test worlds;
- load the reference tokenizer;
- load reference-model weights;
- perform reference inference;
- modify the original seed-0 checkpoint;
- change the original non-admission outcome.

## Decision

Do not run seeds 1 and 2 under the failed architecture.

The next pre-exposure stage is a separately preregistered Gate-8 v1 architecture contract that keeps exactly 19,649 learned parameters while testing:

- root symbol encoded directly into the initial 8-bit message;
- factorized 16-way carrier and 16-way symbol heads;
- terminal answer derived from the symbol head rather than a duplicate answer head;
- deterministic delivery in the capability gate, with sparse activity moved to a later efficiency gate;
- explicit reallocation of every freed parameter without padding.
