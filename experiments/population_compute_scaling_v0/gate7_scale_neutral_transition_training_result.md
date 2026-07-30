# Gate-7 scale-neutral scorer transition training result

## Status

**TRANSITION TRAINING COMPLETE — CHECKPOINTS UNBRIDGED — NO GATE-7 CAPABILITY EVIDENCE.**

The frozen three-seed transition trainer was executed locally from Git head:

`07307650b2bbbfaa09b80e40caa4419ecdda2947`

The invocation used the qualified Windows PowerShell wrapper and an idle-machine attestation on:

- NVIDIA GeForce RTX 4060 Ti;
- PyTorch `2.9.1+cu130`;
- CUDA runtime `13.0`.

The wrapper printed the exact Git head but the checkpoint payload format does not embed Git provenance. Therefore the head above is separately bound invocation provenance from the preserved terminal transcript; checkpoint identity itself is established by the exact SHA-256 and reconstructed parameter fingerprint below.

## Frozen training contract

- training seeds: `0 / 1 / 2`;
- every integer depth `6..18`;
- `1,200` steps/checkpoint;
- batch `256`;
- AdamW, learning rate `3e-4`, weight decay `1e-4`;
- gradient clip `1.0`;
- SmoothL1 loss;
- eager FP32;
- compiler, CUDA graphs and mixed precision OFF;
- `19,649` learned parameters;
- bridge CLOSED;
- high-scale Gate-7 science CLOSED.

## Bound checkpoints

| Transition checkpoint | Final loss | Mean last 50 | SHA-256 | Parameter fingerprint |
|---|---:|---:|---|---|
| T0 | 0.001698 | 0.001663 | `be1b18e22a1f713b48de8934aca1c0302f1255342684158c6ae31a7c3618a719` | `0be5c26d1056da1bf12a53be5ba5e6d1cadb7815eb117e2a77db269391c1c5aa` |
| T1 | 0.000850 | 0.001550 | `a9b712736f440168ccd86ee070b8d301be0eb834b9d17418d40c0175207cadfb` | `b9685382992cb2f94454b6faa3675d458f236e7b0c7d8399bea256bedcb02e46` |
| T2 | 0.000657 | 0.001411 | `cd7869511d07679cb1ad430743cad031299a575413e8a853ef9116d545c7480a` | `1bca7012c7350c3b3fe8c9790a65c36eea8fcd8bef3e123034817ff78673a7bb` |

All three checkpoints completed the exact frozen schedule. No checkpoint was selected, discarded, retrained, fine-tuned or ranked for admission. The complete three-checkpoint set is bound into the preregistered fresh low-scale bridge.

## Interpretation boundary

The low training losses show that the scale-neutral scorer learned its frozen supervised transition target under all three seeds. They do **not** establish that learned K16 routing beats matched hash routing, remains near global, preserves the original scorer mechanism, or scales to any high-N Gate-7 condition.

Those questions remain closed until the separately qualified bridge runner is executed. A bridge failure remains a valid transition result and cannot be repaired by checkpoint selection, extra training, changed margins or alternate bridge worlds.
